"""Authored causal placement policies for a separately versioned virtual-link study.

These policies are not published algorithm reproductions. Recall and false-component
heads retain their original base-model identities. The network estimator assumes the
current public link persists; it cannot inspect the executor's future trace. The
mission score is an eight-second support/count surrogate, not a calibrated mission
success probability. Only actually delivered attempts enter the risk ledger.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import asdict, dataclass, is_dataclass

from missionbench.schemas.contract import PrivacyLevel
from experiments.lake_causal_router import MissionCosts, forecast
from experiments.lake_rotation_study import ENSEMBLE
from experiments.mission_network_emulation import LinkSnapshot
from experiments.mission_routing_v2.features import NAMES as RISK_NAMES
from experiments.mission_routing_v2.features import encode as risk_encode
from experiments.mission_routing_v2.feedback import FeedbackInput
from experiments.mission_routing_v2.models import POOL, TILED
from experiments.mission_routing_v2.policy import plain
from experiments.mission_routing_v3.budget import Action, Continuation
from experiments.mission_routing_v3.policy import support_probability
from experiments.mission_routing_v4.risk import estimate as predict_risk
from experiments.mission_routing_v5.quality import BASE_NAMES, NAMES, encode, predict
from experiments.mission_strategy_routing.policy import log_count_survival
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.scenario import digest
from experiments.v4_gazebo_pilot.timing import seconds_to_ns

FIXED_LOCAL_POLICIES = {"fixed_local_tiled": TILED} | {
    "fixed_local_" + base: base for base in POOL if base != TILED
}

POLICIES = (
    "fixed_local_tiled",
    "fixed_remote_union",
    "snapshot_quality_2s",
    "mission_evidence_network",
    "mission_evidence_network_blind",
    "mission_evidence_local_only",
    *(name for name in FIXED_LOCAL_POLICIES if name != "fixed_local_tiled"),
)


def _nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


@dataclass(frozen=True)
class ActionDescriptor:
    """Placement and cost metadata, separate from immutable prediction identity."""

    action_id: str
    source_config_id: str
    placement: str
    source_signature: str
    latency_s: float
    energy_j: float
    load_latency_s: float = 0.0
    load_energy_j: float = 0.0
    memory_mb: float = 0.0

    def __post_init__(self):
        if not all(
            isinstance(v, str) and v
            for v in (self.action_id, self.source_config_id, self.source_signature)
        ) or self.placement not in ("local", "remote"):
            raise ValueError("Explicit action/base identity and placement required")
        if not all(
            _nonnegative(getattr(self, name))
            for name in ("latency_s", "energy_j", "load_latency_s", "load_energy_j", "memory_mb")
        ):
            raise ValueError("Finite nonnegative action costs required")
        if self.placement == "remote" and self.energy_j != 0:
            raise ValueError("Remote server energy is excluded from UAV energy")


@dataclass(frozen=True)
class PolicyCosts:
    mission: MissionCosts
    timeout_s: float = 8.0
    encode_s: float = 0.02
    decode_s: float = 0.01
    codec_power_w: float = 15.0
    tx_power_w: float = 2.0
    rx_power_w: float = 2.0
    wait_power_w: float = 0.5
    quality_budget_s: float = 2.0
    horizon_s: float = 8.0
    blind_upload_bps: int = 50_000_000
    blind_download_bps: int = 100_000_000
    blind_rtt_ns: int = 30_000_000

    def __post_init__(self):
        if not isinstance(self.mission, MissionCosts):
            raise ValueError("Explicit MissionCosts required")
        for name, value in vars(self).items():
            if name != "mission" and not _nonnegative(value):
                raise ValueError("Finite nonnegative policy settings required")
        if min(self.timeout_s, self.quality_budget_s, self.horizon_s) <= 0:
            raise ValueError("Positive timeout, quality budget and horizon required")
        for name in ("blind_upload_bps", "blind_download_bps", "blind_rtt_ns"):
            if type(getattr(self, name)) is not int:
                raise ValueError("Integer nominal link settings required")
        if min(self.blind_upload_bps, self.blind_download_bps) <= 0:
            raise ValueError("Positive nominal link rates required")


@dataclass(frozen=True)
class AttemptOutcome:
    """Public finished-attempt feedback; contains no evaluator correctness fields."""

    action_id: str
    capture_s: float
    completion_s: float
    delivered: bool
    status: str

    def __post_init__(self):
        if (
            not self.action_id
            or not all(_nonnegative(v) for v in (self.capture_s, self.completion_s))
            or self.completion_s < self.capture_s
        ):
            raise ValueError("Valid completed attempt chronology required")
        if type(self.delivered) is not bool or not isinstance(self.status, str) or not self.status:
            raise ValueError("Explicit attempt delivery/status required")
        if self.delivered != (self.status == "completed"):
            raise ValueError("Only completed attempts may deliver a prediction")


@dataclass(frozen=True)
class NetworkFeedbackInput(FeedbackInput):
    public_link: LinkSnapshot | None = None
    current_payload_bytes: int = 0
    response_bytes: int = 0
    last_attempt: AttemptOutcome | None = None


def estimate_action(action, link, payload_bytes, response_bytes, costs):
    """Prediction from one snapshot, without hidden trace, future boundary or masks.

    PNG preparation has already been paid before policy observation for ALL
    actions. Returns inference/response-decoding/transport costs only: router,
    loading and successful tracking are added by the policy. A forecast exceeding
    the transport
    timeout is inadmissible, rather than pretending its timeout delivers evidence.
    """
    for n in (payload_bytes, response_bytes):
        if type(n) is not int or n < 0:
            raise ValueError("Nonnegative integer payload byte counts required")
    if action.placement == "local":
        return dict(
            latency_s=action.latency_s,
            energy_j=action.energy_j,
            communication_mb=0.0,
            transport_s=0.0,
            predicted_deliverable=True,
        )
    if not isinstance(link, LinkSnapshot):
        raise ValueError("Only a current LinkSnapshot is accepted")
    if type(link.outage) is not bool or any(
        type(v) is not int or v < 0
        for v in (link.timestamp_ns, link.upload_bps, link.download_bps, link.rtt_ns)
    ):
        raise ValueError("Valid integer current-link fields required")
    if link.outage or link.upload_bps <= 0 or link.download_bps <= 0:
        return dict(
            latency_s=None,
            energy_j=None,
            communication_mb=(payload_bytes + response_bytes) / 1e6,
            transport_s=None,
            predicted_deliverable=False,
        )
    upload_ns = (payload_bytes * 8 * 1_000_000_000 + link.upload_bps - 1) // link.upload_bps
    download_ns = (response_bytes * 8 * 1_000_000_000 + link.download_bps - 1) // link.download_bps
    service_ns = seconds_to_ns(action.latency_s, allow_zero=True)
    transport_ns = upload_ns + download_ns + link.rtt_ns + service_ns
    transport = transport_ns / 1e9
    return dict(
        latency_s=transport + costs.decode_s,
        energy_j=costs.decode_s * costs.codec_power_w
        + upload_ns / 1e9 * costs.tx_power_w
        + download_ns / 1e9 * costs.rx_power_w
        + (link.rtt_ns + service_ns) / 1e9 * costs.wait_power_w,
        communication_mb=(payload_bytes + response_bytes) / 1e6,
        transport_s=transport,
        predicted_deliverable=transport_ns <= seconds_to_ns(costs.timeout_s),
    )


class Policy:
    def __init__(self, name, actions, base_specs, quality_profile, risk_profile, costs):
        if name not in POLICIES or not isinstance(costs, PolicyCosts):
            raise ValueError("Declared network policy and PolicyCosts required")
        specs = {s.config_id: s for s in base_specs}
        if not specs or len(specs) != len(base_specs):
            raise ValueError("Unique nonempty base model pool required")
        if quality_profile["schema_version"] != "conditional_recall_v5" or quality_profile[
            "feature_names"
        ] != list(NAMES):
            raise ValueError("Wrong recall schema")
        if risk_profile["schema_version"] != "mission_false_component_risk_v1" or risk_profile[
            "feature_names"
        ] != list(RISK_NAMES):
            raise ValueError("Wrong risk schema")
        for profile in (quality_profile, risk_profile):
            if set(profile["models"]) != set(specs):
                raise ValueError("Profile must use exactly the original base model IDs")
            if any(
                profile["models"][c]["signature"] != model_signature(s) for c, s in specs.items()
            ):
                raise ValueError("Wrong original model signature")
        self.actions = {a.action_id: a for a in actions}
        pairs = {(a.source_config_id, a.placement) for a in actions}
        if (
            len(self.actions) != len(actions)
            or len(pairs) != len(actions)
            or pairs != {(c, p) for c in specs for p in ("local", "remote")}
        ):
            raise ValueError("Exactly one local and remote action per base model required")
        if any(a.source_signature != model_signature(specs[a.source_config_id]) for a in actions):
            raise ValueError("Placement action changed original model signature")
        self.name, self.costs = name, costs
        self.quality = {c: copy.deepcopy(m["base"]) for c, m in quality_profile["models"].items()}
        self.risk = {c: copy.deepcopy(m["risk"]) for c, m in risk_profile["models"].items()}
        self.profile_sha256 = digest(quality_profile)
        self.risk_profile_sha256 = digest(risk_profile)
        self.ledger, self.last_decision = {}, None
        self._previous, self._input_sha, self._budget_key, self._budget = None, None, None, None
        fixed = {name: (base, "local") for name, base in FIXED_LOCAL_POLICIES.items()}
        fixed["fixed_remote_union"] = (ENSEMBLE, "remote")
        self.fixed_action = None
        if name in fixed:
            self.fixed_action = next(
                (a.action_id for a in actions if (a.source_config_id, a.placement) == fixed[name]),
                None,
            )
            if self.fixed_action is None:
                raise ValueError("Fixed reference base model is missing")

    def _validate(self, o):
        now, index = o.state.current_time_s, o.state.frame_id
        capture = index * self.costs.mission.interval_s
        if (
            type(index) is not int
            or index < 0
            or abs(now - capture - self.costs.encode_s) > 1e-7
            or capture >= self.costs.mission.duration_s
        ):
            raise ValueError("Observation off the post-preparation capture clock")
        if set(o.configs.ids()) != set(self.actions):
            raise ValueError("Runtime placement action pool changed")
        if not isinstance(
            o.public_link, LinkSnapshot
        ) or o.public_link.timestamp_ns != seconds_to_ns(now, allow_zero=True):
            raise ValueError("A current-only public LinkSnapshot is required")
        for n in (o.current_payload_bytes, o.response_bytes):
            if type(n) is not int or n < 0:
                raise ValueError("Integer payload byte counts required")
        h = o.evidence_history
        if (
            h["counts_are_ground_truth"]
            or h["available_at_s"] != now
            or (h["last_completion_s"] is not None and h["last_completion_s"] > now)
            or any(t > now for track in h["tracks"] for t in track["capture_times_s"])
        ):
            raise ValueError("Only completed prediction history is allowed")
        if o.last_attempt is not None and (
            not isinstance(o.last_attempt, AttemptOutcome) or o.last_attempt.completion_s > now
        ):
            raise ValueError("Attempt feedback must already be completed")
        # Public values, not scenario IDs, labels, future links or model masks.
        state = asdict(o.state) if is_dataclass(o.state) else vars(o.state)
        contract = asdict(o.contract) if is_dataclass(o.contract) else vars(o.contract)
        resources = asdict(o.resources) if is_dataclass(o.resources) else vars(o.resources)
        return digest(
            dict(
                state=plain(state),
                history=plain(h),
                intent=plain(o.intent_contract),
                contract=plain(contract),
                resources=plain(resources),
                tracking=plain(o.tracking),
                link=o.public_link.to_dict(),
                payload_bytes=o.current_payload_bytes,
                response_bytes=o.response_bytes,
                last_attempt=asdict(o.last_attempt) if o.last_attempt else None,
                rgb_sha256=hashlib.sha256(o.rgb.tobytes()).hexdigest(),
            )
        )

    def _completed_ledger(self, o):
        before = dict(self.ledger)
        if self._previous is None:
            if o.last_attempt is not None:
                raise ValueError("A fresh policy cannot inherit an unrecorded attempt")
            return before
        previous = self._previous
        a = o.last_attempt
        if (
            a is None
            or a.action_id != previous["action_id"]
            or a.capture_s != previous["capture_s"]
            or o.state.frame_id <= previous["frame_id"]
            or o.state.current_config_id != a.action_id
        ):
            raise ValueError("Feedback must identify the previous selected attempt")
        expected = a.completion_s if a.delivered else previous["last_delivery_s"]
        if o.evidence_history["last_completion_s"] != expected:
            raise ValueError("Only delivered attempts may update prediction history")
        if a.delivered:
            for pid in previous["phase_ids"]:
                before[pid] = before.get(pid, 0.0) + previous["rate"]
        return before

    def select(self, o):
        public_sha = self._validate(o)
        if self._previous is not None and o.state.frame_id == self._previous["frame_id"]:
            if public_sha != self._input_sha:
                raise ValueError("Repeated decision has different public input")
            return self.last_decision["selected_config_id"]
        before = self._completed_ledger(o)
        costs, mission = self.costs, self.costs.mission
        now, index = o.state.current_time_s, o.state.frame_id
        capture = index * mission.interval_s
        history, tracking = o.evidence_history, o.tracking["cost"]
        phases = [
            p
            for p in o.intent_contract["mission"]["phases"]
            if p["start_s"] <= capture < p["end_s"]
        ]
        features, risk_features = encode(o.rgb)[: len(BASE_NAMES)], risk_encode(o.rgb)
        link = o.public_link
        if self.name == "mission_evidence_network_blind":
            link = LinkSnapshot(
                link.timestamp_ns,
                costs.blind_upload_bps,
                costs.blind_download_bps,
                costs.blind_rtt_ns,
                False,
            )
        legal = [
            a
            for a in self.actions.values()
            if (a.placement == "local" or o.contract.privacy_level == PrivacyLevel.REMOTE_ALLOWED)
            and (
                o.resources.memory_capacity_mb is None
                or a.memory_mb + tracking["memory_mb"] <= o.resources.memory_capacity_mb
            )
        ]
        if self.name == "mission_evidence_local_only":
            legal = [a for a in legal if a.placement == "local"]
        local = [
            Action(
                a.action_id,
                costs.encode_s + a.latency_s + mission.router_latency_s + tracking["latency_s"],
                costs.encode_s * costs.codec_power_w
                + a.energy_j
                + mission.router_energy_j
                + tracking["energy_j"],
                a.load_latency_s,
                a.load_energy_j,
            )
            for a in legal
            if a.placement == "local"
        ]
        if not local:
            raise ValueError("At least one legal local continuation is required")
        key = (tuple(local), o.contract.deadline_s)
        if key != self._budget_key:
            self._budget = Continuation(mission, local, o.contract.deadline_s)
            self._budget_key = key
        available = (o.state.battery_frac - o.contract.min_final_battery_frac) * mission.capacity_j
        candidates = []
        for a in legal:
            estimate = estimate_action(a, link, o.current_payload_bytes, o.response_bytes, costs)
            quality, outside = predict(self.quality[a.source_config_id], features)
            rate = predict_risk(self.risk[a.source_config_id], risk_features)
            switched = o.state.current_config_id != a.action_id
            latency = estimate["latency_s"]
            slots, required, incremental, details, terms = [], math.inf, math.inf, [], []
            if latency is not None and estimate["predicted_deliverable"]:
                paid_latency = (
                    costs.encode_s + latency + mission.router_latency_s + tracking["latency_s"]
                )
                slots = forecast(mission, index, paid_latency, a.load_latency_s if switched else 0)
                completion_ns, next_index, _ = mission.schedule().complete(
                    index,
                    seconds_to_ns(
                        paid_latency + (a.load_latency_s if switched else 0), allow_zero=True
                    ),
                )
                final = mission.schedule().at(next_index) >= mission.schedule().end_ns
                boundary = (
                    max(mission.duration_s, completion_ns / 1e9)
                    if final
                    else mission.schedule().at(next_index) / 1e9
                )
                incremental = (
                    (boundary - now) * mission.flight_power_w
                    + estimate["energy_j"]
                    + mission.router_energy_j
                    + tracking["energy_j"]
                    + (a.load_energy_j if switched else 0)
                )
                previous_local = a.action_id if a.placement == "local" else None
                tail = (
                    0
                    if final
                    else min(
                        self._budget.required(next_index, previous_local, item.config_id)
                        for item in local
                    )
                )
                required = (
                    incremental + tail if completion_ns / 1e9 <= o.contract.deadline_s else math.inf
                )
                for p in phases:
                    times = [
                        t
                        for t, end in slots
                        if t < min(p["end_s"], capture + costs.horizon_s) and end <= p["deadline_s"]
                    ]
                    args = (
                        quality,
                        p["min_hits"],
                        p["min_evidence_span_s"],
                        o.tracking["configuration"]["max_gap_s"],
                    )
                    discovery = support_probability((), times, *args)
                    pending = []
                    for track in history["tracks"]:
                        hits = [
                            t for t in track["capture_times_s"] if p["start_s"] <= t < p["end_s"]
                        ]
                        if hits and not (
                            len(hits) >= p["min_hits"]
                            and hits[-1] - hits[0] >= p["min_evidence_span_s"]
                        ):
                            pending.append(support_probability(hits, times, *args))
                    confirmation = sum(pending) / len(pending) if pending else 0.0
                    # Full remaining phase risk; short horizon evidence. Both are
                    # conditional on a constant current RGB/link, not future truth.
                    count = sum(p["start_s"] <= t < p["end_s"] for t, _ in slots)
                    projected = before.get(p["phase_id"], 0) + count * rate
                    survival = log_count_survival(projected, p["max_false_components"])
                    utility = p["weight"] * (discovery + confirmation)
                    if utility > 0:
                        terms.append(math.log(utility) + survival)
                    details.append(
                        dict(
                            phase_id=p["phase_id"],
                            captures=times,
                            discovery_support=discovery,
                            pending_tracks=len(pending),
                            confirmation_support=confirmation,
                            projected_false_components=projected,
                            log_count_survival=survival,
                        )
                    )
            score = None
            if terms:
                peak = max(terms)
                score = peak + math.log(sum(math.exp(t - peak) for t in terms))
            first = slots[0][1] if slots else None
            feasible = (
                required <= available + 1e-8
                and estimate["communication_mb"] + o.state.cumulative_communication_mb
                <= o.contract.communication_budget_mb + 1e-10
            )
            candidates.append(
                dict(
                    action_id=a.action_id,
                    source_config_id=a.source_config_id,
                    placement=a.placement,
                    quality=quality,
                    estimated_fp_per_capture=rate,
                    out_of_range_feature_fraction=outside,
                    estimate=estimate,
                    first_completion_s=first,
                    predicted_paid_latency_s=first - capture if first is not None else None,
                    projected_energy_j=required if math.isfinite(required) else None,
                    incremental_energy_j=incremental if math.isfinite(incremental) else None,
                    feasible=feasible,
                    mission_log_score=score,
                    phases=details,
                )
            )

        def energy(c):
            return c["projected_energy_j"] if c["projected_energy_j"] is not None else math.inf

        fallback = None
        if self.fixed_action is not None:
            selected = next((c for c in candidates if c["action_id"] == self.fixed_action), None)
            if selected is None:
                raise ValueError("Fixed reference action is not legal")
        else:
            feasible = [c for c in candidates if c["feasible"]]
            if not feasible:
                selected = min(
                    (c for c in candidates if c["placement"] == "local"),
                    key=lambda c: (energy(c), c["action_id"]),
                )
                fallback = "minimum_projected_energy_local_no_feasible_action"
            elif self.name == "snapshot_quality_2s":
                timely = [
                    c for c in feasible if c["predicted_paid_latency_s"] <= costs.quality_budget_s
                ]
                if timely:
                    selected = max(timely, key=lambda c: (c["quality"], -energy(c), c["action_id"]))
                else:
                    selected = min(
                        feasible,
                        key=lambda c: (c["predicted_paid_latency_s"], energy(c), c["action_id"]),
                    )
                    fallback = "fastest_feasible_no_action_within_quality_budget"
            else:
                selected = max(
                    feasible,
                    key=lambda c: (
                        c["mission_log_score"] if c["mission_log_score"] is not None else -math.inf,
                        -energy(c),
                        c["action_id"],
                    ),
                )
        self.last_decision = dict(
            schema_version="mission_network_policy_v1",
            policy=self.name,
            selected_config_id=selected["action_id"],
            source_config_id=selected["source_config_id"],
            uses_ground_truth=False,
            uses_future_link=False,
            uses_network=self.fixed_action is None
            and self.name
            not in (
                "mission_evidence_network_blind",
                "mission_evidence_local_only",
            ),
            uses_intent=self.name.startswith("mission_evidence"),
            uses_history=self.name.startswith("mission_evidence"),
            profile_sha256=self.profile_sha256,
            risk_profile_sha256=self.risk_profile_sha256,
            quality_variant="base_recall",
            score_is_calibrated_mission_probability=False,
            count_survival_assumes_independent_errors=True,
            resource_guard_is_guarantee=False,
            resource_guard="current_snapshot_cost_plus_minimum_energy_local_continuation",
            applied_resource_guard=self.fixed_action is None,
            estimator_link=link.to_dict(),
            common_preparation_already_paid=True,
            current_capture_s=capture,
            settings=asdict(costs),
            available_energy_j=available,
            fallback=fallback,
            risk_ledger_before=dict(self.ledger),
            risk_ledger_after=dict(before),
            risk_ledger_is_observed_error=False,
            ledger_commits_delivered_attempts_only=True,
            completed_attempt=asdict(o.last_attempt) if o.last_attempt else None,
            history=plain(history),
            candidates=candidates,
        )
        self.ledger = before
        self._previous = dict(
            action_id=selected["action_id"],
            capture_s=capture,
            frame_id=index,
            phase_ids=[p["phase_id"] for p in phases],
            rate=selected["estimated_fp_per_capture"],
            last_delivery_s=history["last_completion_s"],
        )
        self._input_sha = public_sha
        return selected["action_id"]
