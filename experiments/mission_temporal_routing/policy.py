"""Causal learned temporal support, mission risk, and matched input ablations."""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

from experiments.lake_causal_router import forecast
from experiments.lake_intent_contract import immutable
from experiments.mission_routing_v2.features import NAMES, encode
from experiments.mission_routing_v2.policy import plain
from experiments.mission_routing_v4.risk import estimate, phase_projection
from experiments.mission_routing_v5.evidence import Policy as EvidencePolicy
from experiments.mission_strategy_routing.policy import log_count_survival
from experiments.mission_temporal_routing.temporal import support
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.scenario import digest

STRATEGIES = (
    "temporal_only",
    "temporal_mission",
    "temporal_no_repetition",
    "temporal_no_image",
    "temporal_no_history",
)


def repetition_blind(o):
    """Hide mission-specific repetition, retain phase deadlines and safety limits."""
    intent = plain(o.intent_contract)
    for phase in intent["mission"]["phases"]:
        phase.update(min_hits=2, min_evidence_span_s=2)
    fields = (
        "state",
        "contract",
        "configs",
        "resources",
        "rgb",
        "tracking",
        "evidence_history",
        "confirmed_reports_so_far",
    )
    return SimpleNamespace(**{k: getattr(o, k) for k in fields}, intent_contract=immutable(intent))


class Policy(EvidencePolicy):
    def __init__(self, profile, specs, costs, risk_profile, temporal_profile, *, strategy):
        if strategy not in STRATEGIES:
            raise ValueError("Unknown temporal strategy")
        if temporal_profile["schema_version"] != "temporal_support_v1":
            raise ValueError("Wrong temporal profile")
        if temporal_profile["feature_names"] != list(NAMES):
            raise ValueError("Wrong temporal feature schema")
        if risk_profile["schema_version"] != "mission_false_component_risk_v1":
            raise ValueError("Wrong risk profile")
        if risk_profile["feature_names"] != list(NAMES):
            raise ValueError("Wrong risk features")
        for p in (risk_profile, temporal_profile):
            if set(p["models"]) != {s.config_id for s in specs}:
                raise ValueError("Wrong model pool")
            for s in specs:
                if p["models"][s.config_id]["signature"] != model_signature(s):
                    raise ValueError("Wrong model signature")
        mode = {
            "temporal_no_image": "no_context_v3",
            "temporal_no_history": "no_history_v3",
            "temporal_only": "no_history_v3",
        }.get(strategy, "confirmation")
        super().__init__(profile, specs, costs, mode)
        self.strategy = strategy
        self.temporal_models = copy.deepcopy(temporal_profile["models"])
        self.risk_models = copy.deepcopy(risk_profile["models"])
        self.temporal_sha, self.risk_sha = digest(temporal_profile), digest(risk_profile)
        self.ledger, self.decisions = {}, 0
        self._previous, self._input_sha = None, None

    def select(self, observation):
        o = (
            repetition_blind(observation)
            if self.strategy == "temporal_no_repetition"
            else observation
        )
        super().select(o)
        base = self.last_decision
        now, index = o.state.current_time_s, o.state.frame_id
        public_sha = digest(
            dict(
                decision=base,
                intent=plain(o.intent_contract),
                current_config_id=o.state.current_config_id,
                index=index,
                now=now,
            )
        )
        if self._previous is not None:
            prev_index, config, completion, trace = self._previous
            if index == prev_index:
                if public_sha != self._input_sha:
                    raise ValueError("Repeated decision has different public input")
                self.last_decision = copy.deepcopy(trace)
                return config
            if (
                index < prev_index
                or now < completion
                or o.state.current_config_id != config
                or o.evidence_history["last_completion_s"] is None
                or abs(o.evidence_history["last_completion_s"] - completion) > 1e-7
            ):
                raise ValueError("Previous selected execution must complete before a decision")
        phases = {
            p["phase_id"]: p
            for p in o.intent_contract["mission"]["phases"]
            if p["start_s"] <= now < p["end_s"]
        }
        features = encode(o.rgb)
        context = self.strategy != "temporal_no_image"
        before = dict(self.ledger)
        candidates = copy.deepcopy(base["candidates"])
        for c in candidates:
            config = c["config_id"]
            action = self._budget.actions[config]
            rate = estimate(self.risk_models[config]["risk"], features, context=context)
            slots = forecast(
                self.costs,
                index,
                action.latency_s,
                action.load_latency_s if o.state.current_config_id != config else 0,
            )
            projections = [
                phase_projection(p, now, slots, rate, before.get(pid, 0))
                for pid, p in phases.items()
            ]
            survival = {
                r["phase_id"]: log_count_survival(
                    r["projected_false_components"], r["max_false_components"]
                )
                for r in projections
            }
            terms = []
            for detail in c["phases"]:
                p = phases[detail["phase_id"]]
                offsets = tuple(t - now for t in detail["captures"])
                learned = support(
                    self.temporal_models[config],
                    features,
                    p["min_hits"],
                    p["min_evidence_span_s"],
                    offsets,
                    context=context,
                )
                confirmation = detail["confirmation_support"]
                value = p["weight"] * (learned + confirmation)
                risk_term = survival[p["phase_id"]] if self.strategy != "temporal_only" else 0
                if value > 0:
                    terms.append(math.log(value) + risk_term)
                detail.update(learned_discovery_support=learned, temporal_offsets_s=offsets)
            if terms:
                peak = max(terms)
                score = peak + math.log(sum(math.exp(v - peak) for v in terms))
            else:
                score = None
            c.update(
                temporal_log_score=score,
                estimated_fp_per_capture=rate,
                risk_phases=projections,
                log_count_survival_by_phase=survival,
            )

        def energy(c):
            return c["projected_energy_j"] if c["projected_energy_j"] is not None else math.inf

        feasible = [c for c in candidates if c["feasible"]]
        selected = (
            max(
                feasible,
                key=lambda c: (
                    c["temporal_log_score"] if c["temporal_log_score"] is not None else -math.inf,
                    -energy(c),
                    c["config_id"],
                ),
            )
            if feasible
            else min(candidates, key=lambda c: (energy(c), c["config_id"]))
        )
        for pid in phases:
            self.ledger[pid] = before.get(pid, 0) + selected["estimated_fp_per_capture"]
        trace = dict(base)
        trace.update(
            schema_version="mission_temporal_routing_v1",
            strategy=self.strategy,
            temporal_profile_sha256=self.temporal_sha,
            risk_profile_sha256=self.risk_sha,
            candidates=candidates,
            selected_config_id=selected["config_id"],
            risk_ledger_before=before,
            risk_ledger_after=dict(self.ledger),
            decision_count=self.decisions,
            uses_temporal_learning=True,
            uses_mission_repetition=self.strategy != "temporal_no_repetition",
            uses_risk_ledger=self.strategy != "temporal_only",
            uses_context=context,
            uses_history=self.strategy not in ("temporal_only", "temporal_no_history"),
            learned_score_is_mission_probability=False,
            resource_fallback=None if feasible else "minimum_energy_no_feasible_action",
        )
        # Serialization-normalized trace permits byte-independent replay comparison.
        trace = plain(trace)
        self.last_decision, self._input_sha = trace, public_sha
        self._previous = (
            index,
            selected["config_id"],
            selected["first_completion_s"],
            copy.deepcopy(trace),
        )
        self.decisions += 1
        return selected["config_id"]
