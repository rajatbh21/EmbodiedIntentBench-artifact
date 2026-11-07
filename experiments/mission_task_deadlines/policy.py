"""Causal deadline-aware reference and fixed controls; not a published protocol.

The adaptive reference filters predicted per-call deadline, current-link,
privacy, memory, bytes and reserve feasibility. It ranks eligible actions using
RGB-only development recall/risk heads and public intent evidence requirements.
The ranking score is an authored heuristic, not mission-success probability.
Forecasts cannot inspect the private future link or evaluator target labels.
No candidate is replaced with an infeasible 'fastest' fallback: abstention is
explicit and its preparation/routing cost is still charged by the executor.
"""

from __future__ import annotations

import copy
import math

from experiments.mission_network_emulation import LinkSnapshot
from experiments.mission_network_policies import ActionDescriptor, PolicyCosts, estimate_action
from experiments.mission_routing_v2.features import NAMES, encode
from experiments.mission_routing_v4.risk import estimate as predict_risk
from experiments.mission_routing_v5.quality import NAMES as QUALITY_NAMES
from experiments.mission_routing_v5.quality import predict
from experiments.mission_task_deadlines.contract import (
    DeadlineProfile,
    TaskDeadline,
    TaskKind,
    assess_completion,
)
from experiments.v4_gazebo_pilot.timing import seconds_to_ns

FIXED_ACTIONS = {
    "fixed_local_fast": "CFG_YOLO11S_960_ROT0",
    "fixed_local_heavy": "CFG_MASKRCNN_1280_ROT0_90",
    "fixed_remote_union": "REMOTE__CFG_MASKRCNN_1280_ROT0_90",
}
POLICIES = (*FIXED_ACTIONS, "deadline_aware")


def _finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


class DeadlineAwarePolicy:
    def __init__(
        self,
        name,
        actions,
        costs,
        profile,
        quality_profile,
        risk_profile,
        *,
        memory_capacity_mb=4096,
        downstream_memory_mb=68,
        downstream_latency_ns=150_000_000,
        downstream_energy_j=2.1,
    ):
        if name not in POLICIES or not isinstance(costs, PolicyCosts):
            raise ValueError("Declared policy and typed costs required")
        if not isinstance(profile, DeadlineProfile):
            raise ValueError("Typed invocation deadline profile required")
        actions = tuple(actions)
        if not actions or any(not isinstance(a, ActionDescriptor) for a in actions):
            raise ValueError("Nonempty typed action pool required")
        self.actions = {a.action_id: a for a in actions}
        if len(self.actions) != len(actions) or not any(a.placement == "local" for a in actions):
            raise ValueError("Unique actions with a local continuation required")
        source_ids = {a.source_config_id for a in actions}
        if (
            quality_profile.get("schema_version") != "conditional_recall_v5"
            or quality_profile.get("feature_names") != list(QUALITY_NAMES)
            or risk_profile.get("schema_version") != "mission_false_component_risk_v1"
            or risk_profile.get("feature_names") != list(NAMES)
        ):
            raise ValueError("Wrong development-head schema")
        for document in (quality_profile, risk_profile):
            if set(document["models"]) != source_ids or any(
                document["models"][a.source_config_id]["signature"] != a.source_signature
                for a in actions
            ):
                raise ValueError("Prediction identity differs from the development profile")
        if name in FIXED_ACTIONS and FIXED_ACTIONS[name] not in self.actions:
            raise ValueError("Required fixed reference is absent")
        if (
            any(
                not _finite_nonnegative(v)
                for v in (memory_capacity_mb, downstream_memory_mb, downstream_energy_j)
            )
            or type(downstream_latency_ns) is not int
            or downstream_latency_ns < 0
        ):
            raise ValueError("Finite configured downstream resource assumptions required")
        self.name, self.costs, self.profile = name, costs, profile
        self.memory_capacity_mb, self.downstream_memory_mb = (
            memory_capacity_mb,
            downstream_memory_mb,
        )
        self.downstream_latency_ns, self.downstream_energy_j = (
            downstream_latency_ns,
            downstream_energy_j,
        )
        # Strip training lineage/labels. The running policy retains coefficients only.
        self.quality = {k: copy.deepcopy(v["base"]) for k, v in quality_profile["models"].items()}
        self.risk = {k: copy.deepcopy(v["risk"]) for k, v in risk_profile["models"].items()}

    def select(
        self,
        *,
        release_ns,
        current_action_id,
        battery_frac,
        cumulative_communication_mb,
        link,
        payload_bytes,
        response_bytes,
        intent,
        rgb,
    ):
        request = TaskDeadline(TaskKind.PERCEPTION, release_ns, self.profile.perception_ns)
        costs, mission = self.costs, self.costs.mission
        now_ns = release_ns + seconds_to_ns(costs.encode_s, allow_zero=True)
        if not isinstance(link, LinkSnapshot) or link.timestamp_ns != now_ns:
            raise ValueError("Only the current post-preparation LinkSnapshot is admissible")
        if (
            not _finite_nonnegative(battery_frac)
            or battery_frac > 1
            or not _finite_nonnegative(cumulative_communication_mb)
        ):
            raise ValueError("Finite public battery/byte state required")
        if current_action_id is not None and current_action_id not in self.actions:
            raise ValueError("Unknown resident configuration")
        # Refuse an evaluator document instead of accidentally making GT available.
        if "evaluator_target_ids" in intent or any(
            "evaluator_target_ids" in p for p in intent["mission"]["phases"]
        ):
            raise ValueError("Policy requires the redacted public intent document")
        capture_s, now_s = release_ns / 1e9, now_ns / 1e9
        phases = [p for p in intent["mission"]["phases"] if p["start_s"] <= capture_s < p["end_s"]]
        if len(phases) != 1:
            raise ValueError("Capture must belong to exactly one public phase")
        phase, limits = phases[0], intent["limits"]
        features = encode(rgb)
        duration_ns = seconds_to_ns(mission.duration_s)
        phase_deadline_ns = seconds_to_ns(min(phase["deadline_s"], limits["deadline_s"]))
        reserve_j = (battery_frac - limits["min_final_battery_frac"]) * mission.capacity_j
        local_continuations = [
            a
            for a in self.actions.values()
            if a.placement == "local"
            and a.memory_mb + self.downstream_memory_mb <= self.memory_capacity_mb
        ]
        if not local_continuations:
            raise ValueError("No legal local continuation under the memory capacity")
        cheapest_local_j = min(a.energy_j for a in local_continuations)
        warm_future_j = (
            cheapest_local_j
            + mission.router_energy_j
            + self.downstream_energy_j
            + costs.encode_s * costs.codec_power_w
        )
        candidates = []
        for action in self.actions.values():
            estimate = estimate_action(action, link, payload_bytes, response_bytes, costs)
            quality, outside = predict(self.quality[action.source_config_id], features)
            risk = predict_risk(self.risk[action.source_config_id], features)
            legal = action.placement == "local" or limits["privacy_level"] == "remote_allowed"
            legal = (
                legal and action.memory_mb + self.downstream_memory_mb <= self.memory_capacity_mb
            )
            row = dict(
                action_id=action.action_id,
                source_config_id=action.source_config_id,
                placement=action.placement,
                legal=legal,
                estimate=estimate,
                predicted_recall=quality,
                predicted_false_components=risk,
                out_of_training_range_fraction=outside,
                task_deadline_feasible=False,
                phase_deadline_feasible=False,
                battery_feasible=False,
                communication_feasible=False,
                feasible=False,
                score=None,
                predicted_usable_finish_ns=None,
                reserve_required_j=None,
                projected_communication_mb=None,
            )
            if estimate["latency_s"] is not None and estimate["predicted_deliverable"]:
                switched = current_action_id != action.action_id
                load_ns = seconds_to_ns(action.load_latency_s, allow_zero=True) if switched else 0
                raw = (
                    now_ns
                    + seconds_to_ns(mission.router_latency_s, allow_zero=True)
                    + load_ns
                    + seconds_to_ns(estimate["latency_s"], allow_zero=True)
                )
                assessment = assess_completion(request, raw, mission.step_ns)
                finish_ns = assessment.usable_finish_ns
                downstream_end_ns = finish_ns + self.downstream_latency_ns
                # Preserve flight until mission end and budget a cheap local continuation.
                # Forecast only: execution checks actual resources and current link afresh.
                interval_ns = seconds_to_ns(mission.interval_s)
                next_capture_ns = (
                    (downstream_end_ns + interval_ns - 1) // interval_ns
                ) * interval_ns
                remaining = max(0, (duration_ns - next_capture_ns + interval_ns - 1) // interval_ns)
                future_reload_j = (
                    min(a.load_energy_j for a in local_continuations) if remaining else 0
                )
                required_j = (
                    mission.router_energy_j
                    + estimate["energy_j"]
                    + (action.load_energy_j if switched else 0)
                    + self.downstream_energy_j
                    + remaining * warm_future_j
                    + future_reload_j
                    + max(0.0, max(mission.duration_s, downstream_end_ns / 1e9) - now_s)
                    * mission.flight_power_w
                )
                projected_mb = cumulative_communication_mb + estimate["communication_mb"]
                cadence_s = max(
                    mission.interval_s,
                    math.ceil((downstream_end_ns - release_ns) / interval_ns) * mission.interval_s,
                )
                evidence_window_s = max(
                    cadence_s * phase["min_hits"], phase["min_evidence_span_s"] + cadence_s
                )
                score = (
                    quality ** phase["min_hits"]
                    / evidence_window_s
                    * math.exp(-risk / (1 + phase["max_false_components"]))
                )
                row.update(
                    task_deadline_feasible=assessment.on_time,
                    phase_deadline_feasible=downstream_end_ns <= phase_deadline_ns,
                    battery_feasible=required_j <= reserve_j + 1e-9,
                    communication_feasible=projected_mb <= limits["communication_budget_mb"] + 1e-9,
                    score=score,
                    predicted_usable_finish_ns=finish_ns,
                    reserve_required_j=required_j,
                    projected_communication_mb=projected_mb,
                    evidence_window_s=evidence_window_s,
                )
                row["feasible"] = legal and all(
                    row[k]
                    for k in (
                        "task_deadline_feasible",
                        "phase_deadline_feasible",
                        "battery_feasible",
                        "communication_feasible",
                    )
                )
            candidates.append(row)
        if self.name in FIXED_ACTIONS:
            selected = next(r for r in candidates if r["action_id"] == FIXED_ACTIONS[self.name])
            action_id = selected["action_id"] if selected["legal"] else None
            reason = (
                "fixed_control_ignores_resource_forecast" if action_id else "no_legal_fixed_action"
            )
        else:
            admitted = [r for r in candidates if r["feasible"]]
            selected = (
                max(
                    admitted,
                    key=lambda r: (
                        r["score"],
                        -r["reserve_required_j"],
                        r["action_id"],
                    ),
                )
                if admitted
                else None
            )
            action_id = selected["action_id"] if selected else None
            reason = (
                "highest_intent_score_within_forecast_constraints"
                if selected
                else "no_feasible_action"
            )
        return dict(
            policy=self.name,
            action_id=action_id,
            reason=reason,
            candidates=candidates,
            public_decision_ns=now_ns,
            release_ns=release_ns,
            task_deadlines=self.profile.to_dict(),
            link=link.to_dict(),
            public_phase_id=phase["phase_id"],
            reserve_available_j=reserve_j,
            intent_requirements={
                k: phase[k]
                for k in (
                    "min_hits",
                    "min_evidence_span_s",
                    "max_false_components",
                    "deadline_s",
                )
            },
            downstream_budget_guard=False,
            forecast_scope=(
                "current-link perception forecast; fixed downstream; cheap local continuation"
            ),
        )
