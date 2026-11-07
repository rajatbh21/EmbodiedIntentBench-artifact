"""Prediction-only risk ledger around conditional-recall evidence/resource selection.

Admission projects the candidate's current count estimate at its paid cadence
through the remaining capture phase. It is a heuristic, not a joint quality/risk
feasibility proof. The true error count is never supplied at runtime.
"""

from __future__ import annotations

import copy
import math

from experiments.lake_causal_router import forecast
from experiments.mission_frustum_routing.evidence import Policy as EvidencePolicy
from experiments.mission_routing_v2.features import NAMES, encode
from experiments.mission_routing_v2.policy import plain
from experiments.mission_routing_v4.risk import estimate, phase_projection
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.scenario import digest

MODES = ("risk_aware",)


class Policy(EvidencePolicy):
    def __init__(
        self,
        profile,
        specs,
        costs,
        risk_profile,
        mode="risk_aware",
        *,
        gate_trials=True,
        gate_current=True,
    ):
        if type(gate_trials) is not bool or type(gate_current) is not bool:
            raise ValueError("Frustum gates must be explicit booleans")
        if mode not in MODES or risk_profile["feature_names"] != list(NAMES):
            raise ValueError("Unknown risk controller or feature schema")
        if risk_profile["schema_version"] != "mission_false_component_risk_v1":
            raise ValueError("Unknown risk profile version")
        if set(risk_profile["models"]) != {s.config_id for s in specs}:
            raise ValueError("Wrong risk model pool")
        for spec in specs:
            if risk_profile["models"][spec.config_id]["signature"] != model_signature(spec):
                raise ValueError("Wrong risk model signature")
        super().__init__(
            profile,
            specs,
            costs,
            "no_history_v3" if mode == "no_risk_history_v4" else "confirmation",
        )
        self.risk_mode = mode
        self.risk_models = copy.deepcopy(risk_profile["models"])
        self.risk_profile_sha = digest(risk_profile)
        self.ledger = {}
        self.use_provenance, self.use_weakest = True, False
        self.gate_trials, self.gate_current = gate_trials, gate_current
        self.frustum_trials = {}
        self.action_history = {}
        self._previous = None
        self._input_sha = None

    def select(self, observation):
        o = observation
        # The versioned evidence selector validates current prediction history.
        super().select(o)
        base = self.last_decision
        public_key = digest(
            dict(
                decision=base,
                state=dict(
                    frame_id=o.state.frame_id,
                    current_time_s=o.state.current_time_s,
                    current_config_id=o.state.current_config_id,
                ),
                intent=plain(o.intent_contract),
            )
        )
        index, now = o.state.frame_id, o.state.current_time_s
        if self._previous is not None:
            previous_index, config, completion, trace = self._previous
            if index == previous_index:
                if public_key != self._input_sha:
                    raise ValueError("Repeated decision has different public input")
                self.last_decision = copy.deepcopy(trace)
                return config
            if (
                index < previous_index
                or now < completion
                or o.state.current_config_id != config
                or o.evidence_history["last_completion_s"] is None
                or abs(o.evidence_history["last_completion_s"] - completion) > 1e-7
            ):
                raise ValueError("Risk ledger requires the previous selected execution to complete")
        phases = [
            p for p in o.intent_contract["mission"]["phases"] if p["start_s"] <= now < p["end_s"]
        ]
        before = dict(self.ledger)
        candidates = copy.deepcopy(base["candidates"])
        risk_features = encode(o.rgb)
        for candidate in candidates:
            config = candidate["config_id"]
            rate = estimate(
                self.risk_models[config]["risk"],
                risk_features,
                context=self.risk_mode != "global_risk_v4",
            )
            action = self._budget.actions[config]
            slots = forecast(
                self.costs,
                index,
                action.latency_s,
                action.load_latency_s if o.state.current_config_id != config else 0,
            )
            projections = [
                phase_projection(
                    p,
                    now,
                    slots,
                    rate,
                    0 if self.risk_mode == "no_risk_ledger_v4" else before.get(p["phase_id"], 0),
                )
                for p in phases
            ]
            candidate.update(
                estimated_fp_per_capture=rate,
                risk_phases=projections,
                risk_admissible=all(p["within_estimated_budget"] for p in projections),
                excess_expected_fp=sum(
                    max(0, p["projected_false_components"] - p["max_false_components"])
                    for p in projections
                ),
            )

        def energy(c):
            return c["projected_energy_j"] if c["projected_energy_j"] is not None else math.inf

        resource = [c for c in candidates if c["feasible"]]
        admitted = [c for c in resource if c["risk_admissible"]]
        if admitted:
            selected = max(admitted, key=lambda c: (c["utility"], -energy(c), c["config_id"]))
            fallback = None
        elif resource:
            selected = min(
                resource,
                key=lambda c: (c["excess_expected_fp"], -c["utility"], energy(c), c["config_id"]),
            )
            fallback = "minimum_estimated_risk_excess"
        else:
            selected = min(candidates, key=lambda c: (energy(c), c["config_id"]))
            fallback = "minimum_energy_no_resource_feasible_action"
        # Reserve one selected action, never all hypothetical future calls. Every
        # error is attributed to capture time, including late completions.
        for phase in phases:
            pid = phase["phase_id"]
            self.ledger[pid] = before.get(pid, 0) + selected["estimated_fp_per_capture"]
        trace = dict(
            **base,
            risk_profile_sha256=self.risk_profile_sha,
            risk_ledger_before=before,
            risk_ledger_after=dict(self.ledger),
            risk_ledger_is_observed_error=False,
            uses_risk_ledger=self.risk_mode != "no_risk_ledger_v4",
            uses_risk_context=self.risk_mode != "global_risk_v4",
            risk_admissible=bool(admitted),
            risk_fallback=fallback,
            baseline_selected_config_id=base["selected_config_id"],
        )
        trace.update(
            schema_version="frustum_support_routing_v1",
            mode=self.risk_mode,
            selected_config_id=selected["config_id"],
            candidates=candidates,
        )
        self.last_decision, self._input_sha = trace, public_key
        self._previous = (
            index,
            selected["config_id"],
            selected["first_completion_s"],
            copy.deepcopy(trace),
        )
        self.frustum_trials[now] = {t["track_id"]: t["status"] for t in o.frustum_history["tracks"]}
        self.action_history[now] = selected["config_id"]
        return selected["config_id"]
