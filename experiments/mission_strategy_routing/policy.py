"""Distinct objectives sharing a causal paid-action budget and public predictors.

The survival score is an uncalibrated independent-error count surrogate, multiplied
by eight-second evidence support. It is not a full mission-success probability.
"""

from __future__ import annotations

import copy
import math

from experiments.lake_causal_router import forecast
from experiments.mission_routing_v2.features import NAMES, encode
from experiments.mission_routing_v2.policy import plain
from experiments.mission_routing_v4.risk import estimate, phase_projection
from experiments.mission_routing_v5.evidence import Policy as EvidencePolicy
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.scenario import digest

STRATEGIES = (
    "round_robin",
    "greedy_quality",
    "quality_per_joule",
    "evidence_only",
    "mission_survival",
)


def log_count_survival(mean, cap):
    """Log P(Poisson(mean) <= cap), a configured surrogate, not calibrated risk."""
    if not math.isfinite(mean) or mean < 0 or type(cap) is not int or cap < 0:
        raise ValueError("Finite nonnegative mean and integer error cap required")
    if mean == 0:
        return 0.0
    terms = [k * math.log(mean) - math.lgamma(k + 1) for k in range(cap + 1)]
    peak = max(terms)
    return min(0.0, -mean + peak + math.log(sum(math.exp(t - peak) for t in terms)))


class Policy(EvidencePolicy):
    def __init__(
        self,
        profile,
        specs,
        costs,
        risk_profile,
        *,
        strategy="mission_survival",
    ):
        if strategy not in STRATEGIES or risk_profile["feature_names"] != list(NAMES):
            raise ValueError("Unknown strategy or feature schema")
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
            "confirmation",
        )
        self.strategy = strategy
        self.risk_models = copy.deepcopy(risk_profile["models"])
        self.risk_profile_sha = digest(risk_profile)
        self.ledger = {}
        self.decisions = 0
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
                context=True,
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
                    before.get(p["phase_id"], 0),
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

        for candidate in candidates:
            config = candidate["config_id"]
            _, _, incremental, _ = self._budget.transition(index, o.state.current_config_id, config)
            survival = {
                r["phase_id"]: log_count_survival(
                    r["projected_false_components"], r["max_false_components"]
                )
                for r in candidate["risk_phases"]
            }
            terms = []
            for detail in candidate["phases"]:
                phase = next(p for p in phases if p["phase_id"] == detail["phase_id"])
                evidence = phase["weight"] * (
                    detail["discovery_support"] + detail["confirmation_support"]
                )
                if evidence > 0:
                    terms.append(math.log(evidence) + survival[phase["phase_id"]])
            if terms:
                peak = max(terms)
                score = peak + math.log(sum(math.exp(t - peak) for t in terms))
            else:
                score = None
            candidate.update(
                incremental_energy_j=incremental,
                log_count_survival_by_phase=survival,
                mission_log_score=score,
            )

        def energy(c):
            return c["projected_energy_j"] if c["projected_energy_j"] is not None else math.inf

        resource = [c for c in candidates if c["feasible"]]
        fallback = None
        if not resource:
            selected = min(candidates, key=lambda c: (energy(c), c["config_id"]))
            fallback = "minimum_energy_no_resource_feasible_action"
        elif self.strategy == "round_robin":
            order = sorted(self.specs)
            offset = self.decisions % len(order)
            order = order[offset:] + order[:offset]
            selected = min(resource, key=lambda c: order.index(c["config_id"]))
        else:

            def objective(c):
                if self.strategy == "greedy_quality":
                    return c["quality"]
                if self.strategy == "quality_per_joule":
                    return c["quality"] / max(c["incremental_energy_j"], 1e-12)
                if self.strategy == "evidence_only":
                    return c["utility"]
                return c["mission_log_score"] if c["mission_log_score"] is not None else -math.inf

            selected = max(resource, key=lambda c: (objective(c), -energy(c), c["config_id"]))
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
            uses_risk_ledger=self.strategy == "mission_survival",
            uses_risk_context=self.strategy == "mission_survival",
            risk_admissible=selected["risk_admissible"],
            risk_fallback=fallback,
            baseline_selected_config_id=base["selected_config_id"],
        )
        trace.update(
            schema_version="mission_strategy_routing_v1",
            count_survival_is_calibrated=False,
            count_survival_assumes_independent_errors=True,
            decision_count=self.decisions,
            strategy=self.strategy,
            uses_intent=self.strategy in ("evidence_only", "mission_survival"),
            uses_history=self.strategy in ("evidence_only", "mission_survival"),
            uses_context=self.strategy != "round_robin",
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
        self.decisions += 1
        return selected["config_id"]
