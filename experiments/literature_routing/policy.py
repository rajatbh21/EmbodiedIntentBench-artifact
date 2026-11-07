"""ApproxDet Equation 5 scheduler objective on this benchmark's fixed model pool.

This is an objective adapter, not a reproduction of the full ApproxDet system.
The original accuracy/latency predictors, detector/tracker branches, contention
sensor and amortized switching cadence are not implemented here. The prediction
head and resource admission are shared with the benchmark's other controllers.
"""

from __future__ import annotations

import math

from missionbench.simulator.action_validator import privacy_permits
from experiments.mission_routing_v2.policy import plain
from experiments.mission_routing_v3.budget import Action, Continuation
from experiments.mission_routing_v5.evidence import Policy as ProfilePolicy
from experiments.mission_routing_v5.quality import BASE_NAMES, encode, predict

NAME = "approxdet_eq5_adapter"
PAPER_URL = "https://arxiv.org/abs/2010.10754"
SOURCE_URL = (
    "https://raw.githubusercontent.com/StarsThu2016/ApproxDet/main/utils_approxdet/scheduler.py"
)


def choose(candidates, latency_sla_s):
    """Apply Eq.5 after common safety admission, with explicit infeasible fallback.

    Equal-quality candidates prefer lower latency then lexical config ID. The
    paper does not prescribe tie-breaking. The minimum-energy fallback applies
    only when the benchmark cannot certify any resource-feasible continuation;
    it does not turn an infeasible action into a passing mission.
    """
    if not math.isfinite(latency_sla_s) or latency_sla_s <= 0:
        raise ValueError("Finite positive latency SLA required")
    if not candidates or len({c["config_id"] for c in candidates}) != len(candidates):
        raise ValueError("Unique nonempty candidates required")
    for c in candidates:
        if (
            not isinstance(c["config_id"], str)
            or not c["config_id"]
            or not math.isfinite(c["quality"])
            or not 0 <= c["quality"] <= 1
            or not math.isfinite(c["latency_s"])
            or c["latency_s"] < 0
            or type(c["feasible"]) is not bool
            or (
                c["projected_energy_j"] is not None
                and (not math.isfinite(c["projected_energy_j"]) or c["projected_energy_j"] < 0)
            )
        ):
            raise ValueError("Invalid scheduler candidate")
    admitted = [c for c in candidates if c["feasible"]]
    timely = [c for c in admitted if c["latency_s"] < latency_sla_s]
    if timely:
        selected = min(timely, key=lambda c: (-c["quality"], c["latency_s"], c["config_id"]))
        reason = "maximum_predicted_recall_under_sla"
    elif admitted:
        selected = min(admitted, key=lambda c: (c["latency_s"], -c["quality"], c["config_id"]))
        reason = "fastest_resource_feasible_sla_unmet"
    else:
        selected = min(
            candidates,
            key=lambda c: (
                c["projected_energy_j"] if c["projected_energy_j"] is not None else math.inf,
                c["config_id"],
            ),
        )
        reason = "minimum_energy_resource_guard_unmet"
    return selected["config_id"], reason


class Policy(ProfilePolicy):
    """Causal per-decision SLA scheduler using frozen development-only recall.

    The inherited initializer validates and copies only model heads/signatures;
    the inherited mission-evidence selection objective is never called. Mission
    phase thresholds and predicted track counts do not enter this objective.
    """

    def __init__(self, profile, specs, costs, *, latency_sla_s=None):
        super().__init__(profile, specs, costs, "no_history_v3")
        self.latency_sla_s = costs.interval_s if latency_sla_s is None else latency_sla_s
        if not math.isfinite(self.latency_sla_s) or self.latency_sla_s <= 0:
            raise ValueError("Finite positive latency SLA required")

    def select(self, observation):
        o, state = observation, observation.state
        now, index = state.current_time_s, state.frame_id
        if abs(now - index * self.costs.interval_s) > 1e-7:
            raise ValueError("Observation off capture clock")
        if set(o.configs.ids()) != set(self.specs):
            raise ValueError("Runtime pool changed")
        history = o.evidence_history
        if (
            history["counts_are_ground_truth"]
            or history["available_at_s"] != now
            or (history["last_completion_s"] is not None and history["last_completion_s"] > now)
            or any(t > now for track in history["tracks"] for t in track["capture_times_s"])
        ):
            raise ValueError("Only completed prediction history is allowed")
        features = encode(o.rgb)
        if self.variant == "base_recall":
            features = features[: len(BASE_NAMES)]
        tracking = o.tracking["cost"]
        actions = []
        for config in o.configs:
            resource = o.resources.for_config(config.config_id)
            if config.strategy.is_remote or not privacy_permits(o.contract.privacy_level, config):
                continue
            if (
                o.resources.memory_capacity_mb is not None
                and resource.memory_mb + tracking["memory_mb"] > o.resources.memory_capacity_mb
            ):
                continue
            spec = self.specs[config.config_id]
            actions.append(
                Action(
                    spec.config_id,
                    spec.mission_latency_s + self.costs.router_latency_s + tracking["latency_s"],
                    spec.energy_j_per_call + self.costs.router_energy_j + tracking["energy_j"],
                    resource.load_latency_s,
                    resource.load_energy_j,
                )
            )
        if not actions:
            raise ValueError("No legal local configuration")
        key = (tuple(actions), o.contract.deadline_s)
        if key != self._budget_key:
            self._budget = Continuation(self.costs, actions, o.contract.deadline_s)
            self._budget_key = key
        available = (state.battery_frac - o.contract.min_final_battery_frac) * self.costs.capacity_j
        candidates = []
        for action in actions:
            config = action.config_id
            quality, outside = predict(self.models[config]["quality"], features)
            if self.variant == "global_recall":
                quality = self.models[config]["quality"]["global_mean"]
            required = self._budget.required(index, state.current_config_id, config)
            completion = self._budget.transition(index, state.current_config_id, config)[0]
            # Match the discrete paid runner, including first-call switch latency.
            latency = completion - now
            candidates.append(
                dict(
                    config_id=config,
                    quality=quality,
                    out_of_range_feature_fraction=outside,
                    projected_energy_j=required if math.isfinite(required) else None,
                    feasible=bool(required <= available + 1e-8),
                    first_completion_s=completion,
                    latency_s=latency,
                    meets_latency_sla=latency < self.latency_sla_s,
                )
            )
        selected, reason = choose(candidates, self.latency_sla_s)
        self.last_decision = dict(
            schema_version="approxdet_eq5_objective_adapter_v1",
            mode=NAME,
            paper_url=PAPER_URL,
            quality_variant=self.variant,
            quality_is_calibrated_probability=False,
            quality_metric="predicted_visible_target_recall_not_map",
            profile_sha256=self.profile_sha,
            selected_config_id=selected,
            selection_reason=reason,
            uses_ground_truth=False,
            uses_context=self.variant != "global_recall",
            uses_history=False,
            uses_intent=False,
            uses_contract_safety_limits=True,
            latency_sla_s=self.latency_sla_s,
            sla_comparison="strict_less_than_paper_equation_5",
            features=features.tolist(),
            history=plain(history),
            available_energy_j=available,
            resource_feasible=any(c["feasible"] for c in candidates),
            candidates=candidates,
        )
        return selected
