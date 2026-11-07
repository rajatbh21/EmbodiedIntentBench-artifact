"""Receding-horizon evidence heuristic with exact mixed-model resource admission.

Development quality is a surrogate Bernoulli support parameter, NOT calibrated
track correctness or mission-success probability. Future RGB/GT are inaccessible.
The eight-second horizon and discovery weight are fixed before prospective tests.
"""

from __future__ import annotations

import math

from missionbench.simulator.action_validator import privacy_permits
from experiments.lake_causal_router import forecast
from experiments.mission_routing_v2.features import encode, predict
from experiments.mission_routing_v2.policy import Policy as PreviousPolicy
from experiments.mission_routing_v2.policy import plain
from experiments.mission_routing_v3.budget import Action, Continuation

MODES = ("confirmation", "no_history_v3", "no_intent_v3", "no_context_v3", "constant_budget_v3")
HORIZON_S = 8.0


def support_probability(existing, slots, quality, min_hits, min_span, max_gap):
    """Enumerate short-horizon repeat support; at most four captures in this study.

    Gaps reset an unconfirmed track. Existing support must be phase-local. A
    successful branch is absorbing; no GT identity or target count is needed.
    """
    if not 0 <= quality <= 1 or min_hits < 1 or min_span < 0 or max_gap <= 0:
        raise ValueError("Invalid evidence surrogate")
    times = tuple(existing)
    if times and len(times) >= min_hits and times[-1] - times[0] >= min_span:
        return 1.0
    states, success = {times: 1.0}, 0.0
    for capture in slots:
        updated = {}
        for hits, mass in states.items():
            live = hits if not hits or capture - hits[-1] <= max_gap else ()
            updated[live] = updated.get(live, 0) + mass * (1 - quality)
            observed = (*live, capture)
            if len(observed) >= min_hits and observed[-1] - observed[0] >= min_span:
                success += mass * quality
            else:
                updated[observed] = updated.get(observed, 0) + mass * quality
        states = updated
    return min(1.0, success)


class Policy(PreviousPolicy):
    def __init__(self, profile, specs, costs, mode="confirmation"):
        if mode not in MODES:
            raise ValueError("Unknown v3 controller")
        super().__init__(profile, specs, costs, "intent_history")
        self.mode, self._budget_key, self._budget = mode, None, None

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
        use_history = self.mode != "no_history_v3"
        use_intent = self.mode != "no_intent_v3"
        use_context = self.mode != "no_context_v3"
        features = encode(o.rgb)
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
        if use_intent:
            phases = [
                p
                for p in o.intent_contract["mission"]["phases"]
                if p["start_s"] <= now < p["end_s"]
            ]
        else:
            # Remove phase boundaries, weights and repetition requirements;
            # retain the public tracker's generic reporting rule and safety limits.
            cfg = o.tracking["configuration"]
            phases = [
                dict(
                    phase_id="generic_tracking",
                    start_s=0,
                    end_s=self.costs.duration_s,
                    deadline_s=o.contract.deadline_s,
                    min_hits=cfg["min_hits"],
                    min_evidence_span_s=0,
                    weight=1,
                )
            ]
        candidates = []
        for action in actions:
            config = action.config_id
            quality, uncertainty = predict(self.models[config]["quality"], features)
            if not use_context:
                quality = self.models[config]["quality"]["global_mean"]
            slots = forecast(
                self.costs,
                index,
                action.latency_s,
                action.load_latency_s if state.current_config_id != config else 0,
            )
            mixed = self._budget.required(index, state.current_config_id, config)
            constant = (
                (max(self.costs.duration_s, slots[-1][1]) - now) * self.costs.flight_power_w
                + len(slots) * action.energy_j
                + (action.load_energy_j if state.current_config_id != config else 0)
            )
            energy = constant if self.mode == "constant_budget_v3" else mixed
            utility, details = 0.0, []
            for phase in phases:
                times = [
                    t
                    for t, end in slots
                    if t < min(phase["end_s"], now + HORIZON_S) and end <= phase["deadline_s"]
                ]
                args = (
                    quality,
                    phase["min_hits"],
                    phase["min_evidence_span_s"],
                    o.tracking["configuration"]["max_gap_s"],
                )
                discovery = support_probability((), times, *args)
                pending = []
                if use_history:
                    for track in history["tracks"]:
                        hits = [
                            t
                            for t in track["capture_times_s"]
                            if phase["start_s"] <= t < phase["end_s"]
                        ]
                        if hits and not (
                            len(hits) >= phase["min_hits"]
                            and hits[-1] - hits[0] >= phase["min_evidence_span_s"]
                        ):
                            pending.append(support_probability(hits, times, *args))
                # Never stop discovery based on a predicted count. Pending tracks
                # add candidate-specific confirmation value, normalized for count.
                continuation = sum(pending) / len(pending) if pending else 0.0
                utility += phase["weight"] * (discovery + continuation)
                details.append(
                    dict(
                        phase_id=phase["phase_id"],
                        captures=times,
                        discovery_support=discovery,
                        pending_tracks=len(pending),
                        confirmation_support=continuation,
                    )
                )
            feasible = (
                energy <= available + 1e-8 and slots[-1][1] <= o.contract.deadline_s
                if self.mode == "constant_budget_v3"
                else energy <= available + 1e-8
            )
            candidates.append(
                dict(
                    config_id=config,
                    utility=utility,
                    quality=quality,
                    uncertainty=uncertainty,
                    mixed_energy_j=mixed if math.isfinite(mixed) else None,
                    projected_energy_j=energy if math.isfinite(energy) else None,
                    constant_energy_j=constant,
                    feasible=feasible,
                    first_completion_s=slots[0][1],
                    phases=details,
                )
            )
        feasible = [c for c in candidates if c["feasible"]]

        def resource_key(c):
            return c["projected_energy_j"] if c["projected_energy_j"] is not None else math.inf

        selected = (
            max(feasible, key=lambda c: (c["utility"], -resource_key(c), c["config_id"]))
            if feasible
            else min(candidates, key=lambda c: (resource_key(c), c["config_id"]))
        )
        self.last_decision = dict(
            schema_version="mission_routing_decision_v3",
            mode=self.mode,
            profile_sha256=self.profile_sha,
            selected_config_id=selected["config_id"],
            uses_ground_truth=False,
            features=features.tolist(),
            uses_intent=use_intent,
            uses_history=use_history,
            uses_context=use_context,
            horizon_s=HORIZON_S,
            history=plain(history),
            available_energy_j=available,
            resource_feasible=bool(feasible),
            candidates=candidates,
        )
        return selected["config_id"]
