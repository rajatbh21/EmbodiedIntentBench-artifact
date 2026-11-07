"""Causal, calibrated routing using public intent and configured resource forecasts.

Training may consume labeled development captures. The policy receives aggregate
RGB-context statistics only, never target identities, per-frame scores or a route.
These heuristics predict evidence opportunities, not mission success probabilities.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from missionbench.simulator.action_validator import privacy_permits
from experiments.v4_benchmark.calibrated_router import context_key, model_signature
from experiments.v4_benchmark.scenario import digest
from experiments.v4_gazebo_pilot.timing import CaptureSchedule, seconds_to_ns

MODES = ("intent_context", "no_intent", "no_context", "resource_only")
SCHEMA = "lake_causal_calibration_v1"


def split_fingerprint(scenario: Any, scenery_sha256: str) -> dict[str, str]:
    """Evaluator-side lineage. None of these world data enter policy observations."""
    return dict(
        recording=scenario.recording_identity,
        world=digest(scenario.world),
        scenery=scenery_sha256,
        layout=digest(scenario.world["objects"]),
        route=digest(scenario.world["waypoints"]),
    )


def check_split(training: list[dict], evaluation: dict, *, scope: str) -> dict:
    if scope not in ("development", "held_out"):
        raise ValueError("Explicit development or held_out scope required")
    required = {"recording", "world", "scenery", "layout", "route"}
    if not training or any(set(item) != required for item in [*training, evaluation]):
        raise ValueError("Complete split lineage required")
    if any(
        not isinstance(v, str) or not v for item in [*training, evaluation] for v in item.values()
    ):
        raise ValueError("Nonempty split fingerprints required")
    overlaps = sorted(key for key in required if any(t[key] == evaluation[key] for t in training))
    if scope == "held_out" and overlaps:
        raise ValueError("Training/evaluation overlap: " + ", ".join(overlaps))
    return dict(scope=scope, overlaps=overlaps, independent_test=scope == "held_out")


def aggregate(rows: list[dict]) -> dict:
    visible = sum(len(r["visible_target_ids"]) for r in rows)
    matched = sum(len(r["matched_target_ids"]) for r in rows)
    predicted = sum(r["predicted_components"] for r in rows)
    true = sum(r["matched_components"] for r in rows)
    return dict(
        frames=len(rows),
        visible_observations=visible,
        predictions=predicted,
        recall=matched / visible if visible else 0.0,
        precision=true / predicted if predicted else 0.0,
    )


def fit_profile(rows: list[dict], contexts: dict[int, str], specs: tuple, lineage: dict) -> dict:
    """Aggregate a complete development table, discarding all identity/time information."""
    expected = {(i, s.config_id) for i in contexts for s in specs}
    actual = [(r["capture_id"], r["config_id"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Complete unique calibration frame/config pairs required")
    models = {}
    for spec in specs:
        selected = [r for r in rows if r["config_id"] == spec.config_id]
        models[spec.config_id] = dict(
            signature=model_signature(spec),
            global_stats=aggregate(selected),
            contexts={
                context: aggregate([r for r in selected if contexts[r["capture_id"]] == context])
                for context in sorted(set(contexts.values()))
            },
        )
    return dict(
        schema_version=SCHEMA,
        feature_encoding="rgb_v1",
        training_lineage=[copy.deepcopy(lineage)],
        quality_semantics="Development visible-observation recall and component precision",
        models=models,
    )


@dataclass(frozen=True)
class MissionCosts:
    duration_s: float
    interval_s: float
    capacity_j: float
    flight_power_w: float
    router_latency_s: float
    router_energy_j: float
    step_ns: int

    def __post_init__(self):
        for key, value in vars(self).items():
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Finite nonnegative mission costs required")
            if key in ("duration_s", "interval_s", "capacity_j", "step_ns") and value == 0:
                raise ValueError("Positive mission timing and capacity required")
        if type(self.step_ns) is not int:
            raise ValueError("Integer physics step required")

    @classmethod
    def from_scenario(cls, scenario: Any) -> MissionCosts:
        return cls(
            scenario.duration_s,
            scenario.observation_interval_s,
            scenario.battery_capacity_wh * 3600,
            scenario.flight_power_w,
            scenario.router_latency_s,
            scenario.router_energy_j,
            scenario.schedule.step_ns,
        )

    def schedule(self) -> CaptureSchedule:
        return CaptureSchedule(
            0, seconds_to_ns(self.interval_s), seconds_to_ns(self.duration_s), self.step_ns
        )


def forecast(costs: MissionCosts, frame_id: int, latency_s: float, load_s: float) -> list[tuple]:
    """Exact fixed-config counterfactual capture/completion clock, including final overshoot."""
    schedule = costs.schedule()
    result = []
    index = frame_id
    while schedule.at(index) < schedule.end_ns:
        delay = latency_s + (load_s if index == frame_id else 0)
        end, next_index, _ = schedule.complete(index, seconds_to_ns(delay, allow_zero=True))
        result.append((schedule.at(index) / 1e9, end / 1e9))
        index = next_index
    return result


class CausalRouter:
    """Public-input heuristic; all forecasts assume retaining one candidate configuration."""

    def __init__(
        self, profile: dict, specs: tuple, costs: MissionCosts, *, mode: str = "intent_context"
    ):
        if mode not in MODES or profile.get("schema_version") != SCHEMA:
            raise ValueError("Unsupported causal router/profile")
        if profile.get("feature_encoding") != "rgb_v1":
            raise ValueError("Unsupported context encoding")
        if set(profile["models"]) != {s.config_id for s in specs}:
            raise ValueError("Calibration configuration pool mismatch")
        for spec in specs:
            model = profile["models"][spec.config_id]
            if model["signature"] != model_signature(spec) or spec.kind == "simulated_remote":
                raise ValueError("Calibration signature mismatch or unsupported remote model")
            for stats in [model["global_stats"], *model["contexts"].values()]:
                if any(
                    type(stats[k]) not in (int, float)
                    or not math.isfinite(stats[k])
                    or not 0 <= stats[k] <= 1
                    for k in ("recall", "precision")
                ):
                    raise ValueError("Invalid calibration quality")
                if (
                    any(
                        type(stats[k]) is not int or stats[k] < 0
                        for k in ("frames", "visible_observations", "predictions")
                    )
                    or stats["frames"] == 0
                ):
                    raise ValueError("Invalid calibration support")
            if any(
                not math.isfinite(v) or v < 0
                for v in (spec.mission_latency_s, spec.energy_j_per_call)
            ):
                raise ValueError("Invalid model costs")
        # Training lineage is deliberately omitted from the runtime policy object.
        self.models = copy.deepcopy(profile["models"])
        self.profile_sha256 = digest(profile)
        self.model_costs = {s.config_id: (s.mission_latency_s, s.energy_j_per_call) for s in specs}
        self.costs, self.mode, self.last_decision = costs, mode, None

    def select(self, observation: Any) -> str:
        public, state = observation.intent_contract, observation.state
        if public is None or observation.tracking is None:
            raise ValueError("Public intent and tracking cost declaration required")
        if set(observation.configs.ids()) != set(self.models):
            raise ValueError("Runtime configuration pool changed")
        now = state.current_time_s
        if abs(now - state.frame_id * self.costs.interval_s) > 1e-7:
            raise ValueError("Observation does not lie on the declared capture clock")
        use_intent = self.mode in ("intent_context", "no_context")
        use_context = self.mode in ("intent_context", "no_intent")
        context = context_key(observation.rgb) if use_context else "global"
        tracking = observation.tracking["cost"]
        available = (
            state.battery_frac - observation.contract.min_final_battery_frac
        ) * self.costs.capacity_j
        options = []
        for config in observation.configs:
            if config.strategy.is_remote or not privacy_permits(
                observation.contract.privacy_level, config
            ):
                continue
            resource = observation.resources.for_config(config.config_id)
            memory = resource.memory_mb + tracking["memory_mb"]
            if (
                observation.resources.memory_capacity_mb is not None
                and memory > observation.resources.memory_capacity_mb
            ):
                continue
            latency, energy = self.model_costs[config.config_id]
            loading = state.current_config_id != config.config_id
            load_s, load_j = (
                (resource.load_latency_s, resource.load_energy_j) if loading else (0, 0)
            )
            slots = forecast(
                self.costs,
                state.frame_id,
                latency + self.costs.router_latency_s + tracking["latency_s"],
                load_s,
            )
            end = max(self.costs.duration_s, slots[-1][1])
            projected_j = (
                (end - now) * self.costs.flight_power_w
                + len(slots) * (energy + self.costs.router_energy_j + tracking["energy_j"])
                + load_j
            )
            model = self.models[config.config_id]
            stats = model["contexts"].get(context) if use_context else None
            fallback = stats is None or stats["frames"] < 8 or stats["visible_observations"] == 0
            stats = model["global_stats"] if fallback else stats
            quality = stats["recall"] * stats["precision"]
            opportunities = []
            if use_intent:
                for phase in public["mission"]["phases"]:
                    if phase["end_s"] <= now:
                        continue
                    times = [
                        capture
                        for capture, completion in slots
                        if phase["start_s"] <= capture < phase["end_s"]
                        and completion <= phase["deadline_s"]
                    ]
                    count_factor = min(1.0, len(times) / phase["min_hits"])
                    span = times[-1] - times[0] if len(times) > 1 else 0
                    span_factor = (
                        min(1.0, span / phase["min_evidence_span_s"])
                        if phase["min_evidence_span_s"]
                        else 1.0
                    )
                    opportunities.append(
                        dict(
                            phase_id=phase["phase_id"],
                            captures=len(times),
                            span_s=span,
                            weight=phase["weight"],
                            proxy=min(1.0, quality / phase["required_recall"])
                            * count_factor
                            * span_factor,
                        )
                    )
            utility = (
                sum(p["weight"] * p["proxy"] for p in opportunities)
                / sum(p["weight"] for p in opportunities)
                if opportunities
                else quality
            )
            deadline = observation.contract.deadline_s
            if use_intent and public["reporting"]["mode"] == "report_required":
                deadline = min(deadline, public["reporting"]["deadline_s"])
            options.append(
                dict(
                    config_id=config.config_id,
                    utility=utility,
                    quality_proxy=quality,
                    context_fallback=fallback,
                    projected_energy_j=projected_j,
                    battery_fits=projected_j <= available,
                    deadline_fits=slots[0][1] <= deadline,
                    first_completion_s=slots[0][1],
                    next_capture_s=slots[1][0] if len(slots) > 1 else None,
                    phase_opportunities=opportunities,
                )
            )
        if not options:
            raise ValueError("No privacy- and memory-legal local configuration")
        feasible = [o for o in options if o["battery_fits"] and o["deadline_fits"]]
        if feasible and self.mode != "resource_only":
            chosen = max(
                feasible,
                key=lambda o: (
                    o["utility"],
                    -o["projected_energy_j"],
                    -o["first_completion_s"],
                    o["config_id"],
                ),
            )
            reason = "best_development_proxy_within_forecast_budget"
        else:
            chosen = min(
                feasible or options,
                key=lambda o: (
                    not o["deadline_fits"],
                    o["projected_energy_j"],
                    o["first_completion_s"],
                    o["config_id"],
                ),
            )
            reason = "minimum_projected_energy" if feasible else "no_feasible_candidate_best_effort"
        self.last_decision = dict(
            schema_version="causal_decision_v1",
            selected_config_id=chosen["config_id"],
            mode=self.mode,
            reason=reason,
            context=context,
            profile_sha256=self.profile_sha256,
            available_energy_j=available,
            uses_intent=use_intent,
            uses_context=use_context,
            uses_ground_truth=False,
            mission_quality_guaranteed=False,
            prediction_report_count=observation.confirmed_reports_so_far,
            report_count_used_for_stopping=False,
            candidates=options,
        )
        return chosen["config_id"]
