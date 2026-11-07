"""Public routing input and adapters for existing policies and external submissions.

The interface is a data boundary, not a security sandbox for untrusted Python.
No future trace, scenario objects, GT, evaluator, or backend handle is passed in.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from missionbench.policies.budget_planner import BudgetPlannerPolicy, BudgetPlannerSettings
from missionbench.policies.rule_based import RuleBasedPolicy, RuleBasedSettings
from missionbench.policies.static import StaticPolicy
from missionbench.schemas.configuration import ConfigCatalog
from missionbench.schemas.contract import Contract
from missionbench.schemas.profile import PublicProfileView
from missionbench.schemas.runtime_state import RuntimeState
from missionbench.simulator.action_validator import privacy_permits
from experiments.v4_benchmark.mission_requirements import MissionRequirements
from experiments.v4_benchmark.resources import ResourceConfig


@dataclass(frozen=True)
class RouterInput:
    contract: Contract
    state: RuntimeState
    configs: ConfigCatalog
    rgb: np.ndarray  # Independent, immutable bytes-backed RGB; never scene labels.
    position_m: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]
    detections_so_far: int  # Prediction-derived detections, NOT unique GT targets.
    requirements: MissionRequirements = field(default_factory=MissionRequirements)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    confirmed_reports_so_far: int = 0


class Router(Protocol):
    def select(self, observation: RouterInput) -> str: ...


class PolicyAdapter:
    """Keep the V1 Policy interface unchanged; it simply does not consume RGB."""

    def __init__(self, policy: Any):
        self.policy = policy
        self.last_decision: dict[str, Any] | None = None

    def select(self, observation: RouterInput) -> str:
        selected = self.policy.select_config(
            observation.contract, observation.state, observation.configs
        )
        self.last_decision = self._explain(observation, selected)
        return selected

    def _explain(self, observation: RouterInput, selected: str) -> dict[str, Any]:
        """Read-only audit of built-in policy helpers AFTER its single state update.

        Deliberately coupled to the existing policies, with branch parity tested.
        No GT, extra select call, or change to frozen V1 policy behavior.
        """
        policy = self.policy
        detail = {
            "selected_config_id": selected,
            "uses_rgb": False,
            "mission_requirements_consumed": False,
        }
        if not isinstance(policy, (RuleBasedPolicy, BudgetPlannerPolicy)):
            return {**detail, "reason": "static_selection"}
        contract, state, catalog = observation.contract, observation.state, observation.configs
        settings = policy._settings
        stages = []
        candidates = [c for c in catalog if privacy_permits(contract.privacy_level, c)]
        stages.append({"stage": "privacy", "retained": [c.config_id for c in candidates]})
        if not candidates:
            return {**detail, "reason": "no_privacy_legal_candidate", "stages": stages}

        def retain(stage: str, filtered: list) -> None:
            nonlocal candidates
            stages.append(
                {
                    "stage": stage,
                    "passed": [c.config_id for c in filtered],
                    "empty_filter_fallback": not filtered,
                }
            )
            candidates = filtered or candidates

        retain("reachability", policy._reachable(candidates, state))
        if isinstance(policy, RuleBasedPolicy):
            retain("communication_budget", policy._affordable(candidates, contract, state))
            deadline = policy._under_deadline_pressure(contract, state)
            battery = policy._under_battery_pressure(contract, state)
        else:
            retain("communication_pacing", policy._within_pace(candidates, contract, state))
            deadline = policy._deadline_pressure(contract, state)
            battery = policy._battery_projection_low(contract, state)
        projection = (
            state.current_time_s / state.path_progress
            if state.current_time_s > 0 and state.path_progress > 0
            else None
        )
        if deadline or battery:
            reason = "resource_pressure_fastest"
            expected = policy._fastest(candidates, state).config_id
        else:
            retain("latency_budget", policy._within_latency_budget(candidates, state))
            reason = "best_public_quality_tier"
            expected = policy._best_quality(candidates, state).config_id
        return {
            **detail,
            "reason": reason,
            "stages": stages,
            "deadline_pressure": deadline,
            "battery_pressure": battery,
            "projected_finish_s": projection,
            "deadline_trigger_s": contract.deadline_s * settings.deadline_pressure_frac,
            "latency_ceiling_s": settings.max_latency_intervals * settings.decision_interval_s,
            "quality_metric_used_in_selection": False,
            "quality_threshold_used_in_selection": False,
            "explanation_matches_selection": expected == selected,
        }


def resolve_factory(reference: str):
    module, separator, name = reference.partition(":")
    if not separator or not module or not name or ":" in name:
        raise ValueError("factory must have the form module:callable")
    factory = getattr(importlib.import_module(module), name)
    if not callable(factory):
        raise ValueError(f"factory {reference!r} is not callable")
    return factory


def build_router(
    name: str,
    profiles: PublicProfileView,
    interval_s: float,
    parameters: dict[str, Any] | None = None,
) -> Router:
    params = dict(parameters or {})
    if name == "report_calibrated":
        from experiments.v4_benchmark.report_router import ReportCalibratedRouter

        return ReportCalibratedRouter(interval_s=interval_s, **params)
    if name == "calibrated":
        from experiments.v4_benchmark.calibrated_router import CalibratedRouter

        return CalibratedRouter(interval_s=interval_s, **params)
    if name in ("periodic", "seeded_random"):
        from experiments.v4_benchmark.schedule_routers import ScheduleRouter

        return ScheduleRouter(mode=name, **params)
    if name in ("intent_visual", "intent_only", "visual_only", "resource_only"):
        from experiments.v4_benchmark.intent_visual import IntentVisualRouter, Settings

        return IntentVisualRouter(
            profiles,
            Settings(**params),
            use_intent=name in ("intent_visual", "intent_only"),
            use_visual=name in ("intent_visual", "visual_only"),
        )
    if name.startswith("python:"):
        router = resolve_factory(name.removeprefix("python:"))(
            public_profiles=profiles,
            parameters=params,
        )
        if not callable(getattr(router, "select", None)):
            raise ValueError("external router must implement select(RouterInput)")
        return router
    if name.startswith("static:"):
        if params:
            raise ValueError("static routers do not accept parameters")
        return PolicyAdapter(StaticPolicy(name.removeprefix("static:")))
    params.setdefault("decision_interval_s", interval_s)
    if name == "rule_based":
        return PolicyAdapter(
            RuleBasedPolicy(public_profiles=profiles, settings=RuleBasedSettings(**params))
        )
    if name == "budget_planner":
        return PolicyAdapter(
            BudgetPlannerPolicy(public_profiles=profiles, settings=BudgetPlannerSettings(**params))
        )
    raise ValueError(
        f"unknown router {name!r}; use static:ID, rule_based, budget_planner, "
        "intent_visual, intent_only, visual_only, resource_only, periodic, "
        "seeded_random, calibrated or python:module:factory"
    )


def immutable_rgb(rgb: np.ndarray) -> np.ndarray:
    # A merely write-disabled owned array can be made writable again. A bytes-backed
    # copy cannot, and the executor/evaluator retain their independent original.
    return np.frombuffer(rgb.tobytes(), dtype=np.uint8).reshape(rgb.shape)
