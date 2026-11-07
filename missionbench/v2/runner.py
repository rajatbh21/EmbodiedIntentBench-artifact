"""The V2 mission runner: the authoritative visual closed loop.

Observations are scheduled at a fixed mission-time interval. For each processed
observation the runner: computes the UAV position at the capture time, renders the
crop, builds the policy-visible runtime state, asks the (V1-protocol) policy for a
configuration, runs that image executor on the RGB, advances mission time by the
executor's configured latency, moves the UAV to the completion position, marks every
scheduled capture that passed while the executor was busy as **skipped**, scores the
prediction against the **capture-time** ground truth, updates battery / communication /
counters, and checks termination.

Two properties are load-bearing and tested:

- **The UAV never pauses.** Position is a pure function of mission time, so capture
  and completion positions genuinely differ under a slow executor.
- **Stale observations are never processed.** A scheduled capture that passes while
  the executor is busy is recorded as skipped, not queued.

Reused V1 machinery (read-only): ``Contract``, ``RuntimeState``/``EvidenceSummary``,
``NetworkObservation``, ``ConfigCatalog``/``Configuration``, ``ActionValidator`` (with
the scenario's explicit fallback), the policy registry, and the empirical metric
vocabulary via :mod:`missionbench.v2.evaluation`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final

from missionbench.policies.registry import policy_registry
from missionbench.policies.static import StaticPolicy
from missionbench.schemas.configuration import (
    ConfigCatalog,
    Configuration,
    Placement,
    Precision,
    Strategy,
)
from missionbench.schemas.loading import SchemaValidationError
from missionbench.schemas.profile import PublicProfile, PublicProfileView, QualityTier
from missionbench.schemas.runtime_state import EvidenceSummary, RuntimeState
from missionbench.simulator.action_validator import (
    TRANSMITTED_PAYLOAD_PARAMETER,
    ActionOutcome,
    ActionValidator,
    TransmittedPayload,
)
from missionbench.v2.camera import CameraRenderer, Observation
from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import ImageExecutionResult, build_executors
from missionbench.v2.network import V2NetworkModel, constant_network_model
from missionbench.v2.objects import ObjectLayer
from missionbench.v2.remote import ExecutionContext
from missionbench.v2.scenario import V2Scenario
from missionbench.v2.trajectory import build_trajectory
from missionbench.v2.world import open_world

__all__ = ["MissionRunner", "ObservationLog", "V2MissionResult", "build_policy", "run_mission"]

_SECONDS_PER_WH: Final = 3600.0
#: Reasons a V2 mission ends. The first three reuse the V1 vocabulary verbatim.
TERMINATION_PATH_COMPLETE: Final = "path_complete"
TERMINATION_DEADLINE: Final = "deadline_exceeded"
TERMINATION_BATTERY: Final = "battery_depleted"
TERMINATION_EXECUTOR_FAILURE: Final = "executor_failure"


@dataclass(frozen=True, slots=True)
class ObservationLog:
    """One processed observation, from capture to completion."""

    observation_id: int
    capture_time_s: float
    completion_time_s: float
    capture_position_m: tuple[float, float]
    completion_position_m: tuple[float, float]
    requested_config_id: str | None
    executed_config_id: str
    action_valid: bool
    skipped_after: tuple[int, ...]
    execution: dict[str, Any]
    score: dict[str, Any]
    #: Optional replay/dashboard time series (``MissionRunner(record_runtime_snapshots=True)``).
    #: ``at_capture`` is exactly the policy-visible ``RuntimeState`` — never ground truth.
    runtime: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "observation_id": self.observation_id,
            "capture_time_s": self.capture_time_s,
            "completion_time_s": self.completion_time_s,
            "capture_position_m": list(self.capture_position_m),
            "completion_position_m": list(self.completion_position_m),
            "requested_config_id": self.requested_config_id,
            "executed_config_id": self.executed_config_id,
            "action_valid": self.action_valid,
            "skipped_after": list(self.skipped_after),
            "execution": self.execution,
            "score": self.score,
        }
        if self.runtime is not None:
            payload["runtime"] = self.runtime
        return payload


@dataclass
class V2MissionResult:
    """Everything one V2 mission produced."""

    scenario_id: str
    policy_name: str
    mission_success: bool
    quality: dict[str, Any]
    constraints: dict[str, bool]
    termination_reason: str
    final_time_s: float
    final_battery_frac: float
    cumulative_communication_mb: float
    cumulative_energy_j: float
    path_progress: float
    processed_observation_count: int
    skipped_observation_count: int
    skipped_observation_ids: tuple[int, ...]
    mean_executor_latency_s: float
    config_selection_history: tuple[str, ...]
    observations: list[ObservationLog] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    #: V3 P1 additions (additive; zero/empty for local-only missions).
    privacy_violation_count: int = 0
    failed_inference_count: int = 0
    communication_energy_j: float = 0.0
    network_behaviour: dict[str, int] = field(default_factory=dict)

    def to_dict(self, *, include_observations: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "benchmark_mode": "v2_visual",
            "scenario_id": self.scenario_id,
            "policy": self.policy_name,
            "mission_success": self.mission_success,
            "quality": self.quality,
            "constraints": self.constraints,
            "termination_reason": self.termination_reason,
            "final_time_s": self.final_time_s,
            "final_battery_frac": self.final_battery_frac,
            "cumulative_communication_mb": self.cumulative_communication_mb,
            "cumulative_energy_j": self.cumulative_energy_j,
            "path_progress": self.path_progress,
            "processed_observation_count": self.processed_observation_count,
            "skipped_observation_count": self.skipped_observation_count,
            "skipped_observation_ids": list(self.skipped_observation_ids),
            "mean_executor_latency_s": self.mean_executor_latency_s,
            "config_selection_history": list(self.config_selection_history),
            "notes": dict(self.notes),
            "privacy_violation_count": self.privacy_violation_count,
            "failed_inference_count": self.failed_inference_count,
            "communication_energy_j": self.communication_energy_j,
            "network_behaviour": dict(self.network_behaviour),
        }
        if include_observations:
            payload["observations"] = [log.to_dict() for log in self.observations]
        return payload


def build_policy(policy_name: str, scenario: V2Scenario):
    """Resolve a policy name for a V2 mission, on the unchanged V1 policy interface.

    ``always_fast`` / ``always_strong`` pin the first executor of that kind. Any
    registered V1 policy name (e.g. ``rule_based``) is constructed with a public
    profile view derived from the scenario's executor configs, so an adaptive V1
    policy reasons over V2 configurations without modification.
    """
    by_kind = {spec.kind: spec.config_id for spec in scenario.executor_configs}
    if policy_name == "always_fast":
        return StaticPolicy(by_kind["fast_weak"])
    if policy_name == "always_strong":
        return StaticPolicy(by_kind["slow_strong"])
    if policy_name == "always_remote_strong":
        candidates = [
            spec.config_id for spec in scenario.executor_configs if spec.kind == "simulated_remote"
        ]
        if not candidates:
            raise SchemaValidationError(
                "policy 'always_remote_strong' needs a simulated_remote executor; "
                "the scenario declares none"
            )
        return StaticPolicy(candidates[0])
    if policy_name in ("always_light_real", "always_strong_real"):
        wanted_tier = "low" if policy_name == "always_light_real" else "high"
        candidates = [
            spec.config_id
            for spec in scenario.executor_configs
            if spec.kind == "torch_semantic_segmentation" and spec.quality_tier == wanted_tier
        ]
        if not candidates:
            raise SchemaValidationError(
                f"policy {policy_name!r} needs a torch_semantic_segmentation executor with "
                f"quality_tier {wanted_tier!r}; the scenario declares none"
            )
        return StaticPolicy(candidates[0])
    if policy_name in scenario.config_ids:
        return StaticPolicy(policy_name)
    if policy_name in policy_registry.names():
        view = PublicProfileView(
            {
                spec.config_id: PublicProfile(
                    config_id=spec.config_id,
                    expected_latency_ms=spec.mission_latency_s * 1000.0,
                    expected_upload_mb=spec.communication_mb_per_call,
                    quality_tier=QualityTier(spec.quality_tier),
                )
                for spec in scenario.executor_configs
            }
        )
        try:
            return policy_registry.create(policy_name, public_profiles=view)
        except TypeError:
            return policy_registry.create(policy_name)
    raise SchemaValidationError(
        f"unknown policy {policy_name!r}; use always_fast, always_strong, a scenario "
        f"config_id, or one of {policy_registry.names()}"
    )


class MissionRunner:
    """Runs one V2 scenario under one policy, deterministically."""

    def __init__(
        self, scenario: V2Scenario, policy_name: str, *, record_runtime_snapshots: bool = False
    ) -> None:
        self._scenario = scenario
        self._policy_name = policy_name
        self._policy = build_policy(policy_name, scenario)
        #: Off by default so existing result files stay byte-identical. When on, every
        #: ObservationLog carries a ``runtime`` snapshot for replay/dashboard consumers.
        self._record_runtime_snapshots = record_runtime_snapshots

        self._world = open_world(
            _resolve_image_path(scenario),
            scenario.world.meters_per_pixel,
            invalid_pixel_rule=scenario.world.invalid_pixel_rule,
        )
        self._objects = ObjectLayer(scenario.objects, assets=_load_scenario_assets(scenario))
        self._renderer = CameraRenderer(
            scenario_id=scenario.scenario_id,
            world=self._world,
            objects=self._objects,
            camera=scenario.camera,
        )
        self._trajectory = build_trajectory(scenario.trajectory, scenario.drone.speed_mps)
        self._executors = build_executors(scenario.executor_configs)
        self._catalog = _catalog_from_specs(scenario)
        self._validator = ActionValidator(
            catalog=self._catalog,
            allowed_config_ids=scenario.config_ids,
            privacy_level=scenario.contract.privacy_level,
            fallback_config_id=scenario.simulation.fallback_config_id,
        )
        self._evaluator = MissionEvaluator(
            matching_iou_threshold=scenario.simulation.matching_iou_threshold,
            total_targets=len(scenario.targets),
            observation_interval_s=scenario.simulation.observation_interval_s,
        )
        if scenario.simulation.network_trace is not None:
            self._network_model = V2NetworkModel(scenario.simulation.network_trace)
        else:
            bandwidth, rtt, loss = scenario.simulation.network
            self._network_model = constant_network_model(bandwidth, rtt, loss)

    # -- the loop -----------------------------------------------------------------------

    def run(self) -> V2MissionResult:
        scenario = self._scenario
        contract = scenario.contract
        interval = scenario.simulation.observation_interval_s
        capacity_j = scenario.drone.battery_capacity_wh * _SECONDS_PER_WH
        flight_w = scenario.drone.flight_power_w

        mission_time = 0.0
        energy_j = 0.0
        communication_mb = 0.0
        communication_energy_j = 0.0
        battery_frac = 1.0
        current_config: str | None = None
        selections: list[str] = []
        logs: list[ObservationLog] = []
        skipped_ids: list[int] = []
        latencies: list[float] = []
        privacy_violations = 0
        failed_inferences = 0
        network_behaviour = {
            "remote_attempts": 0,
            "remote_successes": 0,
            "remote_failures": 0,
            "remote_timeouts": 0,
            "remote_unreachable": 0,
            "remote_fallbacks": 0,
            "privacy_rejections": 0,
        }
        termination = TERMINATION_PATH_COMPLETE
        notes: dict[str, str] = {}

        schedule_index = 0
        while True:
            capture_time = schedule_index * interval

            # Termination is checked against the *next capture*, so the reason states
            # why no further observation happened.
            if self._trajectory.is_complete_at(capture_time):
                termination = TERMINATION_PATH_COMPLETE
                # The UAV flies the remaining tail of the path; that flight time still
                # burns battery even though no further observation is processed.
                end_time = max(mission_time, self._trajectory.duration_s)
                energy_j += flight_w * max(0.0, end_time - mission_time)
                battery_frac = max(0.0, 1.0 - energy_j / capacity_j)
                mission_time = end_time
                break
            if capture_time > contract.deadline_s:
                termination = TERMINATION_DEADLINE
                break
            if battery_frac <= 0.0:
                termination = TERMINATION_BATTERY
                break

            capture_position = self._trajectory.position_at(capture_time)
            observation = self._renderer.render(
                position_m=capture_position,
                capture_time_s=capture_time,
                observation_id=schedule_index,
            )

            runtime_state = self._runtime_state(
                capture_time,
                capture_position,
                battery_frac,
                energy_j,
                communication_mb,
                schedule_index,
                current_config,
            )
            requested = self._select(runtime_state)
            action = self._validator.validate(requested, current_config_id=current_config)
            config_id = action.config_id
            previous_config = current_config
            if action.outcome is ActionOutcome.PRIVACY_VIOLATION:
                # The attempt is recorded and the constraint fails on the record; the
                # forbidden transfer itself is never simulated (V1 semantics).
                privacy_violations += 1
                network_behaviour["privacy_rejections"] += 1

            executor = self._executors[config_id]
            try:
                if hasattr(executor, "run_with_context"):
                    result: ImageExecutionResult = executor.run_with_context(
                        observation.rgb,
                        ExecutionContext(
                            scenario_id=scenario.scenario_id,
                            observation_id=schedule_index,
                            capture_time_s=capture_time,
                            deadline_s=contract.deadline_s,
                            network=self._network_model.state_at(capture_time),
                        ),
                    )
                else:
                    result = executor.run(observation.rgb)
            except Exception as error:
                termination = TERMINATION_EXECUTOR_FAILURE
                notes["executor_failure"] = f"{type(error).__name__}: {error}"
                mission_time = capture_time
                break
            if not result.success:
                failed_inferences += 1
            remote_info = (result.diagnostics or {}).get("remote_status")
            if remote_info is not None:
                network_behaviour["remote_attempts"] += 1
                if remote_info == "success":
                    network_behaviour["remote_successes"] += 1
                else:
                    network_behaviour["remote_failures"] += 1
                if remote_info == "timeout":
                    network_behaviour["remote_timeouts"] += 1
                if remote_info == "unreachable":
                    network_behaviour["remote_unreachable"] += 1
                if (result.diagnostics or {}).get("fallback") is not None:
                    network_behaviour["remote_fallbacks"] += 1

            completion_time = capture_time + result.mission_latency_s
            completion_position = self._trajectory.position_at(completion_time)

            # Score against the ground truth captured with THIS observation -- never a
            # re-render at completion time.
            score = self._evaluator.update(observation, result.prediction_mask)

            elapsed = completion_time - mission_time
            energy_j += (
                flight_w * max(0.0, elapsed) + result.energy_j + result.communication_energy_j
            )
            communication_mb += result.communication_mb
            communication_energy_j += result.communication_energy_j
            battery_frac = max(0.0, 1.0 - energy_j / capacity_j)
            latencies.append(result.mission_latency_s)
            selections.append(config_id)
            current_config = config_id
            mission_time = completion_time

            next_index = max(schedule_index + 1, math.ceil(mission_time / interval - 1e-9))
            skipped_after = tuple(range(schedule_index + 1, next_index))
            skipped_ids.extend(skipped_after)
            # A target visible only while the executor was busy must still count as
            # encountered-but-missed. Visibility at each skipped capture time is decided
            # geometrically (no render, and the executor never runs on a skipped
            # observation); it feeds the diagnostics only -- recall's denominator is the
            # scenario's total target count regardless.
            for skipped_index in skipped_after:
                skipped_position = self._trajectory.position_at(skipped_index * interval)
                self._evaluator.note_skipped_visibility(
                    self._objects.visible_target_ids_in(
                        self._renderer.footprint_at(skipped_position),
                        self._scenario.camera.output_width_px,
                        self._scenario.camera.output_height_px,
                    )
                )

            runtime_snapshot: dict[str, Any] | None = None
            if self._record_runtime_snapshots:
                # Everything here restates values the loop already computed — the snapshot
                # is a time series for replay/dashboard consumers, never a second ledger.
                runtime_snapshot = {
                    "at_capture": runtime_state.to_dict(),
                    "after_completion": {
                        "battery_frac": battery_frac,
                        "cumulative_energy_j": energy_j,
                        "cumulative_communication_mb": communication_mb,
                        "remaining_deadline_s": contract.deadline_s - completion_time,
                        "path_progress": self._trajectory.progress_at(completion_time),
                    },
                    "config_switched": previous_config is not None and config_id != previous_config,
                    "fallback_used": not action.is_valid,
                    "action_reason": action.reason,
                    "privacy_violations_so_far": privacy_violations,
                }
            logs.append(
                ObservationLog(
                    observation_id=schedule_index,
                    capture_time_s=capture_time,
                    completion_time_s=completion_time,
                    capture_position_m=capture_position,
                    completion_position_m=completion_position,
                    requested_config_id=requested if isinstance(requested, str) else None,
                    executed_config_id=config_id,
                    action_valid=action.is_valid,
                    skipped_after=skipped_after,
                    execution=result.to_summary(),
                    score=score.to_dict(),
                    runtime=runtime_snapshot,
                )
            )
            schedule_index = next_index

        return self._finalise(
            mission_time,
            battery_frac,
            energy_j,
            communication_mb,
            termination,
            selections,
            logs,
            tuple(skipped_ids),
            latencies,
            notes,
            privacy_violations=privacy_violations,
            failed_inferences=failed_inferences,
            communication_energy_j=communication_energy_j,
            network_behaviour=network_behaviour,
        )

    # -- helpers ------------------------------------------------------------------------

    def _select(self, runtime_state: RuntimeState) -> object:
        allowed = self._catalog.subset(self._scenario.config_ids)
        try:
            return self._policy.select_config(self._scenario.contract, runtime_state, allowed)
        except Exception as error:
            return _PolicyFailure(error)

    def _runtime_state(
        self,
        time_s: float,
        position_m: tuple[float, float],
        battery_frac: float,
        energy_j: float,
        communication_mb: float,
        frame_id: int,
        current_config: str | None,
    ) -> RuntimeState:
        del position_m  # policy-visible position arrives via path_progress in V2
        return RuntimeState(
            current_time_s=time_s,
            frame_id=frame_id,
            battery_frac=battery_frac,
            power_mode="v2_fixed",
            # The current sample only -- the policy never sees the trace, the regime
            # name, or anything about the future (V1 boundary, unchanged).
            network=self._network_model.observation_at(time_s),
            current_config_id=current_config,
            remaining_deadline_s=self._scenario.contract.deadline_s - time_s,
            cumulative_energy_j=energy_j,
            cumulative_communication_mb=communication_mb,
            path_progress=self._trajectory.progress_at(time_s),
            evidence_summary=EvidenceSummary(
                predicted_unique_targets=self._evaluator.total_predictions,
                processed_frames=self._evaluator.processed_observations,
                mean_prediction_confidence=0.0,
            ),
        )

    def _finalise(
        self,
        mission_time: float,
        battery_frac: float,
        energy_j: float,
        communication_mb: float,
        termination: str,
        selections: list[str],
        logs: list[ObservationLog],
        skipped_ids: tuple[int, ...],
        latencies: list[float],
        notes: dict[str, str],
        *,
        privacy_violations: int = 0,
        failed_inferences: int = 0,
        communication_energy_j: float = 0.0,
        network_behaviour: dict[str, int] | None = None,
    ) -> V2MissionResult:
        contract = self._scenario.contract
        quality = self._evaluator.quality_details(mission_time_s=mission_time)
        quality_value = float(quality[contract.quality_metric])
        constraints = {
            "quality_success": contract.quality_satisfied(quality_value),
            "deadline_success": mission_time <= contract.deadline_s,
            "battery_constraint_success": battery_frac >= contract.min_final_battery_frac,
            "communication_constraint_success": communication_mb
            <= contract.communication_budget_mb,
            # V3 P1: the fifth hard constraint, aligned with V1 -- a blocked forbidden
            # selection still counts as a violation attempt.
            "privacy_constraint_success": privacy_violations == 0,
        }
        quality["quality_metric"] = contract.quality_metric
        quality["quality_value"] = quality_value
        quality["quality_threshold"] = contract.quality_threshold

        notes.setdefault(
            "measurement_honesty",
            "latency/energy/communication are simulated (configured per executor); "
            "targets are synthetic markers; results are not real UAV perception performance",
        )
        return V2MissionResult(
            scenario_id=self._scenario.scenario_id,
            policy_name=self._policy_name,
            mission_success=all(constraints.values()),
            quality=quality,
            constraints=constraints,
            termination_reason=termination,
            final_time_s=mission_time,
            final_battery_frac=battery_frac,
            cumulative_communication_mb=communication_mb,
            cumulative_energy_j=energy_j,
            path_progress=self._trajectory.progress_at(mission_time),
            processed_observation_count=self._evaluator.processed_observations,
            skipped_observation_count=len(skipped_ids),
            skipped_observation_ids=skipped_ids,
            mean_executor_latency_s=(sum(latencies) / len(latencies)) if latencies else 0.0,
            config_selection_history=tuple(selections),
            observations=logs,
            notes=notes,
            privacy_violation_count=privacy_violations,
            failed_inference_count=failed_inferences,
            communication_energy_j=communication_energy_j,
            network_behaviour=dict(network_behaviour or {}),
        )

    # Exposed for the CLI/visualiser/replay exporter so they render through the same
    # components the mission used — never a parallel implementation.
    @property
    def renderer(self) -> CameraRenderer:
        return self._renderer

    @property
    def trajectory(self):
        return self._trajectory

    @property
    def world(self):
        return self._world

    @property
    def objects(self) -> ObjectLayer:
        return self._objects

    @property
    def network_model(self) -> V2NetworkModel:
        return self._network_model

    def executor_for(self, config_id: str):
        return self._executors[config_id]

    def execution_context_at(self, capture_time_s: float, observation_id: int) -> ExecutionContext:
        """The exact execution context a mission used at ``capture_time_s``.

        Exposed so the replay exporter re-runs a remote executor with the same network
        sample the mission saw -- deterministic under the Model-A transport.
        """
        return ExecutionContext(
            scenario_id=self._scenario.scenario_id,
            observation_id=observation_id,
            capture_time_s=capture_time_s,
            deadline_s=self._scenario.contract.deadline_s,
            network=self._network_model.state_at(capture_time_s),
        )

    def render_at(self, capture_time_s: float, observation_id: int = -1) -> Observation:
        return self._renderer.render(
            position_m=self._trajectory.position_at(capture_time_s),
            capture_time_s=capture_time_s,
            observation_id=observation_id,
        )


class _PolicyFailure:
    """A non-string stand-in for an action the policy failed to produce."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __repr__(self) -> str:
        return f"<policy raised {type(self.error).__name__}: {self.error}>"


def _catalog_from_specs(scenario: V2Scenario) -> ConfigCatalog:
    """A V1 catalog view of the scenario's executor configs, for validation and policies.

    A ``simulated_remote`` config carries REMOTE placement and explicitly declares that
    it transmits raw input, so the frozen V1 privacy logic (`privacy_permits`) applies
    verbatim: forbidden under ``local_only`` and -- because raw is not features -- under
    ``features_only`` too.
    """
    return ConfigCatalog(
        [
            Configuration(
                config_id=spec.config_id,
                model_id=spec.model_strategy_id,
                strategy=Strategy(
                    placement=(
                        Placement.REMOTE if spec.kind == "simulated_remote" else Placement.LOCAL
                    ),
                    precision=Precision.FP32,
                    parameters=(
                        {TRANSMITTED_PAYLOAD_PARAMETER: TransmittedPayload.RAW_INPUT.value}
                        if spec.kind == "simulated_remote"
                        else {}
                    ),
                ),
            )
            for spec in scenario.executor_configs
        ],
        catalog_id=f"{scenario.scenario_id}_CONFIGS",
    )


def _load_scenario_assets(scenario: V2Scenario):
    """Load and validate the target-asset store when the scenario declares one.

    Enforces the person-target rule here, where the manifest and the objects meet: an
    image-asset *target* must reference a ``person``-category asset -- a scenario cannot
    quietly score car cutouts as found people. Distractors may be any category.
    """
    if scenario.assets_manifest is None:
        return None
    from pathlib import Path

    from missionbench.v2.assets import load_asset, load_manifest

    manifest = load_manifest(Path(scenario.assets_manifest))
    store = {}
    for obj in scenario.objects:
        if obj.render_mode != "image_asset":
            continue
        record = manifest.get(obj.asset_id)
        if obj.is_target and record.category != "person":
            raise SchemaValidationError(
                f"target object {obj.object_id!r} references asset {obj.asset_id!r} of "
                f"category {record.category!r}; person-target scenarios require 'person' "
                "assets for targets"
            )
        store[obj.asset_id] = load_asset(record)
    return store


def _resolve_image_path(scenario: V2Scenario):
    from pathlib import Path

    path = Path(scenario.world.image_path)
    if not path.is_file():
        raise SchemaValidationError(
            f"scenario {scenario.scenario_id!r}: world image {path} not found. Large aerial "
            "rasters are not committed; place the source file locally (see the scenario's "
            "provenance block for its origin)."
        )
    return path


def run_mission(
    scenario: V2Scenario, policy_name: str, *, record_runtime_snapshots: bool = False
) -> V2MissionResult:
    """Convenience: build a runner and run the mission once."""
    return MissionRunner(
        scenario, policy_name, record_runtime_snapshots=record_runtime_snapshots
    ).run()
