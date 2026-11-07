"""GT-aware offline-optimal mission skyline (V3 P3).

Answers "what is the best any configuration-selection policy could possibly have
achieved on this mission?" by exhaustive dynamic programming over the deterministic
closed loop. The skyline **deliberately uses information no policy may see** — every
capture's ground-truth detection outcome for every configuration, computed before
choosing — so its result is an *upper bound*, never a policy and never a baseline row.
Every output is labelled ``gt_aware: true``; presenting a skyline number as a policy
result is a defect.

What it bounds and how:

- The mission loop is replayed exactly (capture on the interval grid, configured
  latency moves the clock and skips captures, flight+compute+radio energy drain one
  battery, uploads accrue against the budget, the trajectory tail is flown after the
  last capture). Any divergence from ``MissionRunner`` semantics invalidates the
  bound, so per-slot outcomes come from the mission's own renderer, executors, and
  evaluator — never a parallel implementation.
- The action space is the scenario's configurations **minus privacy-forbidden ones**:
  the skyline optimises over *legal* sequences, because an "optimum" that violates a
  hard constraint bounds nothing.
- The objective is the contract's ``target_recall`` (unique targets found / total),
  maximised subject to the deadline, battery, communication, and privacy constraints;
  the skyline also reports whether *any* legal sequence satisfies the full contract.

Search: forward DP over (capture slot, found-target set) with Pareto pruning on
(mission clock, energy spent, communication spent) — three monotone resources; a
state dominated on all three can never lead to a better terminal. Determinism: pure
function of the scenario (no wall-clock, no RNG); results are byte-stable JSON.

Cost: one render per reachable slot and one executor call per (slot, legal config) —
with real models this is minutes per scenario and stays local-only; the module itself
imports CI-safe.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

from missionbench.schemas.configuration import ConfigCatalog
from missionbench.simulator.action_validator import privacy_permits
from missionbench.v2.camera import CameraRenderer
from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import ImageExecutionResult, build_executors
from missionbench.v2.network import V2NetworkModel, constant_network_model
from missionbench.v2.objects import ObjectLayer
from missionbench.v2.remote import ExecutionContext
from missionbench.v2.runner import (
    _catalog_from_specs,
    _load_scenario_assets,
    _resolve_image_path,
)
from missionbench.v2.scenario import V2Scenario
from missionbench.v2.trajectory import build_trajectory
from missionbench.v2.world import open_world

__all__ = ["SkylineResult", "compute_skyline"]

_SECONDS_PER_WH: Final = 3600.0
#: Pareto slack: resources within this epsilon are treated as equal so float noise
#: never keeps a duplicate state alive.
_EPS: Final = 1e-9


@dataclass(frozen=True, slots=True)
class _SlotOption:
    """The deterministic outcome of running one config at one capture slot."""

    config_id: str
    latency_s: float
    energy_j: float  # onboard compute (+ fallback) energy for this call
    communication_mb: float
    communication_energy_j: float
    matched_target_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class _State:
    """A reachable point in the mission's resource space, with its history."""

    slot: int
    mission_time_s: float  # completion time of the previous observation
    energy_j: float
    communication_mb: float
    found: frozenset[str]
    history: tuple[tuple[int, str], ...]  # (slot, config_id) decisions so far


@dataclass(frozen=True, slots=True)
class SkylineResult:
    """The offline upper bound for one scenario. GT-aware by construction."""

    scenario_id: str
    best_recall: float
    best_success: bool
    #: The sequence achieving ``best_recall`` among success-satisfying sequences when
    #: one exists, otherwise among all legal sequences: (capture slot, config_id).
    best_sequence: tuple[tuple[int, str], ...]
    best_final_battery_frac: float
    best_communication_mb: float
    total_targets: int
    evaluated_slots: int
    legal_config_ids: tuple[str, ...]
    states_explored: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "skyline": {
                "gt_aware": True,
                "label": (
                    "OFFLINE UPPER BOUND: computed with capture-time ground truth for "
                    "every configuration; not a policy, not a baseline, never "
                    "comparable as one"
                ),
                "scenario_id": self.scenario_id,
                "best_recall": self.best_recall,
                "contract_satisfiable": self.best_success,
                "best_sequence": [list(step) for step in self.best_sequence],
                "best_final_battery_frac": self.best_final_battery_frac,
                "best_communication_mb": self.best_communication_mb,
                "total_targets": self.total_targets,
                "evaluated_slots": self.evaluated_slots,
                "legal_config_ids": list(self.legal_config_ids),
                "states_explored": self.states_explored,
            }
        }


def compute_skyline(scenario: V2Scenario) -> SkylineResult:
    """Exhaustive offline optimum over legal config sequences for one scenario.

    Only ``target_recall`` contracts are supported — the DP's objective must be the
    contract's own quality metric, and recall over a found-set is the one V2 metric a
    set-valued DP state can represent exactly.
    """
    contract = scenario.contract
    if contract.quality_metric != "target_recall":
        raise ValueError(
            f"skyline supports target_recall contracts only, got {contract.quality_metric!r}"
        )

    interval = scenario.simulation.observation_interval_s
    capacity_j = scenario.drone.battery_capacity_wh * _SECONDS_PER_WH
    flight_w = scenario.drone.flight_power_w
    total_targets = len(scenario.targets)

    world = open_world(
        _resolve_image_path(scenario),
        scenario.world.meters_per_pixel,
        invalid_pixel_rule=scenario.world.invalid_pixel_rule,
    )
    objects = ObjectLayer(scenario.objects, assets=_load_scenario_assets(scenario))
    renderer = CameraRenderer(
        scenario_id=scenario.scenario_id, world=world, objects=objects, camera=scenario.camera
    )
    trajectory = build_trajectory(scenario.trajectory, scenario.drone.speed_mps)
    executors = build_executors(scenario.executor_configs)
    catalog = _catalog_from_specs(scenario)
    if scenario.simulation.network_trace is not None:
        network: V2NetworkModel = V2NetworkModel(scenario.simulation.network_trace)
    else:
        bandwidth, rtt, loss = scenario.simulation.network
        network = constant_network_model(bandwidth, rtt, loss)

    legal_ids = _legal_config_ids(scenario, catalog)

    # --- per-slot outcomes, from the mission's own components --------------------------
    options: dict[int, tuple[_SlotOption, ...]] = {}
    slot = 0
    while True:
        capture_time = slot * interval
        if trajectory.is_complete_at(capture_time) or capture_time > contract.deadline_s:
            break
        observation = renderer.render(
            position_m=trajectory.position_at(capture_time),
            capture_time_s=capture_time,
            observation_id=slot,
        )
        slot_options = []
        for config_id in legal_ids:
            executor = executors[config_id]
            if hasattr(executor, "run_with_context"):
                result: ImageExecutionResult = executor.run_with_context(
                    observation.rgb,
                    ExecutionContext(
                        scenario_id=scenario.scenario_id,
                        observation_id=slot,
                        capture_time_s=capture_time,
                        deadline_s=contract.deadline_s,
                        network=network.state_at(capture_time),
                    ),
                )
            else:
                result = executor.run(observation.rgb)
            # A throwaway evaluator scores this single frame with the mission's own
            # matching discipline; only the matched ids feed the DP.
            scorer = MissionEvaluator(
                matching_iou_threshold=scenario.simulation.matching_iou_threshold,
                total_targets=total_targets,
                observation_interval_s=interval,
            )
            score = scorer.update(observation, result.prediction_mask)
            slot_options.append(
                _SlotOption(
                    config_id=config_id,
                    latency_s=result.mission_latency_s,
                    energy_j=result.energy_j,
                    communication_mb=result.communication_mb,
                    communication_energy_j=result.communication_energy_j,
                    matched_target_ids=frozenset(score.matched_target_ids),
                )
            )
        options[slot] = tuple(slot_options)
        slot += 1
    evaluated_slots = slot

    # --- forward DP with Pareto pruning ------------------------------------------------
    start = _State(0, 0.0, 0.0, 0.0, frozenset(), ())
    frontier: dict[tuple[int, frozenset[str]], list[_State]] = {(0, frozenset()): [start]}
    terminals: list[_State] = []
    states_explored = 0

    for current_slot in range(evaluated_slots + 1):
        keys = [key for key in frontier if key[0] == current_slot]
        for key in keys:
            for state in frontier.pop(key):
                states_explored += 1
                capture_time = state.slot * interval
                if (
                    state.slot >= evaluated_slots
                    or trajectory.is_complete_at(capture_time)
                    or capture_time > contract.deadline_s
                ):
                    terminals.append(state)
                    continue
                battery = 1.0 - state.energy_j / capacity_j
                if battery <= 0.0:
                    terminals.append(state)
                    continue
                for option in options[state.slot]:
                    completion = capture_time + option.latency_s
                    elapsed = completion - state.mission_time_s
                    energy = (
                        state.energy_j
                        + flight_w * max(0.0, elapsed)
                        + option.energy_j
                        + option.communication_energy_j
                    )
                    communication = state.communication_mb + option.communication_mb
                    if communication > contract.communication_budget_mb + _EPS:
                        continue  # spend is monotone; this branch can never satisfy
                    next_slot = max(state.slot + 1, math.ceil(completion / interval - 1e-9))
                    successor = _State(
                        slot=next_slot,
                        mission_time_s=completion,
                        energy_j=energy,
                        communication_mb=communication,
                        found=state.found | option.matched_target_ids,
                        history=(*state.history, (state.slot, option.config_id)),
                    )
                    _insert_pareto(frontier, successor)
    # Anything left beyond the last evaluated slot is terminal by construction.
    for states in frontier.values():
        terminals.extend(states)
        states_explored += len(states)

    best = _select_best(terminals, scenario, trajectory, capacity_j, flight_w)
    best_state, best_success, best_battery = best
    recall = len(best_state.found) / total_targets if total_targets else 1.0
    return SkylineResult(
        scenario_id=scenario.scenario_id,
        best_recall=recall,
        best_success=best_success,
        best_sequence=best_state.history,
        best_final_battery_frac=best_battery,
        best_communication_mb=best_state.communication_mb,
        total_targets=total_targets,
        evaluated_slots=evaluated_slots,
        legal_config_ids=legal_ids,
        states_explored=states_explored,
    )


# --- helpers ----------------------------------------------------------------------------------


def _legal_config_ids(scenario: V2Scenario, catalog: ConfigCatalog) -> tuple[str, ...]:
    """Configs the contract's privacy level permits, in scenario order."""
    permitted = []
    for spec in scenario.executor_configs:
        configuration = catalog.get(spec.config_id)
        if privacy_permits(scenario.contract.privacy_level, configuration):
            permitted.append(spec.config_id)
    if not permitted:
        raise ValueError(
            f"scenario {scenario.scenario_id!r}: every configuration violates privacy "
            f"level {scenario.contract.privacy_level!r}; no legal skyline exists"
        )
    return tuple(permitted)


def _insert_pareto(
    frontier: dict[tuple[int, frozenset[str]], list[_State]], candidate: _State
) -> None:
    """Keep only states not dominated on (mission_time, energy, communication)."""
    key = (candidate.slot, candidate.found)
    bucket = frontier.setdefault(key, [])
    for existing in bucket:
        if (
            existing.mission_time_s <= candidate.mission_time_s + _EPS
            and existing.energy_j <= candidate.energy_j + _EPS
            and existing.communication_mb <= candidate.communication_mb + _EPS
        ):
            return  # dominated: strictly no better on any axis
    bucket[:] = [
        existing
        for existing in bucket
        if not (
            candidate.mission_time_s <= existing.mission_time_s + _EPS
            and candidate.energy_j <= existing.energy_j + _EPS
            and candidate.communication_mb <= existing.communication_mb + _EPS
        )
    ]
    bucket.append(candidate)


def _select_best(
    terminals: list[_State],
    scenario: V2Scenario,
    trajectory: Any,
    capacity_j: float,
    flight_w: float,
) -> tuple[_State, bool, float]:
    """Pick the best terminal: success-satisfying first, then recall, then battery."""
    contract = scenario.contract
    total = len(scenario.targets)
    scored: list[tuple[bool, float, float, _State]] = []
    for state in terminals:
        capture_time = state.slot * scenario.simulation.observation_interval_s
        # Mirror the runner's endings: a completed path flies its tail; a deadline or
        # battery ending stops the clock at the last completion.
        if trajectory.is_complete_at(capture_time):
            end_time = max(state.mission_time_s, trajectory.duration_s)
            energy = state.energy_j + flight_w * max(0.0, end_time - state.mission_time_s)
            mission_time = end_time
        else:
            energy = state.energy_j
            mission_time = state.mission_time_s
        battery = max(0.0, 1.0 - energy / capacity_j)
        recall = len(state.found) / total if total else 1.0
        success = (
            contract.quality_satisfied(recall)
            and mission_time <= contract.deadline_s
            and battery >= contract.min_final_battery_frac
            and state.communication_mb <= contract.communication_budget_mb
        )
        scored.append((success, recall, battery, state))
    scored.sort(
        key=lambda entry: (entry[0], entry[1], entry[2], -len(entry[3].history)), reverse=True
    )
    success, _recall, battery, state = scored[0]
    return state, success, battery
