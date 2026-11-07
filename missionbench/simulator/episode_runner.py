"""The decision loop.

This is the benchmark core, and it is deliberately dull. It sequences components it knows
only through protocols: it never imports a task, a concrete executor, or a policy, never
branches on ``task_id``, and never reads a ``config_id`` for meaning. Adding a second task,
executor, battery model, or policy must not require editing this file -- that is the
acceptance criterion the architecture is built around.

Per step, in the order the specification fixes:

1. Observe the network at the current time.
2. Build the policy-visible ``RuntimeState``.
3. Ask the policy for a configuration.
4. Validate the action, substituting if it is not usable.
5. Execute the validated configuration on the current frame.
6. Fold the result into evidence.
7. Advance time, path progress, energy, battery, and communication.
8. Record the step, then check termination.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

from missionbench.executor.base import ExecutionRequest, Executor, executor_id_of
from missionbench.schemas.configuration import ConfigCatalog
from missionbench.schemas.contract import Contract
from missionbench.schemas.episode import Episode
from missionbench.simulator.action_validator import ActionOutcome, ActionValidator, is_switch
from missionbench.simulator.network_trace import NetworkModel
from missionbench.simulator.records import EpisodeRecord, StepRecord
from missionbench.simulator.state_manager import SimulationState, StateManager
from missionbench.simulator.termination import (
    DEFAULT_TERMINATION_CONDITIONS,
    TerminationCondition,
    first_triggered,
)
from missionbench.simulator.timing import frames_skipped

__all__ = ["DEFAULT_MAX_STEPS", "EpisodeRunner"]

_LOGGER: Final = logging.getLogger(__name__)

#: Safety stop. Every termination condition depends on the clock advancing, and it always
#: does -- a step lasts at least one decision interval -- so this should be unreachable. It
#: exists so that a future component with a bug produces a clear error instead of hanging.
DEFAULT_MAX_STEPS: Final = 100_000

#: Battery fractions whose crossing is logged, so adaptation latency against battery events
#: can be computed later. Not used by any V1 metric.
_BATTERY_EVENT_THRESHOLDS: Final = (0.75, 0.50, 0.35, 0.25)


class EpisodeRunner:
    """Runs one episode and returns its record."""

    __slots__ = (
        "_allowed_configs",
        "_catalog",
        "_conditions",
        "_contract",
        "_episode",
        "_executor",
        "_executor_id",
        "_max_steps",
        "_network",
        "_policy",
        "_policy_name",
        "_state_manager",
        "_tracker",
        "_validator",
    )

    def __init__(
        self,
        *,
        episode: Episode,
        contract: Contract,
        catalog: ConfigCatalog,
        state_manager: StateManager,
        network_model: NetworkModel,
        executor: Executor,
        policy: object,
        evidence_tracker: object,
        action_validator: ActionValidator,
        termination_conditions: Sequence[TerminationCondition] = DEFAULT_TERMINATION_CONDITIONS,
        policy_name: str = "",
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._episode = episode
        self._contract = contract
        self._catalog = catalog
        self._allowed_configs = catalog.subset(episode.allowed_config_ids)
        self._state_manager = state_manager
        self._network = network_model
        self._executor = executor
        self._policy = policy
        self._tracker = evidence_tracker
        self._validator = action_validator
        self._conditions = tuple(termination_conditions)
        self._policy_name = policy_name or type(policy).__name__
        # Read off the executor itself. Accepting a name here is what previously let a
        # ReplayExecutor be recorded as "profile": two sources of truth, one of them a
        # default that no caller had to override.
        self._executor_id = executor_id_of(executor)
        self._max_steps = max_steps

    def run(self) -> EpisodeRecord:
        """Run the episode to termination and return its record."""
        state = self._state_manager.initial_state()
        steps: list[StepRecord] = []
        selections: list[str] = []
        battery_events: list[float] = []
        invalid_actions = privacy_violations = failed_inferences = 0
        switches = frames_skipped_total = 0

        reason = first_triggered(self._conditions, state, self._contract)
        while reason is None:
            if len(steps) >= self._max_steps:
                raise RuntimeError(
                    f"episode {self._episode.episode_id!r} exceeded {self._max_steps} steps "
                    "without terminating; a component is not advancing the clock"
                )

            network = self._network.observe(state.current_time_s)
            observation = self._state_manager.build_runtime_state(
                state,
                contract=self._contract,
                network=network,
                evidence_summary=self._tracker.policy_summary(),
            )

            action = self._validator.validate(
                self._ask_policy(observation),
                current_config_id=state.current_config_id,
            )
            if not action.is_valid:
                invalid_actions += 1
                if action.outcome is ActionOutcome.PRIVACY_VIOLATION:
                    privacy_violations += 1
                _LOGGER.debug(
                    "episode %s step %d: %s", self._episode.episode_id, len(steps), action.reason
                )

            result = self._executor.execute(
                ExecutionRequest(
                    episode_id=self._episode.episode_id,
                    frame_id=state.frame_id,
                    configuration=self._catalog.get(action.config_id),
                    network=network,
                    current_time_s=state.current_time_s,
                    seed=self._episode.seed,
                )
            )
            failed_inferences += not result.success
            self._tracker.update(result)

            switched = is_switch(state.current_config_id, action.config_id)
            switches += switched
            selections.append(action.config_id)

            next_state = self._state_manager.advance(
                state,
                selected_config_id=action.config_id,
                inference_latency_s=result.latency_s,
                onboard_energy_j=result.onboard_energy_j,
                communication_mb=result.communication_mb,
            )
            skipped = frames_skipped(state.frame_id, next_state.frame_id)
            frames_skipped_total += skipped
            battery_events.extend(
                _crossed_thresholds(state.battery.fraction, next_state.battery.fraction, next_state)
            )

            steps.append(
                StepRecord(
                    step_index=len(steps),
                    time_s=state.current_time_s,
                    frame_id=state.frame_id,
                    state=observation.to_dict(),
                    action=action.to_dict(),
                    executed_config_id=action.config_id,
                    execution=result.to_dict(),
                    energy=(next_state.cumulative_energy - state.cumulative_energy).to_dict(),
                    elapsed_s=next_state.current_time_s - state.current_time_s,
                    frames_skipped=skipped,
                    is_switch=switched,
                )
            )

            state = next_state
            reason = first_triggered(self._conditions, state, self._contract)

        return EpisodeRecord(
            episode_id=self._episode.episode_id,
            contract_id=self._contract.contract_id,
            policy_name=self._policy_name,
            executor_id=self._executor_id,
            steps=tuple(steps),
            termination_reason=reason,
            final_time_s=state.current_time_s,
            final_battery_frac=state.battery.fraction,
            final_path_progress=state.path_progress,
            cumulative_energy=state.cumulative_energy,
            cumulative_communication_mb=state.cumulative_communication_mb,
            evidence=self._tracker.final_record(),
            invalid_action_count=invalid_actions,
            privacy_violation_count=privacy_violations,
            failed_inference_count=failed_inferences,
            configuration_switch_count=switches,
            frames_skipped_total=frames_skipped_total,
            network_change_times_s=_network_change_times(self._network),
            battery_event_times_s=tuple(battery_events),
            seed=self._episode.seed,
            config_selection_history=tuple(selections),
        )

    def _ask_policy(self, observation) -> object:
        """Call the policy, converting a raised exception into an invalid action.

        A submitted policy that crashes must not take the benchmark with it -- the run
        should report that it misbehaved, which is a result. The exception is preserved in
        the step record rather than swallowed, so a genuine bug is still diagnosable.
        """
        try:
            return self._policy.select_config(self._contract, observation, self._allowed_configs)
        except Exception as error:
            _LOGGER.warning(
                "policy %s raised %s; treating as an invalid action",
                self._policy_name,
                error.__class__.__name__,
            )
            return _PolicyFailure(error)


class _PolicyFailure:
    """Stands in for an action a policy failed to produce. Never a valid configuration ID."""

    __slots__ = ("error",)

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __repr__(self) -> str:
        return f"<policy raised {self.error.__class__.__name__}: {self.error}>"


def _crossed_thresholds(before: float, after: float, state: SimulationState) -> list[float]:
    return [
        state.current_time_s
        for threshold in _BATTERY_EVENT_THRESHOLDS
        if before > threshold >= after
    ]


def _network_change_times(model: NetworkModel) -> tuple[float, ...]:
    """Read change times if the model exposes them; not every model can."""
    getter = getattr(model, "change_times_s", None)
    return tuple(getter()) if callable(getter) else ()
