"""Simulator-internal state and the construction of the policy-visible observation.

Two types, deliberately not one:

- :class:`SimulationState` is what the simulator knows -- the full internal position.
- ``RuntimeState`` is what a policy is allowed to see, built fresh from it each step.

Keeping them distinct is what makes the observation boundary auditable. If the runner
handed its own state to the policy, every field added for bookkeeping would silently
become an observation, and a future field carrying ground truth would leak without anyone
editing anything that looks like a policy interface.

:class:`StateManager` owns the transition arithmetic so the episode runner stays an
orchestrator. It composes the injected battery model, the path model, and the timing rules
rather than reimplementing any of them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from missionbench.schemas.contract import Contract
from missionbench.schemas.episode import Episode
from missionbench.schemas.network_trace import NetworkObservation
from missionbench.schemas.platform import PlatformProfile, check_episode_power_mode
from missionbench.schemas.runtime_state import EvidenceSummary, RuntimeState
from missionbench.simulator.battery_model import (
    BatteryModel,
    BatteryState,
    EnergyUsage,
    step_energy_usage,
)
from missionbench.simulator.path import ConstantVelocityPath
from missionbench.simulator.timing import (
    DEFAULT_DECISION_INTERVAL_S,
    DEFAULT_FRAME_INTERVAL_S,
    elapsed_step_time_s,
    frame_id_at,
)

__all__ = ["SimulationState", "StateManager"]


@dataclass(frozen=True, slots=True)
class SimulationState:
    """The simulator's internal position in an episode.

    Frozen: a step produces a new state rather than mutating the old one, so a step record
    that captured the previous state keeps describing what actually happened then.
    """

    current_time_s: float
    frame_id: int
    battery: BatteryState
    cumulative_energy: EnergyUsage
    cumulative_communication_mb: float
    path_progress: float
    current_config_id: str | None
    power_mode: str
    step_index: int


class StateManager:
    """Builds the initial state, advances it one step, and derives policy observations."""

    __slots__ = (
        "_battery_model",
        "_decision_interval_s",
        "_episode",
        "_frame_interval_s",
        "_path",
        "_platform",
    )

    def __init__(
        self,
        *,
        episode: Episode,
        platform: PlatformProfile,
        path: ConstantVelocityPath,
        battery_model: BatteryModel,
        decision_interval_s: float = DEFAULT_DECISION_INTERVAL_S,
        frame_interval_s: float = DEFAULT_FRAME_INTERVAL_S,
    ) -> None:
        # Fail at construction rather than mid-episode: an episode asking for a power mode
        # the platform lacks is a specification error, not a runtime condition to survive.
        check_episode_power_mode(platform, power_mode=episode.power_mode)
        self._episode = episode
        self._platform = platform
        self._path = path
        self._battery_model = battery_model
        self._decision_interval_s = decision_interval_s
        self._frame_interval_s = frame_interval_s

    @property
    def path(self) -> ConstantVelocityPath:
        return self._path

    def initial_state(self) -> SimulationState:
        """Return the state at t=0, before any decision has been made."""
        return SimulationState(
            current_time_s=0.0,
            frame_id=frame_id_at(0.0, frame_interval_s=self._frame_interval_s),
            battery=BatteryState.initial(
                self._platform, initial_battery_frac=self._episode.initial_battery_frac
            ),
            cumulative_energy=EnergyUsage(flight_j=0.0, compute_j=0.0, communication_j=0.0),
            cumulative_communication_mb=0.0,
            path_progress=self._path.progress_at(0.0),
            current_config_id=self._episode.initial_config_id,
            power_mode=self._episode.power_mode,
            step_index=0,
        )

    def advance(
        self,
        state: SimulationState,
        *,
        selected_config_id: str,
        inference_latency_s: float,
        onboard_energy_j: float,
        communication_mb: float,
    ) -> SimulationState:
        """Apply one executed decision and return the resulting state.

        The vehicle flies on whether or not inference succeeded, so time, path progress,
        and flight energy advance even for a failed execution -- the caller passes the
        latency that was actually spent (a timeout, say) with zero communication.

        Args:
            selected_config_id: The configuration that was actually executed, after
                validation -- not necessarily what the policy asked for.
            inference_latency_s: End-to-end latency the executor reported.
            onboard_energy_j: On-board energy the executor reported.
            communication_mb: Bytes moved, upload plus download.
        """
        elapsed_s = elapsed_step_time_s(
            inference_latency_s, decision_interval_s=self._decision_interval_s
        )
        current_time_s = state.current_time_s + elapsed_s
        usage = step_energy_usage(
            self._platform,
            elapsed_time_s=elapsed_s,
            onboard_energy_j=onboard_energy_j,
            communication_mb=communication_mb,
        )
        return replace(
            state,
            current_time_s=current_time_s,
            frame_id=frame_id_at(current_time_s, frame_interval_s=self._frame_interval_s),
            battery=self._battery_model.transition(state.battery, usage, elapsed_s),
            cumulative_energy=state.cumulative_energy + usage,
            cumulative_communication_mb=state.cumulative_communication_mb + communication_mb,
            path_progress=self._path.progress_at(current_time_s),
            current_config_id=selected_config_id,
            step_index=state.step_index + 1,
        )

    def build_runtime_state(
        self,
        state: SimulationState,
        *,
        contract: Contract,
        network: NetworkObservation,
        evidence_summary: EvidenceSummary,
    ) -> RuntimeState:
        """Project internal state onto the observation a policy is permitted to see.

        Everything here is either a resource the policy is spending, a condition it must
        react to, or a summary of its own predictions. Nothing derives from ground truth,
        and nothing reveals the future.
        """
        return RuntimeState(
            current_time_s=state.current_time_s,
            frame_id=state.frame_id,
            battery_frac=state.battery.fraction,
            power_mode=state.power_mode,
            network=network,
            current_config_id=state.current_config_id,
            # Clamped at zero: a negative "remaining" would be a nonsense reading, and the
            # overrun is already captured by current_time_s and the deadline metrics.
            remaining_deadline_s=max(0.0, contract.deadline_s - state.current_time_s),
            cumulative_energy_j=state.cumulative_energy.total_j,
            cumulative_communication_mb=state.cumulative_communication_mb,
            path_progress=state.path_progress,
            evidence_summary=evidence_summary,
        )
