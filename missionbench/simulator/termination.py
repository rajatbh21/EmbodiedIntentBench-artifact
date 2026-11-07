"""Episode termination conditions.

Three conditions end a V1 episode: the path is flown, the deadline passes, or the battery
empties. Each is a separate object so a future condition -- a geofence, an abort signal --
is an addition rather than another branch inside the runner.

A failed remote inference is deliberately *not* a termination condition. Losing the network
is a situation the policy is supposed to handle, not a reason to stop measuring how it
handles it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from missionbench.schemas.contract import Contract
from missionbench.simulator.state_manager import SimulationState

__all__ = [
    "DEFAULT_TERMINATION_CONDITIONS",
    "BatteryDepleted",
    "DeadlineExceeded",
    "PathComplete",
    "TerminationCondition",
    "TerminationReason",
    "first_triggered",
]


class TerminationReason(StrEnum):
    """Why an episode stopped.

    Diagnostic only. Constraint success is computed from final values, not from this, so
    an episode that ends for one reason can still fail a different constraint.
    """

    PATH_COMPLETE = "path_complete"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    BATTERY_DEPLETED = "battery_depleted"


class TerminationCondition(Protocol):
    """Decides whether an episode should stop in its current state."""

    def check(self, state: SimulationState, contract: Contract) -> TerminationReason | None:
        """Return the reason to stop, or ``None`` to continue."""
        ...


class PathComplete:
    """The vehicle has flown the whole predefined path -- the nominal end of a mission."""

    def check(self, state: SimulationState, contract: Contract) -> TerminationReason | None:
        del contract
        return TerminationReason.PATH_COMPLETE if state.path_progress >= 1.0 else None


class DeadlineExceeded:
    """The contract deadline has passed.

    Strictly *exceeds*: finishing exactly on the deadline is on time. The boundary decides
    ``deadline_success`` for the episode, so it is stated once here rather than re-derived
    wherever the deadline is mentioned.
    """

    def check(self, state: SimulationState, contract: Contract) -> TerminationReason | None:
        if state.current_time_s > contract.deadline_s:
            return TerminationReason.DEADLINE_EXCEEDED
        return None


class BatteryDepleted:
    """No energy remains, so nothing further can be executed or flown."""

    def check(self, state: SimulationState, contract: Contract) -> TerminationReason | None:
        del contract
        return TerminationReason.BATTERY_DEPLETED if state.battery.is_depleted else None


#: Evaluated in this order, and the first match wins. The order is reporting precedence
#: only: a step that both completed the path and drained the battery did complete the
#: path, so that is the more faithful description of what happened.
DEFAULT_TERMINATION_CONDITIONS: tuple[TerminationCondition, ...] = (
    PathComplete(),
    DeadlineExceeded(),
    BatteryDepleted(),
)


def first_triggered(
    conditions: tuple[TerminationCondition, ...],
    state: SimulationState,
    contract: Contract,
) -> TerminationReason | None:
    """Return the first condition's reason, or ``None`` if the episode continues."""
    for condition in conditions:
        reason = condition.check(state, contract)
        if reason is not None:
            return reason
    return None
