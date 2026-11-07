"""Battery state, energy accounting, and the battery transition model.

Energy is tracked in **joules** internally and converted to a fraction only for reporting
and constraint checks. Working in fractions would make the arithmetic depend on the
platform's capacity, so two platforms could not share a transition model.

The three energy components -- flight, compute, communication -- are tracked separately
throughout, because the benchmark reports them separately and because a policy's whole
job is trading compute and communication against a budget that flight is also draining.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from missionbench.schemas.platform import PlatformProfile

__all__ = [
    "BatteryModel",
    "BatteryState",
    "EnergyLedger",
    "EnergyUsage",
    "SimpleBatteryModel",
    "step_energy_usage",
]


@dataclass(frozen=True, slots=True)
class EnergyUsage:
    """The energy one step consumed, broken down by component."""

    flight_j: float
    compute_j: float
    communication_j: float

    @property
    def total_j(self) -> float:
        return self.flight_j + self.compute_j + self.communication_j

    def __add__(self, other: EnergyUsage) -> EnergyUsage:
        return EnergyUsage(
            flight_j=self.flight_j + other.flight_j,
            compute_j=self.compute_j + other.compute_j,
            communication_j=self.communication_j + other.communication_j,
        )

    def __sub__(self, other: EnergyUsage) -> EnergyUsage:
        """Componentwise difference, for recovering one step's usage from two cumulatives."""
        return EnergyUsage(
            flight_j=self.flight_j - other.flight_j,
            compute_j=self.compute_j - other.compute_j,
            communication_j=self.communication_j - other.communication_j,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "flight_energy_j": self.flight_j,
            "compute_energy_j": self.compute_j,
            "communication_energy_j": self.communication_j,
            "total_energy_j": self.total_j,
        }


@dataclass(frozen=True, slots=True)
class BatteryState:
    """Remaining energy against a fixed capacity."""

    capacity_j: float
    remaining_j: float

    @property
    def fraction(self) -> float:
        return self.remaining_j / self.capacity_j

    @property
    def is_depleted(self) -> bool:
        return self.remaining_j <= 0.0

    @classmethod
    def initial(cls, platform: PlatformProfile, *, initial_battery_frac: float) -> BatteryState:
        """Build the starting state for an episode on this platform."""
        capacity_j = platform.battery_capacity_j
        return cls(capacity_j=capacity_j, remaining_j=initial_battery_frac * capacity_j)


class BatteryModel(Protocol):
    """Advances the battery by one step's energy usage."""

    def transition(
        self,
        previous_state: BatteryState,
        usage: EnergyUsage,
        elapsed_time_s: float,
    ) -> BatteryState:
        """Return the battery state after consuming ``usage`` over ``elapsed_time_s``."""
        ...


class SimpleBatteryModel:
    """Subtracts consumed energy, floored at empty.

    Deliberately ideal: no internal resistance, temperature, voltage sag, or recovery
    effect. ``elapsed_time_s`` is unused here but is part of the protocol because a
    trace-based or electrochemical replacement needs it -- for self-discharge or a
    rate-dependent capacity -- and adding it later would break every implementation.
    """

    def transition(
        self,
        previous_state: BatteryState,
        usage: EnergyUsage,
        elapsed_time_s: float,
    ) -> BatteryState:
        del elapsed_time_s
        return BatteryState(
            capacity_j=previous_state.capacity_j,
            remaining_j=max(0.0, previous_state.remaining_j - usage.total_j),
        )


def step_energy_usage(
    platform: PlatformProfile,
    *,
    elapsed_time_s: float,
    onboard_energy_j: float,
    communication_mb: float,
) -> EnergyUsage:
    """Compute one step's energy breakdown from platform characteristics.

    Flight energy accrues with wall-clock time regardless of what the policy chose, which
    is what makes a slow inference expensive twice over: it burns its own compute energy
    *and* holds the vehicle airborne longer.

    Args:
        platform: Supplies flight power and the per-megabyte communication cost.
        elapsed_time_s: Wall-clock duration of the step, i.e. ``max(decision interval,
            inference latency)``.
        onboard_energy_j: Energy the executor reports for on-board work -- full inference
            for a local configuration, or the smaller capture/encode cost for a remote one.
        communication_mb: Total bytes moved this step, upload plus download.
    """
    return EnergyUsage(
        flight_j=platform.flight_power_w * elapsed_time_s,
        compute_j=onboard_energy_j,
        communication_j=communication_mb * platform.communication_energy_j_per_mb,
    )


class EnergyLedger:
    """Accumulates per-step energy usage across an episode.

    Separate from :class:`BatteryState` on purpose: the battery answers "how much is
    left", the ledger answers "where did it go". The ledger keeps its components even
    after the battery floors at zero, so a run that ran out still reports what it spent.
    """

    __slots__ = ("_total",)

    def __init__(self) -> None:
        self._total = EnergyUsage(flight_j=0.0, compute_j=0.0, communication_j=0.0)

    def add(self, usage: EnergyUsage) -> None:
        self._total = self._total + usage

    @property
    def total(self) -> EnergyUsage:
        return self._total

    @property
    def total_j(self) -> float:
        return self._total.total_j
