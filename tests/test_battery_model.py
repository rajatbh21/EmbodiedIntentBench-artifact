"""Energy accounting and the battery transition.

The numbers here are worked by hand from the formulas in docs/v1_spec.md §8 rather than
from the implementation, so a change in the transition rule fails the test instead of
silently redefining what the benchmark measures.
"""

from __future__ import annotations

import pytest

from missionbench.simulator.battery_model import (
    BatteryState,
    EnergyLedger,
    EnergyUsage,
    SimpleBatteryModel,
    step_energy_usage,
)

# The `platform` fixture (100 Wh, 180 W flight power, 0.5 J/MB) comes from conftest.

# --- capacity and state ------------------------------------------------------------


def test_capacity_converts_watt_hours_to_joules(platform) -> None:
    state = BatteryState.initial(platform, initial_battery_frac=0.80)
    assert state.capacity_j == 360_000.0
    assert state.remaining_j == 288_000.0
    assert state.fraction == 0.80
    assert not state.is_depleted


def test_empty_battery_is_depleted(platform) -> None:
    assert BatteryState(capacity_j=360_000.0, remaining_j=0.0).is_depleted


# --- per-step energy breakdown -----------------------------------------------------


def test_step_energy_uses_the_specified_formula(platform) -> None:
    usage = step_energy_usage(
        platform, elapsed_time_s=1.0, onboard_energy_j=3.0, communication_mb=0.0
    )
    assert usage.flight_j == 180.0  # 180 W * 1 s
    assert usage.compute_j == 3.0
    assert usage.communication_j == 0.0
    assert usage.total_j == 183.0


def test_communication_energy_scales_with_volume(platform) -> None:
    usage = step_energy_usage(
        platform, elapsed_time_s=1.0, onboard_energy_j=1.0, communication_mb=1.55
    )
    assert usage.communication_j == pytest.approx(0.775)  # 1.55 MB * 0.5 J/MB
    assert usage.total_j == pytest.approx(181.775)


def test_a_slow_step_costs_more_flight_energy(platform) -> None:
    """A long inference is expensive twice: its own energy, and the extra airborne time."""
    fast = step_energy_usage(
        platform, elapsed_time_s=1.0, onboard_energy_j=3.0, communication_mb=0.0
    )
    slow = step_energy_usage(
        platform, elapsed_time_s=2.0, onboard_energy_j=3.0, communication_mb=0.0
    )
    assert slow.flight_j == 2 * fast.flight_j
    assert slow.total_j - fast.total_j == 180.0


def test_components_stay_separable(platform) -> None:
    usage = step_energy_usage(
        platform, elapsed_time_s=1.0, onboard_energy_j=12.0, communication_mb=2.0
    )
    assert usage.to_dict() == {
        "flight_energy_j": 180.0,
        "compute_energy_j": 12.0,
        "communication_energy_j": 1.0,
        "total_energy_j": 193.0,
    }


# --- transition ---------------------------------------------------------------------


def test_transition_subtracts_total_energy(platform) -> None:
    model = SimpleBatteryModel()
    state = BatteryState.initial(platform, initial_battery_frac=0.80)
    usage = step_energy_usage(
        platform, elapsed_time_s=1.0, onboard_energy_j=3.0, communication_mb=0.0
    )

    after = model.transition(state, usage, 1.0)

    assert after.remaining_j == 288_000.0 - 183.0
    assert after.capacity_j == state.capacity_j
    assert state.remaining_j == 288_000.0, "the previous state must not be mutated"


def test_transition_floors_at_empty_rather_than_going_negative(platform) -> None:
    model = SimpleBatteryModel()
    nearly_empty = BatteryState(capacity_j=360_000.0, remaining_j=100.0)
    usage = EnergyUsage(flight_j=180.0, compute_j=3.0, communication_j=0.0)

    after = model.transition(nearly_empty, usage, 1.0)

    assert after.remaining_j == 0.0
    assert after.fraction == 0.0
    assert after.is_depleted


def test_transition_is_deterministic(platform) -> None:
    model = SimpleBatteryModel()
    state = BatteryState.initial(platform, initial_battery_frac=0.5)
    usage = EnergyUsage(flight_j=180.0, compute_j=3.0, communication_j=0.5)
    assert model.transition(state, usage, 1.0) == model.transition(state, usage, 1.0)


def test_repeated_transitions_accumulate_as_expected(platform) -> None:
    model = SimpleBatteryModel()
    state = BatteryState.initial(platform, initial_battery_frac=1.0)
    usage = step_energy_usage(
        platform, elapsed_time_s=1.0, onboard_energy_j=0.0, communication_mb=0.0
    )
    for _ in range(10):
        state = model.transition(state, usage, 1.0)
    assert state.remaining_j == 360_000.0 - 10 * 180.0


# --- ledger -------------------------------------------------------------------------


def test_ledger_accumulates_components_separately() -> None:
    ledger = EnergyLedger()
    ledger.add(EnergyUsage(flight_j=180.0, compute_j=3.0, communication_j=0.0))
    ledger.add(EnergyUsage(flight_j=360.0, compute_j=12.0, communication_j=0.75))

    assert ledger.total.flight_j == 540.0
    assert ledger.total.compute_j == 15.0
    assert ledger.total.communication_j == 0.75
    assert ledger.total_j == 555.75


def test_ledger_starts_at_zero() -> None:
    assert EnergyLedger().total_j == 0.0


def test_ledger_keeps_its_record_after_the_battery_floors(platform) -> None:
    """A run that ran out of energy still reports what it spent getting there."""
    model = SimpleBatteryModel()
    ledger = EnergyLedger()
    state = BatteryState(capacity_j=360_000.0, remaining_j=200.0)
    usage = step_energy_usage(
        platform, elapsed_time_s=1.0, onboard_energy_j=12.0, communication_mb=4.0
    )

    for _ in range(3):
        ledger.add(usage)
        state = model.transition(state, usage, 1.0)

    assert state.is_depleted
    assert ledger.total_j == pytest.approx(3 * 194.0)
    assert ledger.total.communication_j == pytest.approx(6.0)


def test_energy_usage_addition_is_componentwise() -> None:
    left = EnergyUsage(flight_j=1.0, compute_j=2.0, communication_j=3.0)
    right = EnergyUsage(flight_j=10.0, compute_j=20.0, communication_j=30.0)
    assert left + right == EnergyUsage(flight_j=11.0, compute_j=22.0, communication_j=33.0)
