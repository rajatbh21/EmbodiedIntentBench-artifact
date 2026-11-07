"""Matched battery-blind control; actual runner resources remain untouched."""

from types import SimpleNamespace

from experiments.lake_intent_contract import immutable
from experiments.mission_routing_v2.policy import plain


class BatteryBlind:
    """Assume 65% at takeoff, still debit observed cumulative energy causally."""

    def __init__(self, delegate, capacity_j):
        self.delegate = delegate
        self.capacity_j = capacity_j
        self.last_decision = None

    def select(self, observation):
        o = observation
        state = {
            k: getattr(o.state, k)
            for k in ("current_time_s", "frame_id", "current_config_id", "cumulative_energy_j")
        }
        state["battery_frac"] = 0.65 - state["cumulative_energy_j"] / self.capacity_j
        intent = plain(o.intent_contract)
        intent["mission"]["initial_battery_frac"] = 0.65
        public = SimpleNamespace(
            state=SimpleNamespace(**state),
            intent_contract=immutable(intent),
            contract=o.contract,
            configs=o.configs,
            resources=o.resources,
            rgb=o.rgb,
            tracking=o.tracking,
            evidence_history=o.evidence_history,
            confirmed_reports_so_far=o.confirmed_reports_so_far,
        )
        selected = self.delegate.select(public)
        self.last_decision = dict(
            self.delegate.last_decision,
            battery_input_ablation="assume_0.65_initial_then_debit_actual_energy",
        )
        return selected
