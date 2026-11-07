"""Exact minimum-energy continuation over declared local costs, without future RGB.

This is resource feasibility only. It neither guarantees perception quality nor
uses mission labels. The runner's integer capture clock is the transition model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from experiments.v4_gazebo_pilot.timing import seconds_to_ns


@dataclass(frozen=True)
class Action:
    config_id: str
    latency_s: float
    energy_j: float
    load_latency_s: float
    load_energy_j: float

    def __post_init__(self):
        if not self.config_id or any(
            not math.isfinite(v) or v < 0
            for v in (self.latency_s, self.energy_j, self.load_latency_s, self.load_energy_j)
        ):
            raise ValueError("Finite nonnegative action costs required")


class Continuation:
    def __init__(self, costs, actions, deadline_s):
        self.costs, self.actions = costs, {a.config_id: a for a in actions}
        if not self.actions or len(self.actions) != len(actions):
            raise ValueError("Unique nonempty actions required")
        if not math.isfinite(deadline_s) or deadline_s <= 0:
            raise ValueError("Finite positive deadline required")
        self.deadline_s, self.schedule = deadline_s, costs.schedule()
        self.values, self.choices = {}, {}
        count = (self.schedule.end_ns - 1) // self.schedule.interval_ns + 1
        # Acyclic transitions always advance at least one capture. Build once,
        # including the no-model initial state, without recursive depth limits.
        for index in reversed(range(count)):
            for previous in (None, *self.actions):
                options = [(self.required(index, previous, c), c) for c in self.actions]
                value, selected = min(options)
                self.values[index, previous] = value
                self.choices[index, previous] = selected

    def transition(self, index, previous, config):
        action = self.actions[config]
        switch = previous != config
        completion, next_index, _ = self.schedule.complete(
            index,
            seconds_to_ns(
                action.latency_s + (action.load_latency_s if switch else 0), allow_zero=True
            ),
        )
        final = self.schedule.at(next_index) >= self.schedule.end_ns
        end_ns = max(self.schedule.end_ns, completion) if final else self.schedule.at(next_index)
        energy = (
            (end_ns - self.schedule.at(index)) / 1e9 * self.costs.flight_power_w
            + action.energy_j
            + (action.load_energy_j if switch else 0)
        )
        return completion / 1e9, next_index, energy, final

    def required(self, index, previous, config):
        completion, next_index, energy, final = self.transition(index, previous, config)
        if completion > self.deadline_s or (final and self.costs.duration_s > self.deadline_s):
            return math.inf
        return energy + (0 if final else self.values[next_index, config])

    def witness(self, index, previous, first):
        """Return a cost-minimizing continuation for independent schedule checks."""
        actions = []
        config = first
        while True:
            completion, next_index, energy, final = self.transition(index, previous, config)
            actions.append((index, config, completion, energy))
            if final:
                return actions
            previous, index = config, next_index
            config = self.choices[index, previous]
