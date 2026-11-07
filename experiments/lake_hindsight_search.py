"""Evaluator-only bounded search for executable mission witnesses.

Uses hidden complete outcomes deliberately. Never a deployable routing baseline.
A successful path must be replayed through the frozen runner. Beam failure is
inconclusive. The separate energy relaxation is exact for the declared pool.
"""

from __future__ import annotations

import math
from functools import lru_cache

from experiments.v4_gazebo_pilot.timing import seconds_to_ns


def search(mission, scenario, outcomes, beam_width=500):
    if beam_width < 1:
        raise ValueError("Positive beam width required")
    schedule = scenario.schedule
    interval = schedule.interval_ns / 1e9
    count = math.ceil((schedule.end_ns - schedule.epoch_ns) / schedule.interval_ns)
    specs = {s.config_id: s for s in scenario.executors}
    costs = scenario.resources
    lookup = {(r["capture_id"], r["config_id"]): r for r in outcomes}
    expected = {(i, c) for i in range(count) for c in specs}
    if len(outcomes) != len(expected) or set(lookup) != expected:
        raise ValueError("Complete, nonduplicated outcome table required")
    for (i, _), row in lookup.items():
        if row["capture_time_s"] != i * interval:
            raise ValueError("Outcome does not match capture schedule")
    capacity = scenario.battery_capacity_wh * 3600
    budget = (
        mission["initial_battery_frac"] - mission["min_final_battery_frac"]
    ) * capacity - scenario.duration_s * scenario.flight_power_w
    phases = mission["phases"]

    @lru_cache(None)
    def transition(i, previous, config):
        spec, resource = specs[config], costs.for_config(config)
        if costs.memory_capacity_mb is not None and resource.memory_mb > costs.memory_capacity_mb:
            return None
        switching = previous != config
        latency = (
            scenario.router_latency_s
            + spec.mission_latency_s
            + (resource.load_latency_s if switching else 0)
        )
        completed, next_i, _ = schedule.complete(i, seconds_to_ns(latency, allow_zero=True))
        completed_s = (completed - schedule.epoch_ns) / 1e9
        if completed_s > scenario.contract.deadline_s:
            return None
        energy = (
            spec.energy_j_per_call
            + scenario.router_energy_j
            + (resource.load_energy_j if switching else 0)
        )
        if next_i >= count:
            energy += max(0, completed_s - scenario.duration_s) * scenario.flight_power_w
        return completed_s, next_i, energy

    @lru_cache(None)
    def minimum_energy(i, previous):
        if i >= count:
            return 0.0
        values = []
        for config in specs:
            tr = transition(i, previous, config)
            if tr:
                _, next_i, energy = tr
                values.append(energy + minimum_energy(next_i, config))
        return min(values, default=math.inf)

    lower = minimum_energy(0, None)
    base = dict(
        gt_aware=True,
        deployable_policy=False,
        beam_width=beam_width,
        energy_budget_after_nominal_flight_j=budget,
        minimum_compute_load_router_tail_energy_j=lower if math.isfinite(lower) else None,
        energy_relaxation_exact=True,
        energy_relaxation="Mandatory calls at ready captures; no perception/evidence constraints",
    )
    if lower > budget + 1e-8:
        return dict(
            **base, conclusion="certified_resource_infeasible", schedule=None, beam_pruned_states=0
        )

    def empty(k):
        return tuple((0, 0) for _ in phases[k]["evaluator_target_ids"]) if k < len(phases) else ()

    def normalize(k, history, false, index):
        while k < len(phases) and index * interval >= phases[k]["end_s"]:
            phase = phases[k]
            if (
                sum(n == -1 for n, _ in history) / len(history) < phase["required_recall"]
                or false > phase["max_false_components"]
            ):
                return None
            k += 1
            history, false = empty(k), 0
        if k < len(phases):
            slots = math.ceil(phases[k]["min_evidence_span_s"] / interval)
            history = tuple(
                (n, max(first, index - slots)) if n > 0 else (n, 0) for n, first in history
            )
        return k, history, false

    # Bucket values are nonflight energy and a capture-index/config witness.
    buckets = {0: {(None, 0, empty(0), 0): (0.0, ())}}
    terminals, pruned, visited = [], 0, 0
    for i in range(count):
        states = buckets.pop(i, {})

        def priority(item):
            (_, k, history, false), (energy, _) = item
            progress = (
                sum(1 if n == -1 else n / (phases[k]["min_hits"] + 1) for n, _ in history)
                if k < len(phases)
                else 0
            )
            return (-k, -progress, false, energy, repr(item[0]))

        ordered = sorted(states.items(), key=priority)
        pruned += max(0, len(ordered) - beam_width)
        for (previous, k, history, false), (energy, path) in ordered[:beam_width]:
            visited += 1
            for config in specs:
                tr = transition(i, previous, config)
                if tr is None:
                    continue
                completed_s, next_i, cost = tr
                spent = energy + cost
                if spent + minimum_energy(next_i, config) > budget + 1e-8:
                    continue
                new_history, new_false = history, false
                if k < len(phases) and phases[k]["start_s"] <= i * interval < phases[k]["end_s"]:
                    phase = phases[k]
                    row = lookup[i, config]
                    new_false += row["false_positive_components"]
                    if new_false > phase["max_false_components"]:
                        continue
                    if completed_s <= phase["deadline_s"]:
                        updated = []
                        for target, (n, first) in zip(
                            phase["evaluator_target_ids"], history, strict=True
                        ):
                            if n != -1 and target in row["matched_target_ids"]:
                                first = i if n == 0 else first
                                n = min(n + 1, phase["min_hits"])
                                if (
                                    n >= phase["min_hits"]
                                    and (i - first) * interval >= phase["min_evidence_span_s"]
                                ):
                                    n, first = -1, 0
                            updated.append((n, first))
                        new_history = tuple(updated)
                normalized = normalize(k, new_history, new_false, next_i)
                if normalized is None:
                    continue
                new_path = (*path, (i, config))
                if next_i >= count:
                    terminals.append((spent, new_path))
                    continue
                key = (config, *normalized)
                bucket = buckets.setdefault(next_i, {})
                if key not in bucket or spent < bucket[key][0]:
                    bucket[key] = (spent, new_path)
    if not terminals:
        return dict(
            **base,
            conclusion="search_inconclusive",
            schedule=None,
            beam_pruned_states=pruned,
            states_visited=visited,
        )
    energy, path = min(terminals)
    return dict(
        **base,
        conclusion="candidate_requires_runner_verification",
        schedule=[dict(capture_id=i, config_id=c) for i, c in path],
        planned_nonflight_and_tail_energy_j=energy,
        beam_pruned_states=pruned,
        states_visited=visited,
        optimality_claim=False,
    )


class HindsightSchedule:
    """GT-derived evaluator witness, deliberately not exposed as a normal router."""

    def __init__(self, schedule):
        self.schedule = tuple(schedule)
        self.index = 0

    def select(self, observation):
        # Runtime verification checks actual capture IDs against the entire plan.
        config = self.schedule[self.index]["config_id"]
        self.index += 1
        return config
