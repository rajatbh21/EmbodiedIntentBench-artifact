"""Exact single-stage optimistic latency bound; evaluator-only, not a router.

One inference at a time, actual capture grid and configured completion times.
Free idling, zero switching/load, unlimited energy/memory, ignored false alarms,
and independent phase starts make this an upper bound for legal mission policy
evidence. A failed stage certifies impossibility in the named pool/cost cohort.
Passing never proves full mission feasibility. No search truncation is used.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.scenario import load
from experiments.lake_rotation_study import CONTROL, ENSEMBLE, ROTATED, candidate_specs
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


def latency_bound(phase, outcomes, schedule, costs, router_latency_s):
    """DP over capture index and losslessly compressed repeated-evidence history."""
    targets = phase["evaluator_target_ids"]
    hits = phase["min_hits"]
    span_ns = seconds_to_ns(phase["min_evidence_span_s"], allow_zero=True)
    span_slots = math.ceil(span_ns / schedule.interval_ns)
    first = math.ceil(seconds_to_ns(phase["start_s"], allow_zero=True) / schedule.interval_ns)
    end = math.ceil(seconds_to_ns(phase["end_s"]) / schedule.interval_ns)
    deadline_ns = schedule.epoch_ns + seconds_to_ns(phase["deadline_s"])
    lookup = {(r["capture_id"], r["config_id"]): r for r in outcomes}
    expected = {(i, c) for i in range(first, end) for c in costs}
    selected = [
        (r["capture_id"], r["config_id"])
        for r in outcomes
        if first <= r["capture_id"] < end and r["config_id"] in costs
    ]
    if len(selected) != len(expected) or set(selected) != expected:
        raise ValueError("Latency bound requires every phase capture/config pair exactly once")
    for i, c in expected:
        if lookup[i, c]["capture_time_s"] != i * schedule.interval_ns / 1e9:
            raise ValueError("Outcome capture time differs from the canonical schedule")

    def normalized(state, index):
        # Once the first hit is old enough, its precise age cannot affect a
        # future hit. Waiting alone must never turn a target into a success.
        return tuple(
            (count, max(first_hit, index - span_slots)) if count > 0 else (count, 0)
            for count, first_hit in state
        )

    empty = tuple((0, 0) for _ in targets)
    buckets = {first: {empty}}
    best = 0
    states_visited = 0
    for index in range(first, end):
        states = {normalized(s, index) for s in buckets.pop(index, set())}
        states_visited += len(states)
        for state in states:
            best = max(best, sum(count == -1 for count, _ in state))
            buckets.setdefault(index + 1, set()).add(normalized(state, index + 1))
            for config, latency_s in costs.items():
                completion, next_index, _ = schedule.complete(
                    index, seconds_to_ns(latency_s + router_latency_s, allow_zero=True)
                )
                if completion > deadline_ns:
                    continue
                matched = set(lookup[index, config]["matched_target_ids"])
                updated = []
                for target, (count, first_hit) in zip(targets, state, strict=True):
                    if count != -1 and target in matched:
                        first_hit = index if count == 0 else first_hit
                        count = min(hits, count + 1)
                        if count >= hits and index - first_hit >= span_slots:
                            count, first_hit = -1, 0
                    updated.append((count, first_hit))
                supported = sum(count == -1 for count, _ in updated)
                best = max(best, supported)
                if next_index < end:
                    buckets.setdefault(next_index, set()).add(normalized(updated, next_index))
        if best == len(targets):
            # The trivial cardinality ceiling is attained; further search
            # cannot improve this exact maximum.
            break
    recall = best / len(targets)
    return dict(
        phase_id=phase["phase_id"],
        max_supported_targets=best,
        evidence_recall_upper_bound=recall,
        required_recall=phase["required_recall"],
        cannot_meet_recall=recall < phase["required_recall"],
        states_visited=states_visited,
        exact_relaxation=True,
        search_truncated=False,
        gt_aware=True,
        deployable_policy=False,
        ignored_constraints=[
            "energy",
            "memory",
            "false_positives",
            "switching_and_loading",
            "incoming_phase_busy_state",
            "mandatory_inference_at_ready_capture",
        ],
    )


def run(output):
    protocol = read(output / "protocol.json")
    source = Path(protocol["source"])
    suite = read(source / "suite.json")
    missions = read(source / "missions.json")["missions"]
    host = read(output / "costs.json")["host_latency_sensitivity"]
    results = []
    for env in protocol["environments"]:
        scenario = load(source / "scenario.json", suite["environments"][env], Path(suite["assets"]))
        control = next(s for s in scenario.executors if s.config_id == CONTROL)
        expanded = [*scenario.executors, *candidate_specs(control)]
        configured = {s.config_id: s.mission_latency_s for s in expanded}
        cohorts = {
            "configured_three": {c: configured[c] for c in (CONTROL, ROTATED, ENSEMBLE)},
            "configured_augmented_seven": configured,
            "host_timing_three": host,
        }
        outcomes = read(output / env / "outcomes.json")
        cache = {}
        for cohort, costs in cohorts.items():
            for mission in missions:
                phases = []
                for phase in mission["phases"]:
                    key = (
                        cohort,
                        tuple(phase["evaluator_target_ids"]),
                        phase["start_s"],
                        phase["end_s"],
                        phase["deadline_s"],
                        phase["min_hits"],
                        phase["min_evidence_span_s"],
                    )
                    if key not in cache:
                        cache[key] = latency_bound(
                            phase, outcomes, scenario.schedule, costs, scenario.router_latency_s
                        )
                    answer = {
                        **cache[key],
                        "phase_id": phase["phase_id"],
                        "required_recall": phase["required_recall"],
                    }
                    answer["cannot_meet_recall"] = (
                        answer["evidence_recall_upper_bound"] < phase["required_recall"]
                    )
                    phases.append(answer)
                impossible = any(p["cannot_meet_recall"] for p in phases)
                results.append(
                    dict(
                        environment=env,
                        cost_cohort=cohort,
                        mission_id=mission["mission_id"],
                        conclusion="certified_latency_infeasible"
                        if impossible
                        else "feasibility_unresolved",
                        phases=phases,
                    )
                )
                print(
                    f"{env}/{cohort}/{mission['mission_id']}: {results[-1]['conclusion']}",
                    flush=True,
                )
    save(
        output / "latency_bounds.json",
        dict(
            schema_version="1.0",
            implementation_sha256=sha(Path(__file__)),
            source_summary_sha256=sha(output / "summary.json"),
            source_costs_sha256=sha(output / "costs.json"),
            source_outcomes_sha256={
                env: sha(output / env / "outcomes.json") for env in protocol["environments"]
            },
            results=results,
            gt_aware=True,
            deployable_policy=False,
            scope="Exact stage relaxation; no full-mission optimizer or feasible policy computed",
        ),
    )
    lines = [
        "# Latency-aware evidence bounds",
        "",
        "One inference at a time, exact capture grid and completion deadline. Energy, "
        "memory, false alarms and loading costs are relaxed; free idling and independent "
        "phase starts are allowed. Failure proves infeasibility for the named pool and "
        "cost cohort. Passing does not prove a legal full-mission policy exists.",
        "",
        "| Environment | Cohort | Mission | Conclusion | Blocking stages |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        failed = ", ".join(p["phase_id"] for p in r["phases"] if p["cannot_meet_recall"])
        lines.append(
            f"| {r['environment']} | {r['cost_cohort']} | {r['mission_id']} | "
            f"{r['conclusion']} | {failed or 'none'} |"
        )
    (output / "LATENCY_BOUNDS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
