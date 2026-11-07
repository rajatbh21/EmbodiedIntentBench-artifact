"""Sound optimistic evidence bounds on a COMPLETE fixed-flight outcome table.

All models may run simultaneously, for free, at every capture with zero latency.
Target evidence may come from different models at the same capture, but one
capture is never counted twice for a target. Thus failure proves infeasibility
within this table; passing the bound does not provide a realizable policy.
"""

from __future__ import annotations

from experiments.lake_missions.objectives import enough_evidence


def validate_table(rows, capture_times, configs):
    if not capture_times or not configs or len(configs) != len(set(configs)):
        raise ValueError("Nonempty captures and unique configurations required")
    if capture_times != sorted(set(capture_times)):
        raise ValueError("Capture times must be strictly increasing")
    expected = {(i, c) for i in range(len(capture_times)) for c in configs}
    keys = [(r["capture_id"], r["config_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("A complete, nonduplicated frame/configuration table is required")
    for r in rows:
        if r["capture_time_s"] != capture_times[r["capture_id"]]:
            raise ValueError("Capture time differs from the canonical schedule")


def optimistic_bound(mission, rows, capture_times, configs):
    validate_table(rows, capture_times, configs)
    phases = []
    for phase in mission["phases"]:
        times = {target: set() for target in phase["evaluator_target_ids"]}
        suppliers = {target: {} for target in times}
        for row in rows:
            t = row["capture_time_s"]
            if phase["start_s"] <= t < phase["end_s"]:
                for target in set(row["matched_target_ids"]) & times.keys():
                    times[target].add(t)
                    suppliers[target].setdefault(row["config_id"], []).append(t)
        supported = [target for target, hits in times.items() if enough_evidence(hits, phase)]
        recall = len(supported) / len(times)
        feasible = recall >= phase["required_recall"]
        phases.append(
            dict(
                phase_id=phase["phase_id"],
                required_recall=phase["required_recall"],
                evidence_recall_upper_bound=recall,
                supported_targets=supported,
                cannot_meet_recall=not feasible,
                per_target={
                    target: dict(
                        distinct_capture_count=len(hits),
                        capture_times_s=sorted(hits),
                        evidence_span_s=max(hits) - min(hits) if hits else 0,
                        min_hits=phase["min_hits"],
                        min_span_s=phase["min_evidence_span_s"],
                        satisfies_repeated_evidence=enough_evidence(hits, phase),
                        suppliers=suppliers[target],
                    )
                    for target, hits in times.items()
                },
                weight=phase["weight"],
            )
        )
    impossible = any(p["cannot_meet_recall"] for p in phases)
    return dict(
        mission_id=mission["mission_id"],
        gt_aware=True,
        deployable_policy=False,
        bound="optimistic_unlimited_parallel_zero_latency_evidence_union",
        ignored_constraints=[
            "one_model_per_decision",
            "latency",
            "skipped_captures",
            "false_positive_budget",
            "energy",
            "memory",
            "switching_cost",
        ],
        mission_impossible_under_current_table=impossible,
        conclusion="certified_infeasible" if impossible else "feasibility_unresolved",
        phase_success_rate_upper_bound=sum(not p["cannot_meet_recall"] for p in phases)
        / len(phases),
        weighted_evidence_recall_upper_bound=sum(
            p["evidence_recall_upper_bound"] * p["weight"] for p in phases
        )
        / sum(p["weight"] for p in phases),
        phases=phases,
        scope="Fixed recorded RGB, model outputs, matching and mission definitions only.",
    )
