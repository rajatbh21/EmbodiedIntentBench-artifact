"""Evaluator-only optimistic bounds; distinguish impossible missions from policy failures.

Not a routing policy or an optimal oracle. Every model/frame is inspected with GT
only after the locked experiment. Timing, loading and confirmation are relaxed,
so the resulting recall is an upper bound, never a feasible schedule claim.
"""

import argparse
import math
from pathlib import Path

from missionbench.v2.evaluation import MissionEvaluator
from experiments.routing_mission.environment import load
from experiments.routing_mission.experiment import checked_cache, freeze, read, specification
from experiments.routing_mission.inference import checksum
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.runner import evaluation_frame


def classify(
    *,
    static_succeeded,
    required_recall,
    upper_recall,
    deadline,
    earliest_possible_s,
    battery_budget_j,
    minimum_flight_j,
):
    impossible = []
    if upper_recall + 1e-12 < required_recall:
        impossible.append("pool_cannot_match_enough_targets")
    if earliest_possible_s is None or earliest_possible_s > deadline:
        impossible.append("no_pool_match_can_complete_by_report_deadline")
    if minimum_flight_j > battery_budget_j + 1e-9:
        impossible.append("flight_alone_exceeds_battery_reserve")
    if static_succeeded and impossible:
        raise ValueError("optimistic bound contradicts a valid run")
    status = (
        "feasible_static_witness"
        if static_succeeded
        else "infeasible_bound"
        if impossible
        else "unknown"
    )
    return dict(status=status, reasons=impossible)


def analyze(root):
    spec = specification(root)
    freeze(root)
    bounds = {}
    for name, environment in spec["environments"].items():
        scenario = load(root / "scenario.json", environment)
        cache = checked_cache(scenario, root / name)
        backend = ReplayBackend(scenario, root / name / "recording")
        matched = set()
        earliest = math.inf
        scores = []
        try:
            for index, t in enumerate(
                range(
                    scenario.schedule.epoch_ns,
                    scenario.schedule.end_ns,
                    scenario.schedule.interval_ns,
                )
            ):
                backend.advance_to(t)
                frame = backend.capture()
                capture_s = (t - scenario.schedule.epoch_ns) / 1e9
                observation = evaluation_frame(frame, index, capture_s, scenario)
                for config in scenario.executors:
                    result = cache[config.config_id].run(frame.rgb)
                    evaluator = MissionEvaluator(
                        scenario.matching_iou_threshold,
                        len(scenario.targets),
                        scenario.observation_interval_s,
                    )
                    score = evaluator.update(observation, result.prediction_mask)
                    matched.update(score.matched_target_ids)
                    if score.matched_target_ids:
                        # Dropping nonnegative loading and router delay is intentionally optimistic.
                        earliest = min(earliest, capture_s + config.mission_latency_s)
                    scores.append(
                        dict(
                            capture=index,
                            config_id=config.config_id,
                            matched_target_ids=list(score.matched_target_ids),
                        )
                    )
        finally:
            backend.close()
        bounds[name] = dict(
            optimistic_report_recall_upper_bound=len(matched) / len(scenario.targets),
            matched_targets_in_any_prediction=sorted(matched),
            earliest_possible_correct_completion_s=earliest if math.isfinite(earliest) else None,
            unmatchable_targets=sorted(set(scenario.targets.values()) - matched),
        )
        save_json(root / name / "pool_match_diagnostic.json", scores)
    comparison = read(root / "comparison.json")
    cases = []
    for name in spec["environments"]:
        for case in spec["cases"]:
            rows = [
                r
                for r in comparison["rows"]
                if r["environment"] == name and r["case_id"] == case["case_id"]
            ]
            witness = [
                r["policy"]
                for r in rows
                if r["policy"].startswith("static:") and r["mission_success"]
            ]
            scenario = load(
                root / name / "comparison" / case["case_id"] / "scenario.json",
                spec["environments"][name],
            )
            b = bounds[name]
            verdict = classify(
                static_succeeded=bool(witness),
                required_recall=scenario.requirements.min_report_recall,
                upper_recall=b["optimistic_report_recall_upper_bound"],
                deadline=scenario.requirements.confirmed_report_deadline_s,
                earliest_possible_s=b["earliest_possible_correct_completion_s"],
                battery_budget_j=(
                    scenario.initial_battery_frac - scenario.contract.min_final_battery_frac
                )
                * scenario.battery_capacity_wh
                * 3600,
                minimum_flight_j=scenario.duration_s * scenario.flight_power_w,
            )
            cases.append(
                dict(
                    environment=name,
                    case_id=case["case_id"],
                    **verdict,
                    static_witnesses=witness,
                    avoidable_failures_against_static=[
                        r["policy"] for r in rows if witness and not r["mission_success"]
                    ],
                )
            )
    result = dict(
        evaluator_only=True,
        is_policy=False,
        is_optimal_oracle=False,
        implementation_sha256=checksum(Path(__file__)),
        source_lock_sha256=checksum(root / "lock.json"),
        relaxations=[
            "all models at every capture; no resource limits",
            "no confirmation or association requirement",
            "no loading/router latency; matching still identical to mission evaluator",
        ],
        bounds=bounds,
        cases=cases,
    )
    save_json(root / "feasibility_diagnostic.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    analyze(parser.parse_args().experiment)
