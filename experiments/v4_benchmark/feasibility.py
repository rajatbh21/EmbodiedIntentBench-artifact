"""Evaluator-only feasibility checks over deterministic fixed-flight recordings.

This is GT-aware analysis, NEVER a submitted router or baseline score. A successful
schedule is verified by the ordinary runner. Exhausted search budgets mean unknown,
not impossible. Only the shipped stateless RGB heuristics (including remote wrappers
around them) are supported; learned/stateful executors require a different protocol.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from missionbench.simulator.action_validator import privacy_permits
from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import build_executors
from missionbench.v2.remote import ExecutionContext
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.runner import evaluation_frame, run_mission, validate_result
from experiments.v4_benchmark.scenario import Scenario, digest, exported_source, load_scenario
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


class _ScheduleWitness:
    """Offline GT-aware witness, deliberately private to this evaluator module."""

    def __init__(self, sequence):
        self.actions = dict(sequence)

    def select(self, observation):
        return self.actions[observation.state.frame_id]


def analyze(scenario: Scenario, recording: Path, output: Path, *, max_schedules: int = 64) -> dict:
    from experiments.v4_benchmark.mission_requirements import MissionRequirements

    if scenario.requirements != MissionRequirements():
        raise ValueError("feasibility does not yet support additional mission requirements")
    if type(max_schedules) is not int or max_schedules < 1:
        raise ValueError("max_schedules must be a positive integer")
    for spec in scenario.executors:
        kind = spec.parameters.get("backend_kind") if spec.kind == "simulated_remote" else spec.kind
        if kind not in ("fast_weak", "slow_strong"):
            raise ValueError(
                "feasibility supports deterministic stateless threshold executors only"
            )
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "scenario.json", exported_source(scenario))
    save_json(output / "contract.json", {"schema_version": "1.0", **asdict(scenario.contract)})
    schedule = scenario.schedule
    slots = list(range(schedule.epoch_ns, schedule.end_ns, schedule.interval_ns))
    legal = [
        c.config_id for c in scenario.catalog if privacy_permits(scenario.contract.privacy_level, c)
    ]
    backend = ReplayBackend(scenario, recording)
    executors = build_executors(scenario.executors)
    outcomes, next_slots, detectable, visible = [], {}, set(), set()
    try:
        for index, timestamp in enumerate(slots):
            backend.advance_to(timestamp)
            frame = backend.capture()
            time_s = (timestamp - schedule.epoch_ns) / 1e9
            observation = evaluation_frame(frame, index, time_s, scenario)
            visible.update(observation.visible_target_ids)
            for config_id in legal:
                executor = executors[config_id]
                context = ExecutionContext(
                    scenario.scenario_id,
                    index,
                    time_s,
                    scenario.contract.deadline_s,
                    scenario.network.state_at(time_s),
                )
                result = (
                    executor.run_with_context(frame.rgb.copy(), context)
                    if hasattr(executor, "run_with_context")
                    else executor.run(frame.rgb.copy())
                )
                validate_result(result, config_id, frame.rgb.shape[:2])
                prediction = (
                    result.prediction_mask
                    if result.success
                    else np.zeros_like(result.prediction_mask)
                )
                evaluator = MissionEvaluator(
                    scenario.matching_iou_threshold,
                    len(scenario.targets),
                    scenario.observation_interval_s,
                )
                score = evaluator.update(observation, prediction)
                detectable.update(score.matched_target_ids)
                latency = seconds_to_ns(
                    scenario.router_latency_s + result.mission_latency_s, allow_zero=True
                )
                completion, next_index, skipped = schedule.complete(index, latency)
                next_slots[index, config_id] = next_index
                outcomes.append(
                    {
                        "observation_id": index,
                        "config_id": config_id,
                        "capture_time_s": time_s,
                        "completion_time_s": (completion - schedule.epoch_ns) / 1e9,
                        "next_observation_id": next_index,
                        "skipped_ids": list(skipped),
                        "success": result.success,
                        "score": score.to_dict(),
                        "energy_j": result.energy_j + result.communication_energy_j,
                        "communication_mb": result.communication_mb,
                    }
                )
    finally:
        backend.close()
    bound = len(detectable) / len(scenario.targets)
    report = {
        "feasibility_schema_version": "0.1",
        "gt_aware": True,
        "is_policy": False,
        "scope": "fixed flight, configured costs, deterministic threshold executor pool",
        "scenario_sha256": digest(scenario.source),
        "contract_sha256": digest(asdict(scenario.contract)),
        "backend": backend.provenance,
        "visible_target_ids": sorted(visible),
        "detectable_target_ids": sorted(detectable),
        "recall_upper_bound_ignoring_costs_and_skips": bound,
        "status": "unknown",
        "reason": "search_budget_exhausted",
        "max_schedules": max_schedules,
        "evaluated_schedules": 0,
        "search_exhausted": False,
        "witness": None,
    }
    save_json(
        output / "slot_outcomes.json",
        {"outcome_schema_version": "0.1", "gt_aware": True, "outcomes": outcomes},
    )
    contract = scenario.contract
    lower_bound = contract.quality_operator.value in (">", ">=")
    available_j = (scenario.initial_battery_frac - contract.min_final_battery_frac) * (
        scenario.battery_capacity_wh * 3600
    )
    reason = None
    if not legal:
        reason = "no_privacy_legal_action"
    elif scenario.duration_s > contract.deadline_s:
        reason = "required_fixed_flight_exceeds_deadline"
    elif scenario.flight_power_w * scenario.duration_s > available_j + 1e-8:
        reason = "mandatory_flight_energy_exceeds_reserve_budget"
    elif (
        contract.quality_metric == "target_recall"
        and lower_bound
        and not contract.quality_satisfied(bound)
    ):
        reason = "even_all_legal_slot_detections_cannot_meet_recall"
    if reason:
        report.update(status="infeasible", reason=reason)
    else:
        # Static candidates first find cheap witnesses; exhaustive DFS follows unchanged.
        def sequences():
            for config_id in legal:
                sequence, slot = [], 0
                while slot < len(slots):
                    sequence.append((slot, config_id))
                    slot = next_slots[slot, config_id]
                yield tuple(sequence)
            stack = [(0, ())]
            while stack:
                slot, sequence = stack.pop()
                if slot >= len(slots):
                    yield sequence
                else:
                    for config_id in reversed(legal):
                        stack.append((next_slots[slot, config_id], (*sequence, (slot, config_id))))

        seen = set()
        for sequence in sequences():
            if sequence in seen:
                continue
            if report["evaluated_schedules"] >= max_schedules:
                break
            seen.add(sequence)
            replay = ReplayBackend(scenario, recording)
            try:
                result = run_mission(
                    scenario,
                    replay,
                    _ScheduleWitness(sequence),
                    build_executors(scenario.executors),
                    router_name="GT_AWARE_FEASIBILITY_WITNESS_NOT_A_POLICY",
                )
            finally:
                replay.close()
            report["evaluated_schedules"] += 1
            if result["status"] == "error" or result["router_errors"]:
                raise RuntimeError(f"witness replay failed: {result['error']}")
            if result["mission_success"]:
                report.update(
                    status="feasible",
                    reason="ordinary_runner_verified_witness",
                    witness=[{"observation_id": i, "config_id": c} for i, c in sequence],
                )
                save_json(
                    output / "witness.json",
                    {
                        "gt_aware": True,
                        "is_policy": False,
                        "sequence": report["witness"],
                        "result": result,
                    },
                )
                break
        else:
            report.update(
                status="infeasible", reason="all_legal_schedules_failed", search_exhausted=True
            )
    save_json(output / "feasibility.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-schedules", type=int, default=64)
    args = parser.parse_args()
    result = analyze(
        load_scenario(args.scenario), args.recording, args.output, max_schedules=args.max_schedules
    )
    print(f"GT-aware feasibility: {result['status']} ({result['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
