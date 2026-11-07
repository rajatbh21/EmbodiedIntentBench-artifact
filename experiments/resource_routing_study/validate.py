"""Rebuild prediction history and routing decisions, audit clocks and mission scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_search import HindsightSchedule
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract, evaluate, immutable
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import SurfaceProjector, TrackingConfig, TrackingPipeline
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_routing_v2.feedback import snapshot
from experiments.mission_routing_v2.flights import split_check
from experiments.mission_routing_v2.models import CONTROL, ENSEMBLE, ROTATED
from experiments.mission_routing_v2.train import checked_manifest
from experiments.resource_routing_study.protocol import verify
from experiments.resource_routing_study.study import (
    POLICIES,
    load_profiles,
    make_policy,
    scenario_for,
)
from experiments.routing_development_matrix.validate import validate_inference_grid
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(root):
    p = verify(root)
    source = Path(p["cohort"])
    terrain, _ = load_public_terrain(Path(p["assets"]))
    counts = dict(
        cases=0,
        policy_cases=0,
        witness_cases=0,
        events=0,
        public_decisions_reproduced=0,
        witness_decisions_reproduced=0,
    )
    source_hashes = {}
    records = {}
    inference_calls = 0
    dense_pairs = 0
    seen_recordings = set()
    for flight, case in p["cases"].items():
        source_flight = case["source_flight"]
        profile = load_profiles(p, flight)
        risk_profile = None
        for key in ("recall", "risk"):
            split_check(profile[key]["training_lineage"], case["lineage"])
        folder = source / source_flight / "predictions"
        manifest = checked_manifest(folder, source / "protocol.json")
        masks = {c: VerifiedMasks(folder, m) for c, m in manifest["models"].items()}
        by_capture = defaultdict(list)
        completed_scorers = []
        for power in p["active_cpu_power_sensitivity_w"]:
            summary_path = root / f"summary_{flight}_{power}.json"
            summary = read(summary_path)
            source_hashes[summary_path.name] = sha(summary_path)
            require(
                summary["complete"] and summary["protocol_sha256"] == sha(root / "protocol.json"),
                "Wrong summary",
            )
            expected = {(mission, policy) for mission in p["missions"] for policy in POLICIES}
            actual = [(r["mission"], r["policy"]) for r in summary["cases"]]
            require(
                len(actual) == len(expected) and set(actual) == expected, "Incomplete policy grid"
            )
            witness_path = root / "witnesses" / flight / "summary.json"
            witness = read(witness_path)
            require(
                witness["complete"] and witness["protocol_sha256"] == sha(root / "protocol.json"),
                "Wrong witness summary",
            )
            require(
                len(witness["cases"]) == len(p["missions"])
                and {r["mission"] for r in witness["cases"]} == set(p["missions"]),
                "Incomplete witness grid",
            )
            source_hashes[str(witness_path.relative_to(root))] = sha(witness_path)
            combined = [*summary["cases"], *(r for r in witness["cases"] if "path" in r)]
            for row in combined:
                destination = root / row["path"]
                for filename, key in (
                    ("outcome.json", "outcome_sha256"),
                    ("execution/events.json", "events_sha256"),
                    ("execution/results.json", "results_sha256"),
                ):
                    require(sha(destination / filename) == row[key], "Changed result artifact")
                scenario = scenario_for(root, p, flight, row["mission"], power)
                contract = IntentContract(read(root / flight / row["mission"] / "intent.json"))
                scenario = bind_scenario(scenario, contract)
                events = read(destination / "execution/events.json")
                result = read(destination / "execution/results.json")
                outcome = read(destination / "outcome.json")
                canonical = evaluate(contract, events, result)
                for key in ("mission_success", "failed_constraints", "reporting", "phases"):
                    require(canonical[key] == outcome[key], "Canonical outcome mismatch")
                require(
                    row["full_success"] == canonical["mission_success"], "False success summary"
                )
                require(
                    row["phases_succeeded"] == sum(v["success"] for v in canonical["phases"])
                    and row["report_recall"] == canonical["reporting"]["recall_by_deadline"],
                    "Selection endpoint summary differs",
                )
                require(
                    row["failed_constraints"] == canonical["failed_constraints"],
                    "Failure summary differs",
                )
                require(
                    not result["router_errors"]
                    and not result["inference_failures"]
                    and not result["invalid_actions"],
                    "Execution errors",
                )
                if row["policy"] == "gt_aware_witness":
                    search_path = destination.parent / "search.json"
                    require(sha(search_path) == row["search_sha256"], "Changed witness search")
                    policy = HindsightSchedule(read(search_path)["schedule"])
                    counts["witness_cases"] += 1
                else:
                    policy = make_policy(row["policy"], profile, scenario, risk_profile)
                    counts["policy_cases"] += 1
                tracking = read(destination / "protocol.json")["public_tracking"]
                tracker = TrackingPipeline(
                    TrackingConfig(remember_reports=True), SurfaceProjector(terrain)
                )
                frame_scorer = MissionEvaluator(
                    scenario.matching_iou_threshold,
                    len(scenario.targets),
                    scenario.observation_interval_s,
                )
                report_scorer = RunnerReportEvaluator(
                    scenario.matching_iou_threshold, scenario.targets.values(), 2
                )
                completed_scorers.append((report_scorer, result))
                specs = {s.config_id: s for s in scenario.executors}
                previous, index, energy = None, 0, 0.0
                for e in events:
                    t = e["capture_time_s"]
                    state = e["runtime_at_capture"]
                    require(
                        e["observation_id"] == index and state["current_config_id"] == previous,
                        "Illegal decision schedule",
                    )
                    before = energy + t * scenario.flight_power_w
                    require(
                        abs(state["cumulative_energy_j"] - before) < 1e-6, "Capture energy mismatch"
                    )
                    require(
                        abs(
                            state["battery_frac"]
                            - (
                                scenario.initial_battery_frac
                                - before / (scenario.battery_capacity_wh * 3600)
                            )
                        )
                        < 1e-8,
                        "Battery mismatch",
                    )
                    config = e["execution"]["config_id"]
                    spec, resource = specs[config], scenario.resources.for_config(config)
                    load_s = resource.load_latency_s if config != previous else 0
                    load_j = resource.load_energy_j if config != previous else 0
                    completion, next_index, skipped = scenario.schedule.complete(
                        index,
                        seconds_to_ns(
                            spec.mission_latency_s
                            + load_s
                            + scenario.router_latency_s
                            + tracking["cost"]["latency_s"]
                        ),
                    )
                    require(
                        e["capture_timestamp_ns"] == scenario.schedule.at(index),
                        "Capture clock mismatch",
                    )
                    require(
                        abs(
                            e["completion_time_s"] - (completion - scenario.schedule.epoch_ns) / 1e9
                        )
                        < 1e-8
                        and e["skipped_observation_ids"] == list(skipped),
                        "Completion clock mismatch",
                    )
                    energy += (
                        spec.energy_j_per_call
                        + load_j
                        + scenario.router_energy_j
                        + tracking["cost"]["energy_j"]
                    )
                    require(
                        abs(
                            e["cumulative_energy_j"]
                            - energy
                            - e["completion_time_s"] * scenario.flight_power_w
                        )
                        < 1e-6,
                        "Event energy mismatch",
                    )
                    require(
                        e["resident_memory_mb"]
                        == resource.memory_mb + tracking["cost"]["memory_mb"],
                        "Memory mismatch",
                    )
                    by_capture[index].append(
                        (
                            policy,
                            tracker,
                            e,
                            scenario,
                            contract,
                            tracking,
                            frame_scorer,
                            report_scorer,
                        )
                    )
                    previous, index = config, next_index
                    counts["events"] += 1
                require(
                    abs(
                        result["cumulative_energy_j"]
                        - energy
                        - result["final_time_s"] * scenario.flight_power_w
                    )
                    < 1e-6,
                    "Final energy mismatch",
                )
                counts["cases"] += 1
        recording = read(source / source_flight / "recording/recording.json")
        records[flight] = [(f["timestamp_ns"], f["body_pose"]) for f in recording["frames"]]
        expected = {(i, c) for i in range(len(recording["frames"])) for c in masks}
        calls = validate_inference_grid(
            read(folder / "inference.json"), expected, scenario.executors
        )
        if source_flight not in seen_recordings:
            inference_calls += calls
        backend = ReplayBackend(scenario, source / source_flight / "recording")
        try:
            for index, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                key = rgb_key(frame.rgb)
                predictions = {c: m[key] for c, m in masks.items()}
                require(
                    np.array_equal(
                        predictions[ENSEMBLE], predictions[CONTROL] | predictions[ROTATED]
                    ),
                    "Union prediction differs",
                )
                if source_flight not in seen_recordings:
                    dense_pairs += len(predictions)
                for (
                    policy,
                    tracker,
                    event,
                    scenario,
                    contract,
                    tracking,
                    frame_scorer,
                    report_scorer,
                ) in by_capture[index]:
                    require(
                        frame.body_pose == event["capture_body_pose"],
                        "Physical trajectory mismatch",
                    )
                    history = snapshot(tracker, event["capture_time_s"])
                    observation = SimpleNamespace(
                        contract=scenario.contract,
                        intent_contract=immutable(contract.public_document()),
                        state=SimpleNamespace(**event["runtime_at_capture"]),
                        configs=scenario.catalog,
                        resources=scenario.resources,
                        tracking=immutable(tracking),
                        evidence_history=immutable(history),
                        confirmed_reports_so_far=tracker.report_count,
                        rgb=frame.rgb,
                    )
                    choice = policy.select(observation)
                    require(
                        choice == event["execution"]["config_id"], "Public-only routing differs"
                    )
                    if getattr(policy, "last_decision", None) is not None:
                        require(
                            policy.last_decision == event["router_decision"],
                            "Causal history or decision trace differs",
                        )
                    reports = tracker.update(
                        masks[choice][rgb_key(frame.rgb)],
                        event["capture_time_s"],
                        event["completion_time_s"],
                        pose=frame.body_pose,
                        camera_pitch_rad=scenario.world["camera_pitch_rad"],
                    )
                    require(
                        reports == event["emitted_reports"], "Prediction history does not reproduce"
                    )
                    truth = evaluation_frame(frame, index, event["capture_time_s"], scenario)
                    require(
                        frame_scorer.update(truth, predictions[choice]).to_dict() == event["score"],
                        "Raw mask/GT score differs",
                    )
                    report_scorer.update(truth, predictions[choice], reports)
                    counts[
                        "witness_decisions_reproduced"
                        if isinstance(policy, HindsightSchedule)
                        else "public_decisions_reproduced"
                    ] += 1
        finally:
            backend.close()
        print(
            f"Reconstructed {flight}: {counts['public_decisions_reproduced']} decisions so far",
            flush=True,
        )
        seen_recordings.add(source_flight)
        for scorer, execution in completed_scorers:
            require(
                scorer.summary(execution["final_time_s"]) == execution["report_quality"],
                "Report identity/support scoring differs",
            )
    require(
        len(records) == 6 and len({str(v) for v in records.values()}) == 1,
        "Battery or lighting changed the physical trajectory",
    )
    result = dict(
        complete=True,
        actual_neural_calls=inference_calls,
        frame_config_pairs=dense_pairs,
        raw_frame_scores_reconstructed=True,
        report_identity_support_reconstructed=True,
        paired_lighting_poses_identical=True,
        protocol_sha256=sha(root / "protocol.json"),
        scope=p["scope"],
        **counts,
        source_summaries_sha256=source_hashes,
        physical_poses_verified=True,
        per_event_clock_energy_memory_verified=True,
        prediction_history_rebuilt_without_ground_truth=True,
        canonical_outcomes_verified=True,
        validator_sha256=sha(Path(__file__)),
    )
    result["new_neural_calls"] = 0
    result["fresh_test"] = False
    save(root / "validation.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
