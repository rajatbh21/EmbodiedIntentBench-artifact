"""Independently rebuild paid evaluator-only witnesses, including raw mask/report scores."""

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
from experiments.mission_environment_transfer_v3.scenario import load_public_terrain
from experiments.mission_routing_v2.feedback import snapshot
from experiments.mission_routing_v2.models import CONTROL, ENSEMBLE, ROTATED
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_transfer_comparison_v4.integrity import check_execution
from experiments.mission_transfer_comparison_v4.protocol import check_prediction_admission, verify
from experiments.mission_transfer_comparison_v4.study import scenario_for
from experiments.mission_transfer_within_recording.admission import verify as verify_admission
from experiments.routing_development_matrix.validate import validate_inference_grid
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(root, selected_flight):
    p = verify(root)
    if selected_flight not in p["cases"]:
        raise ValueError("Unknown condition")
    source = Path(p["cohort"])
    verify_admission(Path(p["admission"]))
    if p["active_cpu_power_sensitivity_w"] != [15]:
        raise ValueError("Witness audit requires the fixed 15 W study profile")
    plan_path = root / "witnesses" / selected_flight / "plan.json"
    plan = read(plan_path)
    require(
        plan["gt_aware"] and not plan["deployable_policy"] and plan["posthoc"],
        "Witness plan must remain evaluator-only and post-hoc",
    )
    require(plan["protocol_sha256"] == sha(root / "protocol.json"), "Changed search protocol")
    require(
        plan["source_sha256"] == sha(Path(__file__).with_name("witness.py")),
        "Changed witness implementation",
    )
    for path, expected in plan["helper_sha256"].items():
        require(sha(Path(path)) == expected, "Changed witness helper")
    evidence = source / p["cases"][selected_flight]["source_flight"] / "diagnostic_outcomes.json"
    require(plan["evidence_sha256"] == sha(evidence), "Changed search evidence")
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
        if flight != selected_flight:
            continue
        source_flight = case["source_flight"]
        folder = source / source_flight / "predictions"
        manifest = checked_manifest(folder, source / "protocol.json")
        check_prediction_admission(manifest, p)
        masks = {c: VerifiedMasks(folder, m) for c, m in manifest["models"].items()}
        by_capture = defaultdict(list)
        completed_scorers = []
        for power in p["active_cpu_power_sensitivity_w"]:
            summary_path = root / "witnesses" / flight / "summary.json"
            summary = read(summary_path)
            source_hashes[str(summary_path.relative_to(root))] = sha(summary_path)
            require(
                summary["complete"]
                and summary["gt_aware"]
                and not summary["deployable_policy"]
                and summary["protocol_sha256"] == sha(root / "protocol.json"),
                "Wrong witness summary",
            )
            require(
                len(summary["cases"]) == len(p["missions"])
                and {r["mission"] for r in summary["cases"]} == set(p["missions"]),
                "Incomplete witness search grid",
            )
            # A failed beam search is retained and never certified as infeasible here.
            for row in summary["cases"]:
                search_path = summary_path.parent / row["mission"] / "search.json"
                require(sha(search_path) == row["search_sha256"], "Changed witness search")
                search = read(search_path)
                require(row["flight"] == flight, "Wrong witness condition")
                require(
                    row["search_conclusion"] == search["conclusion"],
                    "Search conclusion differs",
                )
                require(
                    not row["full_contract_witness"] or "path" in row,
                    "Unreplayed schedule cannot establish feasibility",
                )
                require(
                    (search["schedule"] is not None) == ("path" in row),
                    "Unreplayed or invented witness schedule",
                )
            combined = [r for r in summary["cases"] if "path" in r]
            scenario = scenario_for(root, p, flight, p["missions"][0], power)
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
                check_execution(scenario, events, result, row, canonical)
                require(
                    row["full_contract_witness"] == canonical["mission_success"],
                    "Witness feasibility claim differs from paid outcome",
                )
                require(
                    outcome["gt_aware"] and not outcome["deployable_policy"],
                    "Witness provenance must remain evaluator-only",
                )
                require(
                    sha(destination / "protocol.json") == outcome["protocol_sha256"],
                    "Changed witness execution protocol",
                )
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
                require(row["policy"] == "gt_aware_witness", "Not an evaluator-only witness")
                search_path = destination.parent / "search.json"
                searched_schedule = read(search_path)["schedule"]
                require(
                    searched_schedule
                    == [
                        dict(capture_id=e["observation_id"], config_id=e["execution"]["config_id"])
                        for e in events
                    ],
                    "Witness capture schedule differs from search",
                )
                policy = HindsightSchedule(searched_schedule)
                counts["witness_cases"] += 1
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
                for e in events:
                    by_capture[e["observation_id"]].append(
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
                    counts["events"] += 1
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
                        choice == event["execution"]["config_id"], "Stored witness schedule differs"
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
                    counts["witness_decisions_reproduced"] += 1
        finally:
            backend.close()
        print(
            f"Reconstructed {flight}: {counts['witness_decisions_reproduced']} decisions so far",
            flush=True,
        )
        seen_recordings.add(source_flight)
        for scorer, execution in completed_scorers:
            require(
                scorer.summary(execution["final_time_s"]) == execution["report_quality"],
                "Report identity/support scoring differs",
            )
    result = dict(
        complete=True,
        actual_neural_calls=inference_calls,
        frame_config_pairs=dense_pairs,
        raw_frame_scores_reconstructed=True,
        report_identity_support_reconstructed=True,
        condition=selected_flight,
        recording_pose_sha256=__import__(
            "experiments.v4_benchmark.scenario", fromlist=["digest"]
        ).digest(records[selected_flight]),
        protocol_sha256=sha(root / "protocol.json"),
        scope=p["scope"],
        **counts,
        source_summaries_sha256=source_hashes,
        search_plan_sha256=sha(plan_path),
        physical_poses_verified=True,
        per_event_clock_energy_memory_verified=True,
        prediction_history_rebuilt_without_ground_truth=True,
        canonical_outcomes_verified=True,
        validator_sha256=sha(Path(__file__)),
        gt_aware=True,
        deployable_policy=False,
        independent_witness_reconstruction=True,
    )
    result["fresh_test"] = False
    result["pre_acquisition_design"] = False
    result["posthoc"] = True
    result["numerical_repair_after_failed_execution"] = False
    result["failed_search_is_inconclusive"] = True
    save(root / "witnesses" / selected_flight / "validation.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight", required=True)
    a = parser.parse_args()
    run(a.output, a.flight)
