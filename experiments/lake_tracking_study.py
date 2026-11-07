"""Paired tracking ablations on frozen lake outputs, plus explicit-cost physical replay.

Thresholds are declared before reading results. Fixed-schedule diagnostics do not
claim policy improvements; fresh paid replays use the versioned tracking runner.
"""

from __future__ import annotations

import argparse
import copy
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_intent_study import cases
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import (
    StaticReportTracker,
    SurfaceProjector,
    TrackingConfig,
    box_iou,
    detections,
)
from experiments.lake_tracking_evaluation import ConsistentReportEvaluator, component_matches
from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST, load_public_terrain

VARIANTS = {
    "image_iou": TrackingConfig(association="image_iou"),
    "flat_world": TrackingConfig(association="flat_world"),
    "terrain_world": TrackingConfig(),
    "terrain_memory": TrackingConfig(remember_reports=True),
}
FILES = (
    "lake_tracking.py",
    "lake_tracking_evaluation.py",
    "lake_tracking_runner.py",
    "lake_tracking_runtime.py",
    "lake_tracking_study.py",
)


def scenario_and_cache(prior, source, environment, mission):
    from experiments.lake_fast_evaluation import load_fast_scenario
    from experiments.lake_missions.scenario import load
    from experiments.v4_benchmark.calibrated_router import model_signature

    suite = read(source / "suite.json")
    base = load(source / "scenario.json", suite["environments"][environment], Path(suite["assets"]))
    control = next(s for s in base.executors if s.config_id == "CFG_MASKRCNN_1280")
    scenario = load_fast_scenario(
        prior / environment / "missions" / mission / "scenario.json",
        suite["environments"][environment],
        Path(suite["assets"]),
        control,
    )
    rotated = Path(read(prior / "protocol.json")["prior"])
    original = Path(read(rotated / "protocol.json")["prior"])
    available, hashes = {}, {}
    for root in (original, rotated, prior):
        directory = root / environment / "predictions"
        manifest, complete = read(directory / "manifest.json"), read(directory / "completion.json")
        if not complete["complete"] or complete["manifest_sha256"] != sha(
            directory / "manifest.json"
        ):
            raise ValueError("Unverified mask manifest")
        if manifest["recording_identity"] != scenario.recording_identity:
            raise ValueError("Wrong cache recording identity")
        hashes[str(directory / "manifest.json")] = sha(directory / "manifest.json")
        available.update(
            {config: (directory, model) for config, model in manifest["models"].items()}
        )
    for spec in scenario.executors:
        if available[spec.config_id][1]["signature"] != model_signature(spec):
            raise ValueError("Wrong model signature")
    return scenario, available, hashes


def prepare_packets(prior, source, output, environment, terrain):
    from experiments.v4_benchmark.backends import ReplayBackend
    from experiments.v4_benchmark.probe_cache import rgb_key
    from experiments.v4_benchmark.runner import evaluation_frame

    scenario, available, hashes = scenario_and_cache(prior, source, environment, "sector_rescue")
    backend = ReplayBackend(scenario, source / environment / "recording")
    projectors = dict(flat_world=SurfaceProjector(), terrain_world=SurfaceProjector(terrain))
    packets, labels, elapsed, masks = {}, {}, [], {}
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            frame = backend.capture()
            key = rgb_key(frame.rgb)
            capture_s = (timestamp - scenario.schedule.epoch_ns) / 1e9
            observation = evaluation_frame(frame, index, capture_s, scenario)
            for spec in scenario.executors:
                directory, model = available[spec.config_id]
                mask = VerifiedMasks(directory, model)[key]
                if mask.shape != frame.rgb.shape[:2]:
                    raise ValueError("Mask resolution mismatch")
                started = time.perf_counter()
                measurement = detections(
                    mask, frame.body_pose, scenario.world["camera_pitch_rad"], projectors
                )
                elapsed.append(time.perf_counter() - started)
                name = f"{index}/{spec.config_id}"
                packets[name] = measurement
                labels[name] = component_matches(observation, mask, scenario.matching_iou_threshold)
                item = model["predictions"][key]
                masks[str(directory / item["file"])] = item["sha256"]
            if index % 50 == 0:
                print(
                    f"{environment}: projected {index + 1}/{len(backend.frames)} captures",
                    flush=True,
                )
    finally:
        backend.close()
    destination = output / environment
    destination.mkdir()
    save(destination / "public_measurements.json", packets)
    save(destination / "private_component_matches.json", labels)
    save(
        destination / "packet_provenance.json",
        dict(
            schema_version="tracking_packets_v1",
            prediction_manifests_sha256=hashes,
            prediction_files_sha256=masks,
            recording_sha256=sha(source / environment / "recording/recording.json"),
            public_measurements_sha256=sha(destination / "public_measurements.json"),
            private_matches_sha256=sha(destination / "private_component_matches.json"),
            measured_shared_extraction_and_two_projections_p95_s=float(np.percentile(elapsed, 95)),
            measurement_scope="Host extraction plus both projections; not a deployment profile",
            pairs=len(packets),
        ),
    )
    return packets, labels


def compare_case(row, index, packets, labels, contracts, output):
    events, result = (
        read(Path(row["output"]) / "events.json"),
        read(Path(row["output"]) / "results.json"),
    )
    contract = contracts[row["mission_id"]]
    records = []
    for variant, config in VARIANTS.items():
        tracker = StaticReportTracker(config)
        evaluator = ConsistentReportEvaluator(
            contract.document()["evaluator_target_ids"], config.min_hits
        )
        emitted, associations, motion_pairs = [], [], []
        previous = {}
        for event in events:
            key = f"{event['observation_id']}/{event['execution']['config_id']}"
            measurements = packets[key] if event["execution"]["success"] else []
            matches = labels[key] if event["execution"]["success"] else {}
            reports = tracker.update(
                measurements, event["capture_time_s"], event["completion_time_s"]
            )
            if variant == "image_iou":
                legacy = [
                    {k: v for k, v in report.items() if k != "evidence"} for report in reports
                ]
                if legacy != event["emitted_reports"]:
                    raise ValueError("Packet image tracker differs from frozen image tracker")
                for measurement in measurements:
                    target = matches.get(measurement["component_id"])
                    if target is None:
                        continue
                    if target in previous:
                        previous_time, old = previous[target]
                        gap = event["capture_time_s"] - previous_time
                        if gap <= config.max_gap_s:
                            motion_pairs.append(
                                dict(
                                    target_id=target,
                                    gap_s=gap,
                                    box_iou=box_iou(old["box"], measurement["box"]),
                                    **{
                                        mode + "_distance_m": float(
                                            np.linalg.norm(
                                                np.asarray(old["positions"][mode])
                                                - measurement["positions"][mode]
                                            )
                                        )
                                        if old["positions"][mode] is not None
                                        and measurement["positions"][mode] is not None
                                        else None
                                        for mode in ("flat_world", "terrain_world")
                                    },
                                )
                            )
                    previous[target] = event["capture_time_s"], measurement
            evaluator.update(event["capture_time_s"], matches, reports)
            emitted.extend(reports)
            associations.append(
                dict(
                    capture_s=event["capture_time_s"],
                    assignments=copy.deepcopy(tracker.assignments),
                )
            )
        quality = evaluator.summary()
        if variant == "image_iou":
            legacy_targets = [r["target_id"] for r in result["report_quality"]["reports"]]
            if [r["last_frame_target_id"] for r in quality["reports"]] != legacy_targets:
                raise ValueError("Legacy report mask matching differs")
        revised = {**result, "report_quality": quality}
        conditional = evaluate(contract, events, revised)
        path = output / "fixed_schedule" / f"{index:03d}_{variant}.json"
        path.parent.mkdir(exist_ok=True)
        save(
            path,
            dict(
                schema_version="tracking_fixed_schedule_diagnostic_v1",
                policy=row["policy"],
                environment=row["environment"],
                mission_id=row["mission_id"],
                gt_aware=row["gt_aware"],
                variant=variant,
                config=config.to_dict(),
                reports=quality,
                associations=associations,
                suppressed_reports=tracker.suppressed_reports,
                diagnostic_motion_pairs=motion_pairs,
                conditional_outcome=conditional,
                caveat="Fixed schedule diagnostic; tracking overhead and policy feedback excluded",
                source_events_sha256=row["events_sha256"],
                source_results_sha256=row["results_sha256"],
            ),
        )
        records.append(
            dict(
                environment=row["environment"],
                mission_id=row["mission_id"],
                policy=row["policy"],
                gt_aware=row["gt_aware"],
                variant=variant,
                conditional_success=conditional["mission_success"],
                report_recall=quality["report_recall"],
                false_reports=quality["false_reports"],
                duplicates=quality["duplicate_correct_reports"],
                mixed_identity_reports=quality["mixed_identity_reports"],
                source_output=row["output"],
                output=str(path.resolve()),
                sha256=sha(path),
            )
        )
    return records


def paid_replay(prior, source, output, row, contract, variant, terrain, map_metadata):
    from experiments.lake_hindsight_search import HindsightSchedule
    from experiments.lake_tracking_runtime import run_tracking_mission
    from experiments.v4_benchmark.backends import ReplayBackend
    from experiments.v4_benchmark.probe_cache import CachedExecutor
    from experiments.v4_benchmark.routers import build_router

    env = row["environment"]
    scenario, available, _ = scenario_and_cache(prior, source, env, row["mission_id"])
    executors = {}
    for spec in scenario.executors:
        directory, model = available[spec.config_id]
        executors[spec.config_id] = CachedExecutor(
            spec, VerifiedMasks(directory, model), sha(directory / "manifest.json")
        )
    router = (
        HindsightSchedule(
            read(prior / "hindsight" / env / row["mission_id"] / "search.json")["schedule"]
        )
        if row["gt_aware"]
        else build_router(row["policy"], scenario.profiles, scenario.observation_interval_s)
    )
    backend = ReplayBackend(scenario, source / env / "recording")
    poses = {t: f["body_pose"] for t, f in backend.frames.items()}
    destination = (
        output
        / "paid_replay"
        / env
        / ("offline_witness" if row["gt_aware"] else "static_ensemble")
        / variant
    )
    try:
        outcome = run_tracking_mission(
            scenario,
            backend,
            router,
            executors,
            contract,
            destination,
            config=VARIANTS[variant],
            terrain=terrain,
            map_metadata=map_metadata,
            router_name=row["policy"],
            gt_aware=row["gt_aware"],
        )
    finally:
        backend.close()
    events, result = (
        read(destination / "execution/events.json"),
        read(destination / "execution/results.json"),
    )
    if any(e["capture_body_pose"] != poses[e["capture_timestamp_ns"]] for e in events):
        raise ValueError("Paid replay changed physical poses")
    components = (
        "flight_energy_j",
        "inference_energy_j",
        "communication_energy_j",
        "router_energy_j",
        "model_load_energy_j",
        "tracking_energy_j",
    )
    if abs(result["cumulative_energy_j"] - sum(result[k] for k in components)) > 1e-6:
        raise ValueError("Tracking energy accounting mismatch")
    verification = dict(
        environment=env,
        policy=row["policy"],
        gt_aware=row["gt_aware"],
        variant=variant,
        mission_success=outcome["mission_success"],
        failed_constraints=outcome["failed_constraints"],
        report_recall=outcome["reporting"]["recall_by_deadline"],
        false_reports=result["report_quality"]["false_reports"],
        duplicates=result["report_quality"]["duplicate_correct_reports"],
        tracking_energy_j=result["tracking_energy_j"],
        final_battery_frac=result["final_battery_frac"],
        exact_capture_poses=True,
        energy_accounting_verified=True,
        output=str(destination.resolve()),
        outcome_sha256=sha(destination / "outcome.json"),
    )
    save(destination / "verification.json", verification)
    print(
        f"Paid replay {env} / {row['policy']} / {variant}: "
        f"success={outcome['mission_success']}, recall={verification['report_recall']:.3f}",
        flush=True,
    )
    return verification


def run(prior, intent_root, output):
    from experiments.lake_missions.run import freeze

    source = Path(read(prior / "protocol.json")["source"])
    freeze(source)
    rows = cases(prior)
    terrain, metadata = load_public_terrain(Path(read(source / "suite.json")["assets"]))
    contracts = {
        mission: IntentContract(
            read(intent_root / "contracts" / mission / "report_required/intent.json")
        )
        for mission in sorted({r["mission_id"] for r in rows})
    }
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        dict(
            schema_version="lake_tracking_study_v1",
            source=str(source.resolve()),
            prior=str(prior.resolve()),
            purpose="Development tracking diagnostic; no held-out routing superiority claim",
            variants={name: config.to_dict() for name, config in VARIANTS.items()},
            map=metadata,
            anchor_height_m=0.9,
            hfov_rad=1.2,
            report_scoring="consistent_report_support_v1",
            thresholds_selected_from="Existing 1 m, 8 s, 2-hit settings; no outcome-driven search",
            cost=asdict(DEFAULT_TRACKING_COST),
            cost_provenance="Explicit assumptions; not measured device energy or memory",
            input_runs=rows,
            contract_sha256={name: c.sha256 for name, c in contracts.items()},
            intent_source_protocol_sha256=sha(intent_root / "protocol.json"),
            implementation_sha256={name: sha(Path(__file__).with_name(name)) for name in FILES},
            paid_cases=(
                "Both environments: rescue offline witness and static ensemble; "
                "paired image and terrain-memory tracking"
            ),
            expected_fixed_schedule_comparisons=len(rows) * len(VARIANTS),
            expected_paid_replays=8,
        ),
    )
    comparisons = []
    for env in ("day", "dusk_haze"):
        packets, labels = prepare_packets(prior, source, output, env, terrain)
        for index, row in enumerate(rows):
            if row["environment"] == env:
                comparisons.extend(compare_case(row, index, packets, labels, contracts, output))
        print(f"{env}: fixed-schedule comparison complete", flush=True)
    save(output / "fixed_schedule_summary.json", comparisons)
    paid = []
    for row in rows:
        if row["mission_id"] == "sector_rescue" and (
            row["gt_aware"] or row["policy"] == "static:CFG_MASKRCNN_1280_ROT0_90"
        ):
            for variant in ("image_iou", "terrain_memory"):
                paid.append(
                    paid_replay(
                        prior,
                        source,
                        output,
                        row,
                        contracts[row["mission_id"]],
                        variant,
                        terrain,
                        metadata,
                    )
                )
    summary = dict(
        schema_version="lake_tracking_study_v1",
        fixed_schedule=comparisons,
        paid_replays=paid,
        protocol_sha256=sha(output / "protocol.json"),
        map_prior_assumed=True,
        energy_measured=False,
    )
    save(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("intent_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.prior, args.intent_root, args.output)
