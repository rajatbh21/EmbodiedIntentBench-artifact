"""Audit model feasibility and all fixed baselines on a completed lake table.

Scoring and GT-aware bounds are separate from RGB-only inference and policies.
Earlier fixed-policy runs are revalidated; only previously absent baselines run.
"""

from __future__ import annotations

import argparse
import copy
import statistics
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import optimistic_bound, validate_table
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.objectives import score
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.routing_mission.experiment import router_for
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import build_cached, rgb_key
from experiments.v4_benchmark.runner import evaluation_frame, run_mission


def analyze(source: Path, output: Path, environment: str):
    freeze(source)
    suite = read(source / "suite.json")
    missions = read(source / "missions.json")["missions"]
    env = suite["environments"][environment]
    assets = Path(suite["assets"])
    scenario = load(source / "scenario.json", env, assets)
    folder = output / environment
    table = folder / "predictions"
    complete = read(table / "completion.json")
    manifest = read(table / "manifest.json")
    assert complete["complete"] and complete["manifest_sha256"] == sha(table / "manifest.json")
    assert manifest["source_inference_sha256"] == sha(table / "inference.json")
    assert manifest["source_metadata_sha256"] == sha(table / "source.json")
    metadata = read(table / "source.json")
    assert metadata["source_lock_sha256"] == sha(source / "lock.json")
    assert metadata["script_sha256"] == sha(
        Path(__file__).with_name("lake_feasibility_inference.py")
    )
    implementation = {
        name: sha(Path(__file__).with_name(name))
        for name in (
            "lake_feasibility_analysis.py",
            "lake_feasibility_metrics.py",
            "lake_feasibility_inference.py",
        )
    }
    save(
        folder / "analysis_provenance.json",
        dict(
            schema_version="1.0",
            implementation=implementation,
            source_lock_sha256=sha(source / "lock.json"),
            dense_manifest_sha256=sha(table / "manifest.json"),
        ),
    )
    cached = build_cached(scenario.executors, table, scenario.recording_identity)
    configs = list(cached)
    backend = ReplayBackend(scenario, source / environment / "recording")
    assert manifest["recording_sha256"] == backend.provenance["recording_sha256"]
    scorers = {
        c: MissionEvaluator(
            scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
        )
        for c in configs
    }
    outcomes, capture_times, visibility = [], [], []
    canonical = {}
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            frame = backend.capture()
            time_s = (timestamp - scenario.schedule.epoch_ns) / 1e9
            capture_times.append(time_s)
            canonical[timestamp] = frame.body_pose
            visibility.append(
                dict(
                    capture_id=index,
                    capture_time_s=time_s,
                    visible_pixels={
                        target: int(np.count_nonzero(frame.object_ids == label))
                        for label, target in scenario.targets.items()
                    },
                )
            )
            truth = evaluation_frame(frame, index, time_s, scenario)
            for config in configs:
                mask = cached[config].predictions[rgb_key(frame.rgb)]
                outcome = scorers[config].update(truth, mask).to_dict()
                outcomes.append(
                    dict(
                        capture_id=index,
                        capture_time_s=time_s,
                        config_id=config,
                        **{
                            key: outcome[key]
                            for key in (
                                "matched_target_ids",
                                "predicted_components",
                                "matched_components",
                                "false_positive_components",
                                "visible_target_ids",
                            )
                        },
                    )
                )
            if (index + 1) % 25 == 0:
                print(f"{environment}: scored {index + 1} captures", flush=True)
    finally:
        backend.close()
    validate_table(outcomes, capture_times, configs)
    source_rows = read(table / "inference.json")
    assert len(source_rows) == len(outcomes)
    assert {(r["capture_id"], r["config_id"]) for r in source_rows} == {
        (r["capture_id"], r["config_id"]) for r in outcomes
    }
    save(folder / "outcomes.json", outcomes)
    save(folder / "visibility.json", visibility)
    bounds = [optimistic_bound(mission, outcomes, capture_times, configs) for mission in missions]
    save(
        folder / "optimistic_bounds.json",
        dict(
            schema_version="1.0",
            outcome_table_sha256=sha(folder / "outcomes.json"),
            bounds=bounds,
        ),
    )
    for bound in bounds:
        print(f"{environment}/{bound['mission_id']}: {bound['conclusion']}", flush=True)
    diagnostics = []
    for config in configs:
        rows = [r for r in outcomes if r["config_id"] == config]
        target_hits = {
            target: [r["capture_time_s"] for r in rows if target in r["matched_target_ids"]]
            for target in scenario.targets.values()
        }
        tp = sum(r["matched_components"] for r in rows)
        predicted = sum(r["predicted_components"] for r in rows)
        visible = sum(len(r["visible_target_ids"]) for r in rows)
        diagnostics.append(
            dict(
                config_id=config,
                captures=len(rows),
                matched_components=tp,
                predicted_components=predicted,
                detection_precision=tp / predicted if predicted else None,
                visible_target_observation_recall=tp / visible if visible else None,
                false_components=sum(r["false_positive_components"] for r in rows),
                target_matched_capture_times=target_hits,
                by_target={
                    target: dict(
                        visible_captures=sum(v["visible_pixels"][target] > 0 for v in visibility),
                        max_visible_pixels=max(v["visible_pixels"][target] for v in visibility),
                        median_visible_pixels=statistics.median(
                            [
                                v["visible_pixels"][target]
                                for v in visibility
                                if v["visible_pixels"][target] > 0
                            ]
                            or [0]
                        ),
                        matched_captures=len(target_hits[target]),
                    )
                    for target in scenario.targets.values()
                },
                note="Dense diagnostic: every capture processed; not a legal mission execution.",
            )
        )
    save(folder / "model_diagnostics.json", diagnostics)
    comparisons = []
    existing = read(source / environment / "comparison.json")["rows"]
    for mission in missions:
        case = folder / "baselines" / mission["mission_id"]
        case.mkdir(parents=True, exist_ok=True)
        data, contract = read(source / "scenario.json"), read(source / "contract.json")
        data["mission"]["initial_battery_frac"] = mission["initial_battery_frac"]
        contract["min_final_battery_frac"] = mission["min_final_battery_frac"]
        save(case / "scenario.json", data)
        save(case / "contract.json", contract)
        local = load(case / "scenario.json", env, assets)
        assert local.recording_identity == scenario.recording_identity
        for config in configs:
            policy = "static:" + config
            old = next(
                (
                    r
                    for r in existing
                    if r["mission_id"] == mission["mission_id"] and r["policy"] == policy
                ),
                None,
            )
            if old:
                path = source / old["output"]
                result = read(path / "results.json")
                origin = "revalidated_prior_fixed_baseline"
            else:
                path = case / config
                replay = ReplayBackend(local, source / environment / "recording")
                try:
                    result = run_mission(
                        local,
                        replay,
                        router_for(policy, local, suite["seed"]),
                        {k: copy.copy(v) for k, v in cached.items()},
                        router_name=policy,
                        output=path,
                    )
                finally:
                    replay.close()
                origin = "new_fixed_baseline_using_actual_prediction_replay"
            events = read(path / "events.json")
            for event in events:
                assert event["capture_body_pose"] == canonical[event["capture_timestamp_ns"]]
                assert event["action"]["config_id"] == config
            objective = score(mission, events, result)
            if old:
                original = read(path / "objectives.json")
                assert all(original[k] == v for k, v in objective.items())
            else:
                save(path / "objectives.json", objective)
            comparisons.append(
                dict(
                    environment=environment,
                    mission_id=mission["mission_id"],
                    policy=policy,
                    origin=origin,
                    output=str(path.resolve()),
                    results_sha256=sha(path / "results.json"),
                    events_sha256=sha(path / "events.json"),
                    recording_sha256=manifest["recording_sha256"],
                    exact_capture_poses=True,
                    **{
                        k: objective[k]
                        for k in (
                            "mission_success",
                            "phase_success_rate",
                            "weighted_evidence_recall",
                            "resource_success",
                            "execution_success",
                        )
                    },
                    final_battery_frac=result["final_battery_frac"],
                    processed=result["processed_observations"],
                )
            )
            save(folder / "baselines.partial.json", comparisons)
            print(
                f"{environment}/{mission['mission_id']}/{config}: "
                f"{objective['phase_success_rate']:.0%} stages",
                flush=True,
            )
    assert len(comparisons) == len(missions) * len(configs)
    save(folder / "baselines.json", dict(schema_version="1.0", rows=comparisons))
    save(
        folder / "analysis_complete.json",
        dict(
            schema_version="1.0",
            completed=True,
            dense_rows=len(outcomes),
            fixed_baselines=len(comparisons),
            all_executions_successful=all(r["execution_success"] for r in comparisons),
            certified_infeasible_missions=[
                b["mission_id"] for b in bounds if b["mission_impossible_under_current_table"]
            ],
            output_hashes={
                name: sha(folder / name)
                for name in (
                    "outcomes.json",
                    "visibility.json",
                    "optimistic_bounds.json",
                    "model_diagnostics.json",
                    "baselines.json",
                )
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--environment", choices=("day", "dusk_haze"), required=True)
    args = parser.parse_args()
    analyze(args.source, args.output, args.environment)
