"""Extend the frozen pool with UAVid CABiNet using development-only costs/quality.

Reuse verified original predictions, measure only the new models on validation,
and leave the evaluation family unscored. No GT enters online router inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.calibrated_router import fit, merge_profiles
from experiments.v4_benchmark.generalization_study import mission_diagnostics
from experiments.v4_benchmark.model_probe import probe
from experiments.v4_benchmark.probe_cache import build_cached
from experiments.v4_benchmark.probe_cache import freeze as freeze_cache
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import load_scenario
from experiments.v4_benchmark.splits import checksum, freeze, verify
from experiments.v4_benchmark.suite import merge
from experiments.v4_benchmark.validation_study import partition_entries, select_per_profile


def combine_same_recording(left, right):
    """Combine disjoint model evidence, never count one recording as two scenes."""
    for key in (
        "training_recording_identities",
        "training_world_hashes",
        "matching_iou_threshold",
        "feature_encoding",
        "purpose",
    ):
        if left[key] != right[key]:
            raise ValueError(f"calibration source mismatch: {key}")
    if set(left["models"]) & set(right["models"]):
        raise ValueError("duplicate calibrated model")
    return {
        **left,
        "models": {**left["models"], **right["models"]},
        "training_probe_sha256": [left["training_probe_sha256"], right["training_probe_sha256"]],
        "training_rows_sha256": [left["training_rows_sha256"], right["training_rows_sha256"]],
    }


def run(study: Path, screen: Path, output: Path, profiles_path: Path):
    verify(study / "pool_split.json")
    manifest = json.loads((study / "pool_split.json").read_text())
    development = partition_entries(manifest, "development")
    profiles = json.loads(profiles_path.read_text())["profiles"]
    source = json.loads((screen / development[0]["case_id"] / "scenario.json").read_text())
    extra = source["executor_configs"]
    for spec in extra:
        p95 = [
            json.loads((screen / r["case_id"] / "probe/probe.json").read_text())["models"][
                spec["config_id"]
            ]["end_to_end_p95_s"]
            for r in development
        ]
        spec["mission_latency_s"] = math.ceil(max(p95) / 0.01) * 0.01
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "development_profile.json"
    ids = list(load_scenario(Path(development[0]["scenario_path"])).catalog.ids())
    ids += [s["config_id"] for s in extra]
    routers = [(f"static:{key}", f"static:{key}", {}) for key in ids]
    routers += [
        ("intent_visual", "intent_visual", {}),
        ("calibrated_context", "calibrated", dict(artifact_path=str(artifact))),
        ("calibrated_global", "calibrated", dict(artifact_path=str(artifact), use_context=False)),
    ]
    save_json(
        output / "protocol.json",
        dict(
            schema_version="0.1",
            evaluation_scored=False,
            extra_models=extra,
            mission_profiles=profiles,
            routers=routers,
            original_split_sha256=checksum(study / "pool_split.json"),
            scope="Repeated validation after model development, not final held-out evidence",
            energy="assumed 3 J / 8 J per CABiNet call; no hardware measurement",
        ),
    )
    fitted = []
    for row in development:
        case = row["case_id"]
        candidate_profile = output / f"{case}_candidate_profile.json"
        fit(
            screen / case / "scenario.json",
            Path(row["recording_path"]),
            screen / case / "probe",
            candidate_profile,
            use_altitude=True,
        )
        # Also validates original/new cache signatures and recording provenance.
        candidate = load_scenario(screen / case / "scenario.json")
        build_cached(candidate.executors, screen / case / "cache", candidate.recording_identity)
        combined = combine_same_recording(
            json.loads((study / case / "profile.json").read_text()),
            json.loads(candidate_profile.read_text()),
        )
        path = output / f"{case}_combined_profile.json"
        save_json(path, combined)
        fitted.append(path)
    merge_profiles(fitted, artifact)
    artifact_sha = checksum(artifact)
    rows_to_freeze = []
    for row in manifest["entries"]:
        folder = output / row["case_id"]
        folder.mkdir()
        old = Path(row["scenario_path"])
        data = json.loads(old.read_text())
        data["executor_configs"] += copy.deepcopy(extra)
        data["public_profiles"] += [
            dict(
                config_id=s["config_id"],
                expected_upload_mb=0,
                expected_latency_ms=s["mission_latency_s"] * 1000,
                quality_tier=s["quality_tier"],
            )
            for s in extra
        ]
        contract = json.loads((old.parent / "contract.json").read_text())
        for profile in profiles:
            path = folder / profile["case_id"]
            path.mkdir()
            save_json(path / "scenario.json", merge(data, profile["scenario_overrides"]))
            save_json(path / "contract.json", merge(contract, profile["contract_overrides"]))
            rows_to_freeze.append(
                dict(
                    case_id=f"{row['case_id']}__{profile['case_id']}",
                    family_id=row["family_id"],
                    partition=row["partition"],
                    scenario=str(path / "scenario.json"),
                    recording=row["recording_path"],
                )
            )
        # Separate two-model input for new inference; no need to rerun old models.
        data["executor_configs"] = copy.deepcopy(extra)
        data["public_profiles"] = data["public_profiles"][-len(extra) :]
        data["mission"]["fallback_config_id"] = extra[0]["config_id"]
        save_json(folder / "scenario.json", data)
        save_json(folder / "contract.json", contract)
    save_json(
        output / "split_source.json", dict(split_schema_version="0.2", entries=rows_to_freeze)
    )
    freeze(output / "split_source.json", output / "missions_split.json")
    save_json(
        output / "frozen_router.json",
        dict(artifact_sha256=artifact_sha, protocol_sha256=checksum(output / "protocol.json")),
    )
    rows = []
    for entry in partition_entries(manifest, "validation"):
        folder = output / entry["case_id"]
        recording = Path(entry["recording_path"])
        scenario = load_scenario(folder / "scenario.json")
        slots = len(
            range(
                scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns
            )
        )
        probe(folder / "scenario.json", recording, folder / "probe", max_frames=slots, threads=2)
        freeze_cache(folder / "scenario.json", recording, folder / "probe", folder / "cache")
        for profile in profiles:
            scenario = load_scenario(folder / profile["case_id"] / "scenario.json")
            for index, (label, name, params) in enumerate(routers):
                executors = build_cached(
                    scenario.executors[: -len(extra)],
                    study / entry["case_id"] / "cache",
                    scenario.recording_identity,
                )
                executors.update(
                    build_cached(
                        scenario.executors[-len(extra) :],
                        folder / "cache",
                        scenario.recording_identity,
                    )
                )
                backend = ReplayBackend(scenario, recording)
                destination = folder / profile["case_id"] / f"router_{index:02d}"
                try:
                    result = run_mission(
                        scenario,
                        backend,
                        build_router(
                            name, scenario.profiles, scenario.observation_interval_s, params
                        ),
                        executors,
                        router_name=label,
                        output=destination,
                    )
                finally:
                    backend.close()
                events = json.loads((destination / "events.json").read_text())
                rows.append(
                    dict(
                        case_id=entry["case_id"],
                        profile=profile["case_id"],
                        router=label,
                        mission_success=result["mission_success"],
                        status=result["status"],
                        error=result["error"],
                        recall=result["quality"]["target_recall"],
                        precision=result["quality"]["detection_precision"],
                        energy_j=result["cumulative_energy_j"],
                        choices=result["config_selection_history"],
                        failed_constraints=[k for k, v in result["constraints"].items() if not v],
                        **mission_diagnostics(result, events),
                    )
                )
                save_json(output / "rows.json", rows)
        print(f"Validated extended pool: {entry['case_id']}", flush=True)
    from experiments.v4_benchmark.validation_study import verify_router_lock

    verify_router_lock(output)
    verify(output / "missions_split.json")
    if any(r["status"] == "error" for r in rows):
        raise RuntimeError("component errors invalidate this screening")
    report = dict(
        schema_version="0.1",
        evaluation_scored=False,
        rows=rows,
        selection=select_per_profile(rows),
        scope="reused validation, synthetic scenes only",
    )
    save_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "screen", "output", "profiles"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.study, args.screen, args.output, args.profiles)
    print(json.dumps(result["selection"], indent=2))


if __name__ == "__main__":
    main()
