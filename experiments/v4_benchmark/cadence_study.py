"""Paired cadence sensitivity on reused validation recordings, never evaluation.

Fixed model choice and inference frequency are separate controls. Flight/sensor
sampling stays unchanged; intentional skips cannot inspect RGB or mission GT.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.generalization_study import mission_diagnostics
from experiments.v4_benchmark.mission_requirements import MissionRequirements
from experiments.v4_benchmark.probe_cache import build_cached
from experiments.v4_benchmark.reporting import ReportingConfig
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import exported_source, load_scenario
from experiments.v4_benchmark.splits import checksum, verify
from experiments.v4_benchmark.validation_study import partition_entries, verify_router_lock


def tradeoffs(rows):
    comparisons = []
    for row in rows:
        if row["stride"] == 1:
            continue
        reference = next(
            r
            for r in rows
            if r["stride"] == 1 and all(r[k] == row[k] for k in ("scene", "profile", "router"))
        )
        both_pass = row["mission_success"] and reference["mission_success"]
        comparisons.append(
            dict(
                scene=row["scene"],
                profile=row["profile"],
                router=row["router"],
                stride=row["stride"],
                both_successful=both_pass,
                success_lost=reference["mission_success"] and not row["mission_success"],
                energy_delta_j=row["energy_j"] - reference["energy_j"] if both_pass else None,
                reference_first_report_s=reference["first_correct_report_s"],
                first_correct_report_s=row["first_correct_report_s"],
            )
        )
    return comparisons


def run(original: Path, expanded: Path, output: Path, *, report_profiles: Path | None = None):
    verify(expanded / "missions_split.json")
    verify(original / "pool_split.json")
    verify_router_lock(expanded)
    manifest = json.loads((expanded / "missions_split.json").read_text())
    profiles = None
    if report_profiles is not None:
        profiles = json.loads(report_profiles.read_text())
        expected = {
            entry["case_id"].split("__")[1] for entry in partition_entries(manifest, "validation")
        }
        if set(profiles) != expected:
            raise ValueError("report profiles must cover the frozen mission profiles exactly")
        profiles = {key: MissionRequirements.parse(value) for key, value in profiles.items()}
    output.mkdir(parents=True, exist_ok=False)
    artifact = expanded / "development_profile.json"
    save_json(
        output / "protocol.json",
        dict(
            schema_version="0.1",
            strides=[1, 2, 4],
            reporting=ReportingConfig().to_dict(),
            report_profiles={key: value.to_dict() for key, value in profiles.items()}
            if profiles
            else None,
            profile_semantics=(
                "optional report requirements replace legacy extra requirements; "
                "primary contract unchanged"
            ),
            source_split_sha256=checksum(expanded / "missions_split.json"),
            calibration_sha256=checksum(artifact),
            scope="reused validation, fixed cadence; not held-out evidence",
            costs="same configured costs; full flight power during intentional skips",
            evaluation_scored=False,
        ),
    )
    rows = []
    for entry in partition_entries(manifest, "validation"):
        scene, profile = entry["case_id"].split("__")
        scenario = load_scenario(Path(entry["scenario_path"]))
        if profiles is not None:
            scenario = replace(scenario, requirements=profiles[profile])
            case_dir = output / entry["case_id"]
            case_dir.mkdir(parents=True)
            save_json(case_dir / "scenario.json", exported_source(scenario))
            save_json(
                case_dir / "contract.json", json.loads(Path(entry["contract_path"]).read_text())
            )
        old_specs = [s for s in scenario.executors if s.kind != "torch_cabinet_segmentation"]
        new_specs = [s for s in scenario.executors if s.kind == "torch_cabinet_segmentation"]
        routers = [
            (f"static:{s.config_id}", f"static:{s.config_id}", {}) for s in scenario.executors
        ]
        routers += [
            ("calibrated_context", "calibrated", dict(artifact_path=str(artifact))),
            (
                "calibrated_global",
                "calibrated",
                dict(artifact_path=str(artifact), use_context=False),
            ),
        ]
        for stride in (1, 2, 4):
            for index, (label, name, params) in enumerate(routers):
                executors = build_cached(
                    old_specs, original / scene / "cache", scenario.recording_identity
                )
                executors.update(
                    build_cached(new_specs, expanded / scene / "cache", scenario.recording_identity)
                )
                backend = ReplayBackend(scenario, Path(entry["recording_path"]))
                destination = output / entry["case_id"] / f"stride_{stride}" / f"router_{index:02d}"
                try:
                    result = run_mission(
                        scenario,
                        backend,
                        build_router(
                            name,
                            scenario.profiles,
                            scenario.observation_interval_s * stride,
                            params,
                        ),
                        executors,
                        router_name=label,
                        output=destination,
                        decision_stride=stride,
                    )
                finally:
                    backend.close()
                events = json.loads((destination / "events.json").read_text())
                rows.append(
                    dict(
                        scene=scene,
                        profile=profile,
                        router=label,
                        stride=stride,
                        decision_period_s=result["decision_period_s"],
                        mission_success=result["mission_success"],
                        status=result["status"],
                        error=result["error"],
                        recall=result["quality"]["target_recall"],
                        precision=result["quality"]["detection_precision"],
                        report_quality=result["report_quality"],
                        evaluation_protocol_sha256=result["evaluation_protocol_sha256"],
                        matched_detections=result["quality"]["matched_detections"],
                        false_positive_detections=result["quality"]["false_positive_detections"],
                        total_predictions=result["quality"]["total_predictions"],
                        energy_j=result["cumulative_energy_j"],
                        flight_energy_j=result["flight_energy_j"],
                        inference_energy_j=result["inference_energy_j"],
                        router_energy_j=result["router_energy_j"],
                        processed=result["processed_observations"],
                        busy_skipped=len(result["busy_skipped_observation_ids"]),
                        cadence_skipped=len(result["cadence_skipped_observation_ids"]),
                        failed_constraints=[k for k, v in result["constraints"].items() if not v],
                        choices=result["config_selection_history"],
                        **mission_diagnostics(result, events),
                    )
                )
                save_json(output / "rows.json", rows)
    verify_router_lock(expanded)
    if any(r["status"] == "error" for r in rows):
        raise RuntimeError("component errors invalidate comparison")
    report = dict(
        schema_version="0.1",
        evaluation_scored=False,
        rows=rows,
        cadence_tradeoffs=tradeoffs(rows),
        scope="162 paired synthetic validation missions, not independent samples",
    )
    save_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("original", "expanded", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--report-profiles", type=Path)
    args = parser.parse_args()
    report = run(args.original, args.expanded, args.output, report_profiles=args.report_profiles)
    print(
        json.dumps(
            dict(
                missions=len(report["rows"]),
                success_losses=sum(r["success_lost"] for r in report["cadence_tradeoffs"]),
            )
        )
    )


if __name__ == "__main__":
    main()
