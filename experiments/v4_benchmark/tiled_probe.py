"""Development-only screening of fixed tiled Mask R-CNN against full-frame controls.

Uses actual RGB inference and measured development p95 for the new configuration.
Energy is an explicit four-forward-pass assumption, not hardware measurement.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.model_probe import probe
from experiments.v4_benchmark.probe_cache import build_cached
from experiments.v4_benchmark.probe_cache import freeze as freeze_cache
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import load_scenario
from experiments.v4_benchmark.splits import checksum, verify
from experiments.v4_benchmark.suite import merge
from experiments.v4_benchmark.validation_study import partition_entries

TILED_ID = "CFG_MASKRCNN_320_TILED"


def tiled_source(source: dict) -> dict:
    data = copy.deepcopy(source)
    spec = next(s for s in data["executor_configs"] if s["config_id"] == "CFG_MASKRCNN_320")
    control = copy.deepcopy(spec)
    spec.update(
        config_id=TILED_ID,
        model_strategy_id="maskrcnn_r50_v2_four_tiles",
        energy_j_per_call=spec["energy_j_per_call"] * 4,
    )
    spec["parameters"].update(tile_width_px=192, tile_height_px=144, tile_overlap_frac=1 / 3)
    data["executor_configs"] = [control, spec]
    data["public_profiles"] = [
        dict(
            config_id=TILED_ID,
            expected_latency_ms=spec["mission_latency_s"] * 1000,
            expected_upload_mb=0,
            quality_tier="high",
        )
    ]
    data["public_profiles"].append(
        dict(
            config_id=control["config_id"],
            expected_latency_ms=control["mission_latency_s"] * 1000,
            expected_upload_mb=0,
            quality_tier="high",
        )
    )
    data["mission"]["fallback_config_id"] = TILED_ID
    return data


def run(study: Path, output: Path, profiles_path: Path):
    verify(study / "pool_split.json")
    manifest = json.loads((study / "pool_split.json").read_text())
    entries = partition_entries(manifest, "development")
    profiles = json.loads(profiles_path.read_text())["profiles"]
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        dict(
            schema_version="0.1",
            scope="development screening; not validation/evaluation",
            source_split_sha256=checksum(study / "pool_split.json"),
            tile_size_px=[192, 144],
            overlap_frac=1 / 3,
            tile_count=4,
            selection="fixed configuration before new inference; no GT-selected crops",
            aggregation="maximum of overlapping person probabilities, then threshold",
            mission_profiles=profiles,
            energy="assumed 4 x Mask R-CNN 320 compute joules",
            latency="new candidate max of development-scene p95, rounded up to 10 ms",
            training_domain="unchanged COCO weights; not aerial-finetuned; not official SAHI",
        ),
    )
    measured = []
    for entry in entries:
        folder = output / entry["case_id"]
        folder.mkdir()
        original = Path(entry["scenario_path"])
        data = tiled_source(json.loads(original.read_text()))
        save_json(
            folder / "contract.json", json.loads((original.parent / "contract.json").read_text())
        )
        save_json(folder / "scenario.json", data)
        scenario = load_scenario(folder / "scenario.json")
        slots = len(
            range(
                scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns
            )
        )
        report = probe(
            folder / "scenario.json",
            Path(entry["recording_path"]),
            folder / "probe",
            max_frames=slots,
            threads=2,
        )
        freeze_cache(
            folder / "scenario.json",
            Path(entry["recording_path"]),
            folder / "probe",
            folder / "cache",
        )
        measured.append(report["models"][TILED_ID]["end_to_end_p95_s"])
        print(f"Measured tiled development: {entry['case_id']}", flush=True)
    latency = math.ceil(max(measured) / 0.01) * 0.01
    save_json(
        output / "cost.json",
        dict(
            schema_version="0.1",
            latency_s=latency,
            development_p95_s=measured,
            energy_j_per_call=32,
            energy_measured=False,
        ),
    )
    rows = []
    for entry in entries:
        folder = output / entry["case_id"]
        for config_id in ("CFG_MASKRCNN_320", "CFG_MASKRCNN_640", TILED_ID):
            is_tiled = config_id == TILED_ID
            base_path = folder / "scenario.json" if is_tiled else Path(entry["scenario_path"])
            base = json.loads(base_path.read_text())
            if is_tiled:
                base["executor_configs"][1]["mission_latency_s"] = latency
                base["public_profiles"][0]["expected_latency_ms"] = latency * 1000
            for profile in profiles:
                destination = folder / config_id / profile["case_id"]
                destination.mkdir(parents=True)
                save_json(destination / "scenario.json", merge(base, profile["scenario_overrides"]))
                contract = json.loads((base_path.parent / "contract.json").read_text())
                save_json(
                    destination / "contract.json", merge(contract, profile["contract_overrides"])
                )
                scenario = load_scenario(destination / "scenario.json")
                backend = ReplayBackend(scenario, Path(entry["recording_path"]))
                cache = folder / "cache" if is_tiled else study / entry["case_id"] / "cache"
                try:
                    result = run_mission(
                        scenario,
                        backend,
                        build_router(
                            f"static:{config_id}",
                            scenario.profiles,
                            scenario.observation_interval_s,
                        ),
                        build_cached(scenario.executors, cache, scenario.recording_identity),
                        router_name=f"static:{config_id}",
                        output=destination / "run",
                    )
                finally:
                    backend.close()
                rows.append(
                    dict(
                        scene=entry["case_id"],
                        profile=profile["case_id"],
                        config_id=config_id,
                        success=result["mission_success"],
                        status=result["status"],
                        error=result["error"],
                        recall=result["quality"]["target_recall"],
                        precision=result["quality"]["detection_precision"],
                        first_correct_report_s=result["first_correct_report_s"],
                        energy_j=result["cumulative_energy_j"],
                        processed=result["processed_observations"],
                        failed_constraints=[k for k, v in result["constraints"].items() if not v],
                    )
                )
                save_json(output / "rows.json", rows)
    report = dict(
        schema_version="0.1",
        scope="development screening only",
        candidate_latency_s=latency,
        rows=rows,
        evaluation_scored=False,
    )
    save_json(output / "report.json", report)
    if any(r["status"] == "error" for r in rows):
        raise RuntimeError("component error; inspect retained results")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "output", "profiles"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.study, args.output, args.profiles)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
