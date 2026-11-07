"""Screen the pinned UAVid model at two fixed input sizes on development only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments.v4_benchmark.backends import save_json
from experiments.v4_benchmark.model_probe import probe
from experiments.v4_benchmark.probe_cache import freeze
from experiments.v4_benchmark.scenario import exported_source, load_scenario
from experiments.v4_benchmark.splits import checksum, verify
from experiments.v4_benchmark.validation_study import partition_entries


def prepare_scenario(source: Path, output: Path, bundle: Path) -> Path:
    from missionbench.v2.cabinet_models import verify_bundle

    bundle = bundle.resolve()
    metadata = verify_bundle(bundle, checksum(bundle))
    original = load_scenario(source)
    data = exported_source(original)
    specs = []
    for width, height, energy, latency in ((320, 240, 3, 0.1), (640, 480, 8, 0.4)):
        spec = copy.deepcopy(data["executor_configs"][0])
        spec.update(
            config_id=f"CFG_CABINET_UAVID_{width}",
            model_strategy_id=f"cabinet_uavid_small_{width}",
            kind="torch_cabinet_segmentation",
            energy_j_per_call=energy,
            mission_latency_s=latency,
            quality_tier="low",
        )
        spec["parameters"] = dict(
            model_id="cabinet_mobilenetv3_small_uavid",
            weights_id=metadata["revision"],
            bundle_manifest=str(bundle),
            bundle_sha256=checksum(bundle),
            task_class="person",
            input_width_px=width,
            input_height_px=height,
            probability_threshold=0.5,
            device="cpu",
            dtype="float32",
            latency_mode="configured",
            warmup_runs=1,
            min_component_px=0,
        )
        specs.append(spec)
    data["executor_configs"] = specs
    data["public_profiles"] = [
        dict(
            config_id=s["config_id"],
            expected_latency_ms=s["mission_latency_s"] * 1000,
            expected_upload_mb=0,
            quality_tier=s["quality_tier"],
        )
        for s in specs
    ]
    data["mission"]["fallback_config_id"] = specs[0]["config_id"]
    contract = json.loads((source.parent / original.source["contract_path"]).read_text())
    data["contract_path"] = "contract.json"
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "scenario.json", data)
    save_json(output / "contract.json", contract)
    load_scenario(output / "scenario.json")
    return output / "scenario.json"


def run(study: Path, bundle: Path, output: Path):
    verify(study / "pool_split.json")
    manifest = json.loads((study / "pool_split.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        dict(
            schema_version="0.1",
            scope="development only",
            bundle_sha256=checksum(bundle),
            source_split_sha256=checksum(study / "pool_split.json"),
            input_sizes=[[320, 240], [640, 480]],
            probability_threshold=0.5,
            costs="initial diagnostic assumptions; not mission-calibrated",
        ),
    )
    for row in partition_entries(manifest, "development"):
        folder = output / row["case_id"]
        path = prepare_scenario(Path(row["scenario_path"]), folder, bundle)
        scenario = load_scenario(path)
        slots = len(
            range(
                scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns
            )
        )
        recording = Path(row["recording_path"])
        probe(path, recording, folder / "probe", max_frames=slots, threads=2)
        freeze(path, recording, folder / "probe", folder / "cache")
        print(f"Screened CABiNet development: {row['case_id']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "bundle", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    run(args.study, args.bundle, args.output)


if __name__ == "__main__":
    main()
