"""Complete RGB-only rotation predictions without modifying frozen experiments.

The custom pipeline kind deliberately cannot be run by the generic executor
factory. Only this module interprets its preprocessing and inverse geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from missionbench.v2.executors import build_executors
from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.lake_orientation_diagnostic import restore_mask
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import validate_result

CONTROL = "CFG_MASKRCNN_1280"
ROTATED = "CFG_MASKRCNN_1280_ROT90"
ENSEMBLE = "CFG_MASKRCNN_1280_ROT0_90"
KIND = "lake_fixed_rotation_pipeline"


def candidate_specs(control):
    return [
        replace(
            control,
            config_id=config,
            kind=KIND,
            model_strategy_id=f"maskrcnn_r50_v2_fixed_views_{views}",
            mission_latency_s=control.mission_latency_s * passes,
            energy_j_per_call=control.energy_j_per_call * passes,
            parameters={**control.parameters, "fixed_views_ccw": views},
        )
        for config, views, passes in ((ROTATED, "90", 1), (ENSEMBLE, "0,90", 2))
    ]


def union_masks(original, rotated):
    if original.dtype != np.bool_ or rotated.dtype != np.bool_ or original.shape != rotated.shape:
        raise ValueError("Ensemble constituents must share boolean sensor coordinates")
    return np.logical_or(original, rotated)


def initialize(source, prior, screen, output):
    freeze(source)
    suite = read(source / "suite.json")
    scenario = load(source / "scenario.json", suite["environments"]["day"], Path(suite["assets"]))
    control = next(s for s in scenario.executors if s.config_id == CONTROL)
    screen_summary = read(screen / "summary.json")
    assert read(screen / "protocol.json")["source_lock_sha256"] == sha(source / "lock.json")
    timing = max(
        e["configurations"][CONTROL]["host_latency_p95_s"]
        for e in screen_summary["environments"].values()
    )
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        dict(
            schema_version="1.0",
            source=str(source.resolve()),
            prior=str(prior.resolve()),
            source_lock_sha256=sha(source / "lock.json"),
            implementation_sha256=sha(Path(__file__)),
            restore_implementation_sha256=sha(
                Path(__file__).with_name("lake_orientation_diagnostic.py")
            ),
            scope="development extension after known failures; no held-out performance claim",
            environments={
                env: dict(
                    recording_sha256=sha(source / env / "recording/recording.json"),
                    prior_manifest_sha256=sha(prior / env / "predictions/manifest.json"),
                    captures=len(read(source / env / "recording/recording.json")["frames"]),
                )
                for env in ("day", "dusk_haze")
            },
            configurations=[asdict(s) for s in candidate_specs(control)],
            inference_inputs="Every canonical whole RGB image; fixed 90-degree CCW rotation; no GT",
            ensemble="Boolean union of verified original and inverse-rotated predicted masks",
            sampling="all canonical capture slots; thresholds and missions unchanged",
            host_threads=4,
            original_control_host_p95_s=timing,
            timing_source_summary_sha256=sha(screen / "summary.json"),
            configured_costs="ROT90: one original 1280 call; ensemble: two calls; energy assumed",
            host_costs="Control: screen p95; ROT90: dense p95; ensemble: sum plus union",
            cost_scope="Host timing sensitivity; energy, memory, loading, flight power assumed",
            new_dense_inference_calls=2
            * len(read(source / "day/recording/recording.json")["frames"]),
            model_selection="no GT-adaptive rotation or winning-image selection",
        ),
    )


def infer(output, environment):
    import torch

    protocol = read(output / "protocol.json")
    assert protocol["implementation_sha256"] == sha(Path(__file__))
    assert protocol["restore_implementation_sha256"] == sha(
        Path(__file__).with_name("lake_orientation_diagnostic.py")
    )
    source, prior = Path(protocol["source"]), Path(protocol["prior"])
    freeze(source)
    assert protocol["source_lock_sha256"] == sha(source / "lock.json")
    suite = read(source / "suite.json")
    scenario = load(
        source / "scenario.json", suite["environments"][environment], Path(suite["assets"])
    )
    control = next(s for s in scenario.executors if s.config_id == CONTROL)
    specs = candidate_specs(control)
    assert [asdict(s) for s in specs] == protocol["configurations"]
    previous = prior / environment / "predictions"
    assert (
        sha(previous / "manifest.json")
        == protocol["environments"][environment]["prior_manifest_sha256"]
    )
    manifest = read(previous / "manifest.json")
    assert read(previous / "completion.json")["manifest_sha256"] == sha(previous / "manifest.json")
    assert manifest["recording_identity"] == scenario.recording_identity
    assert manifest["models"][CONTROL]["signature"] == model_signature(control)
    raw = replace(
        control, parameters={**control.parameters, "input_width_px": 960, "input_height_px": 1280}
    )
    torch.set_num_threads(protocol["host_threads"])
    torch.set_num_interop_threads(1)
    executor = build_executors([raw])[CONTROL]
    folder = output / environment / "predictions"
    folder.mkdir(parents=True, exist_ok=False)
    models = {s.config_id: dict(signature=model_signature(s), predictions={}) for s in specs}
    rows = []
    backend = ReplayBackend(scenario, source / environment / "recording")
    assert (
        backend.provenance["recording_sha256"]
        == protocol["environments"][environment]["recording_sha256"]
    )
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            frame = backend.capture()
            key = rgb_key(frame.rgb)
            old = manifest["models"][CONTROL]["predictions"][key]
            path = (previous / old["file"]).resolve()
            assert path.parent == previous.resolve() and sha(path) == old["sha256"]
            with np.load(path, allow_pickle=False) as data:
                original = data["prediction_mask"].copy()
            assert hashlib.sha256(original.tobytes()).hexdigest() == old["mask_sha256"]
            started = time.perf_counter()
            rotated_rgb = np.rot90(frame.rgb, 1).copy()
            result = executor.run(rotated_rgb)
            validate_result(result, CONTROL, rotated_rgb.shape[:2])
            if not result.success:
                raise RuntimeError("Rotation inference failed")
            rotated = restore_mask(result.prediction_mask, 1, frame.rgb.shape[:2])
            wall_s = time.perf_counter() - started
            started = time.perf_counter()
            ensemble = union_masks(original, rotated)
            union_s = time.perf_counter() - started
            for spec, mask in zip(specs, (rotated, ensemble), strict=True):
                path = folder / f"{index:04d}_{spec.config_id}.npz"
                np.savez_compressed(path, prediction_mask=mask)
                item = dict(
                    file=path.name,
                    sha256=sha(path),
                    mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                )
                table = models[spec.config_id]["predictions"]
                if key in table and table[key]["mask_sha256"] != item["mask_sha256"]:
                    raise ValueError("Identical RGB yielded inconsistent stateless output")
                table[key] = item
            rows.append(
                dict(
                    capture_id=index,
                    timestamp_ns=timestamp,
                    capture_time_s=(timestamp - scenario.schedule.epoch_ns) / 1e9,
                    rgb_sha256=key,
                    actual_raw_execution=result.to_summary(),
                    measured_rotation_call_s=wall_s,
                    measured_union_s=union_s,
                    original_prediction_sha256=old["sha256"],
                    constituent_origin="original: verified cache; 90: fresh whole-image inference",
                )
            )
            save(folder / "progress.json", dict(captures=len(rows), last_capture=index))
            if index % 10 == 0:
                print(
                    f"{environment}: {index + 1}/{len(backend.frames)} "
                    f"rotation calls, {wall_s:.2f}s",
                    flush=True,
                )
    finally:
        backend.close()
    assert len(rows) == protocol["environments"][environment]["captures"]
    save(folder / "inference.json", rows)
    save(
        folder / "manifest.json",
        dict(
            cache_schema_version="0.1",
            execution_mode="precomputed_real_model_predictions",
            recording_identity=scenario.recording_identity,
            recording_sha256=backend.provenance["recording_sha256"],
            capture_count=len(rows),
            models=models,
            protocol_sha256=sha(output / "protocol.json"),
            inference_sha256=sha(folder / "inference.json"),
        ),
    )
    save(
        folder / "completion.json",
        dict(
            complete=True,
            fresh_rotation_calls=len(rows),
            reused_control_predictions=len(rows),
            derived_ensemble_masks=len(rows),
            manifest_sha256=sha(folder / "manifest.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    for name in ("source", "prior", "screen", "output"):
        init.add_argument(name, type=Path)
    run = commands.add_parser("infer")
    run.add_argument("output", type=Path)
    run.add_argument("--environment", choices=("day", "dusk_haze"), required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.source, args.prior, args.screen, args.output)
    else:
        infer(args.output, args.environment)
