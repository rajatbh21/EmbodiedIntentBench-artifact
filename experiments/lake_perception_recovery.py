"""Development RGB-only resolution/tiling screen on immutable lake recordings.

The protocol is written before inference. GT is used only by the separate
analysis command. This screen does not execute or certify a mission policy.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from missionbench.v2.executors import build_executors
from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import validate_result


def capture_indices(count, stride):
    if count < 1 or stride < 1:
        raise ValueError("Positive capture count and stride required")
    return sorted(set(range(0, count, stride)) | {count - 1})


def candidates(base):
    specs = []
    for config, width, height, tiles in (
        ("CFG_MASKRCNN_1280", 1280, 960, {}),
        ("CFG_MASKRCNN_1920", 1920, 1440, {}),
        (
            "CFG_MASKRCNN_1280_FOUR_TILES",
            1280,
            960,
            dict(tile_width_px=768, tile_height_px=576, tile_overlap_frac=1 / 3),
        ),
    ):
        parameters = dict(base.parameters)
        parameters.update(input_width_px=width, input_height_px=height, **tiles)
        specs.append(
            replace(
                base,
                config_id=config,
                model_strategy_id=config.lower(),
                parameters=parameters,
            )
        )
    return specs


def initialize(source, output, stride, threads):
    freeze(source)
    suite = read(source / "suite.json")
    scenario = load(source / "scenario.json", suite["environments"]["day"], Path(suite["assets"]))
    base = next(s for s in scenario.executors if s.config_id == "CFG_MASKRCNN_1280")
    recordings = {}
    for environment in ("day", "dusk_haze"):
        manifest = source / environment / "recording" / "recording.json"
        data = read(manifest)
        recordings[environment] = dict(
            manifest_sha256=sha(manifest),
            captures=len(data["frames"]),
            capture_indices=capture_indices(len(data["frames"]), stride),
        )
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        dict(
            schema_version="1.0",
            scope="development screening on known failure scenes; not held-out evaluation",
            source=str(source.resolve()),
            source_lock_sha256=sha(source / "lock.json"),
            inference_script_sha256=sha(Path(__file__)),
            sampling="fixed capture stride plus final capture; independent of GT and predictions",
            stride=stride,
            recordings=recordings,
            configurations=[asdict(s) for s in candidates(base)],
            host_threads=threads,
            inference_inputs="RGB only; fixed raster tiles cover the entire input",
            thresholds="unchanged person score 0.5, mask probability 0.5, matching IoU 0.3",
            tile_semantics=(
                "four overlapping tiles; each internally resized by torchvision; "
                "max probability union"
            ),
            image_resolution="recorded RGB remains 960x720; upsampling adds no sensor detail",
            costs=(
                "fresh measured host executor time; inherited configured costs are "
                "placeholders, not used for scoring"
            ),
            energy_measured=False,
            mission_execution=False,
            selection_rule="Report every configuration; no pool promotion from screening alone",
        ),
    )


def run(output, environment):
    import torch

    protocol = read(output / "protocol.json")
    source = Path(protocol["source"])
    freeze(source)
    assert protocol["source_lock_sha256"] == sha(source / "lock.json")
    assert protocol["inference_script_sha256"] == sha(Path(__file__))
    suite = read(source / "suite.json")
    scenario = load(
        source / "scenario.json", suite["environments"][environment], Path(suite["assets"])
    )
    recording = source / environment / "recording"
    assert (
        sha(recording / "recording.json") == protocol["recordings"][environment]["manifest_sha256"]
    )
    torch.set_num_threads(protocol["host_threads"])
    torch.set_num_interop_threads(1)
    specs = [ExecutorConfigSpec(**s) for s in protocol["configurations"]]
    folder = output / environment
    folder.mkdir(exist_ok=False)
    # Load and warm every backend before measuring any sample.
    executors = build_executors(specs)
    rows = []
    backend = ReplayBackend(scenario, recording)
    try:
        for order, index in enumerate(protocol["recordings"][environment]["capture_indices"]):
            timestamp = sorted(backend.frames)[index]
            backend.advance_to(timestamp)
            frame = backend.capture()
            # Rotate evaluation order to reduce systematic host warm/thermal bias.
            ordered = specs[order % len(specs) :] + specs[: order % len(specs)]
            for spec in ordered:
                result = executors[spec.config_id].run(frame.rgb.copy())
                validate_result(result, spec.config_id, frame.rgb.shape[:2])
                if not result.success:
                    raise RuntimeError(f"Inference failed: {spec.config_id}")
                path = folder / f"{index:04d}_{spec.config_id}.npz"
                np.savez_compressed(path, prediction_mask=result.prediction_mask)
                rows.append(
                    dict(
                        capture_id=index,
                        capture_time_s=(timestamp - scenario.schedule.epoch_ns) / 1e9,
                        timestamp_ns=timestamp,
                        rgb_sha256=rgb_key(frame.rgb),
                        config_id=spec.config_id,
                        file=path.name,
                        sha256=sha(path),
                        execution=result.to_summary(),
                    )
                )
                save(folder / "progress.json", rows)
                print(
                    f"{environment} frame {index}: {spec.config_id}, "
                    f"{result.measured_wall_clock_s:.2f}s",
                    flush=True,
                )
    finally:
        backend.close()
    save(folder / "inference.json", rows)
    save(
        folder / "completion.json",
        dict(
            complete=True,
            actual_calls=len(rows),
            protocol_sha256=sha(output / "protocol.json"),
            inference_sha256=sha(folder / "inference.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    init.add_argument("source", type=Path)
    init.add_argument("output", type=Path)
    init.add_argument("--stride", type=int, default=8)
    init.add_argument("--threads", type=int, default=4)
    infer = commands.add_parser("run")
    infer.add_argument("output", type=Path)
    infer.add_argument("--environment", choices=("day", "dusk_haze"), required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.source, args.output, args.stride, args.threads)
    else:
        run(args.output, args.environment)
