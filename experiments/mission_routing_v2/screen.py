"""Predeclare and execute complete development inference for the extended pool."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.scenario import load
from experiments.lake_tracking_study import scenario_and_cache
from experiments.mission_routing_v2.models import NEW, POOL, InferencePool, candidates
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key


def initialize(prior, source, weights, output):
    scenario, _, _ = scenario_and_cache(prior, source, "day", "sector_rescue")
    suite = read(source / "suite.json")
    original = load(source / "scenario.json", suite["environments"]["day"], Path(suite["assets"]))
    cabinet = next(s for s in original.executors if s.config_id == "CFG_CABINET_640")
    specs = candidates(scenario.executors, cabinet, weights)
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        dict(
            schema_version="routing_pool_screen_v2",
            scope="development",
            prior=str(prior.resolve()),
            source=str(source.resolve()),
            pool=list(POOL),
            new_configs=list(NEW),
            specs=[asdict(s) for s in specs],
            environments=["day", "dusk_haze"],
            screening="Every canonical capture; all candidates retained regardless of scores",
            selection_used_test_data=False,
            implementation_sha256={
                p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("models.py"))
            },
            energy_profile=(
                "New study only: declared active CPU power times development p95 latency; "
                "not measured energy"
            ),
            upstream=[
                "https://docs.ultralytics.com/modes/predict/",
                "https://github.com/dronefreak/CABiNet",
            ],
            caveat=(
                "Tiled/gamma configurations share pretrained weights; "
                "not separately trained aerial specialists"
            ),
        ),
    )


def infer(output, environment):
    import torch

    from missionbench.v2.scenario import ExecutorConfigSpec

    p = read(output / "protocol.json")
    for name, expected in p["implementation_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("Screen implementation changed")
    source, prior = Path(p["source"]), Path(p["prior"])
    scenario, available, hashes = scenario_and_cache(prior, source, environment, "sector_rescue")
    specs = tuple(ExecutorConfigSpec(**s) for s in p["specs"])
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    inference = InferencePool([s for s in specs if s.config_id in NEW])
    folder = output / environment
    folder.mkdir(exist_ok=False)
    models = {s.config_id: dict(signature=model_signature(s), predictions={}) for s in specs}
    backend = ReplayBackend(scenario, source / environment / "recording")
    rows = []
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            frame = backend.capture()
            key = rgb_key(frame.rgb)
            masks, times, calls = inference.infer(frame.rgb)
            for spec in specs:
                config = spec.config_id
                if config not in NEW:
                    directory, model = available[config]
                    mask = VerifiedMasks(directory, model)[key]
                    masks[config] = mask
                    times[config] = None
                    calls[config] = 0
                path = folder / f"{index:04d}_{config}.npz"
                np.savez_compressed(path, prediction_mask=masks[config])
                models[config]["predictions"][key] = dict(
                    file=path.name,
                    sha256=sha(path),
                    mask_sha256=hashlib.sha256(masks[config].tobytes()).hexdigest(),
                )
                rows.append(
                    dict(
                        capture_id=index,
                        capture_time_s=(timestamp - scenario.schedule.epoch_ns) / 1e9,
                        config_id=config,
                        measured_wall_clock_s=times[config],
                        actual_neural_calls=calls[config],
                        origin="new_actual_inference" if config in NEW else "verified_prior_cache",
                    )
                )
            if index % 10 == 0:
                save(
                    folder / "progress.json", dict(captures=index + 1, expected=len(backend.frames))
                )
                print(f"{environment} {index + 1}/{len(backend.frames)} captures", flush=True)
    finally:
        backend.close()
    save(folder / "inference.json", rows)
    save(
        folder / "manifest.json",
        dict(
            cache_schema_version="0.1",
            models=models,
            recording_identity=scenario.recording_identity,
            recording_sha256=sha(source / environment / "recording/recording.json"),
            capture_count=len(backend.frames),
            source_manifests_sha256=hashes,
            inference_sha256=sha(folder / "inference.json"),
            protocol_sha256=sha(output / "protocol.json"),
        ),
    )
    save(
        folder / "completion.json",
        dict(
            complete=True,
            manifest_sha256=sha(folder / "manifest.json"),
            actual_neural_calls=sum(r["actual_neural_calls"] for r in rows),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("initialize", "infer"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment", choices=("day", "dusk_haze"))
    parser.add_argument("--prior", type=Path, default=Path("results/lake_fast_perception_20261017"))
    parser.add_argument("--source", type=Path, default=Path("results/lake_development_v2_20261016"))
    parser.add_argument("--weights", type=Path, default=Path(".cache/lake_yolo_weights"))
    args = parser.parse_args()
    if args.mode == "initialize":
        initialize(args.prior, args.source, args.weights, args.output)
    else:
        infer(args.output, args.environment)
