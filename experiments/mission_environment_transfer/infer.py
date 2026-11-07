"""Run every frozen candidate on every new physical capture, without GT input."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from experiments.lake_missions.assets import save, sha
from experiments.mission_environment_transfer.cohort import verify
from experiments.mission_environment_transfer.scenario import load_scenario
from experiments.mission_environment_transfer.sensor import verify_gate
from experiments.mission_routing_v2.models import InferencePool
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key


def run(output, flight):
    import torch

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    p = verify(output)
    verify_gate(output)
    case = p["cases"][flight]
    root = output / flight
    scenario = load_scenario(
        root / "rapid_survey/scenario.json", case["environment"], Path(p["assets"])
    )
    folder = root / "predictions"
    folder.mkdir(exist_ok=False)
    for spec in scenario.executors:
        if (
            "checkpoint_sha256" in spec.parameters
            and sha(Path(spec.parameters["checkpoint"])) != spec.parameters["checkpoint_sha256"]
        ):
            raise ValueError("Pretrained checkpoint changed")
    inference = InferencePool(scenario.executors)
    models = {
        s.config_id: dict(signature=model_signature(s), predictions={}) for s in scenario.executors
    }
    backend = ReplayBackend(scenario, root / "recording")
    rows = []
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            frame = backend.capture()
            masks, times, calls = inference.infer(frame.rgb)
            key = rgb_key(frame.rgb)
            for config, mask in masks.items():
                if mask.dtype != np.bool_ or mask.shape != frame.rgb.shape[:2]:
                    raise ValueError("Invalid native segmentation output")
                path = folder / f"{index:04d}_{config}.npz"
                np.savez_compressed(path, prediction_mask=mask)
                models[config]["predictions"][key] = dict(
                    file=path.name,
                    sha256=sha(path),
                    mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                )
                rows.append(
                    dict(
                        capture_id=index,
                        config_id=config,
                        measured_wall_clock_s=times[config],
                        actual_neural_calls=calls[config],
                    )
                )
            if index % 5 == 0:
                save(
                    folder / "progress.json", dict(captures=index + 1, expected=len(backend.frames))
                )
                print(f"{flight}: {index + 1}/{len(backend.frames)}", flush=True)
    finally:
        backend.close()
    save(folder / "inference.json", rows)
    save(
        folder / "manifest.json",
        dict(
            cache_schema_version="0.1",
            models=models,
            recording_identity=scenario.recording_identity,
            recording_sha256=sha(root / "recording/recording.json"),
            protocol_sha256=sha(output / "protocol.json"),
            inference_sha256=sha(folder / "inference.json"),
            capture_count=len(backend.frames),
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight", required=True)
    args = parser.parse_args()
    run(args.output, args.flight)
