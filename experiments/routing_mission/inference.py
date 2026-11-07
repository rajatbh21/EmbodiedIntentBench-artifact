"""Dense real RGB inference, content-addressed masks, and host timing diagnostics."""

import hashlib
import statistics
from pathlib import Path

import numpy as np

from missionbench.v2.executors import build_executors
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import validate_result


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compute(scenario, recording, output, threads=2):
    import torch

    torch.set_num_threads(threads)
    backend = ReplayBackend(scenario, recording)
    executors = build_executors(scenario.executors)
    output.mkdir(parents=True, exist_ok=False)
    models = {
        s.config_id: dict(signature=model_signature(s), predictions={}) for s in scenario.executors
    }
    rows = []
    visibility = {str(label): dict(frames=0, max_pixels=0) for label in scenario.targets}
    try:
        for i, timestamp in enumerate(
            range(
                scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns
            )
        ):
            backend.advance_to(timestamp)
            frame = backend.capture()
            key = rgb_key(frame.rgb)
            for label in scenario.targets:
                count = int(np.count_nonzero(frame.object_ids == label))
                visibility[str(label)]["frames"] += int(count > 0)
                visibility[str(label)]["max_pixels"] = max(
                    visibility[str(label)]["max_pixels"], count
                )
            for j, spec in enumerate(scenario.executors):
                result = executors[spec.config_id].run(frame.rgb.copy())
                validate_result(result, spec.config_id, frame.rgb.shape[:2])
                if not result.success:
                    raise RuntimeError("real inference failed: " + spec.config_id)
                name = f"prediction_{i:05d}_{j}.npz"
                np.savez_compressed(output / name, prediction_mask=result.prediction_mask)
                item = dict(
                    file=name,
                    sha256=checksum(output / name),
                    mask_sha256=hashlib.sha256(result.prediction_mask.tobytes()).hexdigest(),
                )
                existing = models[spec.config_id]["predictions"].get(key)
                if existing and existing["mask_sha256"] != item["mask_sha256"]:
                    raise ValueError("same RGB has inconsistent stateless predictions")
                models[spec.config_id]["predictions"][key] = item
                rows.append(
                    dict(
                        capture_id=i,
                        config_id=spec.config_id,
                        execution=result.to_summary(),
                        prediction_sha256=item["sha256"],
                    )
                )
            save_json(output / "inference.json", rows)
            print(
                f"inference {output.parent.name}: {i + 1} captures, {len(rows)} actual model calls",
                flush=True,
            )
    finally:
        backend.close()
    manifest = dict(
        cache_schema_version="0.1",
        execution_mode="precomputed_real_model_predictions",
        recording_identity=scenario.recording_identity,
        recording_sha256=checksum(recording / "recording.json"),
        capture_count=i + 1,
        models=models,
        source_inference_sha256=checksum(output / "inference.json"),
    )
    save_json(output / "manifest.json", manifest)
    timing = {
        s.config_id: dict(
            median_s=statistics.median(
                r["execution"]["measured_wall_clock_s"]
                for r in rows
                if r["config_id"] == s.config_id
            ),
            configured_s=s.mission_latency_s,
        )
        for s in scenario.executors
    }
    save_json(
        output / "diagnostics.json",
        dict(
            actual_calls=len(rows),
            visibility=visibility,
            timing=timing,
            latency_provenance="host diagnostics; frozen configured mission costs unchanged",
        ),
    )
    return manifest
