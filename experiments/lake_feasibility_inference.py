"""Complete a lake frame/configuration table without changing the source study.

Verified earlier real predictions are copied; only missing RGB/config pairs run
neural inference. The output uses the existing strict frozen-cache interface.
No ground truth enters this inference table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from missionbench.v2.executors import build_executors
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import validate_result


def read(path):
    return json.loads(path.read_text())


def complete(source: Path, output: Path, environment: str, threads: int = 4):
    import torch

    freeze(source)
    torch.set_num_threads(threads)
    suite = read(source / "suite.json")
    scenario = load(
        source / "scenario.json", suite["environments"][environment], Path(suite["assets"])
    )
    recording = source / environment / "recording"
    previous = source / environment / "on_demand_predictions"
    prior = read(previous / "manifest.json")
    assert prior["recording_identity"] == scenario.recording_identity
    assert prior["recording_sha256"] == sha(recording / "recording.json")
    assert set(prior["models"]) == {s.config_id for s in scenario.executors}
    for spec in scenario.executors:
        assert prior["models"][spec.config_id]["signature"] == model_signature(spec)
    folder = output / environment / "predictions"
    folder.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        schema_version="1.0",
        source_study=str(source.resolve()),
        source_lock_sha256=sha(source / "lock.json"),
        source_manifest_sha256=sha(previous / "manifest.json"),
        script_sha256=sha(Path(__file__)),
        environment=environment,
        host_threads=threads,
        purpose="development feasibility; no held-out or measured-drone-performance claim",
    )
    save(folder / "source.json", metadata)
    actual = build_executors(scenario.executors)
    models = {
        s.config_id: dict(signature=model_signature(s), predictions={}) for s in scenario.executors
    }
    rows = []
    new_calls = copied = duplicate = 0
    backend = ReplayBackend(scenario, recording)
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            frame = backend.capture()
            key = rgb_key(frame.rgb)
            for spec in scenario.executors:
                config = spec.config_id
                if key in models[config]["predictions"]:
                    item = models[config]["predictions"][key]
                    origin = "reuse_identical_rgb_within_dense_table"
                    duplicate += 1
                else:
                    path = folder / f"{config}_{key}.npz"
                    old = prior["models"][config]["predictions"].get(key)
                    if old is not None:
                        prior_path = (previous / old["file"]).resolve()
                        assert prior_path.parent == previous.resolve()
                        assert sha(prior_path) == old["sha256"]
                        execution = old["execution"]
                        assert execution["success"] and execution["config_id"] == config
                        with np.load(prior_path, allow_pickle=False) as data:
                            mask = data["prediction_mask"].copy()
                        shutil.copyfile(prior_path, path)
                        origin = "copied_verified_prior_actual_prediction"
                        copied += 1
                    else:
                        result = actual[config].run(frame.rgb.copy())
                        validate_result(result, config, frame.rgb.shape[:2])
                        if not result.success:
                            raise RuntimeError(f"Actual inference failed: {config}")
                        mask = result.prediction_mask
                        execution = result.to_summary()
                        np.savez_compressed(path, prediction_mask=mask)
                        origin = "new_actual_neural_inference"
                        new_calls += 1
                    assert mask.dtype == np.bool_ and mask.shape == frame.rgb.shape[:2]
                    item = dict(
                        file=path.name,
                        sha256=sha(path),
                        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                        execution=execution,
                        origin=origin,
                    )
                    models[config]["predictions"][key] = item
                rows.append(
                    dict(
                        capture_id=index,
                        timestamp_ns=timestamp,
                        rgb_sha256=key,
                        config_id=config,
                        origin=origin,
                        prediction_sha256=item["sha256"],
                    )
                )
            save(
                folder / "progress.json",
                dict(
                    captures=index + 1,
                    rows=len(rows),
                    new_actual_calls=new_calls,
                    copied_prior_predictions=copied,
                    reused_identical_rgb=duplicate,
                ),
            )
            print(
                f"{environment}: {index + 1}/{len(backend.frames)} frames; "
                f"new calls={new_calls}, copied={copied}",
                flush=True,
            )
    finally:
        backend.close()
    expected = {(i, s.config_id) for i in range(len(backend.frames)) for s in scenario.executors}
    assert (
        len(rows) == len(expected) and {(r["capture_id"], r["config_id"]) for r in rows} == expected
    )
    save(folder / "inference.json", rows)
    save(
        folder / "manifest.json",
        dict(
            cache_schema_version="0.1",
            execution_mode="precomputed_real_model_predictions",
            recording_identity=scenario.recording_identity,
            recording_sha256=sha(recording / "recording.json"),
            capture_count=len(backend.frames),
            models=models,
            source_inference_sha256=sha(folder / "inference.json"),
            source_metadata_sha256=sha(folder / "source.json"),
        ),
    )
    save(
        folder / "completion.json",
        dict(
            **read(folder / "progress.json"),
            complete=True,
            manifest_sha256=sha(folder / "manifest.json"),
            note="Host timing is diagnostic; configured mission costs are unchanged.",
        ),
    )
    print(f"{environment}: complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--environment", choices=("day", "dusk_haze"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    complete(args.source, args.output, args.environment, args.threads)
