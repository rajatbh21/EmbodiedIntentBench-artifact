"""Compute a stateless model/RGB pair once, on demand, with an auditable cache.

Only requested predictions are evaluated; unrequested model/frame combinations
are not invented. Configured mission costs are identical on cache hits and misses.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from missionbench.v2.executors import build_executors
from experiments.lake_missions.assets import save, sha
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import CachedExecutor, rgb_key
from experiments.v4_benchmark.runner import validate_result


class OnDemandExecutor:
    def __init__(self, spec, executor, folder, manifest):
        self.spec, self.executor, self.folder, self.manifest = spec, executor, folder, manifest
        self.predictions = {}

    def run(self, rgb):
        key = rgb_key(rgb)
        if key in self.predictions:
            result = CachedExecutor(
                self.spec, self.predictions, sha(self.folder / "manifest.json")
            ).run(rgb)
            return replace(
                result,
                diagnostics=dict(
                    execution_mode="reuse_actual_rgb_prediction",
                    rgb_sha256=key,
                    source_prediction=self.manifest["models"][self.spec.config_id]["predictions"][
                        key
                    ],
                ),
            )
        result = self.executor.run(rgb.copy())
        validate_result(result, self.spec.config_id, rgb.shape[:2])
        if not result.success:
            raise RuntimeError(f"Actual inference failed: {self.spec.config_id}")
        path = self.folder / f"{self.spec.config_id}_{key}.npz"
        np.savez_compressed(path, prediction_mask=result.prediction_mask)
        mask = result.prediction_mask.copy()
        mask.setflags(write=False)
        self.predictions[key] = mask
        self.manifest["models"][self.spec.config_id]["predictions"][key] = dict(
            file=path.name,
            sha256=sha(path),
            execution=result.to_summary(),
        )
        self.manifest["actual_model_calls"] += 1
        save(self.folder / "manifest.json", self.manifest)
        return result


def build(scenario, folder, recording):
    import torch

    torch.set_num_threads(4)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    actual = build_executors(scenario.executors)
    manifest = dict(
        schema_version="1.0",
        mode="on_demand_stateless_real_neural_inference",
        recording_identity=scenario.recording_identity,
        recording_sha256=sha(Path(recording) / "recording.json"),
        actual_model_calls=0,
        host_threads=4,
        models={
            s.config_id: dict(signature=model_signature(s), predictions={})
            for s in scenario.executors
        },
    )
    save(folder / "manifest.json", manifest)
    return {
        s.config_id: OnDemandExecutor(s, actual[s.config_id], folder, manifest)
        for s in scenario.executors
    }
