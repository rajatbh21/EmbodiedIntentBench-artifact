"""Freeze RGB-only predictions from actual dense probes for deterministic mission replay.

Only stateless local mask executors are supported. GT verifies the imported probe
once at freeze time; it is never stored in the runtime executor. Replay latency
and energy come from the mission configuration, not cache lookup wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import ImageExecutionResult
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.v4_benchmark.scenario import digest, load_scenario


def rgb_key(rgb):
    return hashlib.sha256(
        str(rgb.shape).encode() + str(rgb.dtype).encode() + rgb.tobytes()
    ).hexdigest()


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(scenario_path: Path, recording: Path, probe: Path, output: Path):
    scenario = load_scenario(scenario_path)
    report = json.loads((probe / "probe.json").read_text())
    rows = json.loads((probe / "observations.json").read_text())
    if report["scenario_sha256"] != digest(scenario.source):
        raise ValueError("probe scenario mismatch")
    backend = ReplayBackend(scenario, recording)
    try:
        if report["backend"]["recording_sha256"] != backend.provenance["recording_sha256"]:
            raise ValueError("probe recording mismatch")
        timestamps = list(
            range(
                scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns
            )
        )
        if [
            (c["observation_id"], c["timestamp_ns"]) for c in report["visibility"]["captures"]
        ] != list(enumerate(timestamps)):
            raise ValueError("replay requires a complete dense probe")
        expected = {(i, s.config_id) for i in range(len(timestamps)) for s in scenario.executors}
        observed = [(r["observation_id"], r["execution"]["config_id"]) for r in rows]
        if len(set(observed)) != len(observed) or set(observed) != expected:
            raise ValueError("duplicate or missing probe predictions")
        lookup = {key: (j, row) for j, (key, row) in enumerate(zip(observed, rows, strict=True))}
        models = {}
        for spec in scenario.executors:
            if spec.kind not in (
                "torch_semantic_segmentation",
                "torch_instance_segmentation",
                "torch_cabinet_segmentation",
            ):
                raise ValueError("only stateless local neural mask models may be cached")
            models[spec.config_id] = dict(signature=model_signature(spec), predictions={})
        evaluators = {
            s.config_id: MissionEvaluator(
                scenario.matching_iou_threshold,
                len(scenario.targets),
                scenario.observation_interval_s,
            )
            for s in scenario.executors
        }
        output.mkdir(parents=True, exist_ok=False)
        for i, timestamp in enumerate(timestamps):
            backend.advance_to(timestamp)
            frame = backend.capture()
            observation = evaluation_frame(
                frame, i, (timestamp - scenario.schedule.epoch_ns) / 1e9, scenario
            )
            key = rgb_key(frame.rgb)
            for spec in scenario.executors:
                j, row = lookup[(i, spec.config_id)]
                if not row["execution"]["success"]:
                    raise ValueError("cannot cache failed inference")
                source = probe / f"prediction_{i:05d}_{j + 1:05d}.npz"
                if "prediction_sha256" in row and (
                    row.get("prediction_file") != source.name
                    or row["prediction_sha256"] != checksum(source)
                ):
                    raise ValueError("probe prediction checksum mismatch")
                with np.load(source, allow_pickle=False) as data:
                    mask = data["prediction_mask"]
                if mask.dtype != np.bool_ or mask.shape != frame.rgb.shape[:2]:
                    raise ValueError("invalid cached mask")
                score = evaluators[spec.config_id].update(observation, mask).to_dict()
                if score != row["score"]:
                    raise ValueError("cached mask differs from recorded evaluator score")
                name = f"prediction_{j:06d}.npz"
                target = output / name
                np.savez_compressed(target, prediction_mask=mask)
                item = dict(
                    file=name,
                    sha256=checksum(target),
                    mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                )
                table = models[spec.config_id]["predictions"]
                if key in table and table[key]["mask_sha256"] != item["mask_sha256"]:
                    raise ValueError("same RGB produced different masks; stateless replay invalid")
                table[key] = item
        manifest = dict(
            cache_schema_version="0.1",
            execution_mode="precomputed_real_model_predictions",
            recording_identity=scenario.recording_identity,
            recording_sha256=backend.provenance["recording_sha256"],
            source_probe_sha256=checksum(probe / "probe.json"),
            source_rows_sha256=checksum(probe / "observations.json"),
            capture_count=len(timestamps),
            import_integrity=(
                "capture_time_hashes_verified"
                if all("prediction_sha256" in row for row in rows)
                else "legacy_predictions_revalidated_and_hashed_at_freeze"
            ),
            models=models,
        )
        save_json(output / "manifest.json", manifest)
        return manifest
    finally:
        backend.close()


class CachedExecutor:
    def __init__(self, spec, predictions, manifest_sha):
        self.spec = spec
        self.predictions = predictions
        self.manifest_sha = manifest_sha

    def run(self, rgb):
        started = time.perf_counter()
        key = rgb_key(rgb)
        if key not in self.predictions:
            raise ValueError("RGB absent from frozen prediction cache")
        mask = self.predictions[key].copy()
        return ImageExecutionResult(
            config_id=self.spec.config_id,
            model_strategy_id=self.spec.model_strategy_id,
            success=True,
            prediction_mask=mask,
            mission_latency_s=self.spec.mission_latency_s,
            measured_wall_clock_s=time.perf_counter() - started,
            energy_j=self.spec.energy_j_per_call,
            communication_mb=self.spec.communication_mb_per_call,
            measurement_provenance={
                "mission_latency_s": "configured frozen inference cost, NOT cache lookup time",
                "measured_wall_clock_s": "cache lookup only; no neural inference executed",
                "energy_j": "configured inference energy, not measured",
                "communication_mb": "configured local cost",
            },
            diagnostics=dict(
                execution_mode="precomputed_real_model_predictions", cache_sha256=self.manifest_sha
            ),
        )


def build_cached(specs, directory: Path, recording_identity: str):
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest["cache_schema_version"] != "0.1"
        or manifest["recording_identity"] != recording_identity
    ):
        raise ValueError("prediction cache recording mismatch")
    if set(manifest["models"]) != {s.config_id for s in specs}:
        raise ValueError("prediction cache pool mismatch")
    result = {}
    for spec in specs:
        model = manifest["models"][spec.config_id]
        if model["signature"] != model_signature(spec):
            raise ValueError("prediction cache model mismatch")
        predictions = {}
        for key, item in model["predictions"].items():
            path = (directory / item["file"]).resolve()
            if path.parent != directory.resolve() or checksum(path) != item["sha256"]:
                raise ValueError("prediction cache checksum/path mismatch")
            with np.load(path, allow_pickle=False) as data:
                mask = data["prediction_mask"].copy()
            if (
                mask.dtype != np.bool_
                or mask.ndim != 2
                or hashlib.sha256(mask.tobytes()).hexdigest() != item["mask_sha256"]
            ):
                raise ValueError("invalid cached mask")
            mask.setflags(write=False)
            predictions[key] = mask
        result[spec.config_id] = CachedExecutor(
            spec, predictions, checksum(directory / "manifest.json")
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("scenario", "recording", "probe", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(freeze(args.scenario, args.recording, args.probe, args.output)["capture_count"])


if __name__ == "__main__":
    main()
