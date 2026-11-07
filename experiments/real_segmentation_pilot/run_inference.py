"""Run one model-strategy configuration over a dataset, timing each frame.

    python -m experiments.real_segmentation_pilot.run_inference \\
      --adapter my_dataset:build_adapter --model my_models:build_light \\
      --average-power-w 15 --energy-source "assumed 15 W board TDP, vendor datasheet" \\
      --output data/generated/real_segmentation_pilot/sources

For every frame it records the prediction masks (resized onto the ground-truth grid),
category, confidence, a stable prediction id, the measured per-frame latency, and an energy
value with explicit provenance. It writes the three empirical-bundle *source* files for this
configuration; :mod:`build_pilot` then hands them to the existing bundle builder.

It executes whatever ``SegmentationModel`` it is given. In this repository that is a stub in
tests; a real run supplies a concrete model and dataset from outside the benchmark core. The
CLI resolves both by dotted ``module:callable`` path, so no dataset- or model-specific code
enters the tooling.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments.real_segmentation_pilot.convert import (
    stable_prediction_id,
    write_measurements_csv,
    write_predictions_jsonl,
)
from experiments.real_segmentation_pilot.interfaces import DatasetAdapter, SegmentationModel
from experiments.real_segmentation_pilot.masks import resize_nearest
from experiments.real_segmentation_pilot.measure import (
    EnergyEstimate,
    estimate_energy,
    latency_stats,
)

__all__ = ["ConfigRunResult", "main", "run_config"]


@dataclass(frozen=True, slots=True)
class ConfigRunResult:
    """The outcome of running one configuration: where its sources went, and what they cost."""

    config_id: str
    model_id: str
    frame_count: int
    prediction_count: int
    failed_frames: int
    latency: dict[str, Any]
    energy_provenance: str
    predictions_path: Path
    measurements_path: Path
    energy_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "model_id": self.model_id,
            "frame_count": self.frame_count,
            "prediction_count": self.prediction_count,
            "failed_frames": self.failed_frames,
            "latency": self.latency,
            "energy_provenance": self.energy_provenance,
            "predictions_path": str(self.predictions_path),
            "measurements_path": str(self.measurements_path),
            "energy_path": str(self.energy_path),
        }


def run_config(
    adapter: DatasetAdapter,
    model: SegmentationModel,
    *,
    output_dir: Path,
    average_power_w: float,
    energy_source: str,
    warmup: int = 2,
    clock: Callable[[], float] = time.perf_counter,
    energy_for: Callable[[int, float], EnergyEstimate] | None = None,
) -> ConfigRunResult:
    """Run ``model`` over ``adapter``'s frames and write this configuration's source files.

    Args:
        average_power_w, energy_source: assumed average power and a description of where it
            came from. Used to *estimate* energy as power x measured time, labelled
            ``estimated`` in provenance -- never presented as measured.
        energy_for: override to supply measured or externally sourced energy per frame instead
            of the estimate.
        warmup: frames run and discarded before timing begins, to reach steady state.
        clock: monotonic clock, injected for deterministic tests and for GPU synchronisation.
    """
    frames = list(adapter.frames())
    height, width = adapter.image_height, adapter.image_width

    for frame in frames[: max(0, warmup)]:
        # A warm-up failure must not abort measurement; the real run below records it.
        with contextlib.suppress(Exception):
            model.predict(frame)

    predictions: list[dict] = []
    measurements: list[dict] = []
    energy_log: list[dict] = []
    failed = 0

    for frame in frames:
        start = clock()
        try:
            instances = list(model.predict(frame))
            latency_s = clock() - start
            success, failure_reason = True, None
        except Exception as error:
            latency_s = clock() - start
            instances, success, failure_reason = [], False, _failure_reason(error)
            failed += 1

        energy = (
            energy_for(frame.frame_id, latency_s)
            if energy_for is not None
            else estimate_energy(latency_s, average_power_w=average_power_w, source=energy_source)
        )
        energy_log.append({"frame_id": frame.frame_id, **energy.to_dict()})

        measurements.append(
            {
                "frame_id": frame.frame_id,
                "success": "true" if success else "false",
                "latency_s": round(latency_s, 9),
                "compute_energy_j": round(energy.joules, 9),
                "upload_mb": 0.0,
                "download_mb": 0.0,
                "failure_reason": failure_reason or "",
            }
        )
        for index, instance in enumerate(instances):
            mask = resize_nearest(instance.mask, height, width)
            predictions.append(
                {
                    "frame_id": frame.frame_id,
                    "prediction_id": stable_prediction_id(model.config_id, frame.frame_id, index),
                    "category": instance.category,
                    "confidence": instance.confidence,
                    "mask": mask.to_dict(),
                }
            )

    predictions_path = output_dir / f"predictions_{model.config_id}.jsonl"
    measurements_path = output_dir / f"measurements_{model.config_id}.csv"
    energy_path = output_dir / f"energy_{model.config_id}.json"
    write_predictions_jsonl(predictions_path, predictions)
    write_measurements_csv(measurements_path, measurements)
    energy_path.parent.mkdir(parents=True, exist_ok=True)
    energy_path.write_text(
        json.dumps(
            {
                "config_id": model.config_id,
                "note": "energy per frame; provenance is per the estimate/telemetry used",
                "per_frame": energy_log,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    provenance = energy_log[0]["provenance"] if energy_log else "missing"
    return ConfigRunResult(
        config_id=model.config_id,
        model_id=model.model_id,
        frame_count=len(frames),
        prediction_count=len(predictions),
        failed_frames=failed,
        latency=latency_stats([m["latency_s"] for m in measurements]).to_dict(),
        energy_provenance=provenance,
        predictions_path=predictions_path,
        measurements_path=measurements_path,
        energy_path=energy_path,
    )


def _failure_reason(error: Exception) -> str:
    # Map an inference failure onto the replay schema's vocabulary. A real backend can be more
    # specific; the honest default is that no prediction was produced for this frame.
    del error
    return "no_prediction_available"


def _load_callable(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise SystemExit(f"expected 'module:callable', got {path!r}")
    module_name, attribute = path.split(":", 1)
    return getattr(importlib.import_module(module_name), attribute)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="experiments.real_segmentation_pilot.run_inference",
        description="Run one configuration over a dataset and write its bundle source files.",
    )
    parser.add_argument(
        "--adapter", required=True, help="module:callable building a DatasetAdapter"
    )
    parser.add_argument(
        "--model", required=True, help="module:callable building a SegmentationModel"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for the source files."
    )
    parser.add_argument(
        "--average-power-w",
        type=float,
        required=True,
        help="Assumed average power for the energy estimate.",
    )
    parser.add_argument(
        "--energy-source",
        required=True,
        help="Where the assumed power came from (recorded in provenance).",
    )
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args(argv)

    adapter = _load_callable(args.adapter)()
    model = _load_callable(args.model)()
    result = run_config(
        adapter,
        model,
        output_dir=args.output,
        average_power_w=args.average_power_w,
        energy_source=args.energy_source,
        warmup=args.warmup,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
