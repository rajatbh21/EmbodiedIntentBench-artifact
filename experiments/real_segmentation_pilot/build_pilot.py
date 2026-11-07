"""Assemble a pilot's sources into a bundle manifest and hand it to the existing builder.

This is the seam between the pilot and the benchmark: it writes the ground-truth source (from
the dataset adapter) and a bundle manifest that references the per-configuration prediction and
measurement files produced by :mod:`run_inference`, then calls the *existing*
``build_empirical_bundle`` / ``validate_empirical_bundle``. It reconstructs none of the
builder's output itself, so the pilot bundle is exactly the standard empirical bundle the CLI
already runs.

``run_pilot`` is the end-to-end convenience: run every configuration, write ground truth and
the manifest, build, and validate -- returning a summary. Nothing here is model- or
dataset-specific; it is handed an adapter and models from outside.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from missionbench.tools.build_empirical_bundle import BundleSummary, build_empirical_bundle
from experiments.real_segmentation_pilot.convert import write_ground_truth_source
from experiments.real_segmentation_pilot.interfaces import DatasetAdapter, SegmentationModel
from experiments.real_segmentation_pilot.run_inference import ConfigRunResult, run_config

__all__ = ["ConfigSpec", "PilotResult", "run_pilot", "write_pilot_manifest"]


@dataclass(frozen=True, slots=True)
class ConfigSpec:
    """The bundle-level description of one configuration, alongside its executable model."""

    config_id: str
    model_id: str
    placement: str = "local"
    precision: str = "fp16"
    quality_tier: str = "medium"
    prediction_provenance: str = "model_generated"
    measurement_provenance: str = "estimated"


@dataclass(frozen=True, slots=True)
class PilotResult:
    """Everything a pilot build produced."""

    bundle: BundleSummary
    per_config: tuple[ConfigRunResult, ...]
    ground_truth_instances: int
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle": self.bundle.to_dict(),
            "per_config": [result.to_dict() for result in self.per_config],
            "ground_truth_instances": self.ground_truth_instances,
            "manifest_path": str(self.manifest_path),
        }


def write_pilot_manifest(
    manifest_path: Path,
    *,
    adapter: DatasetAdapter,
    configs: Sequence[ConfigSpec],
    bundle_id: str,
    task_id: str = "HUMAN_SEARCH_SEGMENTATION",
    data_origin: str,
    coverage: str = "strict",
    ground_truth_source: str = "ground_truth.json",
    contract: dict | None = None,
    mission: dict | None = None,
) -> Path:
    """Write the bundle-generation manifest referencing the pilot's generated source files."""
    frame_ids = sorted({frame.frame_id for frame in adapter.frames()})
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "bundle_id": bundle_id,
        "frame_stream_id": adapter.frame_stream_id,
        "task_id": task_id,
        "image_width": adapter.image_width,
        "image_height": adapter.image_height,
        "data_origin": data_origin,
        "coverage": coverage,
        "frames": [{"frame_id": fid, "timestamp_s": float(fid)} for fid in frame_ids],
        "ground_truth_source": ground_truth_source,
        "configurations": [
            {
                "config_id": spec.config_id,
                "model_id": spec.model_id,
                "placement": spec.placement,
                "precision": spec.precision,
                "quality_tier": spec.quality_tier,
                "prediction_source": f"predictions_{spec.config_id}.jsonl",
                "measurement_source": f"measurements_{spec.config_id}.csv",
                "prediction_provenance": spec.prediction_provenance,
                "measurement_provenance": spec.measurement_provenance,
            }
            for spec in configs
        ],
    }
    if contract is not None:
        payload["contract"] = contract
    if mission is not None:
        payload["mission"] = mission

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def run_pilot(
    adapter: DatasetAdapter,
    models: Sequence[tuple[SegmentationModel, ConfigSpec]],
    *,
    sources_dir: Path,
    bundle_dir: Path,
    bundle_id: str,
    average_power_w: float,
    energy_source: str,
    data_origin: str,
    contract: dict | None = None,
    mission: dict | None = None,
    created_at: str | None = None,
    clock: Callable[[], float] | None = None,
) -> PilotResult:
    """Run every configuration, write sources and the manifest, and build+validate the bundle."""
    import time

    resolved_clock = clock or time.perf_counter

    per_config = tuple(
        run_config(
            adapter,
            model,
            output_dir=sources_dir,
            average_power_w=average_power_w,
            energy_source=energy_source,
            clock=resolved_clock,
        )
        for model, _spec in models
    )

    gt_count = write_ground_truth_source(
        sources_dir / "ground_truth.json",
        frame_stream_id=adapter.frame_stream_id,
        instances=adapter.ground_truth(),
    )
    manifest_path = write_pilot_manifest(
        sources_dir / "manifest.json",
        adapter=adapter,
        configs=[spec for _model, spec in models],
        bundle_id=bundle_id,
        data_origin=data_origin,
        contract=contract,
        mission=mission,
    )
    bundle = build_empirical_bundle(manifest_path, bundle_dir, created_at=created_at, validate=True)
    return PilotResult(
        bundle=bundle,
        per_config=per_config,
        ground_truth_instances=gt_count,
        manifest_path=manifest_path,
    )
