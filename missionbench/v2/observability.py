"""The V2.2 controlled observability experiment: when can the real models see a target?

Not a mission. A small deterministic matrix of single-target compositions -- asset x
projected size x rotation x background region -- each rendered by the normal camera
pipeline and fed to each configured model-strategy. The report answers the questions
that matter before any mission benchmark: does either model produce *any* overlap with
the target, at what projected size does detection vanish, how do the models differ on
false positives, and do rotation or background complexity move the result.

Honesty rules carried through from the scenario layer: every condition records the
asset's ``view_type`` (conventional / aerial / procedural) and the experiment's
``evaluation_purpose``; a conventional or procedural asset can never produce a result
labelled aerial-human perception. Models receive only the composited RGB -- detection
is judged afterwards by the existing evidence rule (component IoU against the instance
GT), plus pixel-level diagnostics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from missionbench.schemas.loading import (
    DocumentReader,
    SchemaValidationError,
    SchemaVersionError,
    read_json_object,
)
from missionbench.v2.assets import AssetManifest, load_asset, load_manifest
from missionbench.v2.camera import CameraRenderer
from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import build_executors
from missionbench.v2.objects import ObjectLayer
from missionbench.v2.scenario import (
    CameraSpec,
    ExecutorConfigSpec,
    ObjectSpec,
    WorldSpec,
    _read_camera,
    _read_executors,
    _read_world,
)
from missionbench.v2.world import open_world

__all__ = ["OBSERVABILITY_SCHEMA_VERSION", "ObservabilityConfig", "load_config", "run_experiment"]

OBSERVABILITY_SCHEMA_VERSION: Final = "1.0"

_TOP_FIELDS: Final = (
    "observability_schema_version",
    "experiment_id",
    "evaluation_purpose",
    "world",
    "camera",
    "assets_manifest",
    "matching_iou_threshold",
    "executor_configs",
    "backgrounds",
    "conditions",
)
_BACKGROUND_FIELDS: Final = ("background_id", "position_m", "description")
_CONDITION_FIELDS: Final = (
    "condition_id",
    "asset_id",
    "background_id",
    "width_m",
    "height_m",
    "rotation_deg",
)


@dataclass(frozen=True, slots=True)
class Background:
    background_id: str
    position_m: tuple[float, float]
    description: str = ""


@dataclass(frozen=True, slots=True)
class Condition:
    condition_id: str
    asset_id: str
    background_id: str
    width_m: float
    height_m: float
    rotation_deg: float


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    experiment_id: str
    evaluation_purpose: str
    world: WorldSpec
    camera: CameraSpec
    assets_manifest: Path
    matching_iou_threshold: float
    executor_configs: tuple[ExecutorConfigSpec, ...]
    backgrounds: dict[str, Background]
    conditions: tuple[Condition, ...]


def load_config(path: Path) -> ObservabilityConfig:
    payload = read_json_object(path)
    context = f"{path.name} -> ObservabilityConfig"
    declared = payload.get("observability_schema_version")
    if declared != OBSERVABILITY_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{context}: observability_schema_version {declared!r} is not supported; this "
            f"release reads {OBSERVABILITY_SCHEMA_VERSION!r} only"
        )
    reader = DocumentReader(payload, context=context, allowed_fields=_TOP_FIELDS)

    backgrounds: dict[str, Background] = {}
    for entry in reader.get_object_list("backgrounds", allowed_fields=_BACKGROUND_FIELDS):
        raw = entry.get_passthrough("position_m")
        if not isinstance(raw, list) or len(raw) != 2:
            raise SchemaValidationError(f"{entry.context}: position_m must be [x_m, y_m]")
        bid = entry.get_str("background_id")
        if bid in backgrounds:
            raise SchemaValidationError(f"{entry.context}: duplicate background_id {bid!r}")
        backgrounds[bid] = Background(
            background_id=bid,
            position_m=(float(raw[0]), float(raw[1])),
            description=entry.get_optional_str("description") or "",
        )

    conditions: list[Condition] = []
    seen: set[str] = set()
    for entry in reader.get_object_list("conditions", allowed_fields=_CONDITION_FIELDS):
        cid = entry.get_str("condition_id")
        if cid in seen:
            raise SchemaValidationError(f"{entry.context}: duplicate condition_id {cid!r}")
        seen.add(cid)
        background_id = entry.get_str("background_id")
        if background_id not in backgrounds:
            raise SchemaValidationError(f"{entry.context}: unknown background_id {background_id!r}")
        conditions.append(
            Condition(
                condition_id=cid,
                asset_id=entry.get_str("asset_id"),
                background_id=background_id,
                width_m=entry.get_float("width_m", exclusive_minimum=0.0),
                height_m=entry.get_float("height_m", exclusive_minimum=0.0),
                rotation_deg=entry.get_float("rotation_deg", minimum=-360.0, maximum=360.0),
            )
        )

    purpose = reader.get_str("evaluation_purpose")
    return ObservabilityConfig(
        experiment_id=reader.get_str("experiment_id"),
        evaluation_purpose=purpose,
        world=_read_world(
            reader.get_object(
                "world",
                allowed_fields=(
                    "image_path",
                    "source_id",
                    "meters_per_pixel",
                    "invalid_pixel_rule",
                    "valid_region_m",
                ),
            )
        ),
        camera=_read_camera(
            reader.get_object(
                "camera",
                allowed_fields=(
                    "projection",
                    "footprint_width_m",
                    "footprint_height_m",
                    "output_width_px",
                    "output_height_px",
                ),
            )
        ),
        assets_manifest=Path(reader.get_str("assets_manifest")),
        matching_iou_threshold=reader.get_float(
            "matching_iou_threshold", exclusive_minimum=0.0, maximum=1.0
        ),
        executor_configs=_read_executors(reader),
        backgrounds=backgrounds,
        conditions=tuple(conditions),
    )


def run_experiment(
    config: ObservabilityConfig,
    output_dir: Path,
    *,
    panels: int = 0,
) -> dict[str, Any]:
    """Run every condition through every model-strategy and write the report JSON.

    Each row is one (condition, model) pair: the target's physical and projected size,
    the model's positive pixels, target intersection / pixel IoU / pixel recall, false
    positives outside the target, whether the *existing evidence rule* counts it as
    detected, and the measured latencies. Row order is deterministic.
    """
    manifest: AssetManifest = load_manifest(config.assets_manifest)
    world = open_world(
        Path(config.world.image_path),
        config.world.meters_per_pixel,
        invalid_pixel_rule=config.world.invalid_pixel_rule,
    )
    executors = build_executors(config.executor_configs)

    rows: list[dict[str, Any]] = []
    panel_budget = panels
    for condition in config.conditions:
        record = manifest.get(condition.asset_id)
        asset = load_asset(record)
        background = config.backgrounds[condition.background_id]

        obj = ObjectSpec(
            object_id=f"OBS_{condition.condition_id}",
            class_id=record.category,
            is_target=True,
            position_m=background.position_m,
            width_m=condition.width_m,
            height_m=condition.height_m,
            rotation_deg=condition.rotation_deg,
            z_order=1,
            render_mode="image_asset",
            asset_id=condition.asset_id,
        )
        layer = ObjectLayer((obj,), assets={condition.asset_id: asset})
        renderer = CameraRenderer(
            scenario_id=config.experiment_id, world=world, objects=layer, camera=config.camera
        )
        observation = renderer.render(
            position_m=background.position_m, capture_time_s=0.0, observation_id=0
        )
        target_mask = observation.semantic_gt == 1
        projection = observation.provenance["asset_projections"].get(obj.object_id, {})

        for config_id in sorted(executors):
            result = executors[config_id].run(observation.rgb)
            prediction = result.prediction_mask
            intersection = int(np.logical_and(prediction, target_mask).sum())
            union = int(np.logical_or(prediction, target_mask).sum())
            target_px = int(target_mask.sum())

            evaluator = MissionEvaluator(
                matching_iou_threshold=config.matching_iou_threshold,
                total_targets=1,
                observation_interval_s=1.0,
            )
            score = evaluator.update(observation, prediction)

            diagnostics = result.diagnostics or {}
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "asset_id": condition.asset_id,
                    "view_type": record.view_type,
                    "background_id": condition.background_id,
                    "target_width_m": condition.width_m,
                    "target_height_m": condition.height_m,
                    "rotation_deg": condition.rotation_deg,
                    "projected_width_px": projection.get("projected_width_px"),
                    "projected_height_px": projection.get("projected_height_px"),
                    "target_visible_px": target_px,
                    "model_strategy": config_id,
                    "model_id": diagnostics.get("model_id"),
                    "probability_threshold": diagnostics.get("probability_threshold"),
                    "prediction_positive_px": int(prediction.sum()),
                    "target_intersection_px": intersection,
                    "target_pixel_iou": (intersection / union) if union else 0.0,
                    "target_pixel_recall": (intersection / target_px) if target_px else 0.0,
                    "false_positive_px": int(prediction.sum()) - intersection,
                    "detected_by_evidence_rule": score.matched_components > 0,
                    "model_forward_latency_s": diagnostics.get("model_forward_latency_s"),
                    "end_to_end_latency_s": diagnostics.get("end_to_end_executor_latency_s"),
                }
            )
            if panel_budget > 0:
                from missionbench.v2.visualize import render_observation_panel

                render_observation_panel(
                    observation,
                    prediction,
                    output_dir / f"{condition.condition_id}_{config_id}.png",
                    annotation={"config": config_id, "asset": condition.asset_id},
                )
                panel_budget -= 1

    report = {
        "experiment_id": config.experiment_id,
        "evaluation_purpose": config.evaluation_purpose,
        "honesty": {
            "view_types_present": sorted({row["view_type"] for row in rows}),
            "note": (
                "controlled observability / integration diagnostic; results are NOT real "
                "aerial-human perception performance. 'procedural' assets are generated "
                "test silhouettes and never evidence of model quality."
            ),
        },
        "matching_iou_threshold": config.matching_iou_threshold,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{config.experiment_id}_observability.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
