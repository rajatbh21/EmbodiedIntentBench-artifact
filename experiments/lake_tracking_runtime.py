"""Declare map-assisted tracking and its resource costs before physical replay."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import evaluate, immutable
from experiments.lake_intent_runner import IntentRouter, IntentRouterInput, bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import HeightField, SurfaceProjector, TrackingPipeline
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.v4_benchmark.calibrated_router import bind_router
from experiments.v4_benchmark.routers import RouterInput
from experiments.v4_benchmark.scenario import digest


@dataclass(frozen=True)
class TrackingCost:
    latency_s: float = 0.1
    energy_j: float = 1.0
    memory_mb: float = 64.0

    def __post_init__(self):
        for value in (self.latency_s, self.energy_j, self.memory_mb):
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("Tracking costs must be finite nonnegative numbers")


DEFAULT_TRACKING_COST = TrackingCost()
ZERO_TRACKING_COST = TrackingCost(0, 0, 0)


@dataclass(frozen=True)
class TrackingRouterInput(IntentRouterInput):
    tracking: Mapping[str, Any] | None = None


class TrackingIntentRouter(IntentRouter):
    """Make shared tracking assumptions and costs available before each decision."""

    def __init__(self, delegate, contract, tracking):
        super().__init__(delegate, contract)
        self.tracking = immutable(tracking)

    def select(self, observation):
        self.calls += 1
        return self.delegate.select(
            TrackingRouterInput(
                **{field.name: getattr(observation, field.name) for field in fields(RouterInput)},
                intent_contract=self.public,
                public_contract_sha256=self.public_sha256,
                tracking=self.tracking,
            )
        )


def load_public_terrain(assets: Path):
    """Read only the textured scenery's elevation raster and public SDF transform.

    The visual DEM is declared as a known-map prior. Object placements, person
    meshes, labels and collision metadata are not used to locate detections.
    """
    folder = assets / "harmonic_terrain"
    path = folder / "model.sdf"
    root = ET.parse(path).getroot()
    for parent in (root.find("model"), root.find("model/link"), root.find("model/link/visual")):
        if any(float(v) for v in parent.findtext("pose", "0 0 0 0 0 0").split()):
            raise ValueError("Unsupported terrain parent transform")
    geometry = root.find("model/link/visual/geometry/heightmap")
    width, height, scale = map(float, geometry.findtext("size").split())
    x, y, z = map(float, geometry.findtext("pos", "0 0 0").split())
    raster = (folder / geometry.findtext("uri")).resolve()
    if not raster.is_relative_to(folder.resolve()):
        raise ValueError("Terrain raster outside the scenery directory")
    with Image.open(raster) as image:
        raw = np.asarray(image, dtype=float)
    if raw.ndim != 2 or raw.max() <= 0 or min(width, height, scale) <= 0:
        raise ValueError("Invalid terrain raster or dimensions")
    elevation = raw / raw.max() * scale + z
    bounds = dict(
        west=x - width / 2, east=x + width / 2, south=y - height / 2, north=y + height / 2
    )
    field = HeightField(elevation, **bounds)
    metadata = dict(
        schema_version="public_visual_dem_v1",
        model_sdf_sha256=sha(path),
        raster_sha256=sha(raster),
        raster_path=str(raster),
        bounds=bounds,
        vertical_scale_m=scale,
        vertical_origin_m=z,
        normalizer=float(raw.max()),
        origin="Downloaded OpenRobotics Harmonic Terrain visual heightmap",
        assumption="Perfect pre-existing map and ideal odometry; static ground-level targets",
        collision_geometry_used=False,
        object_placements_used=False,
    )
    return field, metadata


def run_tracking_mission(
    scenario,
    backend,
    router,
    executors,
    contract,
    output,
    *,
    config,
    terrain,
    map_metadata,
    router_name,
    gt_aware,
    cost=DEFAULT_TRACKING_COST,
):
    from experiments.lake_tracking_runner import FROZEN_RUNNER_SHA256, run_mission

    if type(gt_aware) is not bool:
        raise ValueError("Explicit GT provenance required")
    original = Path(__file__).parent / "v4_benchmark/runner.py"
    if sha(original) != FROZEN_RUNNER_SHA256:
        raise ValueError("Frozen runner changed; review the versioned tracking fork")
    bound = bind_scenario(scenario, contract)
    bind_router(router, bound)
    projector = SurfaceProjector(terrain if config.association == "terrain_world" else None)
    if config.association == "terrain_world" and terrain is None:
        raise ValueError("Terrain association requires a declared map")
    tracker = TrackingPipeline(config, projector)
    tracker.provenance.update(
        map=map_metadata if config.association == "terrain_world" else None,
        anchor_height_m=projector.anchor_height_m,
        hfov_rad=projector.hfov,
        static_target_assumption=True,
    )
    public_tracking = dict(configuration=tracker.provenance, cost=asdict(cost))
    adapter = TrackingIntentRouter(router, contract, public_tracking)
    scorer = RunnerReportEvaluator(
        bound.matching_iou_threshold, bound.targets.values(), config.min_hits
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "intent.json", contract.document())
    save(output / "public_intent.json", contract.public_document())
    protocol = dict(
        schema_version="tracking_execution_v1",
        router=router_name,
        gt_aware=gt_aware,
        deployable_policy=not gt_aware,
        primary_outcome="outcome.json",
        contract_sha256=contract.sha256,
        public_contract_sha256=contract.public_sha256,
        scenario_sha256=digest(bound.source),
        recording_identity=bound.recording_identity,
        tracking=tracker.provenance,
        public_tracking=public_tracking,
        tracking_cost=asdict(cost),
        report_scoring="consistent_report_support_v1",
        frozen_runner_sha256=FROZEN_RUNNER_SHA256,
        implementation_sha256={
            name: sha(Path(__file__).with_name(name))
            for name in (
                "lake_tracking.py",
                "lake_tracking_evaluation.py",
                "lake_tracking_runner.py",
                "lake_tracking_runtime.py",
                "lake_intent_contract.py",
                "lake_intent_runner.py",
            )
        },
        caveat="Configured tracking costs; perfect visual DEM prior; no measured deployment claim",
    )
    save(output / "protocol.json", protocol)
    result = run_mission(
        bound,
        backend,
        adapter,
        executors,
        router_name=router_name,
        output=output / "execution",
        tracker=tracker,
        report_scoring=scorer,
        tracking_cost=cost,
    )
    if (
        result["scenario_sha256"] != protocol["scenario_sha256"]
        or result["backend"]["recording_identity"] != bound.recording_identity
        or result["tracking"] != protocol["tracking"]
        or result["tracking_cost"] != protocol["tracking_cost"]
    ):
        raise ValueError("Execution binding mismatch")
    events = read(output / "execution/events.json")
    outcome = evaluate(contract, events, result)
    outcome.update(
        schema_version="tracked_intent_result_v1",
        intent_schema_version=outcome["schema_version"],
        report_scoring="consistent_report_support_v1",
        gt_aware=gt_aware,
        deployable_policy=not gt_aware,
        public_contract_deliveries=adapter.calls,
        protocol_sha256=sha(output / "protocol.json"),
        events_sha256=sha(output / "execution/events.json"),
        execution_sha256=sha(output / "execution/results.json"),
    )
    save(output / "outcome.json", outcome)
    return outcome
