"""Bind independent-map sources and the public floor prior to physical identity."""

import copy
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import HeightField
from experiments.mission_environment_transfer.assets import verify
from experiments.mission_routing_v2.models import CONTROL
from experiments.routing_mission.environment import PhysicalScenario, load
from experiments.v4_benchmark.scenario import digest


@dataclass(frozen=True)
class DepotScenario(PhysicalScenario):
    asset_directory: str = ""
    scenery_sha256: str = ""

    @property
    def recording_identity(self):
        return digest(
            dict(
                physical=super().recording_identity,
                scenery=self.scenery_sha256,
                builder_sha256=sha(Path(__file__).with_name("world.py")),
                controller_sha256=sha(Path(__file__).parents[1] / "v4_gazebo_pilot/transport.py"),
            )
        )


def load_scenario(path, environment, assets):
    verify(assets)
    doc = read(path)
    validated = copy.deepcopy(doc)
    control = next(s for s in doc["executor_configs"] if s["config_id"] == CONTROL)
    for spec in validated["executor_configs"]:
        spec["kind"], spec["parameters"] = control["kind"], control["parameters"]
    validated["contract_path"] = str((path.parent / doc["contract_path"]).resolve())
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        temp = Path(folder) / "scenario.json"
        save(temp, validated)
        base = load(temp, environment)
    scenario = DepotScenario(
        **vars(base),
        asset_directory=str(Path(assets).resolve()),
        scenery_sha256=sha(Path(assets) / "manifest.json"),
    )
    return replace(
        scenario,
        source=doc,
        executors=tuple(ExecutorConfigSpec(**s) for s in doc["executor_configs"]),
    )


def load_public_terrain(assets):
    verify(assets)
    bounds = dict(west=-50.0, east=50.0, south=-50.0, north=50.0)
    return HeightField(np.zeros((2, 2)), **bounds), dict(
        schema_version="public_depot_floor_v1",
        bounds=bounds,
        height_m=0.0,
        model_sdf_sha256=sha(Path(assets) / "depot/model.sdf"),
        asset_manifest_sha256=sha(Path(assets) / "manifest.json"),
        origin="Original Depot floor plane; authored targets restricted to ground floor",
        assumption="Perfect known planar floor and ideal odometry; static ground-level targets",
        object_placements_used=False,
        elevated_targets_supported=False,
    )
