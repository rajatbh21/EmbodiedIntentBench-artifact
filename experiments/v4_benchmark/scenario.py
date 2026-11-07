"""Strict, versioned V4 experiment inputs; no simulator handles reach the router."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from missionbench.schemas.configuration import (
    ConfigCatalog,
    Configuration,
    Placement,
    Precision,
    Strategy,
)
from missionbench.schemas.contract import Contract, load_contract
from missionbench.schemas.loading import DocumentReader, SchemaValidationError, read_json_object
from missionbench.schemas.profile import PublicProfile, PublicProfileView, QualityTier
from missionbench.v2.network import NetworkRegime, V2NetworkModel
from missionbench.v2.scenario import ExecutorConfigSpec, read_executor_configs
from experiments.v4_benchmark.mission_requirements import MissionRequirements
from experiments.v4_benchmark.reporting import ReportingConfig
from experiments.v4_benchmark.resources import ResourceConfig
from experiments.v4_gazebo_pilot.timing import CaptureSchedule, seconds_to_ns
from experiments.v4_gazebo_pilot.world import STEP_NS


def finite_tree(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaValidationError("non-finite JSON numbers are forbidden")
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    contract: Contract
    duration_s: float
    observation_interval_s: float
    warmup_s: float
    initial_battery_frac: float
    battery_capacity_wh: float
    flight_power_w: float
    router_latency_s: float
    router_energy_j: float
    matching_iou_threshold: float
    fallback_config_id: str
    executors: tuple[ExecutorConfigSpec, ...]
    catalog: ConfigCatalog
    profiles: PublicProfileView
    network: V2NetworkModel
    world: dict[str, Any]
    source: dict[str, Any]
    requirements: MissionRequirements = field(default_factory=MissionRequirements)

    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)

    @property
    def schedule(self) -> CaptureSchedule:
        epoch = seconds_to_ns(self.warmup_s)
        return CaptureSchedule(
            epoch,
            seconds_to_ns(self.observation_interval_s),
            epoch + seconds_to_ns(self.duration_s),
            STEP_NS,
        )

    @property
    def recording_identity(self) -> str:
        # Costs, contract, battery and network do not affect this fixed flight path.
        return digest(
            {
                "world": self.world,
                "warmup_s": self.warmup_s,
                "duration_s": self.duration_s,
                "observation_interval_s": self.observation_interval_s,
                "physics_step_ns": STEP_NS,
                "world_builder_version": "1",
            }
        )

    @property
    def targets(self) -> dict[int, str]:
        return {obj["label"]: obj["object_id"] for obj in self.world["objects"] if obj["is_target"]}


def load_scenario(path: Path, *, contract_path: Path | None = None) -> Scenario:
    data = read_json_object(path)
    finite_tree(data)
    reader = DocumentReader(
        data,
        context=str(path),
        allowed_fields=(
            "v4_schema_version",
            "scenario_id",
            "contract_path",
            "mission",
            "executor_configs",
            "public_profiles",
            "network_trace",
            "world",
            "mission_requirements",
            "reporting",
            "resources",
        ),
    )
    if reader.get_str("v4_schema_version") != "0.1":
        raise SchemaValidationError("V4 currently supports v4_schema_version '0.1' only")
    contract = load_contract(contract_path or path.parent / reader.get_str("contract_path"))
    finite_tree(
        vars(contract)
        if hasattr(contract, "__dict__")
        else {name: getattr(contract, name) for name in contract.__slots__}
    )
    if contract.quality_metric not in ("target_recall", "detection_precision"):
        raise SchemaValidationError("V4 supports target_recall or detection_precision contracts")
    mission = reader.get_object(
        "mission",
        allowed_fields=(
            "duration_s",
            "observation_interval_s",
            "warmup_s",
            "initial_battery_frac",
            "battery_capacity_wh",
            "flight_power_w",
            "router_latency_s",
            "router_energy_j",
            "matching_iou_threshold",
            "fallback_config_id",
        ),
    )
    specs = read_executor_configs(reader)
    catalog = ConfigCatalog(
        [
            Configuration(
                s.config_id,
                s.model_strategy_id,
                Strategy(
                    Placement.REMOTE if s.kind == "simulated_remote" else Placement.LOCAL,
                    Precision.FP32,
                    parameters={"transmitted_payload": "raw_input"},
                ),
            )
            for s in specs
        ]
    )
    for spec in specs:
        QualityTier(spec.quality_tier)
        if spec.kind != "simulated_remote" and spec.communication_mb_per_call != 0:
            raise SchemaValidationError("local executors must have zero communication")
        fallback = spec.parameters.get("fallback_config_id")
        if fallback is not None and (
            fallback not in catalog or catalog.get(fallback).strategy.is_remote
        ):
            raise SchemaValidationError("remote executor fallback must name a local configuration")
        # The configured-time harness must not silently switch to host-time semantics.
        if spec.parameters.get("latency_mode", "configured") != "configured":
            raise SchemaValidationError("V4 comparisons require configured latency_mode")
        if spec.parameters.get("backend_latency_mode", "configured") != "configured":
            raise SchemaValidationError("V4 comparisons require configured backend_latency_mode")
    profiles = {}
    for entry in reader.get_object_list(
        "public_profiles",
        allowed_fields=(
            "config_id",
            "expected_latency_ms",
            "expected_upload_mb",
            "quality_tier",
        ),
    ):
        key = entry.get_str("config_id")
        if key not in catalog or key in profiles:
            raise SchemaValidationError(f"unknown or duplicate public profile: {key}")
        profiles[key] = PublicProfile(
            config_id=key,
            expected_latency_ms=entry.get_float("expected_latency_ms", minimum=0),
            expected_upload_mb=entry.get_float("expected_upload_mb", minimum=0),
            quality_tier=entry.get_enum("quality_tier", QualityTier),
        )
    regimes = []
    for entry in reader.get_object_list(
        "network_trace",
        allowed_fields=(
            "regime_id",
            "start_s",
            "uplink_mbps",
            "downlink_mbps",
            "rtt_ms",
            "packet_loss_frac",
        ),
    ):
        regimes.append(
            NetworkRegime(
                entry.get_str("regime_id"),
                entry.get_float("start_s", minimum=0),
                entry.get_float("uplink_mbps", minimum=0),
                entry.get_float("downlink_mbps", minimum=0),
                entry.get_float("rtt_ms", minimum=0),
                entry.get_fraction("packet_loss_frac"),
            )
        )
    world = validate_world(data.get("world"), base_dir=path.parent)
    scenario = Scenario(
        scenario_id=reader.get_str("scenario_id"),
        contract=contract,
        duration_s=mission.get_float("duration_s", exclusive_minimum=0),
        observation_interval_s=mission.get_float("observation_interval_s", exclusive_minimum=0),
        warmup_s=mission.get_float("warmup_s", exclusive_minimum=0),
        initial_battery_frac=mission.get_fraction("initial_battery_frac"),
        battery_capacity_wh=mission.get_float("battery_capacity_wh", exclusive_minimum=0),
        flight_power_w=mission.get_float("flight_power_w", minimum=0),
        router_latency_s=mission.get_float("router_latency_s", minimum=0),
        router_energy_j=mission.get_float("router_energy_j", minimum=0),
        matching_iou_threshold=mission.get_float(
            "matching_iou_threshold", exclusive_minimum=0, maximum=1
        ),
        fallback_config_id=mission.get_str("fallback_config_id"),
        executors=specs,
        catalog=catalog,
        profiles=PublicProfileView(profiles),
        network=V2NetworkModel(tuple(regimes)),
        world=world,
        source=data,
        requirements=MissionRequirements.parse(data.get("mission_requirements", {})),
        reporting=ReportingConfig.parse(data.get("reporting", {})),
        resources=ResourceConfig.parse(data.get("resources", {}), catalog.ids()),
    )
    schedule = scenario.schedule
    if schedule.epoch_ns % 100_000_000 or schedule.interval_ns % 100_000_000:
        raise SchemaValidationError("warmup and observation interval must align with 10 Hz sensors")
    fallback = scenario.fallback_config_id
    if fallback not in catalog or catalog.get(fallback).strategy.is_remote:
        raise SchemaValidationError("safe fallback must name a local configuration")
    return scenario


def vector(value: Any, size: int, context: str, *, positive: bool = False) -> list[float]:
    if (
        not isinstance(value, list)
        or len(value) != size
        or any(
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not math.isfinite(v)
            or (positive and v <= 0)
            for v in value
        )
    ):
        raise SchemaValidationError(
            f"{context}: expected {size} finite {'positive ' if positive else ''}numbers"
        )
    return [float(v) for v in value]


def validate_world(data: Any, *, base_dir: Path = Path(".")) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise SchemaValidationError("world must be an object")
    reader = DocumentReader(
        data,
        context="world",
        allowed_fields=(
            "initial_position_m",
            "objects",
            "waypoints",
            "camera_pitch_rad",
            "ground_size_m",
            "max_speed_mps",
        ),
    )
    initial = vector(reader.get_passthrough("initial_position_m"), 3, "initial_position_m")
    if initial[2] <= 0:
        raise SchemaValidationError("initial altitude must be positive")
    objects = []
    for entry in reader.get_object_list(
        "objects",
        allowed_fields=(
            "object_id",
            "label",
            "position_m",
            "size_m",
            "color_rgba",
            "is_target",
            "mesh",
            "shape",
        ),
    ):
        color = vector(entry.get_passthrough("color_rgba"), 4, "color_rgba")
        if any(v < 0 or v > 1 for v in color):
            raise SchemaValidationError("color components must lie in [0,1]")
        label = entry.get_int("label", minimum=1)
        if label > 254:
            raise SchemaValidationError("object labels must be 1..254")
        objects.append(
            {
                "object_id": entry.get_str("object_id"),
                "label": label,
                "position_m": vector(entry.get_passthrough("position_m"), 3, "position_m"),
                "size_m": vector(entry.get_passthrough("size_m"), 3, "size_m", positive=True),
                "color_rgba": color,
                "is_target": entry.get_bool("is_target"),
            }
        )
        raw_object = data["objects"][len(objects) - 1]
        if "shape" in raw_object:
            shape = entry.get_str("shape")
            if shape not in ("box", "cylinder", "ellipsoid") or "mesh" in raw_object:
                raise SchemaValidationError("shape must be box/cylinder/ellipsoid without mesh")
            if shape == "cylinder" and objects[-1]["size_m"][0] != objects[-1]["size_m"][1]:
                raise SchemaValidationError("cylinder X/Y diameters must match")
            objects[-1]["shape"] = shape
        if "mesh" in data["objects"][len(objects) - 1]:
            mesh = entry.get_object("mesh", allowed_fields=("manifest", "scale", "pose"))
            from experiments.v4_gazebo_pilot.assets import load_asset

            objects[-1]["mesh"] = {
                "asset": load_asset(base_dir / mesh.get_str("manifest")),
                "scale": vector(mesh.get_passthrough("scale"), 3, "mesh scale", positive=True),
                "pose": vector(mesh.get_passthrough("pose"), 6, "mesh pose"),
            }
    if len({o["label"] for o in objects}) != len(objects) or len(
        {o["object_id"] for o in objects}
    ) != len(objects):
        raise SchemaValidationError("object labels and IDs must be unique")
    if not any(o["is_target"] for o in objects):
        raise SchemaValidationError("world needs at least one target")
    waypoints = []
    for entry in reader.get_object_list("waypoints", allowed_fields=("start_s", "position_m")):
        time_s = entry.get_float("start_s", minimum=0)
        if seconds_to_ns(time_s, allow_zero=True) % 100_000_000:
            raise SchemaValidationError("waypoints must align with 100 ms controller ticks")
        waypoints.append(
            {
                "start_s": time_s,
                "position_m": vector(entry.get_passthrough("position_m"), 3, "waypoint position"),
            }
        )
    if (
        not waypoints
        or waypoints[0]["start_s"] != 0
        or any(b["start_s"] <= a["start_s"] for a, b in pairwise(waypoints))
    ):
        raise SchemaValidationError("waypoints must start at zero and increase in time")
    world = {"initial_position_m": initial, "objects": objects, "waypoints": waypoints}
    if "camera_pitch_rad" in data:
        world["camera_pitch_rad"] = reader.get_float(
            "camera_pitch_rad", minimum=0, maximum=math.pi / 2
        )
    if "ground_size_m" in data:
        size = vector(data["ground_size_m"], 3, "ground_size_m", positive=True)
        if size[2] != 0.1:
            raise SchemaValidationError("ground thickness must remain 0.1 m")
        world["ground_size_m"] = size
    if "max_speed_mps" in data:
        world["max_speed_mps"] = reader.get_float("max_speed_mps", exclusive_minimum=0, maximum=3)
    return world


def exported_source(scenario: Scenario) -> dict:
    """Keep saved run inputs reloadable with local asset manifest references."""
    source = copy.deepcopy(scenario.source)
    source["contract_path"] = "contract.json"
    source["reporting"] = scenario.reporting.to_dict()
    if scenario.resources.models:
        source["resources"] = scenario.resources.to_dict()
    if scenario.requirements != MissionRequirements() or "mission_requirements" in source:
        source["mission_requirements"] = scenario.requirements.to_dict()
    source["world"] = copy.deepcopy(scenario.world)
    for obj in source["world"]["objects"]:
        if "mesh" in obj:
            asset = obj["mesh"].pop("asset")
            obj["mesh"]["manifest"] = asset["manifest_path"]
    return source
