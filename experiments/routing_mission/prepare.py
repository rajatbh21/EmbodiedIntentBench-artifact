"""Build an editable paired experiment from verified field summaries and geography."""

import copy
import csv
import hashlib
import io
import json
import math
import zipfile
from dataclasses import asdict
from pathlib import Path

from experiments.evidence_environment.map_plan import generate
from experiments.routing_mission.environment import Environment, load
from experiments.routing_mission.intent import parse
from experiments.v4_benchmark.backends import save_json
from experiments.v4_benchmark.model_pool import candidates

ROOT = Path(__file__).resolve().parents[2]
FIELD_SHA = "d1c3c282cc144e3c9b29a4705ebf5df069edf6ae0065a9093ad24235503ffbb8"


def field_row(source, run_id):
    archive = source / "swarm_steward/C_search_experiments.zip"
    if hashlib.sha256(archive.read_bytes()).hexdigest() != FIELD_SHA:
        raise ValueError("field evidence checksum mismatch")
    with zipfile.ZipFile(archive) as z:
        rows = list(
            csv.DictReader(
                io.StringIO(
                    z.read("C_search_experiments/data/search_experiment_manifest.csv").decode()
                )
            )
        )
    return next(r for r in rows if r["run_id"] == run_id)


def prepare(output, source, asset, request, run_id="D-4D-141520", duration=None, interval=5.0):
    row = field_row(source, run_id)
    field_duration = round(float(row["run_duration_min"]) * 60, 1)
    seconds = field_duration if duration is None else duration
    if not math.isfinite(seconds) or not 10 <= seconds <= field_duration:
        raise ValueError("duration must be 10 seconds..source duration; no silent time compression")
    if (
        not math.isfinite(interval)
        or not 0.5 <= interval <= 10
        or round(interval * 10) != interval * 10
    ):
        raise ValueError("interval must be 0.5..10s aligned to 100ms")
    intent, intent_audit = parse(request, duration_s=seconds)
    output.mkdir(parents=True, exist_ok=False)
    # Geographic crop selection is fixed by coordinates, never perception outcomes.
    generate(
        source,
        output / "map",
        asset,
        offset_east_m=-120.2379091900186,
        offset_north_m=-243.3726721068914,
    )
    base = json.loads((output / "map/h35_p45/scenario.json").read_text())
    base["contract_path"] = "contract.json"
    base["scenario_id"] = "FIELD_GROUNDED_ROUTING_" + run_id
    height = float(row["median_altitude_m"])
    speed = float(row["coverage_speed_mps"])
    if not 0 < speed <= 3:
        raise ValueError("source speed outside supported controller range")
    # Two lawnmower lanes in an assumed assigned sector; raw GPS is not in source.
    x, y = 99.5, 30.0
    waypoints = []
    start = 0.0
    previous = [-x, -y, height]
    for point in ([x, -y, height], [x, y, height], [-x, y, height]):
        waypoints.append(dict(start_s=round(start, 1), position_m=point))
        start += math.dist(previous, point) / speed + 8
        previous = point
    base["world"].update(
        initial_position_m=[-x, -y, height],
        waypoints=waypoints,
        max_speed_mps=speed,
        camera_pitch_rad=math.radians(-float(row["median_gimbal_pitch_deg"])),
    )
    base["mission"].update(
        duration_s=seconds,
        observation_interval_s=interval,
        warmup_s=2,
        initial_battery_frac=0.9,
        battery_capacity_wh=30,
        flight_power_w=120,
        router_latency_s=0.01,
        router_energy_j=0.02,
    )
    pool = [
        s
        for s in candidates()
        if s["config_id"] in ("CFG_LRASPP_320", "CFG_DEEPLAB_MOBILE_320", "CFG_MASKRCNN_640")
    ]
    large = copy.deepcopy(pool[-1])
    large.update(
        config_id="CFG_MASKRCNN_1280",
        model_strategy_id="maskrcnn_resnet50_fpn_v2_1280x960",
        mission_latency_s=6,
        energy_j_per_call=30,
    )
    large["parameters"].update(input_width_px=1280, input_height_px=960)
    pool.append(large)
    bundle = ROOT / "results/v4_assets/cabinet_uavid_small/manifest.json"
    metadata = json.loads(bundle.read_text())
    cab = copy.deepcopy(pool[0])
    cab.update(
        config_id="CFG_CABINET_640",
        model_strategy_id="cabinet_uavid_small_640",
        kind="torch_cabinet_segmentation",
        mission_latency_s=0.4,
        energy_j_per_call=8,
        quality_tier="medium",
    )
    cab["parameters"].update(
        model_id="cabinet_mobilenetv3_small_uavid",
        weights_id=metadata["revision"],
        bundle_manifest=str(bundle),
        bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),
        input_width_px=640,
        input_height_px=480,
    )
    pool.append(cab)
    base["executor_configs"] = pool
    base["public_profiles"] = [
        dict(
            config_id=s["config_id"],
            expected_latency_ms=s["mission_latency_s"] * 1000,
            expected_upload_mb=0,
            quality_tier=s["quality_tier"],
        )
        for s in pool
    ]
    base["mission"]["fallback_config_id"] = pool[0]["config_id"]
    base["resources"] = dict(
        memory_capacity_mb=4096,
        models={
            s["config_id"]: dict(
                memory_mb=(
                    2400 if "1280" in s["config_id"] else 1200 if "MASK" in s["config_id"] else 512
                ),
                load_latency_s=0.3,
                load_energy_j=2,
            )
            for s in pool
        },
    )
    base["reporting"] = dict(
        association="ground_plane",
        max_distance_m=5,
        min_hits=2,
        max_gap_s=interval * 3,
        association_iou=0.2,
    )
    contract, requirements = intent.compile()
    base["mission_requirements"] = requirements
    save_json(output / "contract.json", contract)
    save_json(output / "scenario.json", base)
    save_json(output / "intent.json", dict(resolved=asdict(intent), audit=intent_audit))
    environments = {
        "clear": asdict(Environment()),
        "dim_haze": asdict(
            Environment(ambient=0.18, sunlight=0.3, fog_density=0.012, rgb_noise_stddev=0.01)
        ),
    }
    # Intent/resource cases all retain the identical physical flight.
    cases = [
        dict(case_id="user", contract={}, requirements={}, mission={}),
        dict(
            case_id="urgent",
            contract={},
            requirements=dict(confirmed_report_deadline_s=min(120, seconds * 0.4)),
            mission={},
        ),
        dict(
            case_id="strict_reports",
            contract={},
            requirements=dict(min_report_recall=1, max_false_reports=0, max_duplicate_reports=0),
            mission={},
        ),
        dict(
            case_id="low_battery",
            contract={},
            requirements={},
            mission=dict(initial_battery_frac=0.60),
        ),
    ]
    routers = [f"static:{s['config_id']}" for s in pool] + [
        "periodic",
        "seeded_random",
        "context",
        "context_no_intent",
        "context_no_visual",
        "context_no_resources",
    ]
    spec = dict(
        schema_version="1.0",
        scenario="scenario.json",
        environments=environments,
        cases=cases,
        routers=routers,
        seed=17,
        threads=2,
    )
    save_json(output / "experiment.json", spec)
    save_json(
        output / "evidence.json",
        dict(
            source_url="https://zenodo.org/records/20643214",
            archive_sha256=FIELD_SHA,
            source_row=row,
            source_duration_s=field_duration,
            simulated_duration_s=seconds,
            scope="single-aircraft sector adapted from field summary",
            reproduced_fields=[
                "median_altitude_m",
                "median_gimbal_pitch_deg",
                "coverage_speed_mps",
                "run_duration_min",
            ],
            duration_override=duration is not None,
            geography="SAREnv map 1; different site from field trial",
            assumptions=[
                "two-lane assigned-sector path; not recovered GPS trajectory",
                "three static mesh people; source outcome is not simulator GT",
                "geometry and human placement approximate; no photogrammetry",
                "lighting, haze, sensor noise, battery and energy are experimental assumptions",
                "ideal-state controller and rotor physics; no PX4 or real battery model",
                "compute latencies configured; host measurements do not change this run",
                "image resolution 960x720, sampling interval configured; HFOV=1.2rad",
                "this is development evaluation, not independent multi-site generalization",
            ],
        ),
    )
    for env in environments.values():
        load(output / "scenario.json", env)
    return spec
