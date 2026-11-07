"""Build an editable, mission-inspired park search development scene.

Dimensions and speed are declared experiment choices, not emergency-service SOPs.
No target coordinates enter routing. Geometry is procedural, not photorealistic.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from itertools import pairwise
from pathlib import Path

from experiments.v4_benchmark.backends import save_json
from experiments.v4_benchmark.scenario import load_scenario
from experiments.v4_gazebo_pilot.world import HEIGHT, HFOV, WIDTH

SOURCES = {
    "survey": (
        "https://docs.qgroundcontrol.com/master/en/qgc-user-guide/plan_view/pattern_survey.html"
    ),
    "obstacles": (
        ""
        "https://www.nist.gov/el/intelligent-systems-division-73500/standard-test-methods-response-robots/aerial-drone-tests-1"
    ),
}


def sweep_route(altitude: float = 8, speed: float = 1.5) -> tuple[list[dict], float]:
    """Time-triggered position goals with explicit travel allowance and settling.

    This is a fixed search route, not a replanning policy. Actual tracking is audited
    from recorded poses; reaching scheduled goals is not assumed.
    """
    if (
        not math.isfinite(altitude)
        or altitude < 6
        or not math.isfinite(speed)
        or not 0 < speed <= 3
    ):
        raise ValueError("altitude >= 6 m and speed in (0,3] required")
    points = []
    for row, y in enumerate((-9, -3, 3, 9)):
        xs = (-12, 12) if row % 2 == 0 else (12, -12)
        points.extend([[x, y, altitude] for x in xs])
    goals = [{"start_s": 0.0, "position_m": points[0]}]
    clock = 2.0
    for previous, point in pairwise(points):
        goals.append({"start_s": clock, "position_m": point})
        clock += math.ceil((math.dist(previous, point) / speed + 4) * 10) / 10
    return goals, math.ceil(clock)


def generate(output: Path, asset: Path, *, altitude: float = 8, speed: float = 1.5) -> dict:
    base = Path(__file__).parent
    source = json.loads((base / "person_probe_scenario.json").read_text())
    person = copy.deepcopy(source["world"]["objects"][0])
    person["mesh"]["manifest"] = str(asset.resolve())
    objects = []
    for i, (x, y, yaw) in enumerate(
        ((-9, -8, 0), (-2, -5, 1.2), (7, -7, 2.1), (-8, 4, 3), (0, 7, 4), (9, 5, 5))
    ):
        target = copy.deepcopy(person)
        target.update(object_id=f"person_{i + 1}", label=i + 1, position_m=[x, y, 0])
        target["mesh"]["pose"][-1] = yaw
        objects.append(target)

    def obstacle(name, position, size, color, shape="box"):
        objects.append(
            dict(
                object_id=name,
                label=len(objects) + 1,
                position_m=position,
                size_m=size,
                color_rgba=[*color, 1],
                is_target=False,
                shape=shape,
            )
        )

    obstacle("east_west_path", [0, 0, 0.025], [36, 2.5, 0.05], [0.62, 0.59, 0.52])
    obstacle("north_south_path", [0, 0, 0.06], [2.5, 28, 0.05], [0.62, 0.59, 0.52])
    obstacle("service_road", [19, 0, 0.025], [4, 36, 0.05], [0.2, 0.22, 0.24])
    obstacle("maintenance_building", [-18, 7, 2], [5, 8, 4], [0.65, 0.58, 0.48])
    obstacle("building_roof", [-18, 7, 4.1], [5.6, 8.6, 0.2], [0.3, 0.23, 0.2])
    for x, y in (
        (-14, -13),
        (-6, -11),
        (4, -12),
        (13, -11),
        (-13, 5),
        (-4, 8),
        (0, 8),
        (13, 6),
        (5, 13),
    ):
        obstacle(f"trunk_{x}_{y}", [x, y, 1.5], [0.35, 0.35, 3], [0.3, 0.2, 0.12], "cylinder")
        obstacle(f"canopy_{x}_{y}", [x, y, 3.4], [3.6, 3.6, 2.4], [0.15, 0.3, 0.1], "ellipsoid")
    for x, y in ((-5, 2), (5, -2)):
        obstacle(f"bench_{x}", [x, y, 0.5], [2, 0.6, 0.15], [0.4, 0.25, 0.12])
        for dx in (-0.7, 0.7):
            obstacle(f"bench_leg_{x}_{dx}", [x + dx, y, 0.25], [0.12, 0.5, 0.5], [0.18, 0.18, 0.18])
    obstacle("shelter_roof", [8, 5, 2.6], [4, 3, 0.2], [0.4, 0.22, 0.13])
    for x in (6.2, 9.8):
        for y in (3.7, 6.3):
            obstacle(f"shelter_post_{x}_{y}", [x, y, 1.3], [0.15, 0.15, 2.6], [0.5, 0.4, 0.3])

    goals, duration = sweep_route(altitude, speed)
    source["scenario_id"] = "V4_PARK_SEARCH_DEVELOPMENT_V1"
    source["contract_path"] = "contract.json"
    source["world"] = dict(
        initial_position_m=goals[0]["position_m"],
        objects=objects,
        waypoints=goals,
        camera_pitch_rad=math.pi / 2,
        ground_size_m=[48, 40, 0.1],
        max_speed_mps=speed,
    )
    source["mission"].update(
        duration_s=duration,
        warmup_s=2.0,
        initial_battery_frac=0.5,
        router_latency_s=0.01,
        router_energy_j=0.05,
    )
    contract = json.loads((base / "audit_contract.json").read_text())
    contract.update(
        contract_id="PARK_SEARCH_DEVELOPMENT",
        deadline_s=duration + 5,
        min_final_battery_frac=0.2,
        quality_threshold=0.5,
        privacy_level="local_only",
    )
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "contract.json", contract)
    save_json(output / "scenario.json", source)
    native = copy.deepcopy(source)
    native["scenario_id"] = "V4_PARK_NATIVE_RESOLUTION_DIAGNOSTIC"
    for executor in native["executor_configs"]:
        executor["parameters"].update(input_width_px=WIDTH, input_height_px=HEIGHT)
    save_json(output / "native_resolution_probe.json", native)
    scenario = load_scenario(output / "scenario.json")
    width = 2 * altitude * math.tan(HFOV / 2)
    design = {
        "schema_version": "0.1",
        "partition": "development",
        "family_id": "park_search_v1",
        "mission": (
            "Search a park sector for people, including partially occluded "
            "people near vegetation and a shelter."
        ),
        "geometry": (
            "Procedural park layout; one repeated licensed human mesh; not a "
            "real site or photoreal scene."
        ),
        "sources": SOURCES,
        "source_use": (
            "Survey pattern and controlled obstruction concepts only; not a "
            "NIST test reproduction or SAR SOP."
        ),
        "choices": {
            "altitude_m": altitude,
            "speed_limit_mps": speed,
            "lane_spacing_m": 6,
            "settling_allowance_s": 4,
            "duration_s": duration,
            "target_count": 6,
        },
        "planned_geometry": {
            "nadir_footprint_width_m": width,
            "nadir_footprint_height_m": width * HEIGHT / WIDTH,
            "nominal_cross_track_overlap": 1 - 6 / width,
            "assumptions": (
                "Level camera at commanded height over flat ground; actual "
                "pose/visibility must be checked."
            ),
        },
        "limitations": [
            "No landing or return-to-home execution; reserve is a battery constraint only.",
            "No wind, sensor noise, thermal imagery, localization errors or human motion.",
            "Mesh appearance fixed; labels/targets never exposed to router.",
            "Current contract cannot enforce joint precision/recall or first-detection deadline.",
        ],
        "recording_identity": scenario.recording_identity,
    }
    save_json(output / "mission_design.json", design)
    routers = [
        "static:CFG_LRASPP",
        "static:CFG_DEEPLAB",
        "rule_based",
        "budget_planner",
        "intent_visual",
        "intent_only",
        "visual_only",
        "resource_only",
    ]
    suite = {
        "suite_schema_version": "0.1",
        "base_scenario": "scenario.json",
        "routers": [{"name": r, "parameters": {}} for r in routers],
        "cases": [
            {
                "case_id": "search",
                "description": "Same park recording, fixed route and costs for all routers.",
                "scenario_overrides": {},
                "contract_overrides": {},
            }
        ],
    }
    for name in ("periodic", "seeded_random"):
        suite["routers"].append(
            {
                "name": name,
                "parameters": {
                    "config_ids": ["CFG_LRASPP", "CFG_DEEPLAB"],
                    "period_s": 2,
                    "seed": 0,
                },
            }
        )
    save_json(output / "suite.json", suite)
    return design


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--altitude", type=float, default=8)
    parser.add_argument("--speed", type=float, default=1.5)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(args.output, args.asset, altitude=args.altitude, speed=args.speed), indent=2
        )
    )


if __name__ == "__main__":
    main()
