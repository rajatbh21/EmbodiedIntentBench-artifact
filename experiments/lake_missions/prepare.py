"""Author editable development missions on one downloaded lakeside map.

Field data grounds the speed and mission duration, not the hand-authored locations
or objectives. These are development scenarios, not a held-out test distribution.
"""

import argparse
import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

from experiments.lake_missions.assets import save, sha, verify
from experiments.lake_missions.placement import collision_bounds, place
from experiments.routing_mission.environment import Environment
from experiments.routing_mission.prepare import field_row

ROOT = Path(__file__).resolve().parents[2]


def phase(key, name, start, end, targets, *, recall=0.5, hits=2, span=4, false=8, weight=1):
    return dict(
        phase_id=key,
        instruction=name,
        start_s=start,
        end_s=end,
        deadline_s=end,
        required_recall=recall,
        min_hits=hits,
        min_evidence_span_s=span,
        max_false_components=false,
        weight=weight,
        evaluator_target_ids=targets,
    )


def prepare(output, assets):
    verify(assets)
    output.mkdir(parents=True, exist_ok=False)
    basepath = ROOT / "results/routing_mission_field_20261016/scenario.json"
    base = json.loads(basepath.read_text())
    # Preserve the model pool, configured compute costs and field evidence.
    row = field_row(ROOT / "results/v4_external_evidence_20261014", "D-4D-141520")
    duration = round(float(row["run_duration_min"]) * 60, 1)
    speed = float(row["coverage_speed_mps"])
    base["scenario_id"] = "LAKE_FIXED_FLIGHT_DEVELOPMENT_V2"
    base["mission"].update(duration_s=duration, observation_interval_s=2, initial_battery_frac=0.9)
    source_person = base["world"]["objects"][0]
    objects = []
    bounds = collision_bounds(assets)
    placements_audit = []
    placements = [
        ("west_1", -18, -10),
        ("west_2", -18, 12),
        ("west_3", -25, 24),
        ("north_1", -2, 18),
        ("north_2", 12, 24),
        ("north_3", 24, 18),
        ("east_1", 25, -10),
        ("east_2", 22, -25),
        ("east_3", 35, -20),
    ]
    for i, (name, x, y) in enumerate(placements, 1):
        person = copy.deepcopy(source_person)
        person.update(
            object_id=name,
            label=i,
            position_m=place(assets, [x, y], bounds, [o["position_m"] for o in objects]),
        )
        person["mesh"]["pose"] = [0, 0, 0.02, 0, 0, (i % 4) * math.pi / 2]
        objects.append(person)
        placements_audit.append(
            dict(target_id=name, desired_xy=[x, y], actual_position_m=person["position_m"])
        )
    base["world"].update(
        objects=objects,
        initial_position_m=[-30, -25, 34],
        max_speed_mps=speed,
        camera_pitch_rad=1.2,
        waypoints=[
            dict(start_s=t, position_m=p)
            for t, p in [
                (0, [-30, 25, 34]),
                (55, [30, 25, 36]),
                (115, [30, -25, 40]),
                (170, [-30, -25, 34]),
                (225, [-30, 25, 32]),
                (280, [20, 25, 34]),
                (325, [20, -10, 38]),
            ]
        ],
    )
    # Terrain is not a flat z=0 plane. Keep image association explicitly diagnostic.
    base["reporting"] = dict(association="image_iou", min_hits=2, max_gap_s=8, association_iou=0.15)
    base["mission_requirements"].update(confirmed_report_deadline_s=duration)
    save(output / "scenario.json", base)
    save(
        output / "placement_audit.json",
        dict(
            collision_bounds_checked=len(bounds),
            water_clearance_m=1.2,
            max_terrain_gradient=0.3,
            human_clearance_width_m=0.9,
            targets=placements_audit,
            policy_outputs_used=False,
        ),
    )
    contract = json.loads((basepath.parent / "contract.json").read_text())
    contract["deadline_s"] = duration + 10
    save(output / "contract.json", contract)
    west = ["west_1", "west_2", "west_3"]
    north = ["north_1", "north_2", "north_3"]
    east = ["east_1", "east_2", "east_3"]
    missions = [
        dict(
            mission_id="sector_rescue",
            title="구역별 우선 수색과 재확인",
            user_request=(
                "서쪽 수색 후 북쪽 수변을 우선 확인하고, "
                "동쪽 숲 가장자리를 수색한 뒤 서쪽을 재확인해."
            ),
            initial_battery_frac=0.9,
            min_final_battery_frac=0.2,
            phases=[
                phase("west_search", "서쪽 구역에서 반복 증거 확보", 0, 65, west),
                phase(
                    "north_priority", "북쪽 수변 우선 확인", 65, 125, north, recall=2 / 3, weight=2
                ),
                phase("east_search", "동쪽 숲 가장자리 수색", 125, 215, east),
                phase("west_revisit", "서쪽 재방문: 오래된 증거 재사용 금지", 225, 285, west),
            ],
        ),
        dict(
            mission_id="perimeter_inspection",
            title="시설 주변 안전점검",
            user_request=(
                "시설 주변 세 구역을 빠짐없이 확인해. "
                "한 번의 탐지만으로 완료하지 말고 반복 관측하고 오경보를 억제해."
            ),
            initial_battery_frac=0.9,
            min_final_battery_frac=0.25,
            phases=[
                phase(
                    "west_inspect", "서측 접근로 반복 확인", 0, 65, west, hits=3, span=6, false=4
                ),
                phase(
                    "north_inspect",
                    "북측 수변 보행 구역 확인",
                    65,
                    125,
                    north,
                    hits=3,
                    span=6,
                    false=4,
                ),
                phase(
                    "east_inspect", "동측 시설 경계 확인", 125, 215, east, hits=3, span=6, false=4
                ),
                phase(
                    "north_recheck", "북측 구역의 최신 증거로 재점검", 285, duration, north, false=4
                ),
            ],
        ),
        dict(
            mission_id="reserve_revisit",
            title="배터리 제약 아래 재방문 검증",
            user_request=(
                "배터리가 부족해. 서쪽과 북쪽을 확인하고 다시 방문해서 "
                "최신 증거를 확보하되 예비 배터리 20%를 남겨."
            ),
            initial_battery_frac=0.61,
            min_final_battery_frac=0.2,
            phases=[
                phase("west_first", "서측 최초 증거 확보", 0, 65, west, false=6),
                phase("north_first", "북측 최초 증거 확보", 65, 125, north, false=6),
                phase(
                    "west_fresh",
                    "서측 재확인: 새로운 반복 관측 필요",
                    225,
                    285,
                    west,
                    false=6,
                    weight=2,
                ),
                phase(
                    "north_fresh",
                    "북측 재확인 후 배터리 보존",
                    285,
                    duration,
                    north,
                    false=6,
                    weight=2,
                ),
            ],
        ),
    ]
    save(output / "missions.json", dict(schema_version="1.0", missions=missions))
    save(
        output / "suite.json",
        dict(
            schema_version="1.0",
            assets=str(assets.resolve()),
            environments={
                "day": asdict(Environment(ambient=0.5)),
                "dusk_haze": asdict(
                    Environment(
                        ambient=0.22, sunlight=0.35, fog_density=0.004, rgb_noise_stddev=0.005
                    )
                ),
            },
            policies=[
                "static:CFG_CABINET_640",
                "static:CFG_MASKRCNN_640",
                "static:CFG_MASKRCNN_1280",
                "periodic",
                "context",
                "phase_context",
            ],
            purpose="development_only_not_held_out",
            seed=0,
        ),
    )
    save(
        output / "evidence.json",
        dict(
            field_source="https://zenodo.org/records/20643214",
            run_id=row["run_id"],
            field_parameters=dict(duration_s=duration, speed_mps=speed),
            task_design_source="https://www.nist.gov/el/intelligent-systems-division-73500/standard-test-methods-response-robots/aerial-drone-tests-1",
            map_source="https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/Harmonic%20World",
            scenery_manifest_sha256=sha(assets / "manifest.json"),
            authored=[
                "fixed flight route",
                "target placements",
                "phase timing",
                "resource budgets",
                "success thresholds",
            ],
            limitations=[
                "static human meshes",
                "no raw GPS replay",
                "no measured drone power",
                "no NIST certification",
                "human perception tasks, not fire/structural damage recognition",
            ],
        ),
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--assets", type=Path, default=ROOT / "results/lake_assets")
    args = parser.parse_args()
    prepare(args.output, args.assets)
