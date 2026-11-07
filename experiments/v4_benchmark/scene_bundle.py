"""Predeclare procedural scene families before inference; no outcome-based selection.

Shared human mesh and renderer mean these are controlled geometry diagnostics,
not independent real-world test sites. Recording and scoring are separate steps.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

from experiments.v4_benchmark.backends import save_json
from experiments.v4_benchmark.scenario import load_scenario
from experiments.v4_benchmark.splits import checksum

FAMILIES = (
    ("development", "courtyard", 271),
    ("validation", "roadside", 811),
    ("evaluation", "grove", 1429),
)
CONDITIONS = (("low_oblique", 4.0, math.pi / 4), ("high_downward", 8.0, math.pi / 2))


def generate(output: Path, asset: Path) -> dict:
    """Write a fresh, deterministic six-scene plan, without running any model."""
    if output.exists():
        raise FileExistsError(f"refusing to replace scene plan: {output}")
    base = Path(__file__).parent
    template = json.loads((base / "person_probe_scenario.json").read_text())
    contract = json.loads((base / "audit_contract.json").read_text())
    contract.update(contract_id="V4_SCENE_BUNDLE", deadline_s=15.0)
    # Verify license/checksums before creating a plan.
    from experiments.v4_gazebo_pilot.assets import load_asset

    load_asset(asset)
    output.mkdir(parents=True)
    save_json(output / "contract.json", contract)
    entries, designs = [], []
    for partition, family, seed in FAMILIES:
        rng = random.Random(seed)
        positions = [[x + rng.uniform(-0.4, 0.4), y, 0] for x, y in ((1, -1), (4, 1), (7, 0))]
        yaws = [rng.uniform(0, 2 * math.pi) for _ in positions]
        for condition, altitude, pitch in CONDITIONS:
            case = f"{family}_{condition}"
            data = copy.deepcopy(template)
            data.update(scenario_id=f"V4_{case.upper()}", contract_path="../contract.json")
            data["mission"].update(
                duration_s=12.0, warmup_s=2.0, router_latency_s=0.01, router_energy_j=0.05
            )
            objects = []
            for i, (position, yaw) in enumerate(zip(positions, yaws, strict=True)):
                person = copy.deepcopy(template["world"]["objects"][0])
                person.update(object_id=f"person_{i}", label=i + 1, position_m=position)
                person["mesh"]["manifest"] = str(asset.resolve())
                person["mesh"]["pose"][-1] = yaw
                objects.append(person)

            def obstacle(name, position, size, color, shape="box", *, objects=objects):
                objects.append(
                    dict(
                        object_id=name,
                        label=len(objects) + 1,
                        position_m=position,
                        size_m=size,
                        color_rgba=[*color, 1],
                        shape=shape,
                        is_target=False,
                    )
                )

            if family == "courtyard":
                obstacle("paving", [4, 0, 0.025], [18, 10, 0.05], [0.6, 0.6, 0.55])
                obstacle("wall", [4, 3, 1.4], [12, 0.4, 2.8], [0.55, 0.4, 0.3])
                obstacle("bench", [4, 1.7, 0.5], [2, 0.6, 1], [0.3, 0.2, 0.1])
            elif family == "roadside":
                obstacle("road", [4, -3, 0.025], [22, 3, 0.05], [0.18, 0.18, 0.2])
                obstacle("shelter", [7, 0, 2.6], [2, 2.5, 0.2], [0.4, 0.45, 0.5])
                obstacle("barrier", [1, 2, 0.6], [6, 0.3, 1.2], [0.7, 0.65, 0.5])
            else:
                for i, (x, y) in enumerate(((0, 2), (4, 2), (7, 1.5), (9, -2))):
                    obstacle(f"trunk_{i}", [x, y, 1.5], [0.4, 0.4, 3], [0.3, 0.2, 0.1], "cylinder")
                    obstacle(f"canopy_{i}", [x, y, 3.1], [3, 3, 2], [0.15, 0.3, 0.12], "ellipsoid")
            world = data["world"]
            world.update(
                objects=objects,
                initial_position_m=[-2, 0, altitude],
                camera_pitch_rad=pitch,
                ground_size_m=[32, 24, 0.1],
                max_speed_mps=1.5,
                waypoints=[
                    {"start_s": t, "position_m": [x, 0, altitude]}
                    for t, x in ((0, -2), (2, 3), (7, 8))
                ],
            )
            directory = output / case
            directory.mkdir()
            path = directory / "scenario.json"
            save_json(path, data)
            scenario = load_scenario(path)
            entries.append(
                dict(
                    case_id=case,
                    partition=partition,
                    family_id=family,
                    scenario=f"{case}/scenario.json",
                    recording=f"{case}/recording",
                )
            )
            designs.append(
                dict(
                    case_id=case,
                    seed=seed,
                    altitude_m=altitude,
                    camera_pitch_rad=pitch,
                    scenario_sha256=checksum(path),
                    recording_identity=scenario.recording_identity,
                )
            )
    save_json(output / "split_source.json", {"split_schema_version": "0.2", "entries": entries})
    plan = dict(
        scene_bundle_version="0.1",
        purpose="controlled_geometry_generalization",
        independence_verified=False,
        human_appearance_count=1,
        limitations=[
            "shared mesh and renderer across all partitions",
            "static humans; no weather, sensor noise or flight autonomy",
            "one family per partition is insufficient for population statistics",
        ],
        selection="fixed seeds and layouts, no model outcomes consulted",
        contract_sha256=checksum(output / "contract.json"),
        asset_manifest_sha256=checksum(asset),
        split_source_sha256=checksum(output / "split_source.json"),
        cases=designs,
    )
    save_json(output / "scene_plan.json", plan)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    args = parser.parse_args()
    plan = generate(args.output, args.asset)
    print(json.dumps({"cases": len(plan["cases"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
