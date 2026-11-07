"""Acquire two new development layouts, each paired across lighting conditions.

Geometry and placement are frozen before RGB or model outputs exist. This is
known-map development expansion, not held-out evaluation or real mission logs.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np

from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.placement import collision_bounds, place, terrain_height
from experiments.mission_routing_v2.flights import contract_document, load_scenario, split_check
from experiments.mission_routing_v2.flights import verify as verify_base
from experiments.mission_routing_v4.flights import verify as verify_original

DESIGNS = {
    "west_rocks": dict(
        x=-90,
        y=-25,
        z=24,
        targets=[(-80, -22), (-65, -20), (-80, 6), (-62, 8), (-80, -8), (-63, -5)],
    ),
    "east_shore": dict(
        x=8, y=0, z=32, targets=[(20, 2), (35, 4), (20, 28), (38, 30), (20, 15), (36, 17)]
    ),
}


def geometry_guard(assets, bounds, design):
    for y in np.linspace(design["y"], design["y"] + 30, 121):
        x, z = design["x"], design["z"]
        if z < terrain_height(assets, x, y) + 3:
            raise ValueError("Insufficient terrain clearance")
        for _, low, high in bounds:
            if (
                low[0] - 1 <= x <= high[0] + 1
                and low[1] - 1 <= y <= high[1] + 1
                and low[2] - 1 <= z <= high[2] + 1
            ):
                raise ValueError("Route intersects expanded scenery bounds")


def prepare(original, output):
    if output.exists():
        raise FileExistsError("Preserve prior development acquisition")
    old = verify_original(original)
    assets = Path(old["assets"])
    bounds = collision_bounds(assets)
    template = original / "flight_1_day/rapid_survey"
    base = read(template / "scenario.json")
    excluded = [*old["excluded_lineage"], *(c["lineage"] for c in old["cases"].values())]
    output.mkdir(parents=True)
    cases, frozen = {}, {}
    for group, design in DESIGNS.items():
        geometry_guard(assets, bounds, design)
        occupied, objects = [], []
        for index, xy in enumerate(design["targets"]):
            obj = copy.deepcopy(base["world"]["objects"][index])
            position = place(assets, xy, bounds, occupied)
            occupied.append(position)
            obj.update(
                object_id=f"development_{group}_person_{index + 1}",
                label=index + 1,
                position_m=position,
            )
            obj["mesh"]["pose"][-1] = (index % 4) * math.pi / 2
            objects.append(obj)
        for lighting, upstream in (("day", "flight_1_day"), ("dusk_haze", "flight_2_dusk_haze")):
            flight = group + "_" + lighting
            folder = output / flight / "rapid_survey"
            folder.mkdir(parents=True)
            doc = copy.deepcopy(base)
            x, y, z = design["x"], design["y"], design["z"]
            doc["scenario_id"] = "DEVELOPMENT_MATRIX_" + flight.upper()
            doc["world"].update(
                objects=objects,
                initial_position_m=[x, y, z],
                max_speed_mps=1.2,
                waypoints=[
                    dict(start_s=30 * i, position_m=[x, y + 30 if i % 2 == 0 else y, z])
                    for i in range(4)
                ],
            )
            save(folder / "scenario.json", doc)
            save(folder / "contract.json", read(template / "contract.json"))
            save(
                folder / "intent.json",
                IntentContract(
                    contract_document([o["object_id"] for o in objects], "rapid_survey")
                ).document(),
            )
            environment = dict(old["cases"][upstream]["environment"], seed=2101 + len(cases))
            scenario = load_scenario(folder / "scenario.json", environment, assets)
            lineage = split_fingerprint(scenario, scenario.scenery_sha256)
            split_check(excluded, lineage)
            for c in cases.values():
                if c["layout_group"] != group:
                    split_check([c["lineage"]], lineage)
            cases[flight] = dict(
                environment=environment,
                lineage=lineage,
                layout_group=group,
                lighting=lighting,
                role="development",
                planned_altitude_m=z,
                placement=objects,
            )
            for path in folder.glob("*.json"):
                frozen[str(path.relative_to(output))] = sha(path)
    protocol = {
        key: copy.deepcopy(old[key])
        for key in ("assets", "screen", "calibration_sha256", "costs_sha256", "source_sha256")
    }
    protocol.update(
        schema_version="development_matrix_v1",
        scope="development_only_known_map",
        cases=cases,
        missions=["rapid_survey"],
        frozen_files=frozen,
        acquisition_source_sha256=sha(Path(__file__)),
        original=str(original.resolve()),
        original_protocol_sha256=sha(original / "protocol.json"),
        asset_manifest_sha256=sha(assets / "manifest.json"),
        excluded_evaluation_lineage=excluded,
        designs=DESIGNS,
        layout_groups=list(DESIGNS),
        independent_layouts=2,
        independent_maps=0,
        model_outputs_used_for_placement=False,
        sampling="Every canonical capture; all nine configurations; no outcome-based filtering",
        split_rule="Hold out complete layout groups, including both lighting flights",
        caveats=[
            "Authored simulation on the known lake; not real mission logs",
            "Layouts and routes differ jointly; not a causal altitude/occlusion factorial",
            "Visibility loss is not an isolated occlusion treatment",
            "Development data can fit models; never present these as untouched test flights",
        ],
    )
    save(output / "protocol.json", protocol)
    verify(output)
    print("Frozen four development flights across two new layouts", flush=True)


def verify(output):
    p = verify_base(output)
    if p["schema_version"] != "development_matrix_v1" or p["scope"] != "development_only_known_map":
        raise ValueError("Wrong development protocol")
    if sha(Path(__file__)) != p["acquisition_source_sha256"]:
        raise ValueError("Changed acquisition source")
    old = Path(p["original"])
    verify_original(old)
    if sha(old / "protocol.json") != p["original_protocol_sha256"]:
        raise ValueError("Changed historical source")
    if sha(Path(p["assets"]) / "manifest.json") != p["asset_manifest_sha256"]:
        raise ValueError("Changed map")
    keys = {(c["layout_group"], c["lighting"]) for c in p["cases"].values()}
    if len(p["cases"]) != 4 or keys != {
        (g, light) for g in DESIGNS for light in ("day", "dusk_haze")
    }:
        raise ValueError("Incomplete paired acquisition grid")
    for c in p["cases"].values():
        split_check(p["excluded_evaluation_lineage"], c["lineage"])
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "record", "audit", "infer"))
    parser.add_argument("--original", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.original, args.output)
        return
    verify(args.output)
    if args.mode == "record":
        from experiments.mission_routing_v2.flights import record

        record(args.output, args.flight)
    else:
        from experiments.mission_routing_v2_route_validation import audit

        if not audit(args.output)["all_passed"]:
            raise ValueError("Physical route gate failed")
        if args.mode == "infer":
            from experiments.mission_routing_v2.infer import run

            run(args.output, args.flight)


if __name__ == "__main__":
    main()
