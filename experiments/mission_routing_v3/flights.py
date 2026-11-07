"""Freeze new controllers before a geometry-authored paired Gazebo flight cohort.

Old inspected v2 flights remain historical evidence, never a fresh v3 test. This
is a new route/layout on the same lake and nearby corridor, not a new environment.
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
from experiments.mission_routing_v2.flights import verify as verify_v2
from experiments.mission_routing_v2_physics import validate_alignment


def verify(root):
    p = verify_v2(root)
    if p.get("controller_revision") != "confirmation_v3":
        raise ValueError("Expected v3 study")
    sources = {file.name: sha(file) for file in Path(__file__).parent.glob("*.py")}
    if sources != p["v3_source_sha256"]:
        raise ValueError("Frozen v3 implementation changed")
    if sha(Path(p["assets"]) / "manifest.json") != p["asset_manifest_sha256"]:
        raise ValueError("Asset manifest changed")
    for record in p["inspected_prior_studies"]:
        if sha(Path(record["root"]) / "protocol.json") != record["protocol_sha256"]:
            raise ValueError("Prior study changed")
    for case in p["cases"].values():
        split_check(p["excluded_lineage"], case["lineage"])
    return p


def prepare(original, output):
    from experiments.mission_routing_v3.study import POLICIES

    p = verify_v2(original)
    assets = Path(p["assets"])
    validate_alignment(assets)
    profile = read(Path(p["screen"]) / "calibration.json")
    excluded = [*profile["training_lineage"], *(c["lineage"] for c in p["cases"].values())]
    source = original / next(iter(p["cases"])) / "rapid_survey/scenario.json"
    base = read(source)
    bounds = collision_bounds(assets)
    occupied = [obj["position_m"] for obj in base["world"]["objects"]]
    training = Path(read(Path(p["screen"]) / "protocol.json")["source"])
    occupied.extend(o["position_m"] for o in read(training / "scenario.json")["world"]["objects"])
    objects = []
    for index, xy in enumerate(((-20, -11), (-11, -10), (-20, 18), (-10, 21), (-21, 4), (-10, 6))):
        obj = copy.deepcopy(base["world"]["objects"][index])
        position = place(assets, xy, bounds, occupied)
        occupied.append(position)
        obj.update(
            object_id=f"prospective_v3_person_{index + 1}", label=index + 1, position_m=position
        )
        obj["mesh"]["pose"][-1] = (index % 4) * math.pi / 2
        objects.append(obj)
    # Geometry-only clearance, before rendering/inference. Actual route gate is
    # still mandatory: an AABB/DEM precheck is not a physics execution guarantee.
    route_x, route_z = -27, 25
    for y in np.linspace(-12, 18, 121):
        if route_z < terrain_height(assets, route_x, y) + 3:
            raise ValueError("Insufficient terrain clearance")
        for _, low, high in bounds:
            if (
                low[0] - 1 <= route_x <= high[0] + 1
                and low[1] - 1 <= y <= high[1] + 1
                and low[2] - 1 <= route_z <= high[2] + 1
            ):
                raise ValueError("Route intersects expanded scenery bounds")
    output.mkdir(parents=True, exist_ok=False)
    revised = copy.deepcopy(p)
    for name in p["frozen_files"]:
        data = read(original / name)
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.name == "scenario.json":
            data["world"].update(
                objects=objects,
                initial_position_m=[route_x, -12, route_z],
                max_speed_mps=1.2,
                waypoints=[
                    dict(start_s=30 * i, position_m=[route_x, 18 if i % 2 == 0 else -12, route_z])
                    for i in range(4)
                ],
            )
            data["scenario_id"] = "PROSPECTIVE_V3_" + data["scenario_id"]
        elif destination.name == "intent.json":
            data = IntentContract(
                contract_document([o["object_id"] for o in objects], destination.parent.name)
            ).document()
        save(destination, data)
    for index, (flight, case) in enumerate(revised["cases"].items()):
        case["environment"]["seed"] = 1801 + index
        scenario = load_scenario(
            output / flight / "rapid_survey/scenario.json", case["environment"], assets
        )
        case["lineage"] = split_fingerprint(scenario, scenario.scenery_sha256)
        case["split"] = split_check(excluded, case["lineage"])
        case["geometry_route_altitude_m"] = route_z
    # Old protocol revisions describe old interventions; preserve them in the
    # linked historical protocol instead of mislabelling this fresh design.
    for name in ("cohort_design", "route_revision", "physics_alignment"):
        revised.pop(name, None)
    revised.update(
        controller_revision="confirmation_v3",
        policies=list(POLICIES),
        active_cpu_power_sensitivity_w=[15],
        frozen_files={name: sha(output / name) for name in p["frozen_files"]},
        v3_source_sha256={file.name: sha(file) for file in Path(__file__).parent.glob("*.py")},
        asset_manifest_sha256=sha(assets / "manifest.json"),
        excluded_lineage=excluded,
        inspected_prior_studies=[
            dict(root=str(original.resolve()), protocol_sha256=sha(original / "protocol.json"))
        ],
        cohort_design=dict(
            type="prospective_controller_revision_known_corridor",
            paired_geometry=True,
            independent_layout_count=1,
            horizon_s=8,
            training_changed=False,
            quality_semantics="Development quality as a heuristic independent support parameter",
            new_inference_required=True,
            outcome_driven_selection=False,
        ),
        caveats=[
            "One new nearby layout/route, two paired lighting flights; same known lake map",
            "Not independent scene replication, population evidence or a policy-win guarantee",
            "Independent-support lookahead is a heuristic, not calibrated mission probability",
            "Configured energy/memory/load/tracking costs; measured prior CPU model latency",
            "Only 15 W primary assumption in this cohort; no new power sensitivity claim",
            "Authored static-person missions, ideal pose/DEM, not real UAV mission logs",
        ],
    )
    save(output / "protocol.json", revised)
    verify(output)
    print(
        "Frozen 22 policies, 3 intents and paired fresh flights; no outcomes inspected", flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "record", "infer"))
    parser.add_argument("--original", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.original, args.output)
    elif args.mode == "record":
        from experiments.mission_routing_v2.flights import record

        verify(args.output)
        record(args.output, args.flight)
    else:
        from experiments.mission_routing_v2.infer import run
        from experiments.mission_routing_v2_route_validation import audit

        verify(args.output)
        if not audit(args.output)["all_passed"]:
            raise ValueError("Physical route gate failed")
        run(args.output, args.flight)


if __name__ == "__main__":
    main()
