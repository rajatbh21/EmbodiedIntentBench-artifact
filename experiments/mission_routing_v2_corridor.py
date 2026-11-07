"""Prospective low-altitude follow-up; preserve the infeasible high-altitude cohort.

Uses the same frozen policies, costs and intent thresholds. A known-clear physical
corridor provides a lower-altitude operating condition, with paired lighting on
identical geometry. No perception model is retrained or selected from test scores.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.placement import collision_bounds, place
from experiments.mission_routing_v2.flights import (
    contract_document,
    load_scenario,
    split_check,
    verify,
)


def prepare(original, stress, output):
    p = verify(original)
    if not (stress / "feasibility_diagnostics.json").exists():
        raise ValueError("Preserve the high-altitude feasibility analysis first")
    screen = Path(p["screen"])
    training_source = Path(read(screen / "protocol.json")["source"])
    training_world = read(training_source / "scenario.json")["world"]
    assets = Path(p["assets"])
    bounds = collision_bounds(assets)
    # Avoid reusing training target locations, independently of model predictions.
    occupied = [o["position_m"] for o in training_world["objects"]]
    objects = []
    for index, xy in enumerate(((-18, -8), (-13, -8), (-18, 17), (-13, 17), (-18, 5), (-13, 5))):
        obj = copy.deepcopy(training_world["objects"][index])
        position = place(assets, xy, bounds, occupied)
        occupied.append(position)
        obj.update(object_id=f"corridor_person_{index + 1}", label=index + 1, position_m=position)
        obj["mesh"]["pose"][-1] = (index % 4) * math.pi / 2
        objects.append(obj)
    output.mkdir(parents=True, exist_ok=False)
    for name in p["frozen_files"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = read(original / name)
        if destination.name == "scenario.json":
            data["world"].update(
                objects=objects,
                initial_position_m=[-25, -10, 24],
                max_speed_mps=1.2,
                waypoints=[
                    dict(start_s=30 * i, position_m=[-25, 20 if i % 2 == 0 else -10, 24])
                    for i in range(4)
                ],
            )
            data["scenario_id"] = "CORRIDOR_" + data["scenario_id"]
        elif destination.name == "intent.json":
            mission = destination.parent.name
            data = IntentContract(
                contract_document([o["object_id"] for o in objects], mission)
            ).document()
        save(destination, data)
    profile = read(screen / "calibration.json")
    revised = copy.deepcopy(p)
    for flight, case in revised["cases"].items():
        scenario = load_scenario(
            output / flight / "rapid_survey/scenario.json", case["environment"], assets
        )
        case["lineage"] = split_fingerprint(scenario, scenario.scenery_sha256)
        case["split"] = split_check(profile["training_lineage"], case["lineage"])
        case["geometry_route_altitude_m"] = 24
    revised["frozen_files"] = {name: sha(output / name) for name in p["frozen_files"]}
    revised["cohort_design"] = dict(
        type="prospective_low_altitude_paired_lighting_followup",
        motivation="High-altitude cohort is perception-infeasible; retain it as a stress cohort",
        stress_study=str(stress.resolve()),
        stress_protocol_sha256=sha(stress / "protocol.json"),
        stress_feasibility_sha256=sha(stress / "feasibility_diagnostics.json"),
        initial_design_protocol_sha256=sha(original / "protocol.json"),
        operating_condition="Same known-clear x=-25 m corridor; four north/south passes at z=24 m",
        paired_geometry=True,
        independent_layout_count=1,
        training_target_positions_reused=False,
        policies_changed=False,
        model_pool_changed=False,
        costs_changed=False,
        intent_thresholds_changed=False,
        designer_sha256=sha(Path(__file__)),
    )
    revised["caveats"] = [
        "Same known lake map; paired lighting on one new layout and route "
        "is not independent scene replication",
        "Follow-up design motivated by prior negative cohort; "
        "not part of the original predeclaration",
        "Authored simulated missions; static people, perfect pose/DEM and configured energy",
        "Policies and costs remain frozen; no result-dependent model or threshold changes",
    ]
    save(output / "protocol.json", revised)
    print(
        "Prepared paired low-altitude cohort; require physical route gate before inference",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.original, args.stress, args.output)
