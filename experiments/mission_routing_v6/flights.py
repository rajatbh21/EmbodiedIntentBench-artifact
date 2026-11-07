"""Freeze development-selected controllers and a geometry-authored evaluation cohort."""

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
from experiments.mission_routing_v2.flights import contract_document, load_scenario, split_check
from experiments.mission_routing_v2.flights import verify as verify_base
from experiments.mission_routing_v4.flights import verify as verify_original
from experiments.routing_development_matrix.cohort import geometry_guard
from experiments.routing_development_matrix.train import verify_training

DESIGN = dict(
    x=-100,
    y=-30,
    z=24,
    targets=[(-92, -28), (-86, -26), (-92, -2), (-86, 0), (-92, -15), (-86, -12)],
)


def implementation():
    return {f.name: sha(f) for f in Path(__file__).parent.glob("*.py")}


def dependencies():
    parent = Path(__file__).parents[1]
    files = [*parent.joinpath("mission_routing_v5").glob("*.py")]
    files.extend(
        parent / name
        for name in (
            "routing_development_matrix/observability.py",
            "routing_development_matrix/validate.py",
            "mission_routing_v2_route_validation.py",
        )
    )
    return {str(f.resolve()): sha(f) for f in files}


def prepare(original, training, output):
    from experiments.mission_routing_v6.study import POLICIES

    if output.exists():
        raise FileExistsError("Preserve every prior acquisition attempt")
    old = verify_original(original)
    profile = verify_training(training)
    plan = read(training / "plan.json")
    assets = Path(old["assets"])
    bounds = collision_bounds(assets)
    geometry_guard(assets, bounds, DESIGN)
    base = read(original / "flight_1_day/rapid_survey/scenario.json")
    occupied, objects = [], []
    for index, xy in enumerate(DESIGN["targets"]):
        obj = copy.deepcopy(base["world"]["objects"][index])
        position = place(assets, xy, bounds, occupied)
        occupied.append(position)
        obj.update(
            object_id=f"prospective_v6_person_{index + 1}", label=index + 1, position_m=position
        )
        obj["mesh"]["pose"][-1] = (index % 4) * math.pi / 2
        objects.append(obj)
    excluded = [
        *old["excluded_lineage"],
        *(c["lineage"] for c in old["cases"].values()),
        *profile["training_lineage"],
    ]
    output.mkdir(parents=True)
    cases, frozen = {}, {}
    for index, flight in enumerate(old["cases"]):
        environment = dict(old["cases"][flight]["environment"], seed=2201 + index)
        for mission in old["missions"]:
            folder = output / flight / mission
            folder.mkdir(parents=True)
            doc = read(original / flight / mission / "scenario.json")
            x, y, z = DESIGN["x"], DESIGN["y"], DESIGN["z"]
            doc["scenario_id"] = "PROSPECTIVE_V6_" + flight + "_" + mission
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
            save(folder / "contract.json", read(original / flight / mission / "contract.json"))
            save(
                folder / "intent.json",
                IntentContract(
                    contract_document([o["object_id"] for o in objects], mission)
                ).document(),
            )
            for path in folder.glob("*.json"):
                frozen[str(path.relative_to(output))] = sha(path)
        scenario = load_scenario(
            output / flight / "rapid_survey/scenario.json", environment, assets
        )
        lineage = split_fingerprint(scenario, scenario.scenery_sha256)
        cases[flight] = dict(
            environment=environment,
            lineage=lineage,
            split=split_check(excluded, lineage),
            placement=objects,
        )
    protocol = {
        k: copy.deepcopy(old[k])
        for k in (
            "assets",
            "screen",
            "calibration_sha256",
            "costs_sha256",
            "source_sha256",
            "risk_training",
            "risk_profile_sha256",
            "missions",
        )
    }
    protocol.update(
        schema_version="development_selected_prospective_v6",
        scope="held_out_layout_known_map",
        cases=cases,
        frozen_files=frozen,
        source_v6_sha256=implementation(),
        dependency_sha256=dependencies(),
        original=str(original.resolve()),
        original_protocol_sha256=sha(original / "protocol.json"),
        asset_manifest_sha256=sha(assets / "manifest.json"),
        excluded_lineage=excluded,
        recall_training=str(training.resolve()),
        recall_profile_sha256=sha(training / "profile.json"),
        original_recall_training=plan["original_training"],
        policies=list(POLICIES),
        primary_policy="base_recall",
        primary_selection="Lowest macro leave-one-layout-out log loss on three development blocks",
        selection_diagnostics_sha256=sha(training / "blocked_validation.json"),
        active_cpu_power_sensitivity_w=[15],
        design=DESIGN,
        sensor_gate=dict(
            min_pixels=16,
            min_captures=4,
            all_targets_required=True,
            all_intents_zero_latency_bound_required=True,
            predictions_inspected_before_gate=False,
        ),
        failed_attempt_policy=(
            "Retain failed acquisitions; any geometry redesign gets a new protocol/root"
        ),
        primary_endpoint="Full mission contract success; report every predeclared case",
        secondary_endpoints=["phases_succeeded", "report_recall", "false_reports", "final_battery"],
        evaluation_labels_used_for_fit=False,
        outcome_driven_policy_selection=False,
        caveats=[
            "One authored layout with two lighting conditions on the same known lake",
            "Sensor screening is not proof of paid mission feasibility",
            "Static people and ideal pose/terrain; not replayed real drone missions",
            "Configured energy/memory, previously measured CPU model latency",
            "Unchanged risk profile; conditional recall is not calibrated mission probability",
            "Exploratory controller development preceded this frozen fresh-layout evaluation",
        ],
    )
    save(output / "protocol.json", protocol)
    verify(output)
    print(
        "Frozen 15 policies, six intent/lighting cases, sensor gate and primary selection",
        flush=True,
    )


def verify(output):
    from experiments.mission_routing_v6.study import POLICIES

    p = verify_base(output)
    if p["schema_version"] != "development_selected_prospective_v6":
        raise ValueError("Wrong v6 protocol")
    if p["source_v6_sha256"] != implementation():
        raise ValueError("Frozen v6 implementation changed")
    if p["dependency_sha256"] != dependencies():
        raise ValueError("Frozen inherited implementation changed")
    original, training = Path(p["original"]), Path(p["recall_training"])
    verify_original(original)
    profile = verify_training(training)
    for path, key in (
        (original / "protocol.json", "original_protocol_sha256"),
        (training / "profile.json", "recall_profile_sha256"),
        (training / "blocked_validation.json", "selection_diagnostics_sha256"),
        (Path(p["assets"]) / "manifest.json", "asset_manifest_sha256"),
    ):
        if sha(path) != p[key]:
            raise ValueError("Changed frozen input: " + str(path))
    if p["policies"] != list(POLICIES) or p["primary_policy"] != "base_recall":
        raise ValueError("Changed evaluation selection/grid")
    for flight, c in p["cases"].items():
        split_check(p["excluded_lineage"], c["lineage"])
        split_check(profile["training_lineage"], c["lineage"])
        for mission in p["missions"]:
            scenario = load_scenario(
                output / flight / mission / "scenario.json", c["environment"], Path(p["assets"])
            )
            if split_fingerprint(scenario, scenario.scenery_sha256) != c["lineage"]:
                raise ValueError("Mission changed physical stream or declared lineage")
    return p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "record", "infer"))
    parser.add_argument("--original", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.original, args.training, args.output)
    elif args.mode == "record":
        from experiments.mission_routing_v2.flights import record

        verify(args.output)
        record(args.output, args.flight)
    else:
        from experiments.mission_routing_v2.infer import run
        from experiments.mission_routing_v6.sensor import verify_gate

        verify(args.output)
        verify_gate(args.output)
        run(args.output, args.flight)


if __name__ == "__main__":
    main()
