"""Predeclared new-flight evaluation on the known textured Gazebo lake map.

Routes and placements are authored from geometry alone, before evaluation RGB or
model outputs exist. These are simulated search/inspection scenarios, not copies
of real mission logs and not an unseen-map evaluation.
"""

from __future__ import annotations

import argparse
import copy
import math
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.backend import LiveBackend
from experiments.lake_missions.placement import collision_bounds, place, terrain_height
from experiments.lake_missions.scenario import load
from experiments.lake_rotation_evaluation import scenario_document
from experiments.mission_routing_v2.models import CONTROL
from experiments.v4_benchmark.backends import record_world


def split_check(training, evaluation):
    required = {"recording", "world", "scenery", "layout", "route"}
    if not training or any(set(x) != required for x in [*training, evaluation]):
        raise ValueError("Complete lineage required")
    overlap = [k for k in sorted(required) if any(t[k] == evaluation[k] for t in training)]
    if any(k in overlap for k in ("recording", "world", "layout", "route")):
        raise ValueError("Evaluation flight overlaps training: " + str(overlap))
    return dict(
        scope="held_out_flight_known_map",
        overlaps=overlap,
        unseen_recording=True,
        unseen_layout=True,
        unseen_route=True,
        unseen_map=False,
    )


def load_scenario(path, environment, assets):
    doc = read(path)
    validated = copy.deepcopy(doc)
    control = next(s for s in doc["executor_configs"] if s["config_id"] == CONTROL)
    for s in validated["executor_configs"]:
        s["kind"], s["parameters"] = control["kind"], control["parameters"]
    validated["contract_path"] = str((path.parent / doc["contract_path"]).resolve())
    with tempfile.TemporaryDirectory(dir=path.parent) as folder:
        temp = Path(folder) / "scenario.json"
        save(temp, validated)
        scenario = load(temp, environment, assets)
    return replace(
        scenario,
        source=doc,
        executors=tuple(ExecutorConfigSpec(**s) for s in doc["executor_configs"]),
    )


def contract_document(targets, name):
    inspection = name == "verified_inspection"
    tight = name == "reserve_patrol"
    groups = [targets[:2], targets[2:4], targets[4:6], targets[:2]]
    phases = [
        dict(
            phase_id=f"sector_{i + 1}",
            instruction=instruction,
            start_s=30 * i,
            end_s=30 * (i + 1),
            deadline_s=30 * (i + 1),
            min_hits=3 if inspection else 2,
            min_evidence_span_s=4 if inspection else 2,
            required_recall=1.0 if inspection else 0.5,
            max_false_components=2,
            weight=2 if i == 1 else 1,
            evaluator_target_ids=group,
        )
        for i, (group, instruction) in enumerate(
            zip(
                groups,
                (
                    "Confirm people in the first search sector.",
                    "Prioritize repeated evidence in the second sector.",
                    "Survey the third sector before its deadline.",
                    "Revisit the first sector; obtain fresh evidence.",
                ),
                strict=True,
            )
        )
    ]
    return dict(
        schema_version="lake_intent_v1",
        duration_s=120,
        mission=dict(
            mission_id=name,
            title=name.replace("_", " ").title(),
            user_request=(
                "Inspect every sector with three observations and report people."
                if inspection
                else "Search three sectors, revisit the first, and report confirmed people."
            ),
            initial_battery_frac=0.35 if tight else 0.65,
            min_final_battery_frac=0.2,
            phases=phases,
        ),
        limits=dict(
            deadline_s=130,
            communication_budget_mb=0,
            min_final_battery_frac=0.2,
            privacy_level="local_only",
        ),
        reporting=dict(
            mode="report_required",
            target_scope="all_scenario_targets",
            deadline_s=120,
            min_report_recall=0.75 if inspection else 0.5,
            max_false_reports=2,
            max_duplicate_reports=2,
        ),
        evaluator_target_ids=targets,
    )


def prepare(screen, output):
    from experiments.mission_routing_v2.study import POLICIES

    completion = read(screen / "training_complete.json")
    for name in ("calibration", "costs"):
        if completion[name + "_sha256"] != sha(screen / (name + ".json")):
            raise ValueError("Training artifact changed")
    p = read(screen / "protocol.json")
    source = Path(p["source"])
    base = read(source / "scenario.json")
    suite = read(source / "suite.json")
    assets = Path(suite["assets"])
    bounds = collision_bounds(assets)
    profile, costs = read(screen / "calibration.json"), read(screen / "costs.json")
    specs = tuple(
        replace(
            ExecutorConfigSpec(**s),
            mission_latency_s=costs["latency_s"][s["config_id"]],
            energy_j_per_call=costs["energy_j"][s["config_id"]],
        )
        for s in p["specs"]
    )
    output.mkdir(parents=True, exist_ok=False)
    cases = {}
    for number, (shift, envname) in enumerate(((0, "day"), (18, "dusk_haze"))):
        name = f"flight_{number + 1}_{envname}"
        folder = output / name
        folder.mkdir()
        occupied, objects = [], []
        for index, xy in enumerate(((-17, -6), (-15, -2), (-12, 20), (-5, 18), (6, 7), (8, 0))):
            position = place(assets, (xy[0] + shift, xy[1]), bounds, occupied)
            occupied.append(position)
            obj = copy.deepcopy(base["world"]["objects"][index])
            obj.update(object_id=f"{name}_person_{index + 1}", label=index + 1, position_m=position)
            obj["mesh"]["pose"][-1] = (index % 4) * math.pi / 2
            objects.append(obj)
        targets = [o["object_id"] for o in objects]
        # Resolve a conservative route altitude from scenery only, without labels or predictions.
        xy = [(-25 + shift, -10), (-25 + shift, 20), (shift, 20), (shift, -10)]
        samples = [
            np.array(a) * (1 - t) + np.array(b) * t
            for a, b in zip(xy, xy[1:] + xy[:1], strict=True)
            for t in np.linspace(0, 1, 121)
        ]
        ceiling = max(terrain_height(assets, x, y) + 12 for x, y in samples)
        for x, y in samples:
            for _, low, high in bounds:
                if low[0] - 1 <= x <= high[0] + 1 and low[1] - 1 <= y <= high[1] + 1:
                    ceiling = max(ceiling, high[2] + 2)
        height = math.ceil(max(24 + 4 * number, ceiling))
        initial = [*xy[0], height]
        world = copy.deepcopy(base["world"])
        world.update(
            objects=objects,
            initial_position_m=initial,
            max_speed_mps=1.2,
            waypoints=[
                dict(start_s=30 * i, position_m=[*point, height])
                for i, point in enumerate(xy[1:] + xy[:1])
            ],
        )
        environment = dict(suite["environments"][envname], seed=1701 + number)
        for mission in ("rapid_survey", "verified_inspection", "reserve_patrol"):
            document = contract_document(targets, mission)
            intent = IntentContract(document)
            mission_folder = folder / mission
            mission_folder.mkdir()
            contract = read(source / "contract.json")
            contract.update(deadline_s=130, min_final_battery_frac=0.2, quality_threshold=0)
            save(mission_folder / "contract.json", contract)
            data = scenario_document(base, specs, document["mission"], costs["latency_s"])
            data.update(world=world, scenario_id=f"ROUTING_V2_{name}_{mission}")
            data["mission"].update(duration_s=120, router_latency_s=0.05, router_energy_j=0.75)
            data["mission_requirements"]["confirmed_report_deadline_s"] = 120
            save(mission_folder / "scenario.json", data)
            save(mission_folder / "intent.json", intent.document())
            scenario = load_scenario(mission_folder / "scenario.json", environment, assets)
            lineage = split_fingerprint(scenario, scenario.scenery_sha256)
            status = split_check(profile["training_lineage"], lineage)
        cases[name] = dict(
            environment=environment,
            lineage=lineage,
            split=status,
            geometry_route_altitude_m=height,
            conservative_clearance_m=2,
        )
    files = [p for p in output.rglob("*.json")]
    save(
        output / "protocol.json",
        dict(
            schema_version="mission_routing_new_flights_v2",
            scope="held_out_flight_known_map",
            screen=str(screen.resolve()),
            assets=str(assets.resolve()),
            cases=cases,
            missions=["rapid_survey", "verified_inspection", "reserve_patrol"],
            policies=list(POLICIES),
            active_cpu_power_sensitivity_w=[15, 30, 45],
            calibration_sha256=sha(screen / "calibration.json"),
            costs_sha256=sha(screen / "costs.json"),
            frozen_files={str(p.relative_to(output)): sha(p) for p in files},
            source_sha256={p.name: sha(p) for p in Path(__file__).parent.glob("*.py")},
            caveats=[
                "Same known map; only two flights; no statistical population generalization",
                "Authored simulated missions, not replayed real-world drone logs",
                "Static people, ideal pose and DEM; configured power/memory/load costs",
                "Different layout and lighting between flights confound cross-flight attribution",
            ],
        ),
    )


def verify(output):
    protocol = read(output / "protocol.json")
    for name, expected in protocol["frozen_files"].items():
        if sha(output / name) != expected:
            raise ValueError("Predeclared scenario changed: " + name)
    for name, expected in protocol["source_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("Frozen implementation changed: " + name)
    screen = Path(protocol["screen"])
    for name in ("calibration", "costs"):
        if sha(screen / (name + ".json")) != protocol[name + "_sha256"]:
            raise ValueError("Frozen calibration/cost changed")
    return protocol


def record(output, flight):
    p = verify(output)
    case = p["cases"][flight]
    folder = output / flight
    scenario = load_scenario(
        folder / "rapid_survey/scenario.json", case["environment"], Path(p["assets"])
    )
    if (folder / "recording").exists():
        raise FileExistsError("Preserve previous recording")
    backend = LiveBackend(scenario, folder / "live")
    try:
        result = record_world(scenario, backend, folder / "recording")
    finally:
        backend.close()
    print(f"Recorded {flight}: {len(result['frames'])} physical captures", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "record"))
    parser.add_argument("--screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.screen, args.output)
    else:
        record(args.output, args.flight)
