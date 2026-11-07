"""Freeze three separated known-map layouts before acquisition or model evaluation."""

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
from experiments.mission_coverage_routing.train import verify as verify_training
from experiments.mission_routing_v2.flights import contract_document, load_scenario, split_check
from experiments.mission_routing_v2.flights import verify as verify_base
from experiments.mission_routing_v6.flights import verify as verify_previous
from experiments.routing_development_matrix.cohort import geometry_guard

CENTERS = {"west_ridge": (-180, -90), "central_slope": (-60, -90), "east_ridge": (60, -90)}
OFFSETS = ((8, 2), (14, 4), (8, 28), (14, 30), (8, 15), (14, 18))
MISSIONS = ("rapid_survey", "verified_inspection")


def prepare(previous, training, output):
    if output.exists():
        raise FileExistsError("Preserve all acquisition attempts")
    old, profile = verify_previous(previous), verify_training(training)
    assets = Path(old["assets"])
    bounds = collision_bounds(assets)
    base = read(previous / "flight_1_day/rapid_survey/scenario.json")
    excluded = [
        *old["excluded_lineage"],
        *(c["lineage"] for c in old["cases"].values()),
        *profile["training_lineage"],
    ]
    output.mkdir(parents=True)
    cases, frozen, designs = {}, {}, {}
    for group, (x, y) in CENTERS.items():
        occupied, objects = [], []
        for i, (dx, dy) in enumerate(OFFSETS):
            position = place(assets, (x + dx, y + dy), bounds, occupied)
            occupied.append(position)
            obj = copy.deepcopy(base["world"]["objects"][i])
            obj.update(
                object_id=f"coverage_{group}_person_{i + 1}", label=i + 1, position_m=position
            )
            obj["mesh"]["pose"][-1] = (i % 4) * math.pi / 2
            objects.append(obj)
        z = round(
            max(
                [terrain_height(assets, x, yy) for yy in np.linspace(y, y + 30, 121)]
                + [p[2] for p in occupied]
            )
            + 15,
            2,
        )
        design = dict(x=x, y=y, z=z, targets=[(x + dx, y + dy) for dx, dy in OFFSETS])
        geometry_guard(assets, bounds, design)
        designs[group] = design
        for lighting, upstream in (("day", "flight_1_day"), ("dusk_haze", "flight_2_dusk_haze")):
            flight = group + "_" + lighting
            environment = dict(old["cases"][upstream]["environment"], seed=2501 + len(cases))
            for mission in MISSIONS:
                folder = output / flight / mission
                folder.mkdir(parents=True)
                doc = copy.deepcopy(read(previous / upstream / mission / "scenario.json"))
                doc["scenario_id"] = "COVERAGE_PROSPECTIVE_" + flight + "_" + mission
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
                save(
                    folder / "contract.json", read(previous / upstream / mission / "contract.json")
                )
                save(
                    folder / "intent.json",
                    IntentContract(
                        contract_document([o["object_id"] for o in objects], mission)
                    ).document(),
                )
                for f in folder.glob("*.json"):
                    frozen[str(f.relative_to(output))] = sha(f)
            scenario = load_scenario(
                output / flight / "rapid_survey/scenario.json", environment, assets
            )
            lineage = split_fingerprint(scenario, scenario.scenery_sha256)
            split_check(excluded, lineage)
            for c in cases.values():
                if c["layout_group"] != group:
                    split_check([c["lineage"]], lineage)
            cases[flight] = dict(
                environment=environment,
                lineage=lineage,
                placement=objects,
                layout_group=group,
                lighting=lighting,
            )
    repo = Path(__file__).resolve().parents[2]
    sources = [
        f for tree in ("experiments", "missionbench") for f in (repo / tree).rglob("*.py")
    ]
    sources += [
        repo / "tests/test_mission_coverage_routing.py",
        repo / "docs/mission_coverage_routing.md",
    ]
    p = {
        k: old[k]
        for k in ("assets", "screen", "calibration_sha256", "costs_sha256", "source_sha256")
    }
    p.update(
        schema_version="coverage_prospective_cohort_v1",
        scope="fresh_layouts_known_map",
        cases=cases,
        missions=list(MISSIONS),
        frozen_files=frozen,
        designs=designs,
        previous=str(previous.resolve()),
        previous_protocol_sha256=sha(previous / "protocol.json"),
        coverage_training=str(training.resolve()),
        training_completion_sha256=sha(training / "completion.json"),
        implementation_sha256={str(f): sha(f) for f in sources},
        asset_manifest_sha256=sha(assets / "manifest.json"),
        excluded_lineage=excluded,
        sensor_gate=dict(
            min_pixels=16,
            min_captures=4,
            all_targets_required=True,
            all_intents_zero_latency_bound_required=True,
        ),
        failed_attempt_policy=(
            "Retain every failed acquisition; geometry changes require a new frozen root"
        ),
        independent_layout_groups=3,
        physical_recordings=6,
        independent_maps=0,
        model_outputs_used_for_placement=False,
        real_drone_log_replay=False,
    )
    save(output / "protocol.json", p)
    verify(output)
    from experiments.mission_coverage_routing.protocol import prepare as prepare_study

    prepare_study(output, output / "study")
    print("Frozen three new layouts, six physical flights and 504 policy executions", flush=True)


def verify(root):
    p = verify_base(root)
    if p["schema_version"] != "coverage_prospective_cohort_v1" or set(p["designs"]) != set(CENTERS):
        raise ValueError("Wrong cohort")
    for path, expected in p["implementation_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Changed frozen source: " + path)
    for path, expected in (
        (Path(p["previous"]) / "protocol.json", p["previous_protocol_sha256"]),
        (Path(p["coverage_training"]) / "completion.json", p["training_completion_sha256"]),
        (Path(p["assets"]) / "manifest.json", p["asset_manifest_sha256"]),
    ):
        if sha(path) != expected:
            raise ValueError("Changed frozen input")
    verify_training(Path(p["coverage_training"]))
    for f, c in p["cases"].items():
        split_check(p["excluded_lineage"], c["lineage"])
        for m in p["missions"]:
            s = load_scenario(root / f / m / "scenario.json", c["environment"], Path(p["assets"]))
            if split_fingerprint(s, s.scenery_sha256) != c["lineage"]:
                raise ValueError("Changed physical lineage")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "record", "infer"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--flight")
    a = parser.parse_args()
    if a.mode == "prepare":
        prepare(a.previous, a.training, a.output)
    elif a.mode == "record":
        from experiments.mission_routing_v2.flights import record

        verify(a.output)
        record(a.output, a.flight)
    else:
        from experiments.mission_coverage_routing.sensor import verify_gate
        from experiments.mission_routing_v2.infer import run

        verify(a.output)
        verify_gate(a.output)
        run(a.output, a.flight)
