"""Freeze two Depot layouts before physical capture and all model inference."""

import argparse
import copy
import math
from pathlib import Path

from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.train import verify as verify_training
from experiments.mission_environment_transfer.assets import verify as verify_assets
from experiments.mission_environment_transfer.backend import LiveBackend
from experiments.mission_environment_transfer.geometry import route_clear, standing_clear
from experiments.mission_environment_transfer.identity import scene, training_inventory
from experiments.mission_environment_transfer.scenario import load_scenario
from experiments.mission_routing_v2.flights import contract_document, split_check
from experiments.mission_routing_v2.flights import verify as verify_base
from experiments.mission_transfer_identity import assert_disjoint
from experiments.v4_benchmark.backends import record_world

MISSIONS = ("rapid_survey", "verified_inspection")
DESIGNS = {
    "west_bay": dict(
        start=[-13.0, -1.5, 5.0],
        end=[-1.0, -1.5, 5.0],
        targets=[
            [-12.0, -2.5],
            [-10.0, -0.5],
            [-4.0, -2.5],
            [-2.0, -0.5],
            [-8.0, -2.5],
            [-6.0, -0.5],
        ],
    ),
    "east_bay": dict(
        start=[0.0, -1.0, 5.0],
        end=[12.0, -1.0, 5.0],
        targets=[[1.0, -1.0], [3.0, -1.0], [9.0, -1.0], [11.0, -1.0], [5.0, -1.0], [7.0, -1.0]],
    ),
}


def geometry_guard(assets, designs=DESIGNS):
    for group, design in designs.items():
        if not route_clear(assets, [design["start"], design["end"]]):
            raise ValueError("Unsafe indoor route: " + group)
        if any(not standing_clear(assets, *xy) for xy in design["targets"]):
            raise ValueError("Occupied ground placement: " + group)
        if any(
            math.dist(a, b) < 1.5
            for i, a in enumerate(design["targets"])
            for b in design["targets"][i + 1 :]
        ):
            raise ValueError("Overlapping target clearance")


def prepare(previous, training, assets, output, supersedes=None):
    old = verify_base(previous)
    profile = verify_training(training)
    manifest = verify_assets(assets)
    geometry_guard(assets)
    independent_inventory = training_inventory(training, previous)
    output.mkdir(parents=True, exist_ok=False)
    excluded = [
        *old.get("excluded_lineage", []),
        *profile["training_lineage"],
        *(c["lineage"] for c in old["cases"].values()),
    ]
    # Copy model pool and authored contracts unchanged; only physical geometry changes.
    upstream = next(iter(old["cases"]))
    source = read(previous / upstream / "rapid_survey/scenario.json")
    environments = {}
    for case in old["cases"].values():
        environments.setdefault(case["lighting"], case["environment"])
    cases, frozen = {}, {}
    for group, design in DESIGNS.items():
        objects = []
        for i, xy in enumerate(design["targets"]):
            obj = copy.deepcopy(source["world"]["objects"][i])
            obj.update(
                object_id=f"depot_{group}_person_{i + 1}", label=i + 1, position_m=[*xy, 0.04]
            )
            obj["mesh"]["pose"][-1] = (i % 4) * math.pi / 2
            objects.append(obj)
        for lighting in ("day", "dusk_haze"):
            flight = group + "_" + lighting
            environment = dict(environments[lighting], seed=2601 + len(cases))
            for mission in MISSIONS:
                folder = output / flight / mission
                folder.mkdir(parents=True)
                doc = copy.deepcopy(read(previous / upstream / mission / "scenario.json"))
                doc["scenario_id"] = "DEPOT_TRANSFER_" + flight + "_" + mission
                doc["world"].update(
                    objects=objects,
                    initial_position_m=design["start"],
                    max_speed_mps=0.45,
                    ground_size_m=[100, 100, 0.1],
                    waypoints=[
                        dict(
                            start_s=30 * i,
                            position_m=design["end"] if i % 2 == 0 else design["start"],
                        )
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
                for file in folder.glob("*.json"):
                    frozen[str(file.relative_to(output))] = sha(file)
            scenario = load_scenario(
                output / flight / "rapid_survey/scenario.json", environment, assets
            )
            lineage = split_fingerprint(scenario, scenario.scenery_sha256)
            independent_identity = scene(
                read(output / flight / "rapid_survey/scenario.json"),
                assets,
                "fuel/openrobotics/depot",
            )
            independent_split = assert_disjoint(
                [r["identity"] for r in independent_inventory],
                independent_identity,
                unseen_map=True,
            )
            split_check(excluded, lineage)
            if any(
                c["layout_group"] != group and c["lineage"]["layout"] == lineage["layout"]
                for c in cases.values()
            ):
                raise ValueError("Layouts must differ physically")
            cases[flight] = dict(
                environment=environment,
                lineage=lineage,
                layout_group=group,
                lighting=lighting,
                map_family=manifest["map_family"],
                placement=objects,
                transfer_identity=independent_identity,
                transfer_split=independent_split,
            )
    repo = Path(__file__).resolve().parents[2]
    pinned = dict(old.get("implementation_sha256", {}))
    own = [
        *Path(__file__).parent.glob("*.py"),
        repo / "tests/test_mission_environment_transfer.py",
        repo / "docs/mission_environment_transfer.md",
        repo / "experiments/mission_transfer_identity.py",
    ]
    pinned.update({str(path.resolve()): sha(path) for path in own})
    protocol = {
        k: old[k] for k in ("screen", "calibration_sha256", "costs_sha256", "source_sha256")
    }
    protocol.update(
        schema_version="independent_depot_cohort_v1",
        scope="held_out_map_and_layouts",
        assets=str(assets.resolve()),
        asset_manifest_sha256=sha(assets / "manifest.json"),
        map_family=manifest["map_family"],
        independent_maps=1,
        independent_layout_groups=2,
        physical_recordings=4,
        cases=cases,
        missions=list(MISSIONS),
        frozen_files=frozen,
        implementation_sha256=pinned,
        designs=DESIGNS,
        excluded_lineage=excluded,
        excluded_content_inventory=independent_inventory,
        previous=str(previous.resolve()),
        previous_protocol_sha256=sha(previous / "protocol.json"),
        coverage_training=str(training.resolve()),
        training_completion_sha256=sha(training / "completion.json"),
        sensor_gate=dict(
            min_pixels=16,
            min_captures=4,
            all_targets_required=True,
            all_intents_zero_latency_bound_required=True,
        ),
        physical_route_gate=dict(waypoint_tolerance_m=1.0, all_windows_required=True),
        geometry_gate=dict(
            triangle_aabb_clearance=True,
            drone_radius_m=0.4,
            path_sample_step_m=0.1,
            extra_sampling_inflation_m=0.05,
            static_people_ground_clearance=[0.9, 0.9, 1.85],
        ),
        source_adaptations=[
            "Freeze fans and source model",
            "Retain all original visual geometry including roof",
            "Add collision geometry equal to each original visual mesh and pose",
            "Scale original interior point lights by sunlight / 0.9",
            "Resolve local resource paths",
        ],
        model_outputs_used_for_placement=False,
        real_drone_log_replay=False,
        failed_attempt_policy=(
            "Retain all failed recordings; changed geometry requires a new frozen root"
        ),
        caveats=[
            "One new sourced map and two spatial groups, not population generalization",
            "Indoor authored person missions; no real operational intent source",
            "Altitude, route geometry and scenery differ from lake; no isolated map causal effect",
            "Static people, ideal odometry and perfect planar-floor prior",
            "day/dusk_haze are inherited names; interior illumination is controlled",
            "Historical configured latency/power/memory costs unchanged; not measured deployment",
        ],
    )
    if supersedes is not None:
        if any(supersedes.rglob("recording.json")) or any(supersedes.rglob("inference.json")):
            raise ValueError("This revision declaration only supports unacquired preparations")
        protocol["pre_acquisition_revision"] = dict(
            supersedes=str(supersedes.resolve()),
            protocol_sha256=sha(supersedes / "protocol.json"),
            physical_captures=0,
            model_predictions=0,
            reason="Stricter standing-volume head clearance and independent sensor-pair audit",
            policy_change=False,
            physical_world_change=False,
        )
    save(output / "protocol.json", protocol)
    verify(output)
    return protocol


def verify(root):
    protocol = verify_base(root)
    if (
        protocol["schema_version"] != "independent_depot_cohort_v1"
        or protocol["designs"] != DESIGNS
    ):
        raise ValueError("Unexpected independent-map protocol")
    for file, expected in protocol["implementation_sha256"].items():
        if sha(Path(file)) != expected:
            raise ValueError("Frozen source changed: " + file)
    for file, expected in (
        (Path(protocol["assets"]) / "manifest.json", protocol["asset_manifest_sha256"]),
        (Path(protocol["previous"]) / "protocol.json", protocol["previous_protocol_sha256"]),
        (
            Path(protocol["coverage_training"]) / "completion.json",
            protocol["training_completion_sha256"],
        ),
    ):
        if sha(file) != expected:
            raise ValueError("Frozen input changed")
    verify_assets(Path(protocol["assets"]))
    for row in protocol["excluded_content_inventory"]:
        if (
            sha(Path(row["source"])) != row["source_sha256"]
            or sha(Path(row["asset_manifest"])) != row["asset_manifest_sha256"]
        ):
            raise ValueError("Changed excluded physical identity source")
    for flight, case in protocol["cases"].items():
        for mission in protocol["missions"]:
            scenario = load_scenario(
                root / flight / mission / "scenario.json",
                case["environment"],
                Path(protocol["assets"]),
            )
            if split_fingerprint(scenario, scenario.scenery_sha256) != case["lineage"]:
                raise ValueError("Changed recording identity")
        identity = scene(
            read(root / flight / "rapid_survey/scenario.json"),
            Path(protocol["assets"]),
            "fuel/openrobotics/depot",
        )
        if identity != case["transfer_identity"]:
            raise ValueError("Changed independent geometry identity")
        if (
            assert_disjoint(
                [r["identity"] for r in protocol["excluded_content_inventory"]],
                identity,
                unseen_map=True,
            )
            != case["transfer_split"]
        ):
            raise ValueError("Changed map transfer split")
    return protocol


def record(root, flight):
    p = verify(root)
    case = p["cases"][flight]
    folder = root / flight
    scenario = load_scenario(
        folder / "rapid_survey/scenario.json", case["environment"], Path(p["assets"])
    )
    if (folder / "recording").exists() or (folder / "live").exists():
        raise FileExistsError("Preserve every physical acquisition attempt")
    backend = LiveBackend(scenario, folder / "live")
    try:
        result = record_world(scenario, backend, folder / "recording")
    finally:
        backend.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "record"))
    for name in ("output", "previous", "training", "assets"):
        parser.add_argument("--" + name, type=Path, required=name == "output")
    parser.add_argument("--flight")
    parser.add_argument("--supersedes", type=Path)
    a = parser.parse_args()
    if a.mode == "prepare":
        prepare(a.previous, a.training, a.assets, a.output, a.supersedes)
    else:
        record(a.output, a.flight)
