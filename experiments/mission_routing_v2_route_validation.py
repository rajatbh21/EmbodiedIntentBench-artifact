"""Geometry-only physical-route gate and transparent clearance revision.

This module never uses predictions, outcomes, calibration coefficients or labels.
A route revision changes only altitude, preserving policies, intents, layouts and
costs. Failed recordings and partial inference are retained in their original study.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.flights import load_scenario, split_check, verify

TOLERANCE_M = 3.0
REVISION_ALTITUDE_M = 48


def audit(root):
    p = verify(root)
    rows = []
    for flight, case in p["cases"].items():
        source = root / flight / "rapid_survey/scenario.json"
        scenario = load_scenario(source, case["environment"], Path(p["assets"]))
        recording = root / flight / "recording/recording.json"
        frames = read(recording)["frames"]
        waypoints = scenario.world["waypoints"]
        stages = []
        for index, waypoint in enumerate(waypoints):
            end = (
                waypoints[index + 1]["start_s"]
                if index + 1 < len(waypoints)
                else scenario.duration_s
            )
            observed = [
                f
                for f in frames
                if waypoint["start_s"]
                <= (f["timestamp_ns"] - scenario.schedule.epoch_ns) / 1e9
                < end
            ]
            distance = min(
                math.dist(f["body_pose"]["position_m"], waypoint["position_m"]) for f in observed
            )
            stages.append(
                dict(
                    waypoint_index=index,
                    start_s=waypoint["start_s"],
                    end_s=end,
                    closest_distance_m=distance,
                    reached=distance <= TOLERANCE_M,
                )
            )
        rows.append(
            dict(
                flight=flight,
                passed=all(s["reached"] for s in stages),
                stages=stages,
                recording_sha256=sha(recording),
            )
        )
    result = dict(
        schema_version="physical_route_gate_v1",
        ground_truth_labels_used=False,
        model_predictions_used=False,
        tolerance_m=TOLERANCE_M,
        gate="Each commanded waypoint reached within 3 m during its assigned window",
        all_passed=all(r["passed"] for r in rows),
        flights=rows,
        implementation_sha256=sha(Path(__file__)),
    )
    save(root / "physical_route_validation.json", result)
    print(result, flush=True)
    return result


def prepare(original, output):
    p = verify(original)
    failed = audit(original)
    if failed["all_passed"]:
        raise ValueError("No failed physical-route gate to repair")
    output.mkdir(parents=True, exist_ok=False)
    # Copy declared inputs only. No failed or partial recording/prediction is promoted.
    for name in p["frozen_files"]:
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = read(original / name)
        if destination.name == "scenario.json":
            data["world"]["initial_position_m"][2] = REVISION_ALTITUDE_M
            for waypoint in data["world"]["waypoints"]:
                waypoint["position_m"][2] = REVISION_ALTITUDE_M
        save(destination, data)
    revised = copy.deepcopy(p)
    training_lineage = read(Path(p["screen"]) / "calibration.json")["training_lineage"]
    for flight, case in revised["cases"].items():
        scenario = load_scenario(
            output / flight / "rapid_survey/scenario.json", case["environment"], Path(p["assets"])
        )
        case["lineage"] = split_fingerprint(scenario, scenario.scenery_sha256)
        case["split"] = split_check(training_lineage, case["lineage"])
        case["geometry_route_altitude_m"] = REVISION_ALTITUDE_M
    revised["frozen_files"] = {name: sha(output / name) for name in p["frozen_files"]}
    revised["route_revision"] = dict(
        original_study=str(original.resolve()),
        original_protocol_sha256=sha(original / "protocol.json"),
        original_route_validation_sha256=sha(original / "physical_route_validation.json"),
        reason=(
            "Original physical flights failed waypoint coverage; geometry precheck was insufficient"
        ),
        intervention=(
            "Only absolute route altitude increased to 48 m; verify actual poses before inference"
        ),
        model_quality_used_to_select_change=False,
        policies_retrained=False,
        costs_changed=False,
        intent_thresholds_changed=False,
        layouts_changed=False,
        discarded_failures_preserved=True,
        gate_tolerance_m=TOLERANCE_M,
        reviser_sha256=sha(Path(__file__)),
    )
    save(output / "protocol.json", revised)
    print(
        "Prepared clearance revision; physical route validation must pass before inference",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "prepare"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    args = parser.parse_args()
    if args.mode == "audit":
        audit(args.output)
    else:
        prepare(args.original, args.output)
