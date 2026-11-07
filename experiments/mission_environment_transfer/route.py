"""Pre-inference physical-route and sampled body-clearance checks."""

import math
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_environment_transfer.cohort import verify
from experiments.mission_environment_transfer.geometry import clear_box
from experiments.mission_environment_transfer.scenario import load_scenario


def calculate(root):
    p = verify(root)
    tolerance = p["physical_route_gate"]["waypoint_tolerance_m"]
    rows = []
    for flight, case in p["cases"].items():
        scenario = load_scenario(
            root / flight / "rapid_survey/scenario.json", case["environment"], Path(p["assets"])
        )
        path = root / flight / "recording/recording.json"
        frames = read(path)["frames"]
        waypoints = scenario.world["waypoints"]
        stages = []
        for i, waypoint in enumerate(waypoints):
            end = waypoints[i + 1]["start_s"] if i + 1 < len(waypoints) else scenario.duration_s
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
                    waypoint_index=i,
                    start_s=waypoint["start_s"],
                    end_s=end,
                    closest_distance_m=distance,
                    reached=distance <= tolerance,
                )
            )
        clear = []
        for frame in frames:
            point = np.asarray(frame["body_pose"]["position_m"])
            clear.append(clear_box(Path(p["assets"]), point - 0.35, point + 0.35))
        rows.append(
            dict(
                flight=flight,
                stages=stages,
                sampled_body_clear=all(clear),
                passed=all(s["reached"] for s in stages) and all(clear),
                recording_sha256=sha(path),
            )
        )
    result = dict(
        schema_version="depot_physical_route_gate_v1",
        tolerance_m=tolerance,
        flights=rows,
        all_passed=all(r["passed"] for r in rows),
        sampled_capture_clearance_radius_m=0.35,
        continuous_contact_free_proof=False,
        model_predictions_used=False,
        ground_truth_labels_used=False,
        implementation_sha256=sha(Path(__file__)),
    )
    return result


def audit(root):
    destination = root / "physical_route_validation.json"
    if destination.exists():
        raise FileExistsError("Preserve physical-route gate decision")
    result = calculate(root)
    save(destination, result)
    return result


def verify_route(root):
    actual = calculate(root)
    if actual != read(root / "physical_route_validation.json") or not actual["all_passed"]:
        raise ValueError("Physical route audit differs from recorded poses or failed")
    return actual
