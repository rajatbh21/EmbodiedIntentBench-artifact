"""Run a policy and actual models inside a live Gazebo loop, then check paired poses.

Standalone live validation of a locked routing_mission experiment. The offline
paired study remains unchanged. No cached predictions are used by this command.
"""

import argparse
import json
import math
from pathlib import Path

from missionbench.v2.executors import build_executors
from experiments.routing_mission.backend import LiveBackend
from experiments.routing_mission.environment import load
from experiments.routing_mission.experiment import freeze, read, router_for, specification
from experiments.routing_mission.inference import checksum
from experiments.v4_benchmark.backends import save_json
from experiments.v4_benchmark.runner import run_mission


def pose_agreement(events, canonical, position_tolerance_m=0.001, quaternion_tolerance=0.0001):
    """Independent physics runs have numerical drift; paired rankings use exact replay."""
    distances, rotations = [], []
    exact = True
    timestamps = True
    for event in events:
        actual = event["capture_body_pose"]
        expected = canonical[event["capture_timestamp_ns"]]
        exact = exact and actual == expected
        timestamps = timestamps and actual["timestamp_ns"] == expected["timestamp_ns"]
        distances.append(math.dist(actual["position_m"], expected["position_m"]))
        a, b = actual["quaternion_xyzw"], expected["quaternion_xyzw"]
        rotations.append(min(math.dist(a, b), math.dist(a, [-v for v in b])))
    valid = bool(events) and timestamps and all(math.isfinite(v) for v in distances + rotations)
    position = max(distances, default=math.inf)
    rotation = max(rotations, default=math.inf)
    return dict(
        exact_pose_equality=exact and bool(events),
        timestamps_exact=timestamps,
        position_tolerance_m=position_tolerance_m,
        quaternion_chord_tolerance=quaternion_tolerance,
        max_position_difference_m=position if math.isfinite(position) else None,
        max_quaternion_chord_difference=rotation if math.isfinite(rotation) else None,
        within_tolerance=valid
        and position <= position_tolerance_m
        and rotation <= quaternion_tolerance,
    )


def run(root, output, router="context", environment="clear"):
    import torch

    spec = specification(root)
    freeze(root)
    torch.set_num_threads(spec["threads"])
    scenario = load(root / "scenario.json", spec["environments"][environment])
    output.mkdir(parents=True, exist_ok=False)
    save_json(
        output / "protocol.json",
        dict(
            source_experiment=str(root.resolve()),
            source_lock_sha256=checksum(root / "lock.json"),
            environment=vars(scenario.environment),
            policy=router,
            implementation_sha256=checksum(Path(__file__)),
            actual_inference=True,
            prediction_cache=False,
            live_gazebo=True,
        ),
    )
    executors = build_executors(scenario.executors)
    backend = LiveBackend(scenario, output / "physics")
    try:
        result = run_mission(
            scenario,
            backend,
            router_for(router, scenario, spec["seed"]),
            executors,
            router_name=router,
            output=output / "mission",
        )
    finally:
        backend.close()
    canonical = {
        r["timestamp_ns"]: r["body_pose"]
        for r in read(root / environment / "recording/recording.json")["frames"]
    }
    events = read(output / "mission/events.json")
    agreement = pose_agreement(events, canonical)
    save_json(
        output / "live_validation.json",
        dict(
            status=result["status"],
            mission_success=result["mission_success"],
            processed_observations=len(events),
            router_errors=result["router_errors"],
            inference_failures=result["inference_failures"],
            pose_agreement=agreement,
            physical_environment=vars(scenario.environment),
            scope="live selected-model inference and physical stepping; costs remain configured",
        ),
    )
    if (
        result["status"] != "completed"
        or result["router_errors"]
        or result["inference_failures"]
        or not agreement["within_tolerance"]
    ):
        raise RuntimeError("live execution or fixed-flight equivalence failed; inspect outputs")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--router", default="context")
    p.add_argument("--environment", default="clear")
    args = p.parse_args()
    result = run(args.experiment, args.output, args.router, args.environment)
    print(json.dumps(dict(status=result["status"], mission_success=result["mission_success"])))
