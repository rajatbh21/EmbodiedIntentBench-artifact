"""System-Python worker for the physical routing mission harness.

Only this process imports Gazebo's Python 3.10 bindings. NumPy arrays cross via
lossless .npz files, never pickle; the evaluator owns the returned GT-bearing paths.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path


def serve(output: Path, specification: dict) -> None:
    import numpy as np

    from experiments.lake_missions.world import build_world

    os.environ["GZ_PARTITION"] = f"missionbench-v4-{uuid.uuid4().hex}"
    from experiments.v4_gazebo_pilot.transport import GazeboBackend

    output.mkdir(parents=True, exist_ok=False)
    world = specification["world"]
    build_world(
        output / "world.sdf",
        environment=specification["environment"],
        asset_directory=specification["asset_directory"],
        objects=world["objects"],
        ground_size_m=world.get("ground_size_m"),
        initial_position_m=world["initial_position_m"],
        camera_pitch_rad=world.get("camera_pitch_rad", math.pi / 2),
    )
    warmup = specification["warmup_s"]
    waypoints = [{"start_s": 0.0, "position_m": world["initial_position_m"]}] + [
        {"start_s": warmup + w["start_s"], "position_m": w["position_m"]}
        for w in world["waypoints"]
    ]
    with (output / "gazebo.log").open("w") as log:
        # Inherit the worker's process group so the coordinator can always clean up
        # both processes, including a stuck or killed worker.
        process = subprocess.Popen(
            [
                "gz",
                "sim",
                "-s",
                "--headless-rendering",
                "--seed",
                str(specification["environment"]["seed"]),
                "-v",
                "2",
                str((output / "world.sdf").resolve()),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        backend = None
        try:
            backend = GazeboBackend(
                timeout_s=90, waypoints=waypoints, max_speed_mps=world.get("max_speed_mps")
            )
            backend.connect()
            backend.advance_to(round(warmup * 1e9))
            reply(
                {
                    "time_ns": backend.clock_ns,
                    "gazebo_version": subprocess.check_output(
                        ["gz", "sim", "--versions"], text=True
                    ).strip(),
                }
            )
            for line in sys.stdin:
                request = json.loads(line)
                if request["op"] == "close":
                    reply({"closed": True})
                    break
                if request["op"] == "advance":
                    backend.advance_to(request["time_ns"])
                    reply({"time_ns": backend.clock_ns})
                elif request["op"] == "capture":
                    capture = backend.capture()
                    filename = f"frame_{capture.timestamp_ns}.npz"
                    np.savez_compressed(
                        output / filename, rgb=capture.rgb, object_ids=capture.object_ids
                    )
                    reply(
                        {
                            "timestamp_ns": capture.timestamp_ns,
                            "body_pose": capture.body_pose,
                            "file": filename,
                        }
                    )
                else:
                    raise ValueError(f"unknown worker operation: {request['op']}")
        finally:
            if backend is not None:
                backend.close()
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def reply(data: dict) -> None:
    print(json.dumps({"ok": True, **data}, allow_nan=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        serve(args.output, json.loads(args.spec.read_text()))
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
