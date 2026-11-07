"""System-Python Gazebo worker for fixed continuous wide-area flight acquisition.

Launch through runtime.capture. Requires local Gazebo transport IPC, rendering
device access and verified local Lake assets. It controls only its own simulator
partition, never physical hardware. EOF and exceptions clean up the owned server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import uuid
from pathlib import Path

from experiments.wide_area_missions.runtime import Trajectory, write_json


def reply(data: dict) -> None:
    print(json.dumps({"ok": True, **data}, allow_nan=False), flush=True)


def serve(output: Path, specification: dict) -> None:
    # Set partition before importing bindings or constructing transport nodes.
    os.environ["GZ_PARTITION"] = f"missionbench-wide-{uuid.uuid4().hex}"
    import numpy as np

    from experiments.lake_missions.world import build_world
    from experiments.v4_gazebo_pilot.transport import GazeboBackend, pose_record

    class ContinuousBackend(GazeboBackend):
        def __init__(self):
            self.trajectory = Trajectory(specification["world"]["trajectory"])
            self.epoch_s = specification["warmup_s"]
            self.settings = specification["controller"]
            self.trace = []
            self.previous_command = (0.0, 0.0, 0.0)
            self.previous_command_ns = 0
            super().__init__(timeout_s=90,
                             max_speed_mps=self.settings["command_speed_limit_mps"])

        def record_pose(self):
            with self.inbox.changed:
                state = self.inbox.streams["pose"].get(self.clock_ns)
            if state is None:
                return None
            pose = pose_record(state)
            route_time = self.clock_ns / 1e9 - self.epoch_s
            target, feed_forward = self.trajectory.sample(route_time)
            row = {**pose, "route_time_s": route_time,
                   "desired_position_m": target, "desired_velocity_mps": feed_forward,
                   "tracking_error_m": math.dist(pose["position_m"], target)}
            if not self.trace or self.trace[-1]["timestamp_ns"] != self.clock_ns:
                self.trace.append(row)
            return state, row

        def _command_velocity(self):
            current = self.record_pose()
            command = self.twist_type()
            if current is not None:
                state, row = current
                position = row["position_m"]
                target, feed_forward = row["desired_position_m"], row["desired_velocity_mps"]
                desired = tuple(v + self.settings["position_gain"] * (goal - actual)
                                for v, goal, actual in zip(feed_forward, target, position))
                norm = math.sqrt(sum(v*v for v in desired))
                speed_scale = min(1.0, self.max_speed_mps / norm) if norm else 1.0
                desired = tuple(speed_scale * v for v in desired)
                dt = (self.clock_ns - self.previous_command_ns) / 1e9
                change = tuple(v - old for v, old in zip(desired, self.previous_command))
                change_norm = math.sqrt(sum(v*v for v in change))
                limit = self.settings["max_acceleration_mps2"] * max(0.0, dt)
                scale = min(1.0, limit / change_norm) if change_norm else 1.0
                velocity = tuple(old + scale * delta for old, delta in zip(self.previous_command, change))
                self.previous_command, self.previous_command_ns = velocity, self.clock_ns
                q = state.pose.orientation
                yaw = math.atan2(2 * (q.w*q.z + q.x*q.y), 1 - 2 * (q.y*q.y + q.z*q.z))
                vx, vy, vz = velocity
                command.linear.x = math.cos(yaw)*vx + math.sin(yaw)*vy
                command.linear.y = -math.sin(yaw)*vx + math.cos(yaw)*vy
                command.linear.z = vz
                command.angular.z = max(-.8, min(.8, -yaw))
                row["command_velocity_world_mps"] = velocity
            if not self.command.publish(command):
                raise RuntimeError("Continuous flight command could not be published")

    world = specification["world"]
    build_world(output / "world.sdf", environment=specification["environment"],
                asset_directory=specification["asset_directory"], objects=world["objects"],
                initial_position_m=world["initial_position_m"],
                camera_pitch_rad=world.get("camera_pitch_rad", math.pi / 2),
                ground_size_m=world.get("ground_size_m"))
    if specification.get("scenery"):
        from experiments.wide_area_missions.scenery import decorate_world

        decorate_world(output / "world.sdf", specification["scenery"])
    backend = None
    with (output / "gazebo.log").open("w") as log:
        process = subprocess.Popen(
            ["gz", "sim", "-s", "--headless-rendering", "--seed",
             str(specification["environment"]["seed"]), "-v", "2", str(output / "world.sdf")],
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            backend = ContinuousBackend()
            backend.connect()
            backend.advance_to(round(specification["warmup_s"] * 1e9))
            version = subprocess.check_output(["gz", "sim", "--versions"], text=True).strip()
            reply({"time_ns": backend.clock_ns, "gazebo_version": version})
            for line in sys.stdin:
                request = json.loads(line)
                operation = request["op"]
                if operation == "capture_at":
                    backend.advance_to(request["time_ns"])
                    frame = backend.capture()
                    filename = f"frame_{request['index']:05d}.npz"
                    path = output / filename
                    np.savez_compressed(path, rgb=frame.rgb, object_ids=frame.object_ids)
                    reply({"timestamp_ns": frame.timestamp_ns, "body_pose": frame.body_pose,
                           "file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                elif operation == "finish":
                    backend.advance_to(request["time_ns"])
                    backend.record_pose()
                    filename = "private_controller_trace.json"
                    write_json(output / filename, {
                        "schema_version": "wide_area_private_controller_trace_v1",
                        "privacy": "Acquisition validation only; do not expose ideal poses to a routing policy",
                        "poses": backend.trace,
                    })
                    reply({"trace_file": filename, "time_ns": backend.clock_ns,
                           "gazebo_version": version})
                else:
                    raise ValueError(f"Unknown worker operation: {operation}")
        finally:
            if backend is not None:
                backend.close()
                if not (output / "private_controller_trace.json").exists():
                    write_json(output / "private_controller_trace.json", {"complete": False, "poses": backend.trace})
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        serve(args.output, json.loads(args.spec.read_text()))
    except BaseException as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
