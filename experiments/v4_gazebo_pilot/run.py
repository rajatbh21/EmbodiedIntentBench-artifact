"""Launch an isolated local Gazebo physics/capture pilot and save reviewable evidence.

This is a timing/sensor diagnostic with configured busy times, NOT a segmentation
benchmark. It produces no policy scores, inference claims, or mission success rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path

from experiments.v4_gazebo_pilot.timing import CaptureSchedule, seconds_to_ns
from experiments.v4_gazebo_pilot.world import HEIGHT, HFOV, STEP_NS, TARGETS, WIDTH, build_world


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(output: Path, duration_s: float, latencies_s: list[float]) -> dict:
    """Own and clean up exactly one server; fail on missing/misaligned sensor data."""
    from PIL import Image

    duration_ns = seconds_to_ns(duration_s)
    latencies_ns = [seconds_to_ns(v, allow_zero=True) for v in latencies_s]
    if not latencies_ns:
        raise ValueError("at least one latency is required")
    # A 1 s hover warmup precedes the diagnostic mission. It is explicitly excluded.
    schedule = CaptureSchedule(1_000_000_000, 500_000_000, 1_000_000_000 + duration_ns, STEP_NS)
    output.mkdir(parents=True, exist_ok=False)
    build_world(output / "world.sdf")
    partition = f"missionbench-v4-{uuid.uuid4().hex}"
    # Set before importing/constructing gz.transport; scope all messages to our server.
    old_partition = os.environ.get("GZ_PARTITION")
    os.environ["GZ_PARTITION"] = partition
    backend = None
    process = None
    try:
        from experiments.v4_gazebo_pilot.transport import GazeboBackend

        with (output / "gazebo.log").open("w") as log:
            process = subprocess.Popen(
                [
                    "gz",
                    "sim",
                    "-s",
                    "--headless-rendering",
                    "--seed",
                    "0",
                    "-v",
                    "3",
                    str((output / "world.sdf").resolve()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            backend = GazeboBackend()
            backend.connect()
            backend.advance_to(schedule.epoch_ns)
            events = []
            index = 0
            while schedule.at(index) < schedule.end_ns:
                backend.advance_to(schedule.at(index))
                capture = backend.capture()
                labels = set(int(v) for v in capture.object_ids.ravel()) - {0}
                if not labels <= TARGETS.keys():
                    raise ValueError(f"unmapped GT labels: {labels - TARGETS.keys()}")
                stem = f"capture_{index:04d}"
                Image.fromarray(capture.rgb).save(output / f"{stem}_rgb.png")
                Image.fromarray(capture.object_ids).save(output / f"{stem}_object_ids.png")
                latency_ns = latencies_ns[len(events) % len(latencies_ns)]
                completion_ns, next_index, skipped = schedule.complete(index, latency_ns)
                backend.advance_to(completion_ns)
                event = {
                    "observation_id": index,
                    "capture_time_ns": capture.timestamp_ns,
                    "configured_busy_time_ns": latency_ns,
                    "completion_time_ns": completion_ns,
                    "effective_busy_time_ns": completion_ns - capture.timestamp_ns,
                    "capture_body_pose": capture.body_pose,
                    "completion_body_pose": backend.pose(),
                    "skipped_observation_ids": list(skipped),
                    "rgb_path": f"{stem}_rgb.png",
                    "object_ids_path": f"{stem}_object_ids.png",
                    "visible_target_ids": [TARGETS[i] for i in sorted(labels)],
                }
                events.append(event)
                # Preserve partial evidence if a subsequent capture fails.
                write_json(output / "events.json", events)
                index = next_index
            backend.advance_to(max(schedule.end_ns, backend.clock_ns))
            result = {
                "pilot_schema_version": "0.1",
                "status": "capture_diagnostic_completed",
                "benchmark_mode": "v4_gazebo_capture_pilot",
                "evaluation_purpose": "integration_diagnostic",
                "physics": "Gazebo rigid-body + rotor thrust",
                "flight_controller": "Gazebo ideal-state velocity + fixed waypoint controller",
                "px4_sitl": False,
                "inference": "not_executed; configured busy-time probe only",
                "targets": "procedural boxes; no human perception claim",
                "energy": "not_evaluated",
                "network": "not_evaluated",
                "gt_encoding": "unique semantic labels as stable object IDs; background=0",
                "target_id_map": TARGETS,
                "world_frame": "right-handed Z-up, metres",
                "body_quaternion_order": "xyzw",
                "camera": {
                    "intrinsics_provenance": "derived from configured SDF pinhole geometry",
                    "width_px": WIDTH,
                    "height_px": HEIGHT,
                    "horizontal_fov_rad": HFOV,
                    "intrinsics": [
                        [WIDTH / (2 * math.tan(HFOV / 2)), 0, WIDTH / 2],
                        [0, WIDTH / (2 * math.tan(HFOV / 2)), HEIGHT / 2],
                        [0, 0, 1],
                    ],
                    "pose_in_body_xyz_rpy": [0, 0, -0.1, 0, math.pi / 2, 0],
                    "camera_frame": "+X forward, +Y left, +Z up (Gazebo sensor frame)",
                },
                "schedule": asdict(schedule),
                "final_time_ns": backend.clock_ns,
                "processed_observations": len(events),
                "skipped_observations": sum(len(e["skipped_observation_ids"]) for e in events),
                "gazebo_version": subprocess.check_output(
                    ["gz", "sim", "--versions"], text=True
                ).strip(),
                "world_sha256": hashlib.sha256((output / "world.sdf").read_bytes()).hexdigest(),
                "reproducibility": "fixed seed/ticks; cross-machine byte equality not claimed",
            }
            write_json(output / "manifest.json", result)
            return result
    except BaseException as error:
        write_json(output / "failure.json", {"status": "failed", "error": str(error)})
        raise
    finally:
        if backend is not None:
            backend.close()
        if process is not None:
            # Only our isolated child process group, never other running simulators.
            try:
                os.killpg(process.pid, signal.SIGINT)
                process.wait(timeout=10)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
        if old_partition is None:
            os.environ.pop("GZ_PARTITION", None)
        else:
            os.environ["GZ_PARTITION"] = old_partition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory")
    parser.add_argument("--duration-s", type=float, default=8)
    parser.add_argument("--latencies-s", nargs="+", type=float, default=[0.2, 0.8])
    args = parser.parse_args()
    result = run(args.output, args.duration_s, args.latencies_s)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
