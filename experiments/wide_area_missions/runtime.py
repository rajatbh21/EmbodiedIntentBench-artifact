"""Own a headless Gazebo acquisition of one predeclared continuous trajectory.

This is acquisition, not model inference or router evaluation. The original Lake
sources and old recordings remain untouched. The worker uses system Python for
Gazebo's bindings; callers can use the project's normal Python environment.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import queue
import signal
import subprocess
import threading
from contextlib import suppress
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


class Trajectory:
    """Cubic Hermite position and analytic velocity on a fixed simulation clock.

    Explicit velocity samples are optional. Otherwise, centered secant tangents
    are used inside the route and endpoint velocities are zero. Positions and
    velocities are continuous across knots; no body pose is ever assigned.
    """

    def __init__(self, points: list[dict]):
        if len(points) < 2:
            raise ValueError("A continuous trajectory needs at least two samples")
        self.times = [float(point["time_s"]) for point in points]
        self.positions = [tuple(map(float, point["position_m"])) for point in points]
        if self.times[0] != 0 or any(
            not math.isfinite(t) or (i and t <= self.times[i - 1])
            for i, t in enumerate(self.times)
        ):
            raise ValueError("Trajectory times must start at zero and strictly increase")
        if any(len(p) != 3 or not all(map(math.isfinite, p)) for p in self.positions):
            raise ValueError("Trajectory positions must be finite XYZ triples")
        self.velocities = []
        for i, point in enumerate(points):
            if "velocity_mps" in point:
                velocity = tuple(map(float, point["velocity_mps"]))
            elif i in (0, len(points) - 1):
                velocity = (0.0, 0.0, 0.0)
            else:
                dt = self.times[i + 1] - self.times[i - 1]
                velocity = tuple(
                    (b - a) / dt for a, b in zip(self.positions[i - 1], self.positions[i + 1])
                )
            if len(velocity) != 3 or not all(map(math.isfinite, velocity)):
                raise ValueError("Trajectory velocities must be finite XYZ triples")
            self.velocities.append(velocity)

    @property
    def duration_s(self) -> float:
        return self.times[-1]

    def sample(self, time_s: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if not math.isfinite(time_s):
            raise ValueError("Trajectory sample time must be finite")
        if time_s <= 0:
            return self.positions[0], self.velocities[0] if time_s == 0 else (0.0, 0.0, 0.0)
        if time_s >= self.times[-1]:
            return self.positions[-1], (0.0, 0.0, 0.0)
        i = bisect.bisect_right(self.times, time_s) - 1
        dt = self.times[i + 1] - self.times[i]
        u = (time_s - self.times[i]) / dt
        h = (2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u,
             -2 * u**3 + 3 * u**2, u**3 - u**2)
        dh = ((6 * u**2 - 6 * u) / dt, 3 * u**2 - 4 * u + 1,
              (-6 * u**2 + 6 * u) / dt, 3 * u**2 - 2 * u)
        p0, p1 = self.positions[i], self.positions[i + 1]
        v0, v1 = self.velocities[i], self.velocities[i + 1]
        position = tuple(h[0]*a + h[1]*dt*va + h[2]*b + h[3]*dt*vb
                         for a, va, b, vb in zip(p0, v0, p1, v1))
        velocity = tuple(dh[0]*a + dh[1]*va + dh[2]*b + dh[3]*vb
                         for a, va, b, vb in zip(p0, v0, p1, v1))
        return position, velocity


def normalize_specification(source: Path | str | dict, interval_s: float | None = None) -> dict:
    base_directory = Path.cwd() if isinstance(source, dict) else Path(source).resolve().parent
    doc = json.loads(Path(source).read_text()) if not isinstance(source, dict) else json.loads(json.dumps(source))
    world = doc["world"]
    for item in world["objects"]:
        mesh = item.get("mesh")
        if mesh and "asset" not in mesh:
            from experiments.v4_gazebo_pilot.assets import load_asset

            mesh["asset"] = load_asset(base_directory / mesh["manifest"])
    if "trajectory" not in world and "trajectory" in doc:
        world["trajectory"] = doc["trajectory"]
    trajectory = Trajectory(world["trajectory"])
    world.setdefault("initial_position_m", list(trajectory.positions[0]))
    if math.dist(world["initial_position_m"], trajectory.positions[0]) > 1e-6:
        raise ValueError("Initial body position must match the declared trajectory origin")
    doc.setdefault("duration_s", trajectory.duration_s)
    doc.setdefault("warmup_s", 2.0)
    if interval_s is not None:
        doc["capture_interval_s"] = interval_s
    doc.setdefault("capture_interval_s", 2.0)
    doc.setdefault("controller", {})
    controller = doc["controller"]
    controller.setdefault("position_gain", 0.8)
    controller.setdefault("max_acceleration_mps2", 1.5)
    controller.setdefault("tracking_tolerance_m", 5.0)
    controller.setdefault("minimum_clearance_m", 3.0)
    world.setdefault("max_speed_mps", 4.0)
    controller.setdefault("command_speed_limit_mps", min(10.0, world["max_speed_mps"] + 1.0))
    for name in ("duration_s", "warmup_s", "capture_interval_s"):
        value = float(doc[name])
        if not math.isfinite(value) or value <= 0 or abs(value * 100 - round(value * 100)) > 1e-6:
            raise ValueError(f"{name} must be positive and align with the 10 ms physics grid")
    if doc["duration_s"] > trajectory.duration_s + 1e-9:
        raise ValueError("Acquisition duration exceeds the declared trajectory")
    # The frozen Lake camera publishes on the 500 ms grid. Reject impossible joins.
    for name in ("warmup_s", "capture_interval_s"):
        if abs(doc[name] * 2 - round(doc[name] * 2)) > 1e-6:
            raise ValueError(f"{name} must align with the 2 Hz camera clock")
    if not 0 < world["max_speed_mps"] <= 10:
        raise ValueError("Maximum commanded speed must lie in (0, 10] m/s")
    for name in ("position_gain", "max_acceleration_mps2", "tracking_tolerance_m", "minimum_clearance_m", "command_speed_limit_mps"):
        if not math.isfinite(controller[name]) or controller[name] <= 0:
            raise ValueError(f"Invalid controller setting: {name}")
    if controller["max_acceleration_mps2"] > 2:
        raise ValueError("Acceleration limit exceeds the physical controller's 2 m/s² setting")
    if controller["command_speed_limit_mps"] > 10:
        raise ValueError("Command speed limit must not exceed 10 m/s")
    assets = Path(doc["asset_directory"]).resolve()
    if not (assets / "manifest.json").is_file():
        raise FileNotFoundError("Missing verified Lake asset manifest")
    doc["asset_directory"] = str(assets)
    if not isinstance(doc["environment"], dict):
        raise ValueError("Environment must be an explicit physical environment object")
    return doc


class AcquisitionWorker:
    """Bounded JSON exchange with one isolated, owned Gazebo process group."""

    def __init__(self, specification: dict, output: Path, *, python: str):
        self.output = output
        self.timeout_s = 150
        write_json(output / "acquisition_specification.json", specification)
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONNOUSERSITE"] = "1"
        self.log = (output / "worker.log").open("w")
        self.process = subprocess.Popen(
            [python, "-m", "experiments.wide_area_missions.worker", "--spec",
             str(output / "acquisition_specification.json"), "--output", str(output)],
            cwd=Path(__file__).resolve().parents[2], env=environment,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, start_new_session=True,
        )
        self.responses: queue.Queue[str] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.ready = self.response()
        except BaseException:
            self.close()
            raise

    def _read(self):
        for line in self.process.stdout:
            self.responses.put(line)
        self.responses.put("")

    def response(self) -> dict:
        try:
            line = self.responses.get(timeout=self.timeout_s)
        except queue.Empty:
            raise TimeoutError("Wide-area Gazebo worker timed out; see worker.log and gazebo.log") from None
        if not line:
            raise RuntimeError("Wide-area Gazebo worker exited; see worker.log and gazebo.log")
        data = json.loads(line)
        if not data.get("ok"):
            raise RuntimeError(data.get("error", "Acquisition worker failed"))
        return data

    def request(self, data: dict) -> dict:
        self.process.stdin.write(json.dumps(data) + "\n")
        self.process.stdin.flush()
        return self.response()

    def close(self):
        if self.process.poll() is None:
            with suppress(BrokenPipeError):
                self.process.stdin.close()
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=15)
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        self.log.close()


def validate_tracking(specification: dict, trace: list[dict]) -> dict:
    """Check actual body poses, never semantic labels or model predictions."""
    from experiments.lake_missions.placement import collision_bounds, terrain_height

    assets = Path(specification["asset_directory"])
    bounds = collision_bounds(assets)
    bounds += [(entry["name"], entry["low"], entry["high"])
               for entry in specification.get("scenery", {}).get("collision_bounds", [])]
    controller = specification["controller"]
    observed = [row for row in trace if row["route_time_s"] >= 0]
    if not observed:
        raise ValueError("No actual route pose trace was recorded")
    clearances, out_of_bounds, collisions = [], [], []
    for row in observed:
        p = row["position_m"]
        try:
            clearances.append(p[2] - terrain_height(assets, p[0], p[1]))
        except ValueError:
            out_of_bounds.append(row["route_time_s"])
        for name, low, high in bounds:
            if all(a - 0.5 <= value <= b + 0.5 for value, a, b in zip(p, low, high)):
                collisions.append({"time_s": row["route_time_s"], "obstacle": name})
    errors = sorted(row["tracking_error_m"] for row in observed)
    passed = (
        errors[-1] <= controller["tracking_tolerance_m"]
        and bool(clearances)
        and min(clearances) >= controller["minimum_clearance_m"]
        and not collisions and not out_of_bounds
    )
    return {
        "schema_version": "wide_area_actual_route_validation_v1",
        "passed": passed, "actual_physics_poses": len(observed),
        "tracking_error_max_m": errors[-1],
        "tracking_error_p95_m": errors[min(len(errors)-1, math.ceil(.95*len(errors))-1)],
        "tracking_tolerance_m": controller["tracking_tolerance_m"],
        "minimum_observed_terrain_clearance_m": min(clearances) if clearances else None,
        "required_terrain_clearance_m": controller["minimum_clearance_m"],
        "conservative_scenery_aabb_intersections": collisions,
        "outside_terrain_times_s": out_of_bounds,
        "uses_semantic_ground_truth": False, "uses_model_predictions": False,
        "scope": "10 Hz actual pose samples; conservative scenery bounds; not continuous collision proof",
    }


def capture(scenario_path_or_dict: Path | str | dict, output: Path | str, *,
            duration_s: float | None = None, interval_s: float | None = None,
            python: str = "/usr/bin/python3") -> dict:
    """Record one fixed physical flight; preserve failed outputs for diagnosis.

    Capture times are warmup + [0, interval, ..., duration), matching legacy
    fixed-flight replay indexing. A shorter duration is explicitly a smoke test
    of the route prefix, never silently promoted to a complete mission recording.
    """
    specification = normalize_specification(scenario_path_or_dict, interval_s)
    planned_duration = specification["duration_s"]
    if duration_s is not None:
        if not 0 < duration_s <= planned_duration:
            raise ValueError("Smoke-test duration must be within the declared mission")
        specification["duration_s"] = duration_s
        specification = normalize_specification(specification)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), Path(__file__).with_name("worker.py")]
    identity_payload = {
        "specification": specification,
        "asset_manifest_sha256": hashlib.sha256(
            (Path(specification["asset_directory"]) / "manifest.json").read_bytes()).hexdigest(),
        "controller_source_sha256": {str(p.relative_to(Path(__file__).resolve().parents[2])):
                                      hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    }
    worker = None
    frames = []
    try:
        worker = AcquisitionWorker(specification, output, python=python)
        epoch_ns = round(specification["warmup_s"] * 1e9)
        duration_ns = round(specification["duration_s"] * 1e9)
        interval_ns = round(specification["capture_interval_s"] * 1e9)
        for i, offset in enumerate(range(0, duration_ns, interval_ns)):
            timestamp = epoch_ns + offset
            response = worker.request({"op": "capture_at", "time_ns": timestamp, "index": i})
            frame = {key: response[key] for key in ("timestamp_ns", "body_pose", "file", "sha256")}
            if frame["timestamp_ns"] != timestamp or frame["body_pose"]["timestamp_ns"] != timestamp:
                raise ValueError("Worker returned an unsynchronized capture")
            frames.append(frame)
            if i % 10 == 0:
                print(f"Recorded {i + 1} captures at route time {offset / 1e9:.1f} s", flush=True)
        finish = worker.request({"op": "finish", "time_ns": epoch_ns + duration_ns})
        worker.close()
        worker = None
        trace = json.loads((output / finish["trace_file"]).read_text())["poses"]
        validation = validate_tracking(specification, trace)
        write_json(output / "physical_route_validation.json", validation)
        manifest = {
            "recording_schema_version": "0.1",
            "recording_identity": digest(identity_payload),
            "world": specification["world"], "frames": frames,
            "capture_schedule": {"epoch_ns": epoch_ns, "interval_ns": interval_ns,
                                 "end_ns": epoch_ns + duration_ns},
            "provenance": {
                "backend": "live_gazebo_continuous_trajectory",
                "gazebo_version": finish["gazebo_version"],
                "controller": "Hermite trajectory + feed-forward velocity + ideal-odometry position feedback",
                "controller_update_s": .1,
                "physical_pose_teleportation": False,
                "source": identity_payload,
                "full_declared_mission": specification["duration_s"] == planned_duration,
                "declared_duration_s": planned_duration,
                "recorded_duration_s": specification["duration_s"],
                "physical_route_passed": validation["passed"],
                "privacy": "RGB, semantic object IDs and body poses are acquisition/evaluator data; expose only public RGB to routers",
                "no_model_inference_performed": True,
            },
        }
        write_json(output / "recording.json", manifest)
        if not validation["passed"]:
            raise ValueError("Physical route validation failed; preserve this recording and inspect physical_route_validation.json")
        return manifest
    except BaseException as error:
        write_json(output / "acquisition_failure.json", {
            "type": type(error).__name__, "message": str(error),
            "completed_capture_count": len(frames), "partial_frames": frames,
            "promoted_to_valid_mission": False,
        })
        raise
    finally:
        if worker is not None:
            worker.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, help="Explicit prefix-only smoke acquisition")
    parser.add_argument("--interval-s", type=float)
    parser.add_argument("--python", default="/usr/bin/python3")
    args = parser.parse_args()
    capture(args.scenario, args.output, duration_s=args.duration_s,
            interval_s=args.interval_s, python=args.python)


if __name__ == "__main__":
    main()
