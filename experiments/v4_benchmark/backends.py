"""Replaceable physical observation source: live Gazebo or dense fixed-flight replay.

Recorded flight is valid because navigation is fixed and router actions only choose
inference. Replay selects a different subset of captures when latency changes. It
is invalid for policies that steer the drone or change physical flight dynamics.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import subprocess
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from experiments.v4_benchmark.scenario import Scenario


@dataclass(frozen=True)
class Frame:
    timestamp_ns: int
    rgb: np.ndarray
    object_ids: np.ndarray
    body_pose: dict[str, Any]


class WorldBackend(Protocol):
    time_ns: int
    provenance: dict[str, Any]

    def advance_to(self, time_ns: int) -> None: ...
    def capture(self) -> Frame: ...
    def close(self) -> None: ...


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def load_frame(path: Path, metadata: dict) -> Frame:
    with np.load(path, allow_pickle=False) as arrays:
        rgb, gt = arrays["rgb"].copy(), arrays["object_ids"].copy()
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or gt.shape != rgb.shape[:2]:
        raise ValueError("invalid RGB/GT recording shape or RGB dtype")
    if not np.issubdtype(gt.dtype, np.integer) or (gt < 0).any():
        raise ValueError("GT must contain nonnegative integer object IDs")
    timestamp = metadata["timestamp_ns"]
    if metadata["body_pose"]["timestamp_ns"] != timestamp:
        raise ValueError("capture and pose timestamps differ")
    return Frame(timestamp, rgb, gt, metadata["body_pose"])


class LiveBackend:
    """Owned system-Python worker. JSON timeout bounds failures, not simulated time."""

    def __init__(self, scenario: Scenario, output: Path, *, python: str = "/usr/bin/python3"):
        self.output = output.resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        spec = self.output.parent / f"{self.output.name}_worker_spec.json"
        save_json(spec, {"world": scenario.world, "warmup_s": scenario.warmup_s})
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        self.log = self.output.with_suffix(".worker.log").open("w")
        self.process = subprocess.Popen(
            [
                python,
                "-m",
                "experiments.v4_gazebo_pilot.worker",
                "--spec",
                str(spec),
                "--output",
                str(self.output),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.responses: queue.Queue[str] = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            ready = self._response()
            self.time_ns = ready["time_ns"]
            self.provenance = {
                "backend": "live_gazebo",
                "gazebo_version": ready["gazebo_version"],
                "recording_identity": scenario.recording_identity,
            }
        except BaseException:
            self.close()
            raise

    def _read(self) -> None:
        for line in self.process.stdout:
            self.responses.put(line)
        self.responses.put("")

    def _response(self) -> dict:
        try:
            line = self.responses.get(timeout=45)
        except queue.Empty:
            raise TimeoutError("Gazebo worker did not answer within 45 host seconds") from None
        if not line:
            raise RuntimeError(
                f"Gazebo worker exited; see {self.output.with_suffix('.worker.log')}"
            )
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(result.get("error", "worker failed"))
        return result

    def _request(self, data: dict) -> dict:
        self.process.stdin.write(json.dumps(data) + "\n")
        self.process.stdin.flush()
        return self._response()

    def advance_to(self, time_ns: int) -> None:
        if time_ns < self.time_ns:
            raise ValueError("cannot reverse simulation time")
        response = self._request({"op": "advance", "time_ns": time_ns})
        if response["time_ns"] != time_ns:
            raise RuntimeError("Gazebo worker returned the wrong simulation time")
        self.time_ns = time_ns

    def capture(self) -> Frame:
        data = self._request({"op": "capture"})
        if data["timestamp_ns"] != self.time_ns:
            raise ValueError("worker returned stale capture")
        return load_frame(self.output / data["file"], data)

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            with suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=15)
        # Also clean surviving children when the worker itself crashed or timed out.
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        self.log.close()


def record_world(scenario: Scenario, backend: WorldBackend, output: Path) -> dict:
    """Save every eligible capture, never a policy-dependent sparse pilot run."""
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    index = 0
    while scenario.schedule.at(index) < scenario.schedule.end_ns:
        timestamp = scenario.schedule.at(index)
        backend.advance_to(timestamp)
        frame = backend.capture()
        if frame.timestamp_ns != timestamp:
            raise ValueError("backend returned a stale capture")
        path = output / f"frame_{index:05d}.npz"
        np.savez_compressed(path, rgb=frame.rgb, object_ids=frame.object_ids)
        frames.append(
            {
                "timestamp_ns": timestamp,
                "body_pose": frame.body_pose,
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        index += 1
    manifest = {
        "recording_schema_version": "0.1",
        "recording_identity": scenario.recording_identity,
        "world": scenario.world,
        "provenance": backend.provenance,
        "frames": frames,
    }
    save_json(output / "recording.json", manifest)
    return manifest


class ReplayBackend:
    """Exact-time dense replay; incompatible world/time settings fail before a run."""

    def __init__(self, scenario: Scenario, directory: Path):
        self.directory = directory.resolve()
        data = json.loads((directory / "recording.json").read_text())
        if data.get("recording_schema_version") != "0.1":
            raise ValueError("unsupported V4 recording version")
        if data["recording_identity"] != scenario.recording_identity:
            raise ValueError(
                "recording world/capture schedule differs from this scenario; record again"
            )
        frames = data["frames"]
        expected = list(
            range(
                scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns
            )
        )
        if [f["timestamp_ns"] for f in frames] != expected:
            raise ValueError("recording must cover every capture slot exactly once, in order")
        for frame in frames:
            path = (directory / frame["file"]).resolve()
            if path.parent != self.directory:
                raise ValueError("recorded frame must be inside recording directory")
            if hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
                raise ValueError(f"recording checksum mismatch: {path.name}")
        self.frames = {f["timestamp_ns"]: f for f in frames}
        self.time_ns = scenario.schedule.epoch_ns
        self.provenance = {
            "backend": "dense_fixed_flight_replay",
            "source": data["provenance"],
            "recording_identity": data["recording_identity"],
            "recording_sha256": hashlib.sha256(
                (directory / "recording.json").read_bytes()
            ).hexdigest(),
        }

    def advance_to(self, time_ns: int) -> None:
        if time_ns < self.time_ns:
            raise ValueError("cannot reverse simulation time")
        self.time_ns = time_ns

    def capture(self) -> Frame:
        if self.time_ns not in self.frames:
            raise ValueError(f"no recorded capture at {self.time_ns} ns")
        frame = self.frames[self.time_ns]
        return load_frame(self.directory / frame["file"], frame)

    def close(self) -> None:
        pass
