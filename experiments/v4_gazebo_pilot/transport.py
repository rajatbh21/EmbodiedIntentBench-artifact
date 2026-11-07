"""Gazebo Harmonic bridge: exact-time RGB/GT/pose bundles and paused-world stepping.

Gazebo's CPython bindings stay optional. The host waits for messages, but wall time
never advances the mission. A missing/stale sensor sample is an error, not a fallback.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from experiments.v4_gazebo_pilot.world import STEP_NS, WORLD


def stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nsec)


class SensorInbox:
    """Bounded exact-timestamp join; arrival order cannot pair different exposures."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.streams: dict[str, OrderedDict[int, Any]] = {
            name: OrderedDict() for name in ("rgb", "gt", "pose")
        }
        self.changed = threading.Condition()

    def put(self, stream: str, timestamp_ns: int, value: Any) -> None:
        with self.changed:
            entries = self.streams[stream]
            entries[timestamp_ns] = value
            while len(entries) > self.capacity:
                entries.popitem(last=False)
            self.changed.notify_all()

    def wait(self, timestamp_ns: int, timeout_s: float = 10.0) -> dict[str, Any]:
        with self.changed:
            ready = self.changed.wait_for(
                lambda: all(timestamp_ns in entries for entries in self.streams.values()),
                timeout=timeout_s,
            )
            if not ready:
                latest = {
                    name: max(entries, default=None) for name, entries in self.streams.items()
                }
                raise TimeoutError(f"no synchronized capture at {timestamp_ns} ns; latest={latest}")
            return {name: entries[timestamp_ns] for name, entries in self.streams.items()}


def decode_rgb(msg: Any):
    """Decode RGB_INT8 including row padding; reject unsupported wire encodings."""
    import numpy as np

    if msg.pixel_format_type != 3 or msg.width <= 0 or msg.height <= 0:
        raise ValueError("expected a nonempty Gazebo RGB_INT8 image")
    if msg.step < msg.width * 3 or len(msg.data) != msg.height * msg.step:
        raise ValueError("image data length/row stride is inconsistent")
    rows = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    return rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3).copy()


@dataclass(frozen=True)
class Capture:
    """Evaluator-only capture bundle. Any future executor receives only rgb."""

    timestamp_ns: int
    rgb: Any
    object_ids: Any
    body_pose: dict[str, Any]


class GazeboBackend:
    """One owned, initially paused pilot world. Commands never target physical hardware."""

    def __init__(
        self,
        *,
        timeout_s: float = 15.0,
        waypoints: list[dict] | None = None,
        max_speed_mps: float | None = None,
    ) -> None:
        from gz.msgs10.clock_pb2 import Clock
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.odometry_pb2 import Odometry
        from gz.msgs10.twist_pb2 import Twist
        from gz.transport13 import Node

        self.stepper = None
        self.timeout_s = timeout_s
        self.max_speed_mps = max_speed_mps
        self.waypoints = waypoints or [
            {"start_s": 0.0, "position_m": [0.0, 0.0, 4.0]},
            {"start_s": 1.0, "position_m": [3.0, 0.0, 4.0]},
            {"start_s": 6.0, "position_m": [3.0, 2.0, 4.0]},
        ]
        self.node = Node()
        self.inbox = SensorInbox()
        self.clock_condition = threading.Condition()
        self.clock_ns = 0
        self.twist_type = Twist
        self.command = self.node.advertise("/pilot_quad/cmd_vel", Twist)
        for cls, topic, callback in (
            (Clock, f"/world/{WORLD}/clock", self._clock),
            (Image, "/v4/rgb", lambda msg: self.inbox.put("rgb", stamp_ns(msg.header.stamp), msg)),
            (
                Image,
                "/v4/gt/labels_map",
                lambda msg: self.inbox.put("gt", stamp_ns(msg.header.stamp), msg),
            ),
            (Odometry, "/v4/odometry", self._pose),
        ):
            if not self.node.subscribe(cls, topic, callback):
                raise RuntimeError(f"cannot subscribe to {topic}")

    def _clock(self, msg: Any) -> None:
        with self.clock_condition:
            self.clock_ns = stamp_ns(msg.sim)
            self.clock_condition.notify_all()

    def _pose(self, msg: Any) -> None:
        self.inbox.put("pose", stamp_ns(msg.header.stamp), msg)

    def connect(self) -> None:
        deadline = time.monotonic() + self.timeout_s
        while not self.command.has_connections():
            if time.monotonic() >= deadline:
                raise TimeoutError("Gazebo flight controller was not discovered")
            time.sleep(0.05)
        self._command_velocity()
        # Discovery/delivery grace is host-only; the world is still paused.
        time.sleep(0.1)

    def _command_velocity(self) -> None:
        """Fixed waypoint controller, using ideal simulator odometry (not perception GT).

        This is simulator support, not a benchmark policy or PX4/state estimator.
        Waypoint changes depend only on simulation time, independently of inference.
        """
        cmd = self.twist_type()
        with self.inbox.changed:
            state = self.inbox.streams["pose"].get(self.clock_ns)
        if state is not None:
            p = state.pose.position
            q = state.pose.orientation
            target = self.waypoints[0]["position_m"]
            for waypoint in self.waypoints:
                if self.clock_ns >= round(waypoint["start_s"] * 1e9):
                    target = waypoint["position_m"]
            vx, vy, vz = commanded_velocity((p.x, p.y, p.z), target, self.max_speed_mps)
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            cmd.linear.x = math.cos(yaw) * vx + math.sin(yaw) * vy
            cmd.linear.y = -math.sin(yaw) * vx + math.cos(yaw) * vy
            cmd.linear.z = vz
            cmd.angular.z = -yaw
        if not self.command.publish(cmd):
            raise RuntimeError("Gazebo flight command could not be published")

    def advance_to(self, target_ns: int) -> None:
        """Step in 100 ms blocks; flight control continues while inference is busy."""
        if target_ns < self.clock_ns or target_ns % STEP_NS:
            raise ValueError("target must be a future physics tick")
        while self.clock_ns < target_ns:
            # Each chunk starts on the same control grid, regardless of latency.
            next_control = (self.clock_ns // 100_000_000 + 1) * 100_000_000
            next_ns = min(next_control, target_ns)
            if self.clock_ns % 100_000_000 == 0:
                self._command_velocity()
            count = (next_ns - self.clock_ns) // STEP_NS
            if self.stepper is None:
                from experiments.v4_gazebo_pilot.stepper import ServiceStepper

                self.stepper = ServiceStepper()
            self.stepper.step(count, self.timeout_s)
            with self.clock_condition:
                if not self.clock_condition.wait_for(
                    lambda target=next_ns: self.clock_ns >= target, timeout=self.timeout_s
                ):
                    raise TimeoutError(f"world failed to reach {next_ns} ns")
                if self.clock_ns != next_ns:
                    raise RuntimeError("world advanced beyond requested tick; expected paused mode")
            # Don't use a callback from the previous tick for the next control command.
            with self.inbox.changed:
                if not self.inbox.changed.wait_for(
                    lambda target=next_ns: target in self.inbox.streams["pose"],
                    timeout=self.timeout_s,
                ):
                    raise TimeoutError(f"missing body pose at {next_ns} ns")

    def capture(self) -> Capture:
        messages = self.inbox.wait(self.clock_ns, self.timeout_s)
        rgb = decode_rgb(messages["rgb"])
        encoded = decode_rgb(messages["gt"])
        # Gazebo semantic label map repeats its 8-bit class in RGB channels.
        import numpy as np

        if not np.array_equal(encoded[:, :, 0], encoded[:, :, 1]) or not np.array_equal(
            encoded[:, :, 0], encoded[:, :, 2]
        ):
            raise ValueError("unexpected semantic label encoding; refusing to guess object IDs")
        if rgb.shape != encoded.shape:
            raise ValueError("RGB and GT resolutions differ")
        return Capture(self.clock_ns, rgb, encoded[:, :, 0].copy(), pose_record(messages["pose"]))

    def close(self) -> None:
        if self.stepper is not None:
            self.stepper.close()

    def pose(self) -> dict[str, Any]:
        with self.inbox.changed:
            return pose_record(self.inbox.streams["pose"][self.clock_ns])


def pose_record(msg: Any) -> dict[str, Any]:
    p, q = msg.pose.position, msg.pose.orientation
    return {
        "timestamp_ns": stamp_ns(msg.header.stamp),
        "position_m": [p.x, p.y, p.z],
        "quaternion_xyzw": [q.x, q.y, q.z, q.w],
    }


def commanded_velocity(position, target, max_speed_mps=None):
    """Position servo; preserve legacy axis clipping unless a vector cap is set."""
    velocity = [0.8 * (goal - value) for goal, value in zip(target, position, strict=True)]
    if max_speed_mps is None:
        return tuple(max(-0.6, min(0.6, v)) for v in velocity)
    if not math.isfinite(max_speed_mps) or not 0 < max_speed_mps <= 3:
        raise ValueError("speed limit must be in (0,3]")
    speed = math.sqrt(sum(v * v for v in velocity))
    scale = min(1.0, max_speed_mps / speed) if speed else 1.0
    return tuple(v * scale for v in velocity)
