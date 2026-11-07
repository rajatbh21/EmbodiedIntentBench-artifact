"""Gazebo reference controller; uses simulation time and odometry only."""

import math

from experiments.real_mission_telemetry.reference import command, validate
from experiments.v4_gazebo_pilot.transport import GazeboBackend


class TrajectoryBackend(GazeboBackend):
    def __init__(self, reference, *, timeout_s=90):
        self.reference = reference
        validate(reference["times_s"], reference["positions_m"])
        super().__init__(timeout_s=timeout_s)

    def _command_velocity(self):
        message = self.twist_type()
        with self.inbox.changed:
            state = self.inbox.streams["pose"].get(self.clock_ns)
        if state is not None:
            p, q = state.pose.position, state.pose.orientation
            r = self.reference
            v = command(
                (p.x, p.y, p.z),
                self.clock_ns / 1e9 - r["warmup_s"],
                r["times_s"],
                r["positions_m"],
                speed_limit=r["speed_limit_mps"],
            )
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            message.linear.x = math.cos(yaw) * v[0] + math.sin(yaw) * v[1]
            message.linear.y = -math.sin(yaw) * v[0] + math.cos(yaw) * v[1]
            message.linear.z = v[2]
            message.angular.z = -yaw
        if not self.command.publish(message):
            raise RuntimeError("Reference command publication failed")
