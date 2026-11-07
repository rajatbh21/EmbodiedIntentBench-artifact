"""Predefined UAV trajectories: position as a pure function of mission time.

The V2 UAV flies a fixed 2D path at fixed altitude and fixed speed. The policy never
steers; it only chooses the perception configuration. ``position_at(t)`` is therefore
the single authority on where the camera is -- the runner calls it for capture time
*and* for completion time, which is what makes "the UAV kept moving during inference"
literal rather than simulated bookkeeping.

Two shapes: an explicit ``polyline`` of waypoints, and a ``lawnmower`` sweep generated
from a rectangle plus a lane spacing (which degenerates to a polyline and reuses all of
its arithmetic).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from missionbench.schemas.loading import SchemaValidationError
from missionbench.v2.scenario import TrajectorySpec

__all__ = ["PolylineTrajectory", "build_trajectory", "lawnmower_waypoints"]


@dataclass(frozen=True, slots=True)
class PolylineTrajectory:
    """Constant-speed travel along a waypoint chain, clamped at the end."""

    waypoints_m: tuple[tuple[float, float], ...]
    speed_mps: float

    def __post_init__(self) -> None:
        if len(self.waypoints_m) < 2:
            raise SchemaValidationError("a trajectory needs at least two waypoints")
        if self.speed_mps <= 0:
            raise SchemaValidationError(f"speed_mps must be positive, got {self.speed_mps}")

    @property
    def _segment_lengths(self) -> tuple[float, ...]:
        return tuple(
            math.dist(a, b) for a, b in zip(self.waypoints_m, self.waypoints_m[1:], strict=False)
        )

    @property
    def total_length_m(self) -> float:
        return sum(self._segment_lengths)

    @property
    def duration_s(self) -> float:
        """Time to fly the whole path at the fixed speed."""
        return self.total_length_m / self.speed_mps

    def position_at(self, time_s: float) -> tuple[float, float]:
        """UAV position at mission time ``time_s``; clamped to the endpoints."""
        distance = max(0.0, time_s) * self.speed_mps
        remaining = distance
        for (a, b), length in zip(
            zip(self.waypoints_m, self.waypoints_m[1:], strict=False),
            self._segment_lengths,
            strict=False,
        ):
            if remaining <= length or length == 0.0:
                if length == 0.0:
                    continue
                f = remaining / length
                return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)
            remaining -= length
        return self.waypoints_m[-1]

    def progress_at(self, time_s: float) -> float:
        """Fraction of the path flown at ``time_s``, in [0, 1]."""
        total = self.total_length_m
        if total == 0.0:
            return 1.0
        return min(1.0, max(0.0, time_s) * self.speed_mps / total)

    def is_complete_at(self, time_s: float) -> bool:
        return self.progress_at(time_s) >= 1.0


def lawnmower_waypoints(
    region_m: tuple[float, float, float, float], lane_spacing_m: float
) -> tuple[tuple[float, float], ...]:
    """Generate a horizontal back-and-forth sweep of ``region_m``.

    Lanes run along x at successive y offsets, alternating direction so the path is
    continuous. The last lane sits on the region's bottom edge even when the spacing
    does not divide the height exactly, so the sweep always covers the full rectangle.
    """
    x0, y0, x1, y1 = region_m
    if not (x1 > x0 and y1 > y0):
        raise SchemaValidationError(f"lawnmower region {region_m} is not a proper rectangle")
    if lane_spacing_m <= 0:
        raise SchemaValidationError(f"lane_spacing_m must be positive, got {lane_spacing_m}")

    ys: list[float] = []
    y = y0
    while y < y1 - 1e-9:
        ys.append(y)
        y += lane_spacing_m
    ys.append(y1)

    waypoints: list[tuple[float, float]] = []
    for lane, y in enumerate(ys):
        lane_points = [(x0, y), (x1, y)] if lane % 2 == 0 else [(x1, y), (x0, y)]
        waypoints.extend(lane_points)
    return tuple(waypoints)


def build_trajectory(spec: TrajectorySpec, speed_mps: float) -> PolylineTrajectory:
    """Materialise a scenario trajectory spec into a flyable polyline."""
    if spec.type == "polyline":
        return PolylineTrajectory(waypoints_m=spec.waypoints_m, speed_mps=speed_mps)
    assert spec.region_m is not None and spec.lane_spacing_m is not None  # schema-validated
    return PolylineTrajectory(
        waypoints_m=lawnmower_waypoints(spec.region_m, spec.lane_spacing_m),
        speed_mps=speed_mps,
    )
