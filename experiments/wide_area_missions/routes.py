"""Authored, terrain-aware paths for a separate wide-area acquisition cohort.

These are real-world-inspired mission geometries, not recorded UAV trajectories.
The deterministic cubic B-splines have continuous position, tangent and curvature
before dense arc-length sampling. Retiming respects motion limits independently
of perception predictions or routing-policy outcomes.

``terrain_height_callback(x, y)`` must return a conservative clearance surface in
Gazebo world metres. It can be bare terrain or terrain plus an obstacle envelope.
Consequently ``height_above_surface_m`` is not necessarily height above ground.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import numpy as np


ROUTE_NAMES = ("wide_area_search", "shoreline_loop", "sector_revisit")
MAP_BOUNDS_XY_M = ((-600.0, 600.0), (-900.0, 300.0))

_SPECS = {
    "wide_area_search": {
        "label": "Valley area search",
        "family": "Rounded multi-swath coverage",
        "closed": False,
        "control_points": [
            (-430, -620), (-150, -620), (150, -620), (330, -620),
            (350, -580), (350, -500), (330, -460), (100, -460),
            (-150, -460), (-410, -460), (-430, -420), (-430, -340),
            (-410, -300), (-150, -300), (100, -300), (330, -300),
            (350, -260), (350, -180), (330, -140), (150, -140),
            (-150, -140), (-430, -140),
        ],
        "description": "Four broad coverage swaths joined by smooth rounded turns.",
    },
    "shoreline_loop": {
        "label": "Shore-to-upland circuit",
        "family": "Curved inspection circuit",
        "closed": True,
        "control_points": [
            (-300, -65), (-210, -15), (-40, -10), (110, -30),
            (240, -15), (370, -45), (380, -110), (300, -190),
            (150, -220), (-20, -215), (-190, -200), (-300, -140),
        ],
        "description": (
            "A broad circuit between the shore vicinity and upland terrain; "
            "the path is authored, not an extracted water-edge contour."
        ),
    },
    "sector_revisit": {
        "label": "Cross-map sector transit and revisit",
        "family": "Multi-sector transit with revisits",
        "closed": False,
        "control_points": [
            (-120, -30), (-300, -30), (-430, -180), (-430, -380),
            (-330, -620), (-130, -760), (120, -780), (360, -650),
            (460, -400), (420, -180), (180, -40), (-80, -30),
            (-260, -100), (-430, -300), (-380, -620), (-100, -760),
            (180, -700), (380, -470), (460, -200), (300, -20),
            (80, -20), (-120, -30),
        ],
        "description": (
            "Two curved cross-map circuits revisit northern, western and "
            "southern sectors using different transit corridors."
        ),
    },
}


def _spline_xy(control_points: list[tuple[int, int]], closed: bool) -> np.ndarray:
    """Evaluate a uniform cubic B-spline densely without a SciPy dependency."""
    controls = np.asarray(control_points, dtype=float)
    if closed:
        padded = np.concatenate((controls[-1:], controls, controls[:2]))
        segment_count = len(controls)
    else:
        padded = np.concatenate((np.repeat(controls[:1], 2, axis=0), controls,
                                 np.repeat(controls[-1:], 2, axis=0)))
        segment_count = len(padded) - 3
    segments = []
    for index in range(segment_count):
        p = padded[index:index + 4]
        polygon_length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
        # At most about 0.5 m of control-polygon travel per dense sample.
        count = max(12, math.ceil(polygon_length / 0.5))
        t = np.linspace(0.0, 1.0, count, endpoint=False)
        weights = np.stack(((1 - t) ** 3, 3*t**3 - 6*t**2 + 4,
                            -3*t**3 + 3*t**2 + 3*t + 1, t**3), axis=1) / 6
        segments.append(weights @ p)
    if closed:
        endpoint = segments[0][0]
    else:
        endpoint = controls[-1]
    dense = np.vstack((*segments, endpoint[None, :]))
    keep = np.concatenate(([True], np.linalg.norm(np.diff(dense, axis=0), axis=1) > 1e-9))
    return dense[keep]


def _arc_sample(xy: np.ndarray, spacing: float) -> np.ndarray:
    distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    samples = np.linspace(0.0, distance[-1], math.ceil(distance[-1] / spacing) + 1)
    return np.column_stack([np.interp(samples, distance, xy[:, axis]) for axis in range(2)])


def _smooth_upper_surface(surface: np.ndarray, spacing: float, radius_m: float = 16.0) -> np.ndarray:
    """Smooth without descending below a sampled discontinuous obstacle envelope.

    Each maximum-filter window contains every original sample contributing to
    the subsequent nonnegative convolution. Thus the convolution is an upper
    bound, unlike directly smoothing terrain heights. The compact bump kernel
    avoids a hard transition at the edge of its spatial support.
    """
    radius = max(2, math.ceil(radius_m / spacing))
    maximum = np.lib.stride_tricks.sliding_window_view(
        np.pad(surface, (radius, radius), mode="edge"), 2 * radius + 1).max(axis=1)
    coordinate = np.linspace(-1, 1, 2 * radius + 1)
    weights = np.zeros_like(coordinate)
    interior = np.abs(coordinate) < 1
    weights[interior] = np.exp(-1 / (1 - coordinate[interior] ** 2))
    weights /= weights.sum()
    smoothed = np.convolve(np.pad(maximum, (radius, radius), mode="edge"), weights, mode="valid")
    return np.maximum(smoothed, surface)


def kinematic_summary(points: list[dict[str, Any]]) -> dict[str, float]:
    """Audit the actual timestamped polyline, including departure and arrival.

    Acceleration compares adjacent interval velocities at their midpoint times;
    the boundary audit assumes zero velocity at the first and last timestamps.
    This does not claim a full rotor-dynamics or tracking-error validation.
    """
    time = np.asarray([point["time_s"] for point in points], dtype=float)
    xyz = np.asarray([point["position_m"] for point in points], dtype=float)
    dt = np.diff(time)
    if len(points) < 3 or np.any(dt <= 0):
        raise ValueError("A route audit requires at least three strictly ordered timestamps")
    velocities = np.diff(xyz, axis=0) / dt[:, None]
    acceleration = np.diff(velocities, axis=0) / ((dt[:-1] + dt[1:])[:, None] / 2)
    boundary_acceleration = 2 * np.linalg.norm(velocities[[0, -1], :2], axis=1) / dt[[0, -1]]
    vertical_boundary_acceleration = 2 * np.abs(velocities[[0, -1], 2]) / dt[[0, -1]]
    return {
        "max_speed_mps": float(np.linalg.norm(velocities, axis=1).max()),
        "max_horizontal_speed_mps": float(np.linalg.norm(velocities[:, :2], axis=1).max()),
        "max_vertical_speed_mps": float(np.abs(velocities[:, 2]).max()),
        "max_horizontal_acceleration_mps2": float(max(
            np.linalg.norm(acceleration[:, :2], axis=1).max(), boundary_acceleration.max())),
        "max_vertical_acceleration_mps2": float(max(
            np.abs(acceleration[:, 2]).max(), vertical_boundary_acceleration.max())),
        "max_horizontal_sample_spacing_m": float(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1).max()),
    }


def _hermite_motion_summary(xyz: np.ndarray, times: np.ndarray, velocities: np.ndarray) -> dict[str, float]:
    """Exact extrema of the runtime's continuous cubic Hermite interpolation.

    On normalized interval coordinate u, velocity is A*u**2 + B*u + C.
    Its squared norm has a cubic derivative. Acceleration is affine, so the
    maximum norm or absolute component occurs at one of the interval endpoints.
    This audits the planned interpolant, not actual physical tracking dynamics.
    """
    dt = np.diff(times)
    delta = np.diff(xyz, axis=0) / dt[:, None]
    first, last = velocities[:-1], velocities[1:]
    a = -6 * delta + 3 * (first + last)
    b = 6 * delta - 4 * first - 2 * last
    c = first
    acceleration_start = b / dt[:, None]
    acceleration_end = (2 * a + b) / dt[:, None]
    accelerations = np.vstack((acceleration_start, acceleration_end))
    maximum_speed, maximum_vertical_speed = 0.0, 0.0
    for ai, bi, ci in zip(a, b, c):
        coefficients = [4 * np.dot(ai, ai), 6 * np.dot(ai, bi),
                        2 * (np.dot(bi, bi) + 2 * np.dot(ai, ci)), 2 * np.dot(bi, ci)]
        candidates = [0.0, 1.0]
        for root in np.roots(coefficients):
            if abs(root.imag) < 1e-8 and 0 < root.real < 1:
                candidates.append(float(root.real))
        for u in candidates:
            velocity = (ai * u + bi) * u + ci
            maximum_speed = max(maximum_speed, float(np.linalg.norm(velocity)))
        vertical_candidates = [0.0, 1.0]
        if abs(ai[2]) > 1e-14:
            extremum = -bi[2] / (2 * ai[2])
            if 0 < extremum < 1:
                vertical_candidates.append(float(extremum))
        for u in vertical_candidates:
            maximum_vertical_speed = max(maximum_vertical_speed, abs(float((ai[2] * u + bi[2]) * u + ci[2])))
    return {
        "max_hermite_speed_mps": maximum_speed,
        "max_hermite_vertical_speed_mps": maximum_vertical_speed,
        "max_hermite_horizontal_acceleration_mps2": float(np.linalg.norm(accelerations[:, :2], axis=1).max()),
        "max_hermite_vertical_acceleration_mps2": float(np.abs(accelerations[:, 2]).max()),
    }


def build_route(
    name: str,
    terrain_height_callback: Callable[[float, float], float],
    *,
    sample_spacing_m: float = 2.0,
    cruise_speed_mps: float = 5.0,
    max_horizontal_acceleration_mps2: float = 2.0,
    max_vertical_speed_mps: float = 1.5,
    max_vertical_acceleration_mps2: float = 2.0,
    clearance_m: float = 28.0,
) -> dict[str, Any]:
    """Build JSON-serializable geometry and a conservative timestamp schedule.

    The callback is evaluated at every dense route point. A caller supplying an
    obstacle envelope should inflate obstacles for the vehicle radius and allow
    for interpolation/tracking error. This module validates sampled clearance;
    it is not a continuous collision detector or a flight controller.
    """
    if name not in _SPECS:
        raise ValueError(f"Unknown route {name!r}; choose one of {ROUTE_NAMES}")
    values = (sample_spacing_m, cruise_speed_mps, max_horizontal_acceleration_mps2,
              max_vertical_speed_mps, max_vertical_acceleration_mps2, clearance_m)
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError("Route settings must be finite positive numbers")
    if sample_spacing_m > 10:
        raise ValueError("Route sample spacing must be at most 10 m")
    if (cruise_speed_mps > 5 or max_horizontal_acceleration_mps2 > 2
            or max_vertical_speed_mps > 1.5 or max_vertical_acceleration_mps2 > 2):
        raise ValueError("Limits must not exceed 5 m/s, 2 m/s^2 horizontally, 1.5 m/s vertically and 2 m/s^2 vertically")
    if not 20 <= clearance_m <= 35:
        raise ValueError("Clearance above the supplied surface must be 20 to 35 m")
    spec = _SPECS[name]
    xy = _arc_sample(_spline_xy(spec["control_points"], spec["closed"]), sample_spacing_m)
    surface = np.asarray([float(terrain_height_callback(float(x), float(y))) for x, y in xy])
    if not np.isfinite(surface).all():
        raise ValueError("The clearance surface returned a non-finite height")
    # Plan ahead of abrupt terrain/obstacles rather than commanding altitude
    # steps. Actual clearance can exceed the requested minimum near a ridge.
    altitude_surface = _smooth_upper_surface(surface, sample_spacing_m)
    xyz = np.column_stack((xy, altitude_surface + clearance_m))
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dz = np.diff(xyz[:, 2])
    slope = np.abs(dz) / ds
    neighboring_slope = np.maximum(np.r_[slope[0], slope], np.r_[slope, slope[-1]])
    direction = np.diff(xy, axis=0) / ds[:, None]
    curvature = np.zeros(len(xy))
    curvature[1:-1] = np.linalg.norm(np.diff(direction, axis=0), axis=1) / ((ds[:-1] + ds[1:]) / 2)
    # Reserve acceleration for both turning and changes in forward speed.
    turn_limit = np.sqrt((0.4 * max_horizontal_acceleration_mps2) / np.maximum(curvature, 1e-12))
    vertical_limit = max_vertical_speed_mps / np.maximum(neighboring_slope, 1e-12)
    spatial_speed_limit = cruise_speed_mps / np.sqrt(1 + neighboring_slope**2)
    speed = np.minimum.reduce((turn_limit, vertical_limit, spatial_speed_limit))
    speed[0] = speed[-1] = 0.0
    tangential_acceleration = 0.4 * max_horizontal_acceleration_mps2
    for index in range(1, len(speed)):
        speed[index] = min(speed[index], math.sqrt(speed[index - 1]**2 + 2 * tangential_acceleration * ds[index - 1]))
    for index in range(len(speed) - 2, -1, -1):
        speed[index] = min(speed[index], math.sqrt(speed[index + 1]**2 + 2 * tangential_acceleration * ds[index]))
    dt = 2 * ds / (speed[:-1] + speed[1:])
    times = np.r_[0.0, np.cumsum(dt)]

    # A final finite-difference audit catches irregular terrain and numerical
    # corner cases. Global dilation can only reduce all three motion limits.
    def audit_at(t: np.ndarray) -> dict[str, float]:
        return kinematic_summary([{"time_s": float(ti), "position_m": p.tolist()} for ti, p in zip(t, xyz)])

    initial = audit_at(times)
    dilation = max(1.0, initial["max_speed_mps"] / cruise_speed_mps,
                   initial["max_vertical_speed_mps"] / max_vertical_speed_mps,
                   math.sqrt(initial["max_horizontal_acceleration_mps2"] / max_horizontal_acceleration_mps2),
                   math.sqrt(initial["max_vertical_acceleration_mps2"] / max_vertical_acceleration_mps2))
    times *= dilation * (1 + 1e-10)
    velocities = np.gradient(xyz, times, axis=0)
    velocities[[0, -1]] = 0.0
    # A polyline audit alone does not bound Hermite motion between knots. Keep
    # exactly the same geometry, then dilate the continuous interpolant as well.
    continuous = _hermite_motion_summary(xyz, times, velocities)
    continuous_dilation = max(
        1.0,
        continuous["max_hermite_speed_mps"] / cruise_speed_mps,
        continuous["max_hermite_vertical_speed_mps"] / max_vertical_speed_mps,
        math.sqrt(continuous["max_hermite_horizontal_acceleration_mps2"] / max_horizontal_acceleration_mps2),
        math.sqrt(continuous["max_hermite_vertical_acceleration_mps2"] / max_vertical_acceleration_mps2),
    ) * (1 + 1e-10)
    times *= continuous_dilation
    velocities /= continuous_dilation
    continuous = _hermite_motion_summary(xyz, times, velocities)
    heading = np.unwrap(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0])))
    path_distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    points = [{
        "time_s": float(t), "position_m": position.tolist(),
        "yaw_rad": float(yaw), "velocity_mps": velocity.tolist(),
        "speed_mps": float(np.linalg.norm(velocity)),
        "height_above_surface_m": float(position[2] - surface_height),
        "distance_along_route_m": float(distance),
    } for t, position, yaw, velocity, distance, surface_height in zip(times, xyz, heading, velocities, path_distance, surface)]
    audit = kinematic_summary(points)
    return {
        "schema_version": 1,
        "route_id": name,
        "name": spec["label"],
        "coordinate_frame": "gazebo_world_m",
        "points": points,
        "bounds": {f"{axis}_m": [float(xyz[:, index].min()), float(xyz[:, index].max())]
                   for index, axis in enumerate("xyz")},
        "horizontal_distance_m": float(ds.sum()),
        "distance_m": float(path_distance[-1]),
        "duration_s": float(times[-1]),
        "design": {
            "family": spec["family"], "description": spec["description"],
            "trajectory_source": "Deterministic authored spline; not a real UAV flight log",
            "closed_geometry": spec["closed"], "starts_and_ends_at_rest": True,
            "control_points_xy_m": [list(point) for point in spec["control_points"]],
            "clearance_reference": "Caller-supplied conservative terrain or obstacle envelope",
            "altitude_envelope": "Spatial maximum filter plus compact positive smoothing; requested clearance is a minimum",
            "clearance_m": float(clearance_m), "sample_spacing_m": float(sample_spacing_m),
            "cruise_speed_limit_mps": float(cruise_speed_mps),
            "horizontal_acceleration_limit_mps2": float(max_horizontal_acceleration_mps2),
            "vertical_speed_limit_mps": float(max_vertical_speed_mps),
            "vertical_acceleration_limit_mps2": float(max_vertical_acceleration_mps2),
            "continuous_motion_validation": "Exact cubic Hermite velocity extrema and both endpoint accelerations per interval",
            "fair_comparison": "Reuse the same timestamped path and potential captures for every router",
        },
        "validation": {**audit, **continuous,
                       "minimum_sampled_clearance_m": float(np.min(xyz[:, 2] - surface)),
                       "maximum_sampled_clearance_m": float(np.max(xyz[:, 2] - surface))},
    }
