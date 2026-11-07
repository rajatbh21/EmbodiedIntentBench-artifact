"""Pinhole ground-plane projection using public camera calibration and odometry.

Gazebo cameras look along +X with +Y left and +Z up. No depth, mesh, object
position or semantic information is used. Slopes and non-ground objects violate
this approximation; invalid/upward rays are rejected rather than fabricated.
"""

import math

import numpy as np


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-9:
        raise ValueError("invalid odometry quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def ground_point(pixel, shape, pose, pitch, hfov=1.2, ground_z=0.0):
    height, width = shape
    focal = width / (2 * math.tan(hfov / 2))
    u, v = pixel
    ray = np.array([1, -(u - (width - 1) / 2) / focal, -(v - (height - 1) / 2) / focal])
    c, s = math.cos(pitch), math.sin(pitch)
    mount = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    body = rotation_xyzw(pose["quaternion_xyzw"])
    origin = np.asarray(pose["position_m"], dtype=float) + body @ np.array([0, 0, -0.1])
    ray = body @ mount @ ray
    if not np.isfinite(origin).all() or ray[2] >= -1e-6:
        return None
    distance = (ground_z - origin[2]) / ray[2]
    if distance <= 0:
        return None
    point = origin + distance * ray
    return tuple(float(x) for x in point[:2])
