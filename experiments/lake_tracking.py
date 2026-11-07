"""Prediction-only static-target tracking with explicit public geometry priors.

No target identities, object poses, depth labels or evaluator outputs enter this
module. A terrain prior is optional and must be declared as available map data.
This is not a moving-person tracker or appearance-based re-identification model.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from experiments.v4_benchmark.projection import rotation_xyzw
from experiments.v4_benchmark.reporting import box_iou, components


@dataclass(frozen=True)
class TrackingConfig:
    association: str = "terrain_world"
    max_gap_s: float = 8.0
    max_distance_m: float = 1.0
    association_iou: float = 0.15
    min_hits: int = 2
    remember_reports: bool = False

    def __post_init__(self):
        if self.association not in ("image_iou", "flat_world", "terrain_world"):
            raise ValueError("Unknown association mode")
        if type(self.min_hits) is not int or self.min_hits < 2:
            raise ValueError("At least two observations required")
        if type(self.remember_reports) is not bool:
            raise ValueError("remember_reports must be boolean")
        if self.remember_reports and self.association == "image_iou":
            raise ValueError("Persistent reports require world coordinates")
        for key in ("max_gap_s", "max_distance_m", "association_iou"):
            value = getattr(self, key)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {key}")
        if self.association_iou > 1:
            raise ValueError("Invalid IoU threshold")

    def to_dict(self):
        return asdict(self)


class HeightField:
    """A public north-up elevation raster, copied independently of scene labels."""

    def __init__(self, elevations, *, west, east, south, north):
        values = np.asarray(elevations, dtype=float)
        if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
            raise ValueError("A finite 2D elevation grid is required")
        if not all(math.isfinite(v) for v in (west, east, south, north)):
            raise ValueError("Finite bounds required")
        if west >= east or south >= north:
            raise ValueError("Invalid map bounds")
        self.values = np.frombuffer(values.tobytes(), dtype=float).reshape(values.shape)
        self.west, self.east, self.south, self.north = west, east, south, north

    def sample(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        h, w = self.values.shape
        u = (x - self.west) / (self.east - self.west) * (w - 1)
        v = (self.north - y) / (self.north - self.south) * (h - 1)
        valid = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
        safe_u, safe_v = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
        j = np.minimum(safe_u.astype(int), w - 2)
        i = np.minimum(safe_v.astype(int), h - 2)
        dx, dy = safe_u - j, safe_v - i
        z = (
            self.values[i, j] * (1 - dx) * (1 - dy)
            + self.values[i, j + 1] * dx * (1 - dy)
            + self.values[i + 1, j] * (1 - dx) * dy
            + self.values[i + 1, j + 1] * dx * dy
        )
        return np.where(valid, z, np.nan)


def camera_ray(pixel, shape, pose, pitch, hfov=1.2):
    height, width = shape
    if min(shape) <= 0 or not 0 < hfov < math.pi or not math.isfinite(pitch):
        raise ValueError("Invalid camera calibration")
    u, v = pixel
    if not (0 <= u < width and 0 <= v < height):
        raise ValueError("Pixel outside image")
    focal = width / (2 * math.tan(hfov / 2))
    ray = np.array([1, -(u - (width - 1) / 2) / focal, -(v - (height - 1) / 2) / focal])
    c, s = math.cos(pitch), math.sin(pitch)
    mount = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    body = rotation_xyzw(pose["quaternion_xyzw"])
    position = np.asarray(pose["position_m"], dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Invalid camera position")
    origin = position + body @ np.array([0, 0, -0.1])
    ray = body @ mount @ ray
    return origin, ray / np.linalg.norm(ray)


class SurfaceProjector:
    """Project mask centroids to a declared nominal height above a static surface.

    The 0.9 m anchor is a generic standing-person center prior, not measured target
    height. Terrain intersections use 0.5 m ray sampling followed by bisection.
    Roofs, people in motion and inaccurate maps/odometry are outside this model.
    """

    def __init__(self, terrain=None, *, anchor_height_m=0.9, hfov=1.2):
        if not math.isfinite(anchor_height_m) or anchor_height_m < 0:
            raise ValueError("Invalid anchor height")
        if not math.isfinite(hfov) or not 0 < hfov < math.pi:
            raise ValueError("Invalid horizontal field of view")
        self.terrain, self.anchor_height_m, self.hfov = terrain, anchor_height_m, hfov

    def point(self, pixel, shape, pose, pitch):
        origin, ray = camera_ray(pixel, shape, pose, pitch, self.hfov)
        if ray[2] >= -1e-6:
            return None
        if self.terrain is None:
            distance = (self.anchor_height_m - origin[2]) / ray[2]
            if not 0 < distance <= 300:
                return None
            point = origin + distance * ray
            return tuple(float(v) for v in point[:2])
        distances = np.arange(0, 300.5, 0.5)
        points = origin + distances[:, None] * ray
        residual = (
            points[:, 2] - self.terrain.sample(points[:, 0], points[:, 1]) - self.anchor_height_m
        )
        if not math.isfinite(residual[0]) or residual[0] <= 0:
            return None
        crossings = np.flatnonzero((residual[:-1] > 0) & (residual[1:] <= 0))
        if not len(crossings):
            return None
        index = int(crossings[0])
        if not np.isfinite(residual[: index + 2]).all():
            return None
        low, high = distances[index], distances[index + 1]
        for _ in range(20):
            middle = (low + high) / 2
            point = origin + middle * ray
            delta = point[2] - float(self.terrain.sample(point[0], point[1])) - self.anchor_height_m
            if delta > 0:
                low = middle
            else:
                high = middle
        point = origin + (low + high) / 2 * ray
        return tuple(float(v) for v in point[:2])


def detections(mask, pose, pitch, projectors):
    """Convert a predicted boolean mask to public measurements, without GT access."""
    if mask.dtype != np.bool_ or mask.ndim != 2:
        raise ValueError("Expected a 2D boolean prediction mask")
    labels, ids = components(mask)
    result = []
    for cid in ids:
        y, x = np.where(labels == cid)
        pixel = (float(x.mean()), float(y.mean()))
        result.append(
            dict(
                component_id=cid,
                box=[int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)],
                centroid_px=pixel,
                positions={
                    key: projector.point(pixel, mask.shape, pose, pitch)
                    for key, projector in projectors.items()
                },
            )
        )
    return result


class StaticReportTracker:
    """Causal one-to-one association; report memory never counts as fresh evidence."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracks, self.memory = {}, {}
        self.next_id, self.report_count = 1, 0
        self.last_capture = -math.inf
        self.last_completion = -math.inf
        self.assignments = []
        self.suppressed_reports = 0

    def update(self, measurements, capture_s, completion_s):
        if not all(math.isfinite(v) for v in (capture_s, completion_s)):
            raise ValueError("Finite times required")
        if (
            capture_s < 0
            or capture_s <= self.last_capture
            or capture_s < self.last_completion
            or completion_s < capture_s
        ):
            raise ValueError("Updates must follow the serial causal capture schedule")
        self.last_capture, self.last_completion = capture_s, completion_s
        current = {m["component_id"]: m for m in measurements}
        if len(current) != len(measurements):
            raise ValueError("Duplicate component IDs")
        self.tracks = {
            tid: t
            for tid, t in self.tracks.items()
            if capture_s - t["last_s"] <= self.config.max_gap_s
        }
        pairs = []
        for tid, track in self.tracks.items():
            for cid, measurement in current.items():
                if self.config.association == "image_iou":
                    overlap = box_iou(track["box"], measurement["box"])
                    if overlap >= self.config.association_iou:
                        pairs.append((-overlap, tid, cid))
                else:
                    point = measurement["positions"].get(self.config.association)
                    if point is not None and track["position"] is not None:
                        distance = math.dist(point, track["position"])
                        if distance <= self.config.max_distance_m:
                            pairs.append((distance, tid, cid))
        assigned, used = {}, set()
        for _distance, tid, cid in sorted(pairs):
            if tid not in used and cid not in assigned:
                assigned[cid] = tid
                used.add(tid)
        self.assignments = []
        emitted = []
        for cid, measurement in current.items():
            point = measurement["positions"].get(self.config.association)
            if self.config.association != "image_iou" and point is None:
                continue  # Reject invalid rays; never fabricate a world location.
            tid = assigned.get(cid)
            if tid is None:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = dict(first_s=capture_s, evidence=[], reported=False)
            track = self.tracks[tid]
            track.update(box=measurement["box"], position=point, last_s=capture_s)
            track["evidence"].append(dict(capture_s=capture_s, component_id=cid))
            self.assignments.append(dict(component_id=cid, track_id=tid))
            if track["reported"] or len(track["evidence"]) < self.config.min_hits:
                continue
            track["reported"] = True
            if self.config.remember_reports:
                if any(
                    math.dist(point, old) <= self.config.max_distance_m
                    for old in self.memory.values()
                ):
                    self.suppressed_reports += 1
                    continue
                self.memory[tid] = tuple(point)
            self.report_count += 1
            emitted.append(
                dict(
                    report_id=tid,
                    component_id=cid,
                    first_prediction_s=track["first_s"],
                    completion_time_s=completion_s,
                    confirmation_latency_s=completion_s - track["first_s"],
                    evidence=[dict(item) for item in track["evidence"]],
                )
            )
        return emitted


class TrackingPipeline:
    """Runner-compatible public-input adapter; no scene or evaluator handle."""

    def __init__(self, config, projector=None):
        self.tracker = StaticReportTracker(config)
        self.projectors = (
            {} if config.association == "image_iou" else {config.association: projector}
        )
        if config.association != "image_iou" and projector is None:
            raise ValueError("World association requires a public surface projector")
        self.provenance = dict(schema_version="static_report_tracking_v1", **config.to_dict())

    @property
    def report_count(self):
        return self.tracker.report_count

    def update(self, mask, capture_s, completion_s, *, pose=None, camera_pitch_rad=None):
        measurements = detections(mask, pose, camera_pitch_rad, self.projectors)
        return self.tracker.update(measurements, capture_s, completion_s)
