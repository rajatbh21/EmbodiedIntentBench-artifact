"""Classify a configured uncertainty box in a public camera frustum, not visibility."""

import itertools
import math

import numpy as np

from experiments.v4_benchmark.projection import rotation_xyzw

HORIZONTAL_MARGIN_M = 1.0
VERTICAL_MARGIN_M = 0.9
STATUSES = ("inside_frustum", "outside_frustum", "boundary_uncertain", "unknown_geometry")


def classify(
    point,
    shape,
    pose,
    pitch,
    hfov=1.2,
    horizontal_margin=HORIZONTAL_MARGIN_M,
    vertical_margin=VERTICAL_MARGIN_M,
):
    """Plane tests on eight corners; margins are assumptions, not confidence bounds."""
    h, w = shape
    if (
        min(h, w) < 2
        or not all(math.isfinite(v) for v in (pitch, hfov, horizontal_margin, vertical_margin))
        or not 0 < hfov < math.pi
        or min(horizontal_margin, vertical_margin) < 0
    ):
        raise ValueError("Invalid camera or uncertainty-box configuration")
    if point is None or np.shape(point) != (3,) or not np.isfinite(point).all():
        return dict(status="unknown_geometry", center_pixel=None, occlusion_known=False)
    body = rotation_xyzw(pose["quaternion_xyzw"])
    origin = np.asarray(pose["position_m"], float)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("Invalid public camera position")
    origin = origin + body @ np.array([0, 0, -0.1])
    c, s = math.cos(pitch), math.sin(pitch)
    camera = body @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    offsets = np.asarray(
        list(
            itertools.product(
                (-horizontal_margin, horizontal_margin),
                (-horizontal_margin, horizontal_margin),
                (-vertical_margin, vertical_margin),
            )
        )
    )
    corners = (np.asarray(point) + offsets - origin) @ camera
    x, y, z = corners.T
    focal, cx, cy = w / (2 * math.tan(hfov / 2)), (w - 1) / 2, (h - 1) / 2
    planes = np.array(
        [
            x - 1e-6,
            cx * x - focal * y,
            (w - 1 - cx) * x + focal * y,
            cy * x - focal * z,
            (h - 1 - cy) * x + focal * z,
        ]
    )
    if np.any(np.all(planes < 0, axis=1)):
        status = "outside_frustum"
    elif np.all(planes >= 0):
        status = "inside_frustum"
    else:
        status = "boundary_uncertain"
    center = (np.asarray(point) - origin) @ camera
    pixel = (
        [float(cx - focal * center[1] / center[0]), float(cy - focal * center[2] / center[0])]
        if center[0] > 1e-6
        else None
    )
    return dict(status=status, center_pixel=pixel, occlusion_known=False)


def snapshot(tracker, now, pose, shape, pitch):
    """Read only completed predicted positions and a declared terrain prior."""
    state = tracker.tracker
    if state.last_completion > now + 1e-8:
        raise ValueError("Future predictions cannot enter frustum feedback")
    projector = tracker.projectors.get(state.config.association)
    tracks = []
    for tid, track in sorted(state.tracks.items()):
        if now - track["last_s"] > state.config.max_gap_s:
            continue
        point = None
        if projector is not None and track.get("position") is not None:
            x, y = track["position"]
            z = float(projector.terrain.sample(x, y)) if projector.terrain is not None else 0
            if math.isfinite(z):
                point = [float(x), float(y), z + projector.anchor_height_m]
        result = classify(
            point, shape, pose, pitch, hfov=projector.hfov if projector is not None else 1.2
        )
        tracks.append(
            dict(
                track_id=tid,
                estimate_capture_s=track["last_s"],
                predicted_position_m=point,
                **result,
            )
        )
    return dict(
        schema_version="predicted_track_frustum_v1",
        available_at_s=now,
        tracks=tracks,
        uses_ground_truth=False,
        occlusion_known=False,
        horizontal_margin_m=HORIZONTAL_MARGIN_M,
        vertical_margin_m=VERTICAL_MARGIN_M,
        margins_are_calibrated_bounds=False,
    )
