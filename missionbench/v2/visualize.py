"""Debug visualisation: a world overview and a handful of observation panels.

Headless by design -- everything writes PNG files, nothing opens a window, and nothing
here is required for a mission to run. The overview shows the mission's slice of the
world with the trajectory, objects, and sample footprints; an observation panel shows
RGB / ground truth / prediction side by side with capture and completion annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from missionbench.v2.camera import Observation
from missionbench.v2.objects import ObjectLayer
from missionbench.v2.scenario import V2Scenario
from missionbench.v2.trajectory import PolylineTrajectory
from missionbench.v2.world import _WorldBase

__all__ = ["render_observation_panel", "render_overview"]

_TARGET_RGB = (0, 220, 60)
_DISTRACTOR_RGB = (250, 210, 0)
_PATH_RGB = (40, 120, 255)
_FOOTPRINT_RGB = (255, 255, 255)


def render_overview(
    scenario: V2Scenario,
    world: _WorldBase,
    trajectory: PolylineTrajectory,
    objects: ObjectLayer,
    output_path: Path,
    *,
    footprint_samples: int = 4,
    max_size_px: int = 1200,
) -> Path:
    """Write a world-overview PNG for the mission's region of interest."""
    from PIL import Image, ImageDraw

    # The mission's bounding box in meters, padded by one footprint.
    xs = [p[0] for p in trajectory.waypoints_m] + [o.position_m[0] for o in objects.objects]
    ys = [p[1] for p in trajectory.waypoints_m] + [o.position_m[1] for o in objects.objects]
    pad_x = scenario.camera.footprint_width_m
    pad_y = scenario.camera.footprint_height_m
    x0, x1 = min(xs) - pad_x, max(xs) + pad_x
    y0, y1 = min(ys) - pad_y, max(ys) + pad_y

    width_m, height_m = x1 - x0, y1 - y0
    scale = min(max_size_px / width_m, max_size_px / height_m)
    out_w = max(64, int(width_m * scale))
    out_h = max(64, int(height_m * scale))

    read = world.read_window_m(((x0 + x1) / 2.0, (y0 + y1) / 2.0), width_m, height_m, out_w, out_h)
    image = Image.fromarray(read.rgb)
    draw = ImageDraw.Draw(image)

    def to_px(point_m: tuple[float, float]) -> tuple[float, float]:
        return ((point_m[0] - x0) / width_m * out_w, (point_m[1] - y0) / height_m * out_h)

    # Trajectory, start and end markers.
    points = [to_px(p) for p in trajectory.waypoints_m]
    draw.line(points, fill=_PATH_RGB, width=3)
    draw.ellipse(_dot(points[0], 6), fill=(0, 255, 255))
    draw.ellipse(_dot(points[-1], 6), fill=(255, 0, 255))

    # Objects.
    for obj in objects.objects:
        colour = _TARGET_RGB if obj.is_target else _DISTRACTOR_RGB
        draw.ellipse(_dot(to_px(obj.position_m), 5), outline=colour, width=3)

    # Sample camera footprints along the path.
    duration = trajectory.duration_s
    for i in range(footprint_samples):
        t = duration * i / max(1, footprint_samples - 1)
        cx, cy = trajectory.position_at(t)
        hw, hh = scenario.camera.footprint_width_m / 2, scenario.camera.footprint_height_m / 2
        draw.rectangle(
            [to_px((cx - hw, cy - hh)), to_px((cx + hw, cy + hh))],
            outline=_FOOTPRINT_RGB,
            width=1,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def render_observation_panel(
    observation: Observation,
    prediction_mask: np.ndarray | None,
    output_path: Path,
    *,
    annotation: dict[str, Any] | None = None,
) -> Path:
    """Write RGB | ground truth | prediction side by side, with a caption strip."""
    from PIL import Image, ImageDraw

    h, w = observation.rgb.shape[:2]
    gt = np.zeros((h, w, 3), dtype=np.uint8)
    gt[observation.semantic_gt == 1] = _TARGET_RGB
    gt[observation.semantic_gt == 2] = _DISTRACTOR_RGB
    pred = np.zeros((h, w, 3), dtype=np.uint8)
    if prediction_mask is not None:
        pred[prediction_mask] = (255, 80, 80)

    caption_h = 40
    panel = Image.new("RGB", (w * 3 + 8, h + caption_h), (24, 24, 24))
    panel.paste(Image.fromarray(observation.rgb), (0, caption_h))
    panel.paste(Image.fromarray(gt), (w + 4, caption_h))
    panel.paste(Image.fromarray(pred), (2 * w + 8, caption_h))

    draw = ImageDraw.Draw(panel)
    fields = {
        "obs": observation.observation_id,
        "t_capture": f"{observation.scheduled_capture_time_s:.1f}s",
        "pos": f"({observation.position_m[0]:.1f}, {observation.position_m[1]:.1f})m",
        **(annotation or {}),
    }
    caption = "  ".join(f"{k}={v}" for k, v in fields.items())
    draw.text((4, 4), caption, fill=(230, 230, 230))
    caption2 = "RGB | GT (green=target, yellow=distractor) | prediction"
    draw.text((4, 22), caption2, fill=(160, 160, 160))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return output_path


def _dot(centre: tuple[float, float], radius: float) -> list[float]:
    return [centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius]
