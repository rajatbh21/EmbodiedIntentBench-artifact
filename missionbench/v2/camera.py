"""The orthographic camera: from a UAV position to one rendered observation.

At each capture the renderer computes the rectangular footprint centred on the UAV,
reads exactly that window from the world (decimated to the output resolution during the
read), composites the visible synthetic objects, and returns the RGB image together
with the ground truth the evaluator will need.

The ground-truth fields on an :class:`Observation` exist **for the evaluator only**.
The runner hands the policy a runtime state and hands the executor ``observation.rgb``
-- never the masks, never the visible-object ids. That boundary is what makes V2 an
honest closed loop rather than an oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from missionbench.v2.objects import ObjectLayer, RenderedObjects
from missionbench.v2.scenario import CameraSpec
from missionbench.v2.world import _WorldBase

__all__ = ["CameraRenderer", "Observation"]


@dataclass(frozen=True, slots=True)
class Observation:
    """One rendered camera capture, with its hidden ground truth alongside."""

    scenario_id: str
    observation_id: int
    scheduled_capture_time_s: float
    position_m: tuple[float, float]
    footprint_m: tuple[float, float, float, float]  # x0, y0, x1, y1 in world meters
    rgb: np.ndarray  # uint8 (H, W, 3) -- the only pixel data an executor may see
    # -- evaluator-only fields ----------------------------------------------------------
    semantic_gt: np.ndarray  # uint8 (H, W)
    instance_gt: np.ndarray  # int32 (H, W)
    visible_target_ids: tuple[str, ...]
    visible_distractor_ids: tuple[str, ...]
    valid: np.ndarray  # bool (H, W)
    source_window_px: tuple[int, int, int, int]
    provenance: dict[str, Any]

    @property
    def meters_per_output_pixel_x(self) -> float:
        x0, _, x1, _ = self.footprint_m
        return (x1 - x0) / self.rgb.shape[1]

    @property
    def meters_per_output_pixel_y(self) -> float:
        _, y0, _, y1 = self.footprint_m
        return (y1 - y0) / self.rgb.shape[0]

    def pixel_to_world_m(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Map an output-crop pixel back to world meters (crop pixel centres)."""
        x0, y0, _, _ = self.footprint_m
        return (
            x0 + (x_px + 0.5) * self.meters_per_output_pixel_x,
            y0 + (y_px + 0.5) * self.meters_per_output_pixel_y,
        )


class CameraRenderer:
    """Renders position-dependent observations from the world and the object layer."""

    def __init__(
        self,
        *,
        scenario_id: str,
        world: _WorldBase,
        objects: ObjectLayer,
        camera: CameraSpec,
    ) -> None:
        self._scenario_id = scenario_id
        self._world = world
        self._objects = objects
        self._camera = camera

    def footprint_at(self, position_m: tuple[float, float]) -> tuple[float, float, float, float]:
        x, y = position_m
        hw = self._camera.footprint_width_m / 2.0
        hh = self._camera.footprint_height_m / 2.0
        return (x - hw, y - hh, x + hw, y + hh)

    def render(
        self,
        *,
        position_m: tuple[float, float],
        capture_time_s: float,
        observation_id: int,
    ) -> Observation:
        footprint = self.footprint_at(position_m)
        read = self._world.read_window_m(
            position_m,
            self._camera.footprint_width_m,
            self._camera.footprint_height_m,
            self._camera.output_width_px,
            self._camera.output_height_px,
        )
        rgb = read.rgb.copy()
        rendered: RenderedObjects = self._objects.render(rgb, footprint, read.valid)

        return Observation(
            scenario_id=self._scenario_id,
            observation_id=observation_id,
            scheduled_capture_time_s=capture_time_s,
            position_m=position_m,
            footprint_m=footprint,
            rgb=rgb,
            semantic_gt=rendered.semantic,
            instance_gt=rendered.instance,
            visible_target_ids=rendered.visible_target_ids,
            visible_distractor_ids=rendered.visible_distractor_ids,
            valid=read.valid,
            source_window_px=read.source_window_px,
            provenance={
                "world": dict(self._world.provenance),
                "capture_time_s": capture_time_s,
                "position_m": list(position_m),
                # Evaluator-only: lets scoring map GT instance ids back to object ids.
                "instance_object_ids": self._objects.instance_object_ids,
                # Evaluator/debug-only: physical->pixel projection stats per image asset.
                "asset_projections": rendered.projections,
            },
        )

    def valid_fraction(self, position_m: tuple[float, float]) -> float:
        return self._world.valid_fraction_m(
            position_m, self._camera.footprint_width_m, self._camera.footprint_height_m
        )
