"""The synthetic object layer: targets and distractors composited over the aerial world.

The source raster is never modified. Objects live in world coordinates and are painted
into each camera crop at render time, together with the semantic and instance ground
truth that scoring needs. **These are synthetic rescue-target markers, not realistic
humans** -- a recognisable high-visibility pattern the lightweight V2 executors can find
from RGB alone, with distractors wearing similar-but-imperfect appearances.

Rendering supports rotation, partial visibility at crop boundaries, deterministic
z-order (higher ``z_order`` paints later; ties break on ``object_id``), alpha blending
into the RGB, and exact per-pixel masks. Semantic labels: 0 background, 1 target,
2 distractor. Instance labels: 0 background, else 1-based index into the scenario's
object list (stable across the mission).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from missionbench.v2.assets import LoadedAsset
from missionbench.v2.scenario import ObjectSpec

__all__ = [
    "SEMANTIC_BACKGROUND",
    "SEMANTIC_DISTRACTOR",
    "SEMANTIC_TARGET",
    "ObjectLayer",
    "RenderedObjects",
]

SEMANTIC_BACKGROUND: Final = 0
SEMANTIC_TARGET: Final = 1
SEMANTIC_DISTRACTOR: Final = 2


@dataclass(frozen=True, slots=True)
class RenderedObjects:
    """What the object layer contributed to one crop."""

    semantic: np.ndarray  # uint8 (H, W): 0 bg / 1 target / 2 distractor
    instance: np.ndarray  # int32 (H, W): 0 bg, else 1-based object ordinal
    visible_target_ids: tuple[str, ...]
    visible_distractor_ids: tuple[str, ...]
    #: Per rendered object: requested physical size, projected pixel sizes, visible
    #: pixel area, and visible fraction of the transformed mask (evaluator/debug only).
    projections: dict[str, dict[str, float]] = field(default_factory=dict)


class ObjectLayer:
    """Holds the scenario's objects and paints the ones a footprint can see.

    Two render modes per object: the V2.0 ``procedural_marker`` (parametric shapes) and
    the V2.2 ``image_asset`` (an RGB(A) asset composited with its own mask). Ground
    truth always comes from the object's *mask geometry* -- the procedural inside-test
    or the transformed asset mask -- never from thresholding the composited RGB.
    """

    def __init__(
        self,
        objects: tuple[ObjectSpec, ...],
        assets: Mapping[str, LoadedAsset] | None = None,
    ) -> None:
        # Paint order: z_order then object_id -- fully deterministic under overlap.
        self._objects = tuple(sorted(objects, key=lambda o: (o.z_order, o.object_id)))
        #: 1-based instance ids follow the *scenario declaration* order, not paint order,
        #: so an id is stable regardless of z-order edits.
        self._instance_ids = {obj.object_id: i + 1 for i, obj in enumerate(objects)}
        self._assets = dict(assets or {})
        for obj in self._objects:
            if obj.render_mode == "image_asset" and obj.asset_id not in self._assets:
                from missionbench.schemas.loading import SchemaValidationError

                raise SchemaValidationError(
                    f"object {obj.object_id!r} references asset {obj.asset_id!r}, which is "
                    "not in the loaded asset manifest"
                )

    @property
    def objects(self) -> tuple[ObjectSpec, ...]:
        return self._objects

    def instance_id(self, object_id: str) -> int:
        return self._instance_ids[object_id]

    @property
    def instance_object_ids(self) -> dict[str, str]:
        """The full ``str(instance_id) -> object_id`` map, for observation provenance."""
        return {str(i): object_id for object_id, i in self._instance_ids.items()}

    # -- visibility ---------------------------------------------------------------------

    def intersecting(
        self, footprint_m: tuple[float, float, float, float]
    ) -> tuple[ObjectSpec, ...]:
        """Objects whose bounding circle touches the footprint rectangle."""
        fx0, fy0, fx1, fy1 = footprint_m
        hits = []
        for obj in self._objects:
            radius = 0.5 * float(np.hypot(obj.width_m, obj.height_m))
            x, y = obj.position_m
            if fx0 - radius <= x <= fx1 + radius and fy0 - radius <= y <= fy1 + radius:
                hits.append(obj)
        return tuple(hits)

    def visible_target_ids_in(
        self, footprint_m: tuple[float, float, float, float], samples_x: int, samples_y: int
    ) -> tuple[str, ...]:
        """Target ids geometrically visible in ``footprint_m``, without rendering.

        Used for *skipped* scheduled observations: the executor must never run on a
        skipped capture, but a target that was visible only while the executor was busy
        must still count as encountered-but-missed rather than silently vanishing from
        the mission diagnostics. This runs the same inside-test the renderer uses, on a
        coarse sample grid over the footprint -- no raster read, no RGB, no validity
        mask (a target standing on nodata would still count; diagnostic-only, and
        documented as such).
        """
        fx0, fy0, fx1, fy1 = footprint_m
        xs = fx0 + (np.arange(samples_x) + 0.5) * (fx1 - fx0) / samples_x
        ys = fy0 + (np.arange(samples_y) + 0.5) * (fy1 - fy0) / samples_y
        world_x = xs[None, :]
        world_y = ys[:, None]
        visible: list[str] = []
        for obj in self.intersecting(footprint_m):
            if not obj.is_target:
                continue
            inside, _ = _object_masks(obj, world_x, world_y)
            if inside.any():
                visible.append(obj.object_id)
        return tuple(visible)

    # -- rendering ----------------------------------------------------------------------

    def render(
        self,
        rgb: np.ndarray,
        footprint_m: tuple[float, float, float, float],
        valid: np.ndarray,
    ) -> RenderedObjects:
        """Paint visible objects into ``rgb`` (in place) and return the ground truth.

        ``rgb`` is the crop already resampled to output resolution; the footprint gives
        its world extent, from which the per-pixel world coordinates follow. Objects are
        only painted onto valid raster (an object cannot stand on nodata scenery).
        """
        h, w = rgb.shape[:2]
        fx0, fy0, fx1, fy1 = footprint_m
        # World coordinates of each output pixel centre.
        xs = fx0 + (np.arange(w) + 0.5) * (fx1 - fx0) / w
        ys = fy0 + (np.arange(h) + 0.5) * (fy1 - fy0) / h
        world_x = xs[None, :]
        world_y = ys[:, None]

        semantic = np.zeros((h, w), dtype=np.uint8)
        instance = np.zeros((h, w), dtype=np.int32)
        visible_targets: list[str] = []
        visible_distractors: list[str] = []
        projections: dict[str, dict[str, float]] = {}

        for obj in self.intersecting(footprint_m):
            if obj.render_mode == "image_asset":
                inside = self._composite_image_asset(obj, rgb, footprint_m, valid, projections)
            else:
                inside = self._paint_procedural(obj, rgb, world_x, world_y, valid)
            if inside is None or not inside.any():
                continue

            label = SEMANTIC_TARGET if obj.is_target else SEMANTIC_DISTRACTOR
            semantic[inside] = label
            instance[inside] = self._instance_ids[obj.object_id]
            (visible_targets if obj.is_target else visible_distractors).append(obj.object_id)

        return RenderedObjects(
            semantic=semantic,
            instance=instance,
            visible_target_ids=tuple(visible_targets),
            visible_distractor_ids=tuple(visible_distractors),
            projections=projections,
        )

    def _paint_procedural(
        self,
        obj: ObjectSpec,
        rgb: np.ndarray,
        world_x: np.ndarray,
        world_y: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray | None:
        """The V2.0 parametric marker painter. Returns the object's GT mask in the crop."""
        inside, stripe = _object_masks(obj, world_x, world_y)
        inside &= valid
        if not inside.any():
            return None

        appearance = obj.appearance
        body = np.array(appearance.get("body_rgb") or (220, 60, 20), dtype=np.float32)
        alpha = float(appearance.get("alpha") or 1.0)
        paint = np.zeros((*inside.shape, 3), dtype=np.float32)
        paint[inside] = body
        if appearance.get("stripe") and appearance.get("stripe_rgb") is not None:
            paint[stripe & inside] = np.array(appearance["stripe_rgb"], dtype=np.float32)

        blended = rgb.astype(np.float32)
        blended[inside] = (1.0 - alpha) * blended[inside] + alpha * paint[inside]
        rgb[inside] = np.clip(blended[inside], 0, 255).astype(np.uint8)
        return inside

    def _composite_image_asset(
        self,
        obj: ObjectSpec,
        rgb: np.ndarray,
        footprint_m: tuple[float, float, float, float],
        valid: np.ndarray,
        projections: dict[str, dict[str, float]],
    ) -> np.ndarray | None:
        """Composite one image asset into the crop and return its GT mask.

        The asset's smooth alpha (bilinear-resized, rotated) blends the RGB so edges
        stay anti-aliased; the *binary* GT mask travels through the same spatial
        transform with nearest-neighbour resampling, so ground truth is exact geometry
        and never derived from the composited pixels. Physical size drives everything:
        metres -> output pixels via the footprint scale; a target that projects to zero
        pixels is skipped and recorded, never silently enlarged.
        """
        from PIL import Image

        asset = self._assets[obj.asset_id]  # presence checked at construction
        h, w = rgb.shape[:2]
        fx0, fy0, fx1, fy1 = footprint_m
        px_per_m_x = w / (fx1 - fx0)
        px_per_m_y = h / (fy1 - fy0)

        target_w_px = round(obj.width_m * px_per_m_x)
        target_h_px = round(obj.height_m * px_per_m_y)
        stats = {
            "requested_width_m": obj.width_m,
            "requested_height_m": obj.height_m,
            "projected_width_px": float(target_w_px),
            "projected_height_px": float(target_h_px),
            "visible_px": 0.0,
            "visible_fraction": 0.0,
        }
        projections[obj.object_id] = stats
        if target_w_px < 1 or target_h_px < 1:
            return None  # zero-pixel projection: recorded above, never enlarged

        rgb_img = Image.fromarray(asset.rgb).resize((target_w_px, target_h_px), Image.BILINEAR)
        alpha_img = Image.fromarray((asset.alpha * 255).astype(np.uint8)).resize(
            (target_w_px, target_h_px), Image.BILINEAR
        )
        mask_img = Image.fromarray(asset.mask.astype(np.uint8) * 255).resize(
            (target_w_px, target_h_px), Image.NEAREST
        )
        if obj.rotation_deg:
            rgb_img = rgb_img.rotate(-obj.rotation_deg, expand=True, resample=Image.BILINEAR)
            alpha_img = alpha_img.rotate(-obj.rotation_deg, expand=True, resample=Image.BILINEAR)
            mask_img = mask_img.rotate(-obj.rotation_deg, expand=True, resample=Image.NEAREST)

        patch_rgb = np.array(rgb_img, dtype=np.float32)
        patch_alpha = np.array(alpha_img, dtype=np.float32) / 255.0
        patch_mask = np.array(mask_img) >= 128
        total_mask_px = int(patch_mask.sum())
        ph, pw = patch_mask.shape

        # Centre-anchored placement in crop pixels, then clip to the crop.
        cx = (obj.position_m[0] - fx0) * px_per_m_x
        cy = (obj.position_m[1] - fy0) * px_per_m_y
        x0 = round(cx - pw / 2.0)
        y0 = round(cy - ph / 2.0)
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, x0 + pw), min(h, y0 + ph)
        if cx1 <= cx0 or cy1 <= cy0:
            return None  # entirely outside the crop
        sx0, sy0 = cx0 - x0, cy0 - y0
        sx1, sy1 = sx0 + (cx1 - cx0), sy0 + (cy1 - cy0)

        alpha_clip = patch_alpha[sy0:sy1, sx0:sx1] * obj.opacity
        alpha_clip = alpha_clip * valid[cy0:cy1, cx0:cx1]  # nothing renders onto nodata
        rgb_clip = np.clip(patch_rgb[sy0:sy1, sx0:sx1] * obj.brightness_factor, 0, 255)

        region = rgb[cy0:cy1, cx0:cx1].astype(np.float32)
        blended = (1.0 - alpha_clip[..., None]) * region + alpha_clip[..., None] * rgb_clip
        rgb[cy0:cy1, cx0:cx1] = np.clip(blended, 0, 255).astype(np.uint8)

        inside = np.zeros((h, w), dtype=bool)
        inside[cy0:cy1, cx0:cx1] = patch_mask[sy0:sy1, sx0:sx1] & valid[cy0:cy1, cx0:cx1]
        visible_px = int(inside.sum())
        stats["visible_px"] = float(visible_px)
        stats["visible_fraction"] = visible_px / total_mask_px if total_mask_px > 0 else 0.0
        return inside if visible_px > 0 else None


def _object_masks(
    obj: ObjectSpec, world_x: np.ndarray, world_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel inside/stripe tests for one object, in its rotated local frame."""
    theta = np.deg2rad(obj.rotation_deg)
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    dx = world_x - obj.position_m[0]
    dy = world_y - obj.position_m[1]
    # Inverse-rotate the world offset into the object's axis-aligned local frame.
    local_x = cos_t * dx + sin_t * dy
    local_y = -sin_t * dx + cos_t * dy
    half_w, half_h = obj.width_m / 2.0, obj.height_m / 2.0

    if obj.appearance.get("shape") == "ellipse":
        inside = (local_x / half_w) ** 2 + (local_y / half_h) ** 2 <= 1.0
    else:
        inside = (np.abs(local_x) <= half_w) & (np.abs(local_y) <= half_h)
    # The stripe is a central band across the short axis -- the "rescue marker" pattern.
    stripe = np.abs(local_y) <= (half_h / 3.0)
    return inside, stripe
