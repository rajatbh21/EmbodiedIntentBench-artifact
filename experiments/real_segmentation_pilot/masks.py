"""Deterministic nearest-neighbour resizing of a binary mask onto the GT coordinate system.

A model often runs at a different input resolution than the ground truth was annotated at. Its
output mask must be mapped back to the GT grid before evaluation, and for a *binary* mask the
only correct interpolation is nearest-neighbour -- bilinear would invent fractional occupancy
the pixel set cannot represent. The mapping is integer arithmetic, so it is exactly
reproducible.
"""

from __future__ import annotations

from missionbench.tasks.human_search_segmentation.masks import BinaryMask

__all__ = ["resize_nearest"]


def resize_nearest(mask: BinaryMask, target_height: int, target_width: int) -> BinaryMask:
    """Return ``mask`` resampled to ``target_height x target_width`` by nearest neighbour.

    Each target pixel takes the value of the source pixel it maps back to under
    ``floor(index * source / target)``. A mask already at the target size is returned
    unchanged.
    """
    if target_height < 1 or target_width < 1:
        raise ValueError(f"target dimensions must be positive, got {target_height}x{target_width}")
    if mask.height == target_height and mask.width == target_width:
        return mask

    source_height, source_width = mask.height, mask.width
    row_source = [
        min(source_height - 1, (r * source_height) // target_height) for r in range(target_height)
    ]
    col_source = [
        min(source_width - 1, (c * source_width) // target_width) for c in range(target_width)
    ]

    pixels = {
        (r, c)
        for r in range(target_height)
        for c in range(target_width)
        if (row_source[r], col_source[c]) in mask.pixels
    }
    return BinaryMask(height=target_height, width=target_width, pixels=frozenset(pixels))
