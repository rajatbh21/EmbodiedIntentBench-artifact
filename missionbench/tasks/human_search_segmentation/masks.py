"""Binary segmentation masks, and IoU computed from them.

The canonical internal form is a frozen set of set-pixel coordinates over an explicit
``height x width`` grid. V1 carries no array dependency and its masks are tiny, so a pixel
set is both sufficient and exactly auditable: a test can name the pixels it expects and an
IoU comes out of two set operations.

Wire encoding
-------------
One explicit, versioned encoding: a ``rows`` bitmap, one string per row, ``'1'`` for a set
pixel and ``'0'`` for background. The dimensions are stated *and* recoverable from the
bitmap, so a ragged or mis-sized bitmap is rejected rather than silently reshaped. Other
encodings a real dataset might ship -- COCO run-length, polygons, external PNG references --
would decode into this same :class:`BinaryMask` before any comparison; none is needed in V1,
and adding one is a new decoder, not a change to the evaluator.

Everything here is deterministic: decoding a bitmap yields the same pixel set every time, and
IoU is integer set arithmetic with no floating-point accumulation order to depend on.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from missionbench.schemas.loading import SchemaValidationError

__all__ = ["BinaryMask", "decode_mask", "mask_iou"]

_MASK_FIELDS: Final = ("height", "width", "rows")


@dataclass(frozen=True, slots=True)
class BinaryMask:
    """A binary mask as its set of foreground pixels over an explicit grid.

    ``pixels`` holds ``(row, column)`` pairs, zero-indexed from the top-left. An empty
    ``pixels`` set is a legitimate mask -- a frame where the model segmented nothing -- and is
    handled deliberately by :func:`mask_iou` rather than treated as an error.
    """

    height: int
    width: int
    pixels: frozenset[tuple[int, int]]

    @property
    def area(self) -> int:
        return len(self.pixels)

    @property
    def is_empty(self) -> bool:
        return not self.pixels

    def to_rows(self) -> list[str]:
        """Render back to the ``rows`` bitmap. Round-trips ``decode_mask`` exactly."""
        return [
            "".join("1" if (row, col) in self.pixels else "0" for col in range(self.width))
            for row in range(self.height)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"height": self.height, "width": self.width, "rows": self.to_rows()}


def decode_mask(payload: object, *, context: str = "mask") -> BinaryMask:
    """Decode a wire mask into a :class:`BinaryMask`, validating strictly.

    Rejects anything malformed -- a non-object, unknown fields, non-positive dimensions, a
    row count or row length that disagrees with the stated size, or a character other than
    ``'0'``/``'1'`` -- because a mask that decoded loosely would compare against ground truth
    on a grid neither side agreed to.
    """
    if not isinstance(payload, Mapping):
        raise SchemaValidationError(
            f"{context}: a mask must be an object with 'height', 'width', and 'rows', "
            f"got {type(payload).__name__}"
        )
    unknown = sorted(set(payload) - set(_MASK_FIELDS))
    if unknown:
        raise SchemaValidationError(
            f"{context}: unknown mask field(s) {unknown}; allowed fields are {list(_MASK_FIELDS)}"
        )

    height = _positive_int(payload.get("height"), context=context, field="height")
    width = _positive_int(payload.get("width"), context=context, field="width")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise SchemaValidationError(f"{context}: field 'rows' must be a list of strings")
    if len(rows) != height:
        raise SchemaValidationError(
            f"{context}: field 'rows' has {len(rows)} row(s) but 'height' is {height}"
        )

    pixels: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        if not isinstance(row, str):
            raise SchemaValidationError(f"{context}: rows[{row_index}] must be a string")
        if len(row) != width:
            raise SchemaValidationError(
                f"{context}: rows[{row_index}] has length {len(row)} but 'width' is {width}"
            )
        for col_index, char in enumerate(row):
            if char == "1":
                pixels.add((row_index, col_index))
            elif char != "0":
                raise SchemaValidationError(
                    f"{context}: rows[{row_index}] contains {char!r}; only '0' and '1' are allowed"
                )

    return BinaryMask(height=height, width=width, pixels=frozenset(pixels))


def mask_iou(a: BinaryMask, b: BinaryMask) -> float:
    """Intersection-over-union of two masks on the same grid.

    Dimension mismatch is an error, never a silent zero: two masks of different sizes were
    never describing the same image, and comparing them would fabricate an overlap. Both
    masks empty yields ``0.0`` -- there is no positive evidence of a target, so it must not
    read as a perfect match and let an empty prediction "find" an empty region.
    """
    if a.height != b.height or a.width != b.width:
        raise SchemaValidationError(
            f"cannot compute IoU for masks of different dimensions: "
            f"{a.height}x{a.width} vs {b.height}x{b.width}"
        )
    if a.is_empty and b.is_empty:
        return 0.0
    intersection = len(a.pixels & b.pixels)
    union = len(a.pixels | b.pixels)
    return intersection / union


def _positive_int(value: object, *, context: str, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise SchemaValidationError(
            f"{context}: field {field!r} must be a positive integer, got {value!r}"
        )
    return value
