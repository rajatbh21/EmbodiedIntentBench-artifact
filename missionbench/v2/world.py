"""World sources: the static 2D aerial raster the mission flies over.

A ``WorldSource`` exposes the raster's size and scale, converts between local world
meters and source pixels, and serves **windowed** reads -- a camera observation never
loads the whole raster. Three backends:

- :class:`RasterioWorld` -- GeoTIFF via ``rasterio``. Reads only the requested window
  (decimated to the output size during the read, so a 2000-pixel footprint over a
  JPEG-tiled orthomosaic decodes a handful of tiles, not the mosaic). Validity comes
  from the dataset mask when present, else the configured invalid-pixel rule.
- :class:`ImageWorld` -- plain PNG/JPEG/TIFF via Pillow, for small worlds. The image is
  decoded once and windows are sliced from the array.
- :class:`ArrayWorld` -- an in-memory array, for tests and procedural fixtures.

Local coordinates: origin at the raster's top-left, x right, y down, in meters;
``meters_per_pixel`` is the single scale between the two. The source CRS and affine
transform, when present, are carried as provenance only -- the mission runs in local
meters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from missionbench.schemas.loading import SchemaValidationError

__all__ = [
    "ArrayWorld",
    "ImageWorld",
    "RasterioWorld",
    "WindowRead",
    "open_world",
]

#: How far a GeoTIFF's own resolution may disagree with the scenario's declared
#: meters_per_pixel before the mismatch is an error rather than a rounding artefact.
_MPP_TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class WindowRead:
    """One camera window: RGB at the requested output size, plus per-pixel validity.

    ``valid`` is False where the window fell outside the raster or on nodata; those RGB
    pixels are zeroed so invalid scenery can never be mistaken for content.
    """

    rgb: np.ndarray  # uint8, (H, W, 3)
    valid: np.ndarray  # bool, (H, W)
    #: The source-pixel rectangle actually requested (x0, y0, x1, y1), for provenance.
    source_window_px: tuple[int, int, int, int]


class _WorldBase:
    """Shared conversions. Subclasses provide ``_read_native``."""

    width_px: int
    height_px: int
    meters_per_pixel: float
    provenance: dict[str, Any]

    # -- coordinate conversions ---------------------------------------------------------

    @property
    def width_m(self) -> float:
        return self.width_px * self.meters_per_pixel

    @property
    def height_m(self) -> float:
        return self.height_px * self.meters_per_pixel

    def meters_to_pixels(self, x_m: float, y_m: float) -> tuple[float, float]:
        return (x_m / self.meters_per_pixel, y_m / self.meters_per_pixel)

    def pixels_to_meters(self, x_px: float, y_px: float) -> tuple[float, float]:
        return (x_px * self.meters_per_pixel, y_px * self.meters_per_pixel)

    # -- windowed reading ---------------------------------------------------------------

    def read_window_m(
        self,
        centre_m: tuple[float, float],
        width_m: float,
        height_m: float,
        out_width_px: int,
        out_height_px: int,
    ) -> WindowRead:
        """Read the axis-aligned window centred at ``centre_m``, resampled to the output size.

        The window may extend past the raster edge; the overhang is returned as invalid
        (and black) rather than clamped, so the observation geometry stays exactly what
        the camera footprint says it is.
        """
        cx_px, cy_px = self.meters_to_pixels(*centre_m)
        half_w = (width_m / self.meters_per_pixel) / 2.0
        half_h = (height_m / self.meters_per_pixel) / 2.0
        x0 = int(np.floor(cx_px - half_w))
        y0 = int(np.floor(cy_px - half_h))
        x1 = int(np.ceil(cx_px + half_w))
        y1 = int(np.ceil(cy_px + half_h))
        return self._read_native(x0, y0, x1, y1, out_width_px, out_height_px)

    def _read_native(
        self, x0: int, y0: int, x1: int, y1: int, out_w: int, out_h: int
    ) -> WindowRead:  # pragma: no cover - abstract
        raise NotImplementedError

    def valid_fraction_m(
        self, centre_m: tuple[float, float], width_m: float, height_m: float
    ) -> float:
        """Fraction of a window that is valid raster, at a coarse probe resolution."""
        probe = self.read_window_m(centre_m, width_m, height_m, 32, 32)
        return float(probe.valid.mean())


def _paste_window(
    full: np.ndarray,
    valid_full: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    out_w: int,
    out_h: int,
    array: np.ndarray,
    array_valid: np.ndarray | None,
    width_px: int,
    height_px: int,
) -> WindowRead:
    """Slice an in-memory array for the window, padding out-of-bounds as invalid."""
    ix0, iy0 = max(0, x0), max(0, y0)
    ix1, iy1 = min(width_px, x1), min(height_px, y1)
    if ix1 > ix0 and iy1 > iy0:
        full[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = array[iy0:iy1, ix0:ix1]
        if array_valid is not None:
            valid_full[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = array_valid[iy0:iy1, ix0:ix1]
        else:
            valid_full[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = True
    rgb = _resize_rgb(full, out_w, out_h)
    valid = _resize_mask_nearest(valid_full, out_w, out_h)
    rgb[~valid] = 0
    return WindowRead(rgb=rgb, valid=valid, source_window_px=(x0, y0, x1, y1))


class ArrayWorld(_WorldBase):
    """A world held in memory. The test and procedural-fixture backend."""

    def __init__(
        self,
        rgb: np.ndarray,
        meters_per_pixel: float,
        *,
        valid: np.ndarray | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> None:
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise SchemaValidationError("ArrayWorld expects a uint8 (H, W, 3) array")
        self._rgb = rgb
        # Without an explicit mask, apply the all-zero nodata rule: pure-black pixels are
        # treated as invalid, matching the GeoTIFF fallback behaviour.
        self._valid = valid if valid is not None else ~(rgb == 0).all(axis=2)
        self.height_px, self.width_px = rgb.shape[:2]
        self.meters_per_pixel = meters_per_pixel
        self.provenance = provenance or {"backend": "array"}

    def _read_native(self, x0, y0, x1, y1, out_w, out_h) -> WindowRead:
        h, w = y1 - y0, x1 - x0
        full = np.zeros((h, w, 3), dtype=np.uint8)
        valid_full = np.zeros((h, w), dtype=bool)
        return _paste_window(
            full,
            valid_full,
            x0,
            y0,
            x1,
            y1,
            out_w,
            out_h,
            self._rgb,
            self._valid,
            self.width_px,
            self.height_px,
        )


class ImageWorld(_WorldBase):
    """A plain image world (PNG/JPEG/simple TIFF) decoded once via Pillow."""

    def __init__(
        self,
        path: Path,
        meters_per_pixel: float,
        *,
        invalid_pixel_rule: str = "all_zero",
        provenance: dict[str, Any] | None = None,
    ) -> None:
        from PIL import Image

        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
        self._rgb = rgb
        self.height_px, self.width_px = rgb.shape[:2]
        self.meters_per_pixel = meters_per_pixel
        self._valid = (
            ~(rgb == 0).all(axis=2)
            if invalid_pixel_rule == "all_zero"
            else np.ones(rgb.shape[:2], bool)
        )
        self.provenance = {"backend": "pillow", "path": str(path), **(provenance or {})}

    def _read_native(self, x0, y0, x1, y1, out_w, out_h) -> WindowRead:
        h, w = y1 - y0, x1 - x0
        full = np.zeros((h, w, 3), dtype=np.uint8)
        valid_full = np.zeros((h, w), dtype=bool)
        return _paste_window(
            full,
            valid_full,
            x0,
            y0,
            x1,
            y1,
            out_w,
            out_h,
            self._rgb,
            self._valid,
            self.width_px,
            self.height_px,
        )


class RasterioWorld(_WorldBase):
    """A GeoTIFF world, read window by window through rasterio.

    The dataset stays open for the mission; each observation issues one decimated
    windowed read (``boundless`` so footprints may overhang the edge) plus a matching
    mask read. The CRS/affine transform are preserved as provenance only.
    """

    def __init__(
        self,
        path: Path,
        meters_per_pixel: float,
        *,
        invalid_pixel_rule: str = "all_zero",
    ) -> None:
        try:
            import rasterio
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise SchemaValidationError(
                "reading a GeoTIFF world requires rasterio; install the v2 extra "
                "(pip install 'missionbench[v2]')"
            ) from error
        import rasterio

        self._ds = rasterio.open(path)
        ds = self._ds
        if ds.count < 3:
            raise SchemaValidationError(
                f"{path.name}: expected >= 3 bands for an RGB world, found {ds.count}"
            )
        self.width_px = ds.width
        self.height_px = ds.height
        native = (abs(ds.transform.a) + abs(ds.transform.e)) / 2.0
        if native > 0 and abs(native - meters_per_pixel) / native > _MPP_TOLERANCE:
            raise SchemaValidationError(
                f"{path.name}: scenario meters_per_pixel {meters_per_pixel} disagrees with the "
                f"raster's own resolution {native:.6g} by more than {_MPP_TOLERANCE:.0%}. Fix the "
                "scenario, or use the raster value. (Web Mercator rasters carry the projection's "
                "nominal metre; the scenario value is authoritative once consistent.)"
            )
        self.meters_per_pixel = meters_per_pixel
        self._invalid_pixel_rule = invalid_pixel_rule
        self.provenance = {
            "backend": "rasterio",
            "path": str(path),
            "crs": str(ds.crs),
            "transform": list(ds.transform)[:6],
            "native_resolution": ds.res,
            "nodata": ds.nodata,
            "size_px": [ds.width, ds.height],
        }

    def close(self) -> None:
        self._ds.close()

    def _read_native(self, x0, y0, x1, y1, out_w, out_h) -> WindowRead:
        from rasterio.enums import Resampling
        from rasterio.windows import Window

        window = Window(x0, y0, x1 - x0, y1 - y0)
        rgb = self._ds.read(
            indexes=[1, 2, 3],
            window=window,
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
            boundless=True,
            fill_value=0,
        )
        rgb = np.transpose(rgb, (1, 2, 0)).astype(np.uint8, copy=False)
        mask = self._ds.read_masks(1, window=window, out_shape=(out_h, out_w), boundless=True)
        valid = mask > 0
        if self._invalid_pixel_rule == "all_zero":
            valid &= ~(rgb == 0).all(axis=2)
        rgb = rgb.copy()
        rgb[~valid] = 0
        return WindowRead(rgb=rgb, valid=valid, source_window_px=(x0, y0, x1, y1))


def open_world(
    image_path: Path,
    meters_per_pixel: float,
    *,
    invalid_pixel_rule: str = "all_zero",
) -> _WorldBase:
    """Open the right backend for ``image_path``."""
    if not image_path.is_file():
        raise SchemaValidationError(f"world image not found: {image_path}")
    suffix = image_path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        return RasterioWorld(image_path, meters_per_pixel, invalid_pixel_rule=invalid_pixel_rule)
    if suffix in (".png", ".jpg", ".jpeg"):
        return ImageWorld(image_path, meters_per_pixel, invalid_pixel_rule=invalid_pixel_rule)
    raise SchemaValidationError(
        f"unsupported world image format {suffix!r}; supported: .tif .tiff .png .jpg .jpeg"
    )


# --- resampling helpers (shared with the renderer) -------------------------------------------


def _resize_rgb(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Bilinear RGB resize via Pillow."""
    from PIL import Image

    if rgb.shape[0] == out_h and rgb.shape[1] == out_w:
        return rgb.copy()
    # np.array (not asarray): Pillow hands back a read-only buffer, and callers write.
    return np.array(Image.fromarray(rgb).resize((out_w, out_h), Image.BILINEAR), dtype=np.uint8)


def _resize_mask_nearest(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Nearest-neighbour mask resize -- the only correct interpolation for label data."""
    h, w = mask.shape[:2]
    if h == out_h and w == out_w:
        return mask.copy()
    rows = np.minimum((np.arange(out_h) * h) // out_h, h - 1)
    cols = np.minimum((np.arange(out_w) * w) // out_w, w - 1)
    return mask[np.ix_(rows, cols)]


def resize_mask_nearest(mask: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Public nearest-neighbour mask resize (semantic/instance labels keep exact values)."""
    return _resize_mask_nearest(mask, out_w, out_h)
