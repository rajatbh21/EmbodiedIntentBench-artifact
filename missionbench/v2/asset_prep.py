"""Deterministic preparation of owner-supplied RGBA human assets.

    python -m missionbench.v2.asset_prep \\
      --plan data/v2_assets/prep_plan.json --output data/v2_assets/generated

Generated person cutouts routinely arrive with a broad, low-opacity halo across the
whole canvas. This module turns such an original into a canonical processed asset with
one fixed, content-agnostic operation -- never hand-painting, never reshaping the
person, and never informed by any model's predictions:

1. ``solid`` = alpha >= ``solid_threshold`` (the unambiguous subject).
2. Keep the original alpha inside ``solid`` and within a ``keep_band_px`` dilation of
   it (this preserves the genuine anti-aliased boundary), zero it everywhere else
   (this removes the broad generation halo).
3. Crop to the solid bounding box plus ``pad_px`` deterministic padding.
4. Verify the mask is non-empty and the subject was not clipped by the crop.

The exact operation string and the output sha256 are recorded for the manifest. The
originals are never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from missionbench.schemas.loading import SchemaValidationError

__all__ = ["CleanResult", "clean_rgba", "inspect_rgba", "prepare_asset"]

_SOLID_THRESHOLD: Final = 128
_KEEP_BAND_PX: Final = 3
_PAD_PX: Final = 8


def inspect_rgba(path: Path) -> dict[str, Any]:
    """Report an original's format, alpha distribution, and subject/halo geometry."""
    from PIL import Image

    with Image.open(path) as img:
        info: dict[str, Any] = {
            "filename": path.name,
            "format": img.format,
            "width_px": img.width,
            "height_px": img.height,
            "mode": img.mode,
            "has_alpha": "A" in img.getbands(),
        }
        if not info["has_alpha"]:
            return info
        alpha = np.array(img.convert("RGBA"))[..., 3]
    total = alpha.size
    solid = alpha >= _SOLID_THRESHOLD
    any_alpha = alpha > 0
    info["alpha_distribution"] = {
        "transparent": float((alpha == 0).sum() / total),
        "low(0,64)": float(((alpha > 0) & (alpha < 64)).sum() / total),
        "mid[64,192)": float(((alpha >= 64) & (alpha < 192)).sum() / total),
        "high[192,255]": float((alpha >= 192).sum() / total),
    }
    if solid.any():
        ys, xs = np.where(solid)
        info["solid_bbox_px"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        info["solid_area_fraction"] = float(solid.sum() / total)
    if any_alpha.any():
        ys, xs = np.where(any_alpha)
        info["any_alpha_bbox_px"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        # A halo exists when faint alpha vastly outreaches the solid subject.
        info["low_alpha_halo_fraction"] = float(
            ((alpha > 0) & (alpha < 64)).sum() / max(1, int(any_alpha.sum()))
        )
        info["has_broad_halo"] = bool(
            solid.any()
            and (info["any_alpha_bbox_px"][2] - info["any_alpha_bbox_px"][0])
            > 1.5 * (info["solid_bbox_px"][2] - info["solid_bbox_px"][0])
        )
    return info


@dataclass(frozen=True, slots=True)
class CleanResult:
    rgba: np.ndarray  # uint8 (H, W, 4), cropped
    processing: str  # the exact deterministic operation, for provenance
    solid_px: int


def clean_rgba(
    rgba: np.ndarray,
    *,
    solid_threshold: int = _SOLID_THRESHOLD,
    keep_band_px: int = _KEEP_BAND_PX,
    pad_px: int = _PAD_PX,
) -> CleanResult:
    """Remove the broad low-alpha halo and crop to the subject, deterministically."""
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise SchemaValidationError("clean_rgba expects an (H, W, 4) RGBA array")
    alpha = rgba[..., 3]
    solid = alpha >= solid_threshold
    if not solid.any():
        raise SchemaValidationError(
            f"no pixel reaches alpha >= {solid_threshold}; the asset has no solid subject"
        )

    # Preserve the genuine anti-aliased boundary: a keep-band dilated from the solid
    # subject retains the original alpha; everything beyond it is halo and goes to 0.
    band = solid.copy()
    for _ in range(keep_band_px):
        grown = band.copy()
        grown[1:, :] |= band[:-1, :]
        grown[:-1, :] |= band[1:, :]
        grown[:, 1:] |= band[:, :-1]
        grown[:, :-1] |= band[:, 1:]
        band = grown

    cleaned = rgba.copy()
    cleaned[..., 3] = np.where(band, alpha, 0)

    ys, xs = np.where(solid)
    height, width = alpha.shape
    y0 = max(0, int(ys.min()) - pad_px)
    y1 = min(height, int(ys.max()) + 1 + pad_px)
    x0 = max(0, int(xs.min()) - pad_px)
    x1 = min(width, int(xs.max()) + 1 + pad_px)
    cropped = cleaned[y0:y1, x0:x1]

    # The crop is derived from the solid bbox itself, so the subject cannot be clipped;
    # assert it anyway -- silent body loss would corrupt every downstream mask.
    if int((cropped[..., 3] >= solid_threshold).sum()) != int(solid.sum()):
        raise SchemaValidationError("internal error: the crop clipped solid subject pixels")

    processing = (
        f"halo-removal(keep original alpha within {keep_band_px}px dilation of "
        f"alpha>={solid_threshold}; zero elsewhere); crop to solid bbox +{pad_px}px pad"
    )
    return CleanResult(rgba=cropped, processing=processing, solid_px=int(solid.sum()))


def prepare_asset(source: Path, destination: Path) -> dict[str, Any]:
    """Clean one original into a canonical processed PNG; return manifest-ready facts."""
    from PIL import Image

    with Image.open(source) as img:
        if "A" not in img.getbands():
            raise SchemaValidationError(f"{source.name}: no alpha channel; cannot prepare")
        rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    result = clean_rgba(rgba)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.rgba, mode="RGBA").save(destination)
    return {
        "source": str(source),
        "destination": str(destination),
        "processing": result.processing,
        "width_px": int(result.rgba.shape[1]),
        "height_px": int(result.rgba.shape[0]),
        "solid_px": result.solid_px,
        "checksum_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="missionbench.v2.asset_prep",
        description="Deterministically clean owner-supplied RGBA assets per a prep plan.",
    )
    parser.add_argument("--plan", type=Path, required=True, help="prep_plan.json")
    parser.add_argument("--output", type=Path, required=True, help="Processed-asset directory.")
    args = parser.parse_args(argv)

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    for entry in plan["assets"]:
        source = Path(entry["source_path"])
        destination = args.output / entry["processed_name"]
        facts = prepare_asset(source, destination)
        print(f"{source.name} -> {destination.name}  sha256={facts['checksum_sha256'][:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
