"""V2.2 target assets: image-based human targets with provenance and exact masks.

An *asset* is an RGB(A) image of a target (nominally a person) plus a binary mask, a
declared physical size, and — non-negotiably — provenance: source, license, and whether
redistribution is permitted. The manifest is strict in the repository's usual way:
unknown fields are rejected, files must exist, checksums must match, masks must be
non-empty and dimension-aligned, and nothing is silently inferred.

Two mask sources are supported:

- ``mask_source: "alpha"`` — an RGBA PNG whose alpha channel yields both the smooth
  compositing alpha and (thresholded at 0.5) the binary GT mask.
- ``mask_source: "file"`` — an RGB image plus a separate binary mask image.

View-type honesty: every asset declares ``view_type`` — ``conventional`` (a normal
person photo; useful only as a controlled model-integration diagnostic),
``aerial`` (a genuinely overhead capture), or ``procedural`` (a generated test
silhouette that must never be presented as a realistic person or as evidence of model
performance). The categories are never conflated, and results downstream carry the
label. A conventional cutout composited into a top-down scene is **not** aerial-human
perception and is labelled accordingly.

No real licensed human asset ships with the repository. ``data/v2_assets/`` holds a
manifest template and a README describing what an owner must supply; the loader fails
actionably until they do.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from missionbench.schemas.loading import (
    DocumentReader,
    SchemaValidationError,
    SchemaVersionError,
    read_json_object,
)

__all__ = [
    "ASSET_SCHEMA_VERSION",
    "AssetManifest",
    "AssetRecord",
    "LoadedAsset",
    "load_asset",
    "load_manifest",
]

ASSET_SCHEMA_VERSION: Final = "1.0"

_TOP_FIELDS: Final = ("asset_schema_version", "assets")
_ASSET_FIELDS: Final = (
    "asset_id",
    "asset_path",
    "mask_source",
    "mask_path",
    "category",
    "view_type",
    "pose",
    "synthetic",
    "processing",
    "source",
    "physical",
    "rendering",
    "checksum_sha256",
)
_SOURCE_FIELDS: Final = (
    "provider",
    "creator",
    "source_url",
    "license",
    "permitted_use",
    "redistribution_allowed",
)
_PHYSICAL_FIELDS: Final = ("nominal_width_m", "nominal_height_m")
_RENDERING_FIELDS: Final = ("anchor", "default_rotation_deg")

_VIEW_TYPES: Final = ("conventional", "aerial", "procedural")
_MASK_SOURCES: Final = ("alpha", "file")
_IMAGE_SUFFIXES: Final = (".png", ".jpg", ".jpeg")


@dataclass(frozen=True, slots=True)
class AssetRecord:
    """One manifest entry: where the asset lives, what it is, and who allows its use."""

    asset_id: str
    asset_path: Path
    mask_source: str
    category: str
    view_type: str
    provider: str
    license: str
    redistribution_allowed: bool
    nominal_width_m: float
    nominal_height_m: float
    checksum_sha256: str
    mask_path: Path | None = None
    creator: str | None = None
    source_url: str | None = None
    permitted_use: str | None = None
    anchor: str = "center"
    default_rotation_deg: float = 0.0
    #: The subject's pose (e.g. standing / walking / crouching), when known. Free text,
    #: but never a claim the image does not support.
    pose: str | None = None
    #: True for generated imagery. A synthetic asset can never yield results labelled
    #: real aerial-human perception.
    synthetic: bool = False
    #: The exact deterministic processing that produced this file from its original.
    processing: str | None = None

    def provenance(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "category": self.category,
            "view_type": self.view_type,
            "pose": self.pose,
            "synthetic": self.synthetic,
            "provider": self.provider,
            "creator": self.creator,
            "license": self.license,
            "source_url": self.source_url,
            "redistribution_allowed": self.redistribution_allowed,
            "processing": self.processing,
            "checksum_sha256": self.checksum_sha256,
        }


@dataclass(frozen=True, slots=True)
class AssetManifest:
    """Every validated asset record, keyed by id."""

    path: Path
    records: dict[str, AssetRecord] = field(default_factory=dict)

    def __contains__(self, asset_id: object) -> bool:
        return asset_id in self.records

    def get(self, asset_id: str) -> AssetRecord:
        try:
            return self.records[asset_id]
        except KeyError:
            raise SchemaValidationError(
                f"no asset {asset_id!r} in manifest {self.path.name}; "
                f"available: {sorted(self.records)}"
            ) from None


@dataclass(frozen=True, slots=True)
class LoadedAsset:
    """An asset decoded into arrays: smooth alpha for RGB compositing, binary mask for GT."""

    record: AssetRecord
    rgb: np.ndarray  # uint8 (H, W, 3)
    alpha: np.ndarray  # float32 (H, W) in [0, 1] -- anti-aliased compositing weight
    mask: np.ndarray  # bool (H, W) -- the exact GT geometry

    @property
    def height_px(self) -> int:
        return self.rgb.shape[0]

    @property
    def width_px(self) -> int:
        return self.rgb.shape[1]


def load_manifest(path: Path) -> AssetManifest:
    """Load and strictly validate an asset manifest, including files and checksums."""
    payload = read_json_object(path)
    context = f"{path.name} -> AssetManifest"
    declared = payload.get("asset_schema_version")
    if declared != ASSET_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"{context}: asset_schema_version {declared!r} is not supported; this release "
            f"reads {ASSET_SCHEMA_VERSION!r} only"
        )
    reader = DocumentReader(payload, context=context, allowed_fields=_TOP_FIELDS)

    records: dict[str, AssetRecord] = {}
    for entry in reader.get_object_list("assets", allowed_fields=_ASSET_FIELDS):
        record = _read_record(entry, manifest_dir=path.parent)
        if record.asset_id in records:
            raise SchemaValidationError(f"{entry.context}: duplicate asset_id {record.asset_id!r}")
        records[record.asset_id] = record
    return AssetManifest(path=path, records=records)


def _read_record(entry: DocumentReader, manifest_dir: Path) -> AssetRecord:
    asset_id = entry.get_str("asset_id")
    mask_source = entry.get_str("mask_source")
    if mask_source not in _MASK_SOURCES:
        raise SchemaValidationError(
            f"{entry.context}: mask_source {mask_source!r} must be one of {list(_MASK_SOURCES)}"
        )
    view_type = entry.get_str("view_type")
    if view_type not in _VIEW_TYPES:
        raise SchemaValidationError(
            f"{entry.context}: view_type {view_type!r} must be one of {list(_VIEW_TYPES)} -- "
            "a conventional photo must never be relabelled aerial"
        )

    asset_path = _existing_image(manifest_dir, entry.get_str("asset_path"), entry.context)
    mask_path: Path | None = None
    if mask_source == "file":
        raw = entry.get_optional_str("mask_path")
        if raw is None:
            raise SchemaValidationError(f"{entry.context}: mask_source 'file' requires 'mask_path'")
        mask_path = _existing_image(manifest_dir, raw, entry.context)
    elif entry.get_passthrough("mask_path") is not None:
        raise SchemaValidationError(
            f"{entry.context}: mask_path is only valid with mask_source 'file'"
        )

    source = entry.get_object("source", allowed_fields=_SOURCE_FIELDS)
    physical = entry.get_object("physical", allowed_fields=_PHYSICAL_FIELDS)
    rendering = (
        entry.get_object("rendering", allowed_fields=_RENDERING_FIELDS)
        if entry.get_passthrough("rendering") is not None
        else None
    )

    declared_checksum = entry.get_str("checksum_sha256").lower()
    actual = hashlib.sha256(asset_path.read_bytes()).hexdigest()
    if declared_checksum != actual:
        raise SchemaValidationError(
            f"{entry.context}: checksum mismatch for {asset_path.name}: manifest declares "
            f"{declared_checksum[:12]}..., file is {actual[:12]}... -- the asset changed or "
            "the manifest lies; re-verify provenance before using it"
        )

    return AssetRecord(
        asset_id=asset_id,
        asset_path=asset_path,
        mask_source=mask_source,
        mask_path=mask_path,
        category=entry.get_str("category"),
        view_type=view_type,
        provider=source.get_str("provider"),
        creator=source.get_optional_str("creator"),
        source_url=source.get_optional_str("source_url"),
        license=source.get_str("license"),
        permitted_use=source.get_optional_str("permitted_use"),
        redistribution_allowed=source.get_bool("redistribution_allowed"),
        nominal_width_m=physical.get_float("nominal_width_m", exclusive_minimum=0.0),
        nominal_height_m=physical.get_float("nominal_height_m", exclusive_minimum=0.0),
        checksum_sha256=declared_checksum,
        anchor=(rendering.get_optional_str("anchor") or "center") if rendering else "center",
        default_rotation_deg=(
            rendering.get_float("default_rotation_deg", minimum=-360.0, maximum=360.0)
            if rendering and rendering.get_passthrough("default_rotation_deg") is not None
            else 0.0
        ),
        pose=entry.get_optional_str("pose"),
        synthetic=(
            entry.get_bool("synthetic") if entry.get_passthrough("synthetic") is not None else False
        ),
        processing=entry.get_optional_str("processing"),
    )


def _existing_image(manifest_dir: Path, raw: str, context: str) -> Path:
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else (manifest_dir / candidate)
    if not path.is_file():
        raise SchemaValidationError(
            f"{context}: asset file not found: {path}. Real human target assets are not "
            "committed to this repository; place a licensed asset locally and record its "
            "provenance in the manifest (see data/v2_assets/README.md)."
        )
    if path.suffix.lower() not in _IMAGE_SUFFIXES:
        raise SchemaValidationError(
            f"{context}: unsupported asset format {path.suffix!r}; "
            f"supported: {list(_IMAGE_SUFFIXES)}"
        )
    return path


# --- decoding --------------------------------------------------------------------------------

_ASSET_CACHE: dict[tuple[str, str], LoadedAsset] = {}


def load_asset(record: AssetRecord) -> LoadedAsset:
    """Decode an asset once (cached by id+checksum) into RGB, alpha, and binary mask."""
    key = (record.asset_id, record.checksum_sha256)
    if key in _ASSET_CACHE:
        return _ASSET_CACHE[key]

    from PIL import Image

    with Image.open(record.asset_path) as img:
        if record.mask_source == "alpha":
            if "A" not in img.getbands():
                raise SchemaValidationError(
                    f"asset {record.asset_id!r}: mask_source 'alpha' but "
                    f"{record.asset_path.name} has no alpha channel (bands: {img.getbands()})"
                )
            rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
            rgb = rgba[..., :3]
            alpha = rgba[..., 3].astype(np.float32) / 255.0
        else:
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)
            assert record.mask_path is not None  # schema-validated
            with Image.open(record.mask_path) as mask_img:
                alpha = np.array(mask_img.convert("L"), dtype=np.float32) / 255.0
            if alpha.shape != rgb.shape[:2]:
                raise SchemaValidationError(
                    f"asset {record.asset_id!r}: mask {record.mask_path.name} is "
                    f"{alpha.shape}, RGB is {rgb.shape[:2]} -- dimensions must match"
                )

    mask = alpha >= 0.5
    if not mask.any():
        raise SchemaValidationError(
            f"asset {record.asset_id!r}: the binary mask is empty -- an invisible target "
            "cannot be a target"
        )
    loaded = LoadedAsset(record=record, rgb=rgb, alpha=alpha, mask=mask)
    _ASSET_CACHE[key] = loaded
    return loaded


def clear_asset_cache() -> None:
    _ASSET_CACHE.clear()
