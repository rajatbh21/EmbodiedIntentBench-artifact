"""The input manifest for empirical-bundle generation, and the source-format parsers.

A manifest points at externally produced data -- ground-truth masks, per-configuration model
predictions, and measured execution metadata -- and describes the mission scaffolding needed
to turn them into a runnable Mission Routing Benchmark data root. Nothing here executes a model or
measures hardware; it reads files a researcher produced *outside* the benchmark and validates
them strictly.

The manifest and every source file declare a ``schema_version``; V1 supports ``"1.0"`` only.
Path fields are resolved relative to the manifest's own directory and may not escape it, so a
manifest cannot be made to read an arbitrary file on the machine that builds it.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from missionbench.schemas.loading import (
    SUPPORTED_SCHEMA_VERSIONS,
    DocumentReader,
    SchemaValidationError,
    SchemaVersionError,
    open_document,
    read_json_object,
)
from missionbench.tasks.human_search_segmentation.masks import BinaryMask, decode_mask

__all__ = [
    "BundleManifest",
    "ManifestConfig",
    "ManifestFrame",
    "MeasurementRecord",
    "SourceGroundTruthInstance",
    "SourcePrediction",
    "load_manifest",
    "parse_ground_truth_source",
    "parse_measurements_source",
    "parse_predictions_source",
    "safe_resolve",
]

#: Coverage over the declared frame x configuration grid.
COVERAGE_STRICT: Final = "strict"
COVERAGE_SPARSE: Final = "sparse"
_COVERAGE_POLICIES: Final = (COVERAGE_STRICT, COVERAGE_SPARSE)

#: Honest provenance for where a bundle's numbers came from. A tool must never silently
#: label a hand-authored or estimated value as measured, so these are required, not defaulted.
_DATA_ORIGINS: Final = ("hand_authored_fixture", "external_capture", "mixed")
_MEASUREMENT_PROVENANCE: Final = ("measured", "externally_supplied", "estimated", "missing")
_PREDICTION_PROVENANCE: Final = ("model_generated", "hand_authored", "externally_supplied")

_MANIFEST_FIELDS: Final = (
    "bundle_id",
    "frame_stream_id",
    "task_id",
    "image_width",
    "image_height",
    "data_origin",
    "coverage",
    "frames",
    "ground_truth_source",
    "configurations",
    "mission",
    "contract",
)
_FRAME_FIELDS: Final = ("frame_id", "timestamp_s", "image", "source_metadata")
_CONFIG_FIELDS: Final = (
    "config_id",
    "model_id",
    "placement",
    "precision",
    "input_compression",
    "parameters",
    "quality_tier",
    "prediction_source",
    "measurement_source",
    "prediction_provenance",
    "measurement_provenance",
    "platform_id",
)
_MISSION_FIELDS: Final = (
    "platform",
    "path_length_m",
    "network",
    "velocity_mps",
    "initial_battery_frac",
    "initial_altitude_m",
    "power_mode",
    "seed",
)
_PLATFORM_FIELDS: Final = (
    "battery_capacity_wh",
    "flight_power_w",
    "communication_energy_j_per_mb",
    "supported_power_modes",
)
_NETWORK_SEGMENT_FIELDS: Final = (
    "start_s",
    "end_s",
    "bandwidth_mbps",
    "rtt_ms",
    "packet_loss_frac",
)
_CONTRACT_FIELDS: Final = (
    "quality_metric",
    "quality_operator",
    "quality_threshold",
    "deadline_s",
    "communication_budget_mb",
    "min_final_battery_frac",
    "privacy_level",
)

_GT_DOCUMENT_FIELDS: Final = ("frame_stream_id", "instances")
_GT_INSTANCE_FIELDS: Final = ("frame_id", "track_id", "category", "mask", "ignore")
_PREDICTION_FIELDS: Final = ("frame_id", "prediction_id", "category", "confidence", "mask")
_MEASUREMENT_COLUMNS: Final = (
    "frame_id",
    "success",
    "latency_s",
    "compute_energy_j",
    "upload_mb",
    "download_mb",
    "failure_reason",
)

#: Fields a prediction is forbidden from carrying: the answer key must never travel inside a
#: prediction, and an on-the-wire IoU must never be trusted.
_FORBIDDEN_PREDICTION_FIELDS: Final = ("ground_truth_track_id", "mask_iou", "track_id")


def safe_resolve(root: Path, relative: str, *, what: str) -> Path:
    """Resolve ``relative`` under ``root``, refusing anything that escapes it.

    Source paths come from a manifest, which a researcher writes by hand; an absolute path or
    a ``..`` that climbs out of the manifest's directory is rejected rather than followed, so
    a bundle build cannot be pointed at an arbitrary file on the machine.
    """
    candidate = Path(relative)
    if candidate.is_absolute():
        raise SchemaValidationError(f"{what}: path {relative!r} must be relative to the manifest")
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    if root_resolved != resolved and root_resolved not in resolved.parents:
        raise SchemaValidationError(
            f"{what}: path {relative!r} escapes the manifest directory {root_resolved}"
        )
    return resolved


@dataclass(frozen=True, slots=True)
class ManifestFrame:
    frame_id: int
    timestamp_s: float
    image: str | None = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ManifestConfig:
    config_id: str
    model_id: str
    placement: str
    precision: str
    quality_tier: str
    prediction_source: str
    measurement_source: str
    prediction_provenance: str
    measurement_provenance: str
    input_compression: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    platform_id: str | None = None


@dataclass(frozen=True, slots=True)
class BundleManifest:
    """A validated bundle-generation manifest, with source paths resolved and dimensions set."""

    manifest_dir: Path
    bundle_id: str
    frame_stream_id: str
    task_id: str
    image_width: int
    image_height: int
    data_origin: str
    coverage: str
    frames: tuple[ManifestFrame, ...]
    ground_truth_source: str
    configurations: tuple[ManifestConfig, ...]
    mission: Mapping[str, Any]
    contract: Mapping[str, Any]

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(frame.frame_id for frame in self.frames)


@dataclass(frozen=True, slots=True)
class SourceGroundTruthInstance:
    frame_id: int
    track_id: str
    category: str
    mask: BinaryMask
    ignore: bool


@dataclass(frozen=True, slots=True)
class SourcePrediction:
    frame_id: int
    prediction_id: str
    category: str
    confidence: float
    mask: BinaryMask


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    frame_id: int
    success: bool
    latency_s: float
    compute_energy_j: float
    upload_mb: float
    download_mb: float
    failure_reason: str | None


def load_manifest(path: Path) -> BundleManifest:
    """Load and validate a bundle-generation manifest."""
    reader = open_document(path, document_type="BundleManifest", allowed_fields=_MANIFEST_FIELDS)
    manifest_dir = path.parent

    width = reader.get_int("image_width", minimum=1)
    height = reader.get_int("image_height", minimum=1)

    coverage = reader.get_optional_str("coverage") or COVERAGE_STRICT
    if coverage not in _COVERAGE_POLICIES:
        raise SchemaValidationError(
            f"{reader.context}: coverage {coverage!r} must be one of {list(_COVERAGE_POLICIES)}"
        )
    data_origin = _one_of(
        reader.get_str("data_origin"), _DATA_ORIGINS, reader.context, "data_origin"
    )

    frames = _read_frames(reader)
    configurations = _read_configs(reader, manifest_dir)

    return BundleManifest(
        manifest_dir=manifest_dir,
        bundle_id=reader.get_str("bundle_id"),
        frame_stream_id=reader.get_str("frame_stream_id"),
        task_id=reader.get_str("task_id"),
        image_width=width,
        image_height=height,
        data_origin=data_origin,
        coverage=coverage,
        frames=frames,
        ground_truth_source=reader.get_str("ground_truth_source"),
        configurations=configurations,
        mission=_read_mission(reader),
        contract=_read_contract(reader),
    )


def _read_frames(reader: DocumentReader) -> tuple[ManifestFrame, ...]:
    frames: list[ManifestFrame] = []
    seen: set[int] = set()
    for entry in reader.get_object_list("frames", allowed_fields=_FRAME_FIELDS):
        frame_id = entry.get_int("frame_id", minimum=0)
        if frame_id in seen:
            raise SchemaValidationError(f"{entry.context}: duplicate frame_id {frame_id}")
        seen.add(frame_id)
        frames.append(
            ManifestFrame(
                frame_id=frame_id,
                timestamp_s=entry.get_optional_float(
                    "timestamp_s", default=float(frame_id), minimum=0.0
                ),
                image=entry.get_optional_str("image"),
                source_metadata=entry.get_scalar_mapping("source_metadata"),
            )
        )
    # Ordered by frame id so the generated records and manifest are order-independent.
    return tuple(sorted(frames, key=lambda frame: frame.frame_id))


def _read_configs(reader: DocumentReader, manifest_dir: Path) -> tuple[ManifestConfig, ...]:
    configs: list[ManifestConfig] = []
    seen: set[str] = set()
    for entry in reader.get_object_list("configurations", allowed_fields=_CONFIG_FIELDS):
        config_id = entry.get_str("config_id")
        if config_id in seen:
            raise SchemaValidationError(f"{entry.context}: duplicate config_id {config_id!r}")
        seen.add(config_id)
        # Resolve source paths now so an escaping or missing path fails at load, with context.
        prediction_source = entry.get_str("prediction_source")
        measurement_source = entry.get_str("measurement_source")
        safe_resolve(manifest_dir, prediction_source, what=f"{entry.context}.prediction_source")
        safe_resolve(manifest_dir, measurement_source, what=f"{entry.context}.measurement_source")
        configs.append(
            ManifestConfig(
                config_id=config_id,
                model_id=entry.get_str("model_id"),
                placement=entry.get_str("placement"),
                precision=entry.get_str("precision"),
                quality_tier=entry.get_str("quality_tier"),
                prediction_source=prediction_source,
                measurement_source=measurement_source,
                prediction_provenance=_one_of(
                    entry.get_str("prediction_provenance"),
                    _PREDICTION_PROVENANCE,
                    entry.context,
                    "prediction_provenance",
                ),
                measurement_provenance=_one_of(
                    entry.get_str("measurement_provenance"),
                    _MEASUREMENT_PROVENANCE,
                    entry.context,
                    "measurement_provenance",
                ),
                input_compression=entry.get_optional_str("input_compression"),
                parameters=entry.get_scalar_mapping("parameters"),
                platform_id=entry.get_optional_str("platform_id"),
            )
        )
    if not configs:
        raise SchemaValidationError(f"{reader.context}: 'configurations' must not be empty")
    return tuple(configs)


def _read_mission(reader: DocumentReader) -> Mapping[str, Any]:
    if reader.get_passthrough("mission") is None:
        return {}
    mission = reader.get_object("mission", allowed_fields=_MISSION_FIELDS)
    result: dict[str, Any] = {}
    if mission.get_passthrough("platform") is not None:
        platform = mission.get_object("platform", allowed_fields=_PLATFORM_FIELDS)
        result["platform"] = {
            "battery_capacity_wh": platform.get_float("battery_capacity_wh", exclusive_minimum=0.0),
            "flight_power_w": platform.get_float("flight_power_w", exclusive_minimum=0.0),
            "communication_energy_j_per_mb": platform.get_float(
                "communication_energy_j_per_mb", minimum=0.0
            ),
            "supported_power_modes": list(platform.get_str_tuple("supported_power_modes")),
        }
    if mission.get_passthrough("network") is not None:
        result["network"] = [
            {
                "start_s": seg.get_float("start_s", minimum=0.0),
                "end_s": seg.get_float("end_s", exclusive_minimum=0.0),
                "bandwidth_mbps": seg.get_float("bandwidth_mbps", minimum=0.0),
                "rtt_ms": seg.get_float("rtt_ms", minimum=0.0),
                "packet_loss_frac": seg.get_fraction("packet_loss_frac"),
            }
            for seg in mission.get_object_list("network", allowed_fields=_NETWORK_SEGMENT_FIELDS)
        ]
    for key, getter in (
        ("path_length_m", lambda: mission.get_float("path_length_m", exclusive_minimum=0.0)),
        ("velocity_mps", lambda: mission.get_float("velocity_mps", exclusive_minimum=0.0)),
        ("initial_battery_frac", lambda: mission.get_fraction("initial_battery_frac")),
        ("initial_altitude_m", lambda: mission.get_float("initial_altitude_m", minimum=0.0)),
        ("power_mode", lambda: mission.get_str("power_mode")),
        ("seed", lambda: mission.get_int("seed")),
    ):
        if mission.get_passthrough(key) is not None:
            result[key] = getter()
    return result


def _read_contract(reader: DocumentReader) -> Mapping[str, Any]:
    if reader.get_passthrough("contract") is None:
        return {}
    contract = reader.get_object("contract", allowed_fields=_CONTRACT_FIELDS)
    result: dict[str, Any] = {}
    for key, getter in (
        ("quality_metric", lambda: contract.get_str("quality_metric")),
        ("quality_operator", lambda: contract.get_str("quality_operator")),
        ("quality_threshold", lambda: contract.get_fraction("quality_threshold")),
        ("deadline_s", lambda: contract.get_float("deadline_s", exclusive_minimum=0.0)),
        (
            "communication_budget_mb",
            lambda: contract.get_float("communication_budget_mb", minimum=0.0),
        ),
        ("min_final_battery_frac", lambda: contract.get_fraction("min_final_battery_frac")),
        ("privacy_level", lambda: contract.get_str("privacy_level")),
    ):
        if contract.get_passthrough(key) is not None:
            result[key] = getter()
    return result


# --- source-format parsers -------------------------------------------------------------------


def parse_ground_truth_source(
    path: Path, *, width: int, height: int
) -> tuple[SourceGroundTruthInstance, ...]:
    """Parse a ground-truth source: a JSON object with an ``instances`` list.

    Each instance is one person in one frame -- ``frame_id``, ``track_id``, ``category``,
    ``mask``, optional ``ignore``. Masks decode through the shared bitmap decoder and must
    match the declared frame dimensions.
    """
    reader = open_document(
        path, document_type="GroundTruthSource", allowed_fields=_GT_DOCUMENT_FIELDS
    )
    instances: list[SourceGroundTruthInstance] = []
    for entry in reader.get_object_list("instances", allowed_fields=_GT_INSTANCE_FIELDS):
        frame_id = entry.get_int("frame_id", minimum=0)
        track_id = entry.get_str("track_id")
        mask = _decode_sized(
            entry.get_passthrough("mask"),
            width=width,
            height=height,
            context=f"{entry.context}.mask (frame {frame_id}, track {track_id!r})",
        )
        ignore = entry.get_bool("ignore") if entry.get_passthrough("ignore") is not None else False
        instances.append(
            SourceGroundTruthInstance(
                frame_id=frame_id,
                track_id=track_id,
                category=entry.get_str("category"),
                mask=mask,
                ignore=ignore,
            )
        )
    return tuple(instances)


def parse_predictions_source(
    path: Path, *, config_id: str, width: int, height: int
) -> tuple[SourcePrediction, ...]:
    """Parse a per-configuration predictions source: JSON Lines, one prediction per line.

    A prediction carrying a ground-truth track id or an on-the-wire ``mask_iou`` is rejected:
    the answer key must never ride inside a prediction, and IoU is computed from masks, not
    trusted from a field.
    """
    predictions: list[SourcePrediction] = []
    for line_number, raw in _read_jsonl(path):
        context = f"{path.name} line {line_number} (config {config_id!r})"
        if not isinstance(raw, Mapping):
            raise SchemaValidationError(f"{context}: each line must be a JSON object")
        forbidden = sorted(set(raw) & set(_FORBIDDEN_PREDICTION_FIELDS))
        if forbidden:
            raise SchemaValidationError(
                f"{context}: prediction must not carry {forbidden}; a prediction may not name "
                "its ground-truth target, and any on-the-wire mask_iou is not trusted"
            )
        unknown = sorted(set(raw) - set(_PREDICTION_FIELDS))
        if unknown:
            raise SchemaValidationError(f"{context}: unknown prediction field(s) {unknown}")
        entry = DocumentReader(raw, context=context, allowed_fields=_PREDICTION_FIELDS)
        prediction_id = entry.get_str("prediction_id")
        predictions.append(
            SourcePrediction(
                frame_id=entry.get_int("frame_id", minimum=0),
                prediction_id=prediction_id,
                category=entry.get_str("category"),
                confidence=entry.get_fraction("confidence"),
                mask=_decode_sized(
                    entry.get_passthrough("mask"),
                    width=width,
                    height=height,
                    context=f"{context}: prediction {prediction_id!r} mask",
                ),
            )
        )
    return tuple(predictions)


def parse_measurements_source(path: Path, *, config_id: str) -> tuple[MeasurementRecord, ...]:
    """Parse a per-configuration measurements source: CSV with a fixed header.

    Columns: ``frame_id, success, latency_s, compute_energy_j, upload_mb, download_mb,
    failure_reason``. Latency and energy are never inferred -- a blank required cell is an
    error, not a silent zero. A failed row must state a failure_reason; a successful one must
    not.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SchemaValidationError(
            f"{path}: measurement source cannot be read ({error})"
        ) from error

    rows = list(csv.DictReader(text.splitlines()))
    reader_fields = set(csv.reader([text.splitlines()[0]]).__next__()) if text.strip() else set()
    unknown = sorted(reader_fields - set(_MEASUREMENT_COLUMNS))
    if unknown:
        raise SchemaValidationError(f"{path.name}: unknown measurement column(s) {unknown}")

    records: list[MeasurementRecord] = []
    for index, row in enumerate(rows):
        context = f"{path.name} row {index + 1} (config {config_id!r})"
        frame_id = _csv_int(row, "frame_id", context)
        success = _csv_bool(row, "success", context)
        failure_reason = (row.get("failure_reason") or "").strip() or None
        if success and failure_reason is not None:
            raise SchemaValidationError(
                f"{context}: a successful measurement has no failure_reason"
            )
        if not success and failure_reason is None:
            raise SchemaValidationError(
                f"{context}: a failed measurement must state a failure_reason"
            )
        records.append(
            MeasurementRecord(
                frame_id=frame_id,
                success=success,
                latency_s=_csv_float(row, "latency_s", context, minimum=0.0),
                compute_energy_j=_csv_float(row, "compute_energy_j", context, minimum=0.0),
                upload_mb=_csv_float(row, "upload_mb", context, minimum=0.0, default=0.0),
                download_mb=_csv_float(row, "download_mb", context, minimum=0.0, default=0.0),
                failure_reason=failure_reason,
            )
        )
    return tuple(records)


# --- low-level helpers -----------------------------------------------------------------------


def _one_of(value: str, allowed: tuple[str, ...], context: str, field_name: str) -> str:
    if value not in allowed:
        raise SchemaValidationError(
            f"{context}: {field_name} {value!r} must be one of {list(allowed)}"
        )
    return value


def _decode_sized(payload: object, *, width: int, height: int, context: str) -> BinaryMask:
    mask = decode_mask(payload, context=context)
    if mask.width != width or mask.height != height:
        raise SchemaValidationError(
            f"{context}: mask is {mask.height}x{mask.width} but the frame is {height}x{width}"
        )
    return mask


def _read_jsonl(path: Path) -> list[tuple[int, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise SchemaValidationError(
            f"{path}: prediction source cannot be read ({error})"
        ) from error
    parsed: list[tuple[int, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed.append((line_number, json.loads(line)))
        except json.JSONDecodeError as error:
            raise SchemaValidationError(
                f"{path.name} line {line_number}: invalid JSON ({error})"
            ) from error
    return parsed


def _require_cell(row: Mapping[str, str], column: str, context: str) -> str:
    if column not in row:
        raise SchemaValidationError(f"{context}: missing required column {column!r}")
    value = (row[column] or "").strip()
    if not value:
        raise SchemaValidationError(
            f"{context}: column {column!r} is blank; measurements are never inferred"
        )
    return value


def _csv_int(row: Mapping[str, str], column: str, context: str) -> int:
    value = _require_cell(row, column, context)
    try:
        return int(value)
    except ValueError:
        raise SchemaValidationError(
            f"{context}: column {column!r} must be an integer, got {value!r}"
        ) from None


def _csv_bool(row: Mapping[str, str], column: str, context: str) -> bool:
    value = _require_cell(row, column, context).lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise SchemaValidationError(f"{context}: column {column!r} must be a boolean, got {value!r}")


def _csv_float(
    row: Mapping[str, str],
    column: str,
    context: str,
    *,
    minimum: float,
    default: float | None = None,
) -> float:
    raw = (row.get(column) or "").strip()
    if not raw:
        if default is not None:
            return default
        raise SchemaValidationError(
            f"{context}: column {column!r} is blank; latency and energy are never inferred"
        )
    try:
        number = float(raw)
    except ValueError:
        raise SchemaValidationError(
            f"{context}: column {column!r} must be a number, got {raw!r}"
        ) from None
    if number != number:  # NaN
        raise SchemaValidationError(f"{context}: column {column!r} must be finite, got NaN")
    if number in (float("inf"), float("-inf")):
        raise SchemaValidationError(f"{context}: column {column!r} must be finite")
    if number < minimum:
        raise SchemaValidationError(
            f"{context}: column {column!r} must be >= {minimum}, got {number}"
        )
    return number


def is_supported_version(path: Path) -> None:
    """Confirm a source file declares a supported schema version, before deeper parsing."""
    payload = read_json_object(path)
    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaVersionError(
            f"{path.name}: unsupported schema_version {version!r}; "
            f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
