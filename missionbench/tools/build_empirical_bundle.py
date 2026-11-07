"""Turn externally produced empirical inputs into a runnable Mission Routing Benchmark replay bundle.

    python -m missionbench.tools.build_empirical_bundle \\
      --manifest experiments/pilot/manifest.json \\
      --output data/generated/pilot_bundle --validate

A researcher runs any model outside the benchmark, records its per-configuration predictions
and measured execution metadata, and points a manifest at them (see
:mod:`missionbench.tools.bundle_manifest`). This tool reads those sources, validates them
strictly, and writes a **self-contained data root** the ordinary benchmark CLI can run with
``--executor replay`` -- no change to ``EpisodeRunner``, ``ReplayExecutor``, or the evaluator,
which continue to consume only the standardised replay schema.

It executes no model and measures no hardware. Latency, energy, and communication come from
the supplied measurements; the generated bundle records their declared provenance
(``measured`` / ``externally_supplied`` / ``estimated`` / ``missing``) so a reader can always
tell a hand-authored fixture from a real capture.

Coverage. Real captures may lack some ``frame x config`` pairs. The policy is explicit:
``strict`` (the default) requires a measurement for every declared frame and configuration;
``sparse`` turns a declared-but-missing pair into an explicit failed replay record. A missing
pair is never silently dropped.

Determinism. Given the same sources and options, the generated files are byte-stable: records
are ordered by ``(frame_id, config_id)``, ground truth by ``(frame_id, track_id)``, JSON is
written sorted. The only non-deterministic value, the build timestamp, lives in one
``created_at`` field of the provenance manifest and can be pinned with ``--created-at``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any, Final

from missionbench import SCHEMA_VERSION, __version__
from missionbench.executor.base import FailureReason
from missionbench.schemas.loading import SchemaValidationError
from missionbench.tools.bundle_manifest import (
    COVERAGE_SPARSE,
    COVERAGE_STRICT,
    BundleManifest,
    ManifestConfig,
    MeasurementRecord,
    SourceGroundTruthInstance,
    SourcePrediction,
    load_manifest,
    parse_ground_truth_source,
    parse_measurements_source,
    parse_predictions_source,
    safe_resolve,
)

__all__ = ["BUILDER_VERSION", "BundleSummary", "build_empirical_bundle", "main"]

#: Bumped when the generated layout or provenance shape changes.
BUILDER_VERSION: Final = "1.0"
_MS_PER_S: Final = 1000.0
_SAFE_FALLBACK_CONFIG_ID: Final = "CFG_LOCAL_LIGHT"

_DEFAULT_PLATFORM: Final = {
    "battery_capacity_wh": 100.0,
    "flight_power_w": 180.0,
    "communication_energy_j_per_mb": 0.5,
    "supported_power_modes": ["15W"],
}
_DEFAULT_MISSION: Final = {
    "velocity_mps": 5.0,
    "initial_battery_frac": 0.80,
    "initial_altitude_m": 40.0,
    "power_mode": "15W",
    "seed": 0,
}
_DEFAULT_CONTRACT: Final = {
    "quality_metric": "target_recall",
    "quality_operator": ">=",
    "quality_threshold": 0.50,
    "communication_budget_mb": 1000.0,
    "min_final_battery_frac": 0.10,
    "privacy_level": "remote_allowed",
}


@dataclass(frozen=True, slots=True)
class BundleSummary:
    """What a build produced, returned to a programmatic caller and echoed by the CLI."""

    bundle_id: str
    output_root: Path
    coverage: str
    frame_count: int
    config_ids: tuple[str, ...]
    prediction_counts: Mapping[str, int]
    replay_record_count: int
    gt_instance_count: int
    unique_track_count: int
    measurement_coverage: Mapping[str, str]
    provenance_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "output_root": str(self.output_root),
            "coverage": self.coverage,
            "frame_count": self.frame_count,
            "config_ids": list(self.config_ids),
            "prediction_counts": dict(self.prediction_counts),
            "replay_record_count": self.replay_record_count,
            "gt_instance_count": self.gt_instance_count,
            "unique_track_count": self.unique_track_count,
            "measurement_coverage": dict(self.measurement_coverage),
            "provenance_path": str(self.provenance_path),
        }


def build_empirical_bundle(
    manifest_path: Path,
    output_root: Path,
    *,
    coverage: str | None = None,
    created_at: str | None = None,
    validate: bool = False,
) -> BundleSummary:
    """Build a self-contained empirical replay bundle from ``manifest_path``.

    Args:
        coverage: Override the manifest's coverage policy (``strict`` or ``sparse``).
        created_at: Pin the provenance timestamp for a reproducible build; defaults to now.
        validate: Run the bundle validator over the result before returning.
    """
    manifest = load_manifest(manifest_path)
    policy = coverage or manifest.coverage
    if policy not in (COVERAGE_STRICT, COVERAGE_SPARSE):
        raise SchemaValidationError(
            f"coverage {policy!r} must be {COVERAGE_STRICT!r} or {COVERAGE_SPARSE!r}"
        )

    ground_truth = parse_ground_truth_source(
        safe_resolve(
            manifest.manifest_dir, manifest.ground_truth_source, what="ground_truth_source"
        ),
        width=manifest.image_width,
        height=manifest.image_height,
    )
    _check_ground_truth_frames(ground_truth, manifest)

    per_config = {
        config.config_id: _load_config_sources(manifest, config)
        for config in manifest.configurations
    }

    records = _build_records(manifest, per_config, policy)

    files = _assemble_files(manifest, ground_truth, records, policy)
    written = _write_bundle(output_root, files)

    provenance = _build_provenance(
        manifest, manifest_path, ground_truth, per_config, records, policy, written, created_at
    )
    provenance_path = output_root / "provenance.json"
    provenance_path.write_text(_dump(provenance), encoding="utf-8")

    summary = BundleSummary(
        bundle_id=manifest.bundle_id,
        output_root=output_root,
        coverage=policy,
        frame_count=len(manifest.frames),
        config_ids=tuple(config.config_id for config in manifest.configurations),
        prediction_counts={cid: len(sources[0]) for cid, sources in per_config.items()},
        replay_record_count=len(records),
        gt_instance_count=len(ground_truth),
        unique_track_count=len({gt.track_id for gt in ground_truth if not gt.ignore}),
        measurement_coverage={
            config.config_id: config.measurement_provenance for config in manifest.configurations
        },
        provenance_path=provenance_path,
    )

    if validate:
        from missionbench.tools.validate_empirical_bundle import validate_empirical_bundle

        report = validate_empirical_bundle(output_root)
        if not report.ok:
            raise SchemaValidationError(
                "the generated bundle failed validation:\n  " + "\n  ".join(report.errors)
            )
    return summary


# --- source loading and cross-checks ---------------------------------------------------------


def _load_config_sources(
    manifest: BundleManifest, config: ManifestConfig
) -> tuple[tuple[SourcePrediction, ...], dict[int, MeasurementRecord]]:
    predictions = parse_predictions_source(
        safe_resolve(manifest.manifest_dir, config.prediction_source, what="prediction_source"),
        config_id=config.config_id,
        width=manifest.image_width,
        height=manifest.image_height,
    )
    _check_prediction_frames(predictions, manifest, config)
    _check_prediction_ids_unique(predictions, config)

    measurement_list = parse_measurements_source(
        safe_resolve(manifest.manifest_dir, config.measurement_source, what="measurement_source"),
        config_id=config.config_id,
    )
    measurements: dict[int, MeasurementRecord] = {}
    known_frames = set(manifest.frame_ids)
    for record in measurement_list:
        if record.frame_id not in known_frames:
            raise SchemaValidationError(
                f"measurement for config {config.config_id!r} names frame {record.frame_id}, "
                "which is not declared in the manifest frames"
            )
        if record.frame_id in measurements:
            raise SchemaValidationError(
                f"config {config.config_id!r} has two measurements for frame {record.frame_id}"
            )
        if record.failure_reason is not None:
            _validate_failure_reason(record.failure_reason, config.config_id, record.frame_id)
        measurements[record.frame_id] = record
    return predictions, measurements


def _build_records(
    manifest: BundleManifest,
    per_config: Mapping[str, tuple[tuple[SourcePrediction, ...], dict[int, MeasurementRecord]]],
    coverage: str,
) -> list[dict[str, Any]]:
    """Assemble one replay record per declared frame x configuration, deterministically."""
    records: list[dict[str, Any]] = []
    for config in manifest.configurations:
        predictions, measurements = per_config[config.config_id]
        by_frame: dict[int, list[SourcePrediction]] = {}
        for prediction in predictions:
            by_frame.setdefault(prediction.frame_id, []).append(prediction)

        for frame in manifest.frames:
            measurement = measurements.get(frame.frame_id)
            if measurement is None:
                if coverage == COVERAGE_STRICT:
                    raise SchemaValidationError(
                        f"strict coverage: no measurement for frame {frame.frame_id}, config "
                        f"{config.config_id!r}. Supply it, or build with coverage 'sparse'."
                    )
                records.append(_missing_record(frame.frame_id, config.config_id))
                continue
            records.append(
                _record_from_measurement(
                    frame.frame_id, config.config_id, measurement, by_frame.get(frame.frame_id, [])
                )
            )
    records.sort(key=lambda record: (record["frame_id"], record["config_id"]))
    return records


def _record_from_measurement(
    frame_id: int,
    config_id: str,
    measurement: MeasurementRecord,
    predictions: Sequence[SourcePrediction],
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "frame_id": frame_id,
        "config_id": config_id,
        "success": measurement.success,
        "latency_ms": round(measurement.latency_s * _MS_PER_S, 6),
        "onboard_energy_j": measurement.compute_energy_j,
        "upload_mb": measurement.upload_mb,
        "download_mb": measurement.download_mb,
    }
    if measurement.success:
        entry["prediction"] = {
            "frame_id": frame_id,
            "instances": [
                {
                    "prediction_id": prediction.prediction_id,
                    "category": prediction.category,
                    "confidence": prediction.confidence,
                    "mask": prediction.mask.to_dict(),
                }
                for prediction in sorted(predictions, key=lambda p: p.prediction_id)
            ],
        }
    else:
        entry["failure_reason"] = measurement.failure_reason
    return entry


def _missing_record(frame_id: int, config_id: str) -> dict[str, Any]:
    """A sparse-coverage gap, recorded as an explicit failed inference (never omitted)."""
    return {
        "frame_id": frame_id,
        "config_id": config_id,
        "success": False,
        "latency_ms": 0.0,
        "onboard_energy_j": 0.0,
        "failure_reason": FailureReason.NO_PREDICTION_AVAILABLE.value,
    }


# --- bundle file assembly --------------------------------------------------------------------


def _assemble_files(
    manifest: BundleManifest,
    ground_truth: Sequence[SourceGroundTruthInstance],
    records: Sequence[dict[str, Any]],
    coverage: str,
) -> dict[str, Any]:
    mission = {
        **_DEFAULT_MISSION,
        **{k: v for k, v in manifest.mission.items() if k not in ("platform", "network")},
    }
    platform = {**_DEFAULT_PLATFORM, **manifest.mission.get("platform", {})}
    contract = {**_DEFAULT_CONTRACT, **manifest.contract}

    max_frame = max(manifest.frame_ids)
    deadline = float(contract.get("deadline_s") or (max_frame + 1))
    contract["deadline_s"] = deadline
    velocity = mission["velocity_mps"]
    # Path long enough that completion never pre-empts the frames the deadline admits.
    path_length = float(manifest.mission.get("path_length_m") or velocity * (deadline + 2.0))
    network = manifest.mission.get("network") or [
        {
            "start_s": 0.0,
            "end_s": deadline + 1.0,
            "bandwidth_mbps": 20.0,
            "rtt_ms": 30.0,
            "packet_loss_frac": 0.0,
        }
    ]

    platform_id = f"{manifest.bundle_id}_PLATFORM"
    path_id = f"{manifest.bundle_id}_PATH"
    trace_id = f"{manifest.bundle_id}_TRACE"
    episode_id = f"{manifest.bundle_id}_EPISODE"
    config_ids = [config.config_id for config in manifest.configurations]

    return {
        "configs/config_catalog.json": _config_catalog(manifest),
        "platforms/platform.json": {
            "schema_version": SCHEMA_VERSION,
            "platform_id": platform_id,
            **platform,
        },
        "profiles/profiles.json": _profiles(manifest, platform_id, records),
        "paths/path.json": {
            "schema_version": SCHEMA_VERSION,
            "path_id": path_id,
            "length_m": path_length,
            "description": "Synthetic scaffold path generated by build_empirical_bundle.",
        },
        "network_traces/trace.json": {
            "schema_version": SCHEMA_VERSION,
            "trace_id": trace_id,
            "segments": network,
        },
        "task_specs/task.json": _task_spec(manifest),
        "episodes/episode.json": {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "platform_id": platform_id,
            "path_id": path_id,
            "frame_stream_id": manifest.frame_stream_id,
            "initial_altitude_m": mission["initial_altitude_m"],
            "velocity_mps": velocity,
            "initial_battery_frac": mission["initial_battery_frac"],
            "power_mode": mission["power_mode"],
            "network_trace_id": trace_id,
            "allowed_config_ids": config_ids,
            "initial_config_id": None,
            "seed": mission["seed"],
        },
        "contracts/contract.json": {
            "schema_version": SCHEMA_VERSION,
            "contract_id": f"{manifest.bundle_id}_CONTRACT",
            "task_id": manifest.task_id,
            "evidence_type": "instance_mask_set",
            **contract,
        },
        "ground_truth/ground_truth.json": _ground_truth(manifest, ground_truth),
        "predictions/replay.json": {
            "schema_version": SCHEMA_VERSION,
            "record_set_id": f"{manifest.bundle_id}_REPLAY",
            "episode_id": episode_id,
            "records": list(records),
        },
        "frames/frame_manifest.json": {
            "schema_version": SCHEMA_VERSION,
            "frame_stream_id": manifest.frame_stream_id,
            "image_width": manifest.image_width,
            "image_height": manifest.image_height,
            "coverage": coverage,
            "frames": [
                {"frame_id": frame.frame_id, "timestamp_s": frame.timestamp_s, "image": frame.image}
                for frame in manifest.frames
            ],
        },
    }


def _config_catalog(manifest: BundleManifest) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "catalog_id": f"{manifest.bundle_id}_CATALOG",
        "configs": [
            {
                "config_id": config.config_id,
                "model_id": config.model_id,
                "strategy": {
                    "placement": config.placement,
                    "precision": config.precision,
                    **(
                        {"input_compression": config.input_compression}
                        if config.input_compression
                        else {}
                    ),
                    "parameters": dict(config.parameters),
                },
            }
            for config in manifest.configurations
        ],
    }


def _profiles(
    manifest: BundleManifest, platform_id: str, records: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Policy-facing nominal costs, summarised from the supplied measurements.

    These feed only the policy's public view (a quality tier and rounded expectations); the
    executor uses the per-record measurements, not these. They are means of the successful
    measurements, marked in provenance as derived scaffolding -- never a separate measurement.
    """
    profiles: dict[str, Any] = {}
    for config in manifest.configurations:
        successes = [r for r in records if r["config_id"] == config.config_id and r["success"]]
        n = len(successes)
        profiles[config.config_id] = {
            "compute_latency_ms": round(sum(r["latency_ms"] for r in successes) / n, 6)
            if n
            else 0.0,
            "onboard_energy_j": round(sum(r["onboard_energy_j"] for r in successes) / n, 6)
            if n
            else 0.0,
            "upload_mb": round(sum(r.get("upload_mb", 0.0) for r in successes) / n, 6)
            if n
            else 0.0,
            "download_mb": round(sum(r.get("download_mb", 0.0) for r in successes) / n, 6)
            if n
            else 0.0,
            "quality_tier": config.quality_tier,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": f"{manifest.bundle_id}_PROFILES",
        "platform_id": platform_id,
        "profiles": profiles,
    }


def _task_spec(manifest: BundleManifest) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": manifest.task_id,
        "evidence_type": "instance_mask_set",
        "target_type": "person",
        "ground_truth_type": "instance_masks_with_track_ids",
        "matching_rule": {"metric": "mask_iou", "operator": ">=", "threshold": 0.50},
        "deduplication": {"method": "ground_truth_track_id"},
        "supported_quality_metrics": ["target_recall", "target_precision", "target_f1"],
    }


def _ground_truth(
    manifest: BundleManifest, ground_truth: Sequence[SourceGroundTruthInstance]
) -> dict[str, Any]:
    by_frame: dict[int, list[SourceGroundTruthInstance]] = {}
    for instance in ground_truth:
        by_frame.setdefault(instance.frame_id, []).append(instance)
    frames = [
        {
            "frame_id": frame_id,
            "instances": [
                {
                    "track_id": gt.track_id,
                    "category": gt.category,
                    **({"ignore": True} if gt.ignore else {}),
                    "mask": gt.mask.to_dict(),
                }
                for gt in sorted(by_frame[frame_id], key=lambda g: g.track_id)
            ],
        }
        for frame_id in sorted(by_frame)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "ground_truth_id": f"{manifest.bundle_id}_GT",
        "frame_stream_id": manifest.frame_stream_id,
        "task_id": manifest.task_id,
        "frames": frames,
    }


def _write_bundle(output_root: Path, files: Mapping[str, Any]) -> dict[str, str]:
    """Write every bundle file and return a ``relative path -> sha256`` map."""
    hashes: dict[str, str] = {}
    for relative, payload in sorted(files.items()):
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = _dump(payload)
        destination.write_text(text, encoding="utf-8")
        hashes[relative] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashes


# --- provenance ------------------------------------------------------------------------------


def _build_provenance(
    manifest: BundleManifest,
    manifest_path: Path,
    ground_truth: Sequence[SourceGroundTruthInstance],
    per_config: Mapping[str, tuple[tuple[SourcePrediction, ...], dict[int, MeasurementRecord]]],
    records: Sequence[dict[str, Any]],
    coverage: str,
    file_hashes: Mapping[str, str],
    created_at: str | None,
) -> dict[str, Any]:
    configs_provenance = []
    for config in manifest.configurations:
        predictions, measurements = per_config[config.config_id]
        declared = len(manifest.frames)
        present = len(measurements)
        configs_provenance.append(
            {
                "config_id": config.config_id,
                "model_id": config.model_id,
                "prediction_count": len(predictions),
                "prediction_provenance": config.prediction_provenance,
                "measurement_provenance": config.measurement_provenance,
                "measurement_coverage": f"{present}/{declared}",
                "failed_records": sum(
                    1 for r in records if r["config_id"] == config.config_id and not r["success"]
                ),
                "prediction_source_sha256": _hash_file(
                    safe_resolve(
                        manifest.manifest_dir, config.prediction_source, what="prediction_source"
                    )
                ),
                "measurement_source_sha256": _hash_file(
                    safe_resolve(
                        manifest.manifest_dir, config.measurement_source, what="measurement_source"
                    )
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "builder_version": BUILDER_VERSION,
        "benchmark_version": __version__,
        "bundle_id": manifest.bundle_id,
        "created_at": created_at if created_at is not None else _now_iso(),
        "manifest_sha256": _hash_file(manifest_path),
        "data_origin": manifest.data_origin,
        "coverage": coverage,
        "frame_stream_id": manifest.frame_stream_id,
        "image_dimensions": {"width": manifest.image_width, "height": manifest.image_height},
        "frame_count": len(manifest.frames),
        "configuration_ids": [config.config_id for config in manifest.configurations],
        "replay_record_count": len(records),
        "ground_truth": {
            "instance_count": len(ground_truth),
            "unique_track_count": len({gt.track_id for gt in ground_truth if not gt.ignore}),
            "ignored_instance_count": sum(1 for gt in ground_truth if gt.ignore),
            "source_sha256": _hash_file(
                safe_resolve(
                    manifest.manifest_dir, manifest.ground_truth_source, what="ground_truth_source"
                )
            ),
        },
        "configurations": configs_provenance,
        "profiles_note": (
            "policy-facing nominal costs are means of the supplied measurements, "
            "not a separate measurement"
        ),
        "generated_file_sha256": dict(sorted(file_hashes.items())),
        "notes": [
            "This tool executes no model and measures no hardware.",
            "Measurement and prediction provenance are declared by the manifest, not inferred.",
        ],
    }


def _hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _now_iso() -> str:
    # Isolated to this one provenance field; every other generated byte is deterministic.
    from datetime import datetime

    return datetime.now(UTC).isoformat()


# --- cross-checks ----------------------------------------------------------------------------


def _check_ground_truth_frames(
    ground_truth: Sequence[SourceGroundTruthInstance], manifest: BundleManifest
) -> None:
    known = set(manifest.frame_ids)
    for gt in ground_truth:
        if gt.frame_id not in known:
            raise SchemaValidationError(
                f"ground-truth instance {gt.track_id!r} is on frame {gt.frame_id}, which is not "
                "declared in the manifest frames"
            )
    for frame in manifest.frames:
        seen: set[str] = set()
        for gt in ground_truth:
            if gt.frame_id != frame.frame_id:
                continue
            if gt.track_id in seen:
                raise SchemaValidationError(
                    f"ground truth has two instances of track {gt.track_id!r} "
                    f"on frame {frame.frame_id}"
                )
            seen.add(gt.track_id)


def _check_prediction_frames(
    predictions: Sequence[SourcePrediction], manifest: BundleManifest, config: ManifestConfig
) -> None:
    known = set(manifest.frame_ids)
    for prediction in predictions:
        if prediction.frame_id not in known:
            raise SchemaValidationError(
                f"config {config.config_id!r} predicts on frame {prediction.frame_id}, which is "
                "not declared in the manifest frames"
            )


def _check_prediction_ids_unique(
    predictions: Sequence[SourcePrediction], config: ManifestConfig
) -> None:
    seen: set[tuple[int, str]] = set()
    for prediction in predictions:
        key = (prediction.frame_id, prediction.prediction_id)
        if key in seen:
            raise SchemaValidationError(
                f"config {config.config_id!r} repeats prediction_id {prediction.prediction_id!r} "
                f"on frame {prediction.frame_id}; prediction ids are unique within a frame/config"
            )
        seen.add(key)


def _validate_failure_reason(reason: str, config_id: str, frame_id: int) -> None:
    try:
        FailureReason(reason)
    except ValueError:
        raise SchemaValidationError(
            f"config {config_id!r} frame {frame_id}: failure_reason {reason!r} is not a known "
            f"reason ({[r.value for r in FailureReason]})"
        ) from None


def _dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


# --- CLI -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="missionbench.tools.build_empirical_bundle",
        description="Build a self-contained empirical replay bundle from a source manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Bundle-generation manifest.")
    parser.add_argument("--output", type=Path, required=True, help="Output bundle root.")
    parser.add_argument(
        "--coverage",
        choices=(COVERAGE_STRICT, COVERAGE_SPARSE),
        default=None,
        help="Override the manifest coverage policy (default: the manifest's, else strict).",
    )
    parser.add_argument(
        "--created-at", default=None, help="Pin the provenance timestamp for a reproducible build."
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate the generated bundle before returning."
    )
    args = parser.parse_args(argv)

    try:
        summary = build_empirical_bundle(
            args.manifest,
            args.output,
            coverage=args.coverage,
            created_at=args.created_at,
            validate=args.validate,
        )
    except SchemaValidationError as error:
        print(f"build_empirical_bundle: {error}", file=sys.stderr)
        return 2

    print(json.dumps(summary.to_dict(), indent=2))
    print(f"wrote bundle {summary.bundle_id!r} to {summary.output_root}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
