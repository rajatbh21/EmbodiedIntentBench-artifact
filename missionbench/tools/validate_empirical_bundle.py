"""Validate a generated empirical replay bundle, independently of how it was built.

    python -m missionbench.tools.validate_empirical_bundle --bundle data/generated/pilot_bundle

Runs the same strict loaders the benchmark uses -- so any schema drift, unknown field, or
cross-reference gap surfaces here -- and then adds the checks specific to an empirical bundle:
every replay record names a known configuration, masks match the frame dimensions, predictions
carry no ground-truth identity, latency and energy are finite and non-negative, the declared
coverage policy is honoured, and the provenance manifest's counts and file hashes match what is
actually on disk.

Errors accumulate with context (frame, configuration, prediction or track id, file, field);
the report lists everything wrong rather than stopping at the first problem, so one pass fixes
a bundle. Returns a :class:`ValidationReport`; the CLI exits non-zero when it is not ``ok``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from missionbench.benchmark import BenchmarkData
from missionbench.executor.base import FailureReason
from missionbench.schemas.contract import load_contract
from missionbench.schemas.episode import load_episode
from missionbench.schemas.loading import SchemaValidationError, read_json_object
from missionbench.schemas.profile import check_catalog_is_profiled
from missionbench.schemas.task_spec import check_contract_is_supported
from missionbench.tasks.human_search_segmentation.ground_truth import load_any_ground_truth
from missionbench.tasks.human_search_segmentation.masks import decode_mask

__all__ = ["ValidationReport", "validate_empirical_bundle"]

_FORBIDDEN_PREDICTION_FIELDS = ("ground_truth_track_id", "mask_iou", "track_id")


@dataclass(slots=True)
class ValidationReport:
    """The outcome of validating a bundle: whether it is usable, and every problem found."""

    bundle_root: Path
    errors: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def _check(self, name: str) -> None:
        self.checks.append(name)

    def _fail(self, message: str) -> None:
        self.errors.append(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bundle_root": str(self.bundle_root),
            "ok": self.ok,
            "checks_run": self.checks,
            "errors": self.errors,
        }


def validate_empirical_bundle(bundle_root: Path) -> ValidationReport:
    """Validate the bundle at ``bundle_root`` and return a report of everything found."""
    report = ValidationReport(bundle_root=bundle_root)
    if not bundle_root.is_dir():
        report._fail(f"bundle root {bundle_root} is not a directory")
        return report

    # The strict loaders validate the scaffold: catalog, platforms, profiles, paths, traces,
    # task specs, and replay-set schemas, plus at-most-one replay set per episode.
    try:
        data = BenchmarkData(bundle_root)
        report._check("scaffold loads (BenchmarkData)")
    except SchemaValidationError as error:
        report._fail(f"scaffold: {error}")
        return report

    episode = _load_one(bundle_root / "episodes", load_episode, "episode", report)
    contract = _load_one(bundle_root / "contracts", load_contract, "contract", report)
    frame_manifest = _load_frame_manifest(bundle_root, report)
    ground_truth = _load_ground_truth(bundle_root, report)
    if episode is None or contract is None or frame_manifest is None or ground_truth is None:
        return report

    _validate_cross_references(data, episode, contract, ground_truth, report)
    _validate_ground_truth(ground_truth, frame_manifest, report)
    replay = _load_replay(bundle_root, report)
    if replay is not None:
        _validate_records(replay, data, episode, frame_manifest, report)
    _validate_provenance(bundle_root, frame_manifest, ground_truth, replay, report)
    return report


def _validate_cross_references(
    data, episode, contract, ground_truth, report: ValidationReport
) -> None:
    report._check("episode/contract/task cross-references")
    try:
        task_spec = data.task_spec(contract.task_id)
        check_contract_is_supported(
            task_spec, task_id=contract.task_id, quality_metric=contract.quality_metric
        )
    except SchemaValidationError as error:
        report._fail(f"contract vs task spec: {error}")
    try:
        profiles = data.profiles(episode.platform_id)
        check_catalog_is_profiled(
            profiles, config_ids=episode.allowed_config_ids, platform_id=episode.platform_id
        )
    except SchemaValidationError as error:
        report._fail(f"profiles: {error}")
    for config_id in episode.allowed_config_ids:
        if config_id not in data.catalog:
            report._fail(f"episode allows config {config_id!r} absent from the catalog")
    if ground_truth.frame_stream_id != episode.frame_stream_id:
        report._fail(
            f"ground-truth frame_stream_id {ground_truth.frame_stream_id!r} != episode "
            f"{episode.frame_stream_id!r}"
        )


def _validate_ground_truth(ground_truth, frame_manifest, report: ValidationReport) -> None:
    report._check("ground-truth masks and track ids")
    width = frame_manifest["image_width"]
    height = frame_manifest["image_height"]
    declared_frames = {frame["frame_id"] for frame in frame_manifest["frames"]}
    for frame in ground_truth.frames:
        if frame.frame_id not in declared_frames:
            report._fail(
                f"ground truth references frame {frame.frame_id}, not in the frame manifest"
            )
        for instance in frame.instances:
            if not isinstance(instance.track_id, str) or not instance.track_id.strip():
                report._fail(
                    f"frame {frame.frame_id}: a ground-truth track_id is not a stable string"
                )
            if instance.mask.width != width or instance.mask.height != height:
                report._fail(
                    f"frame {frame.frame_id} track {instance.track_id!r}: GT mask is "
                    f"{instance.mask.height}x{instance.mask.width}, frame is {height}x{width}"
                )


def _validate_records(replay, data, episode, frame_manifest, report: ValidationReport) -> None:
    report._check("replay records: configs, masks, ids, finiteness, coverage")
    width = frame_manifest["image_width"]
    height = frame_manifest["image_height"]
    declared_frames = [frame["frame_id"] for frame in frame_manifest["frames"]]
    coverage = frame_manifest.get("coverage", "strict")

    present: set[tuple[int, str]] = set()
    for (frame_id, config_id), record in replay["records"].items():
        present.add((frame_id, config_id))
        where = f"frame {frame_id}, config {config_id!r}"
        if config_id not in data.catalog:
            report._fail(f"{where}: replay record names a config absent from the catalog")
        for numeric in ("latency_ms", "onboard_energy_j", "upload_mb", "download_mb"):
            value = record["raw"].get(numeric, 0.0)
            if not _finite_non_negative(value):
                report._fail(f"{where}: {numeric} must be finite and non-negative, got {value!r}")
        if not record["success"]:
            reason = record["raw"].get("failure_reason")
            if reason is None:
                report._fail(f"{where}: a failed record must state a failure_reason")
            elif reason not in {r.value for r in FailureReason}:
                report._fail(f"{where}: unknown failure_reason {reason!r}")
        else:
            _validate_prediction_payload(
                record["raw"].get("prediction"), where, width, height, report
            )

    if coverage == "strict":
        for config_id in episode.allowed_config_ids:
            for frame_id in declared_frames:
                if (frame_id, config_id) not in present:
                    report._fail(
                        f"strict coverage: no replay record for frame {frame_id}, "
                        f"config {config_id!r}"
                    )


def _validate_prediction_payload(
    prediction, where, width, height, report: ValidationReport
) -> None:
    if not isinstance(prediction, dict) or "instances" not in prediction:
        report._fail(
            f"{where}: a successful record must carry a prediction with an 'instances' list"
        )
        return
    seen: set[str] = set()
    for index, instance in enumerate(prediction["instances"]):
        if not isinstance(instance, dict):
            report._fail(f"{where}: instance {index} is not an object")
            continue
        forbidden = sorted(set(instance) & set(_FORBIDDEN_PREDICTION_FIELDS))
        if forbidden:
            report._fail(f"{where}: prediction carries forbidden field(s) {forbidden}")
        prediction_id = instance.get("prediction_id")
        if not isinstance(prediction_id, str) or not prediction_id:
            report._fail(f"{where}: instance {index} has no valid prediction_id")
        elif prediction_id in seen:
            report._fail(
                f"{where}: duplicate prediction_id {prediction_id!r} within the frame/config"
            )
        else:
            seen.add(prediction_id)
        try:
            mask = decode_mask(instance.get("mask"), context=f"{where} instance {index} mask")
        except SchemaValidationError as error:
            report._fail(str(error))
            continue
        if mask.width != width or mask.height != height:
            report._fail(
                f"{where}: prediction mask is {mask.height}x{mask.width}, frame is {height}x{width}"
            )


def _validate_provenance(
    bundle_root, frame_manifest, ground_truth, replay, report: ValidationReport
) -> None:
    report._check("provenance counts and file hashes")
    path = bundle_root / "provenance.json"
    if not path.is_file():
        report._fail("provenance.json is missing")
        return
    try:
        provenance = read_json_object(path)
    except SchemaValidationError as error:
        report._fail(f"provenance.json: {error}")
        return

    if provenance.get("frame_count") != len(frame_manifest["frames"]):
        report._fail("provenance frame_count does not match the frame manifest")
    unique_tracks = len(
        {
            inst.track_id
            for frame in ground_truth.frames
            for inst in frame.instances
            if not inst.ignore
        }
    )
    gt = provenance.get("ground_truth", {})
    if gt.get("unique_track_count") != unique_tracks:
        report._fail(
            f"provenance unique_track_count {gt.get('unique_track_count')} "
            f"!= actual {unique_tracks}"
        )
    if replay is not None and provenance.get("replay_record_count") != len(replay["records"]):
        report._fail("provenance replay_record_count does not match the replay set")

    for relative, expected in provenance.get("generated_file_sha256", {}).items():
        target = bundle_root / relative
        if not target.is_file():
            report._fail(f"provenance lists {relative}, which is missing from the bundle")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            report._fail(f"provenance hash mismatch for {relative}")


# --- loading helpers -------------------------------------------------------------------------


def _load_one(directory: Path, loader, kind: str, report: ValidationReport):
    files = sorted(directory.glob("*.json")) if directory.is_dir() else []
    if not files:
        report._fail(f"no {kind} file under {directory}")
        return None
    try:
        return loader(files[0])
    except SchemaValidationError as error:
        report._fail(f"{kind}: {error}")
        return None


def _load_frame_manifest(bundle_root: Path, report: ValidationReport) -> dict[str, Any] | None:
    path = bundle_root / "frames" / "frame_manifest.json"
    if not path.is_file():
        report._fail("frames/frame_manifest.json is missing")
        return None
    try:
        return dict(read_json_object(path))
    except SchemaValidationError as error:
        report._fail(f"frame manifest: {error}")
        return None


def _load_ground_truth(bundle_root: Path, report: ValidationReport):
    files = sorted((bundle_root / "ground_truth").glob("*.json"))
    if not files:
        report._fail("no ground-truth file under ground_truth/")
        return None
    try:
        return load_any_ground_truth(files[0])
    except SchemaValidationError as error:
        report._fail(f"ground truth: {error}")
        return None


def _load_replay(bundle_root: Path, report: ValidationReport) -> dict[str, Any] | None:
    files = sorted((bundle_root / "predictions").glob("*.json"))
    if not files:
        report._fail("no replay set under predictions/")
        return None
    try:
        payload = read_json_object(files[0])
    except SchemaValidationError as error:
        report._fail(f"replay set: {error}")
        return None
    records: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in payload.get("records", []):
        key = (raw.get("frame_id"), raw.get("config_id"))
        if not isinstance(key[0], int) or not isinstance(key[1], str):
            report._fail(f"replay record has an invalid frame_id/config_id: {key}")
            continue
        records[key] = {"success": bool(raw.get("success")), "raw": raw}
    return {"records": records}


def _finite_non_negative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == value
        and value >= 0.0
        and value != float("inf")
    )


# --- CLI -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="missionbench.tools.validate_empirical_bundle",
        description="Validate a generated empirical replay bundle.",
    )
    parser.add_argument("--bundle", type=Path, required=True, help="Bundle root to validate.")
    args = parser.parse_args(argv)

    report = validate_empirical_bundle(args.bundle)
    print(json.dumps(report.to_dict(), indent=2))
    if report.ok:
        print(f"bundle at {args.bundle} is valid ({len(report.checks)} checks)", file=sys.stderr)
        return 0
    print(f"bundle at {args.bundle} has {len(report.errors)} error(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
