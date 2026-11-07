"""Write model output and ground truth into the empirical bundle *source* formats.

These produce exactly the files ``missionbench.tools.bundle_manifest`` parses -- a
ground-truth JSON, a per-configuration prediction JSONL, and a measurement CSV -- so the
existing builder and validator consume them unchanged. Nothing here trusts or writes a
ground-truth track id or a ``mask_iou`` into a prediction; the pilot computes IoU from masks,
in the evaluator, later. Ordering is deterministic (frames ascending, instances by their
stable id) so a rebuild is byte-stable.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

from experiments.real_segmentation_pilot.interfaces import GroundTruthInstance

__all__ = [
    "MEASUREMENT_COLUMNS",
    "stable_prediction_id",
    "write_ground_truth_source",
    "write_measurements_csv",
    "write_predictions_jsonl",
]

MEASUREMENT_COLUMNS = (
    "frame_id",
    "success",
    "latency_s",
    "compute_energy_j",
    "upload_mb",
    "download_mb",
    "failure_reason",
)


def stable_prediction_id(config_id: str, frame_id: int, index: int) -> str:
    """A deterministic, collision-free prediction id, unique within a frame and configuration.

    Derived only from the configuration, frame, and the instance's order within the frame --
    never from content that could vary between runs -- so the same inputs always yield the same
    ids and the records diff cleanly.
    """
    return f"{config_id}__f{frame_id:06d}__i{index:03d}"


def write_ground_truth_source(
    path: Path, *, frame_stream_id: str, instances: Iterable[GroundTruthInstance]
) -> int:
    """Write the ground-truth source JSON. Returns the instance count."""
    ordered = sorted(instances, key=lambda gt: (gt.frame_id, gt.track_id))
    payload = {
        "schema_version": "1.0",
        "frame_stream_id": frame_stream_id,
        "instances": [
            {
                "frame_id": gt.frame_id,
                "track_id": gt.track_id,
                "category": gt.category,
                **({"ignore": True} if gt.ignore else {}),
                "mask": gt.mask.to_dict(),
            }
            for gt in ordered
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return len(ordered)


def write_predictions_jsonl(path: Path, records: Sequence[dict]) -> int:
    """Write per-configuration predictions as JSON Lines, one object per line, ordered.

    Each record must be ``{frame_id, prediction_id, category, confidence, mask}``. Records are
    sorted by ``(frame_id, prediction_id)`` for a deterministic file.
    """
    ordered = sorted(records, key=lambda record: (record["frame_id"], record["prediction_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return len(ordered)


def write_measurements_csv(path: Path, records: Sequence[dict]) -> int:
    """Write per-configuration measurements as CSV with the fixed header, ordered by frame."""
    ordered = sorted(records, key=lambda record: record["frame_id"])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MEASUREMENT_COLUMNS)
    writer.writeheader()
    for record in ordered:
        writer.writerow({column: record.get(column, "") for column in MEASUREMENT_COLUMNS})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buffer.getvalue(), encoding="utf-8")
    return len(ordered)
