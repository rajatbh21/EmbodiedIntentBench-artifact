"""Replay execution: results come from a precomputed record per frame and configuration.

The long-term mapping the benchmark wants is ``frame x config -> prediction``. Computing
that once with real models and replaying it makes every subsequent run reproducible, free of
GPUs, and comparable across policies -- two policies that select the same configuration on
the same frame see exactly the same prediction, so a difference in their scores comes from
their decisions and nothing else.

This module owns the *envelope*: which frame, which configuration, what it cost, and an
opaque prediction payload. The payload's shape belongs to the task, and is passed through
untouched.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from missionbench.executor.base import (
    ExecutionRequest,
    ExecutionResult,
    FailureReason,
    Prediction,
)
from missionbench.schemas.loading import DocumentReader, SchemaValidationError, open_document

__all__ = [
    "ReplayExecutor",
    "ReplayRecord",
    "ReplayRecordSet",
    "load_replay_record_set",
    "load_replay_records",
]

_MS_PER_S: Final = 1000.0
_DOCUMENT_FIELDS: Final = ("record_set_id", "episode_id", "records")
_RECORD_FIELDS: Final = (
    "frame_id",
    "config_id",
    "success",
    "latency_ms",
    "onboard_energy_j",
    "upload_mb",
    "download_mb",
    "failure_reason",
    "prediction",
)


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """One precomputed execution outcome."""

    frame_id: int
    config_id: str
    success: bool
    latency_ms: float
    onboard_energy_j: float
    upload_mb: float = 0.0
    download_mb: float = 0.0
    failure_reason: FailureReason | None = None
    prediction: Prediction | None = None

    @property
    def key(self) -> tuple[int, str]:
        return (self.frame_id, self.config_id)


@dataclass(frozen=True, slots=True)
class ReplayRecordSet:
    """One file's worth of precomputed outcomes, and the episode they were recorded from.

    Keyed on the episode rather than the frame stream because a set captured synthetically
    is tied to that episode's seed. A set captured from real model output would generalise
    to the whole stream, but claiming that now would be claiming more than V1 can deliver.
    """

    episode_id: str
    records: dict[tuple[int, str], ReplayRecord]
    record_set_id: str = ""
    source: Path | None = None

    def __len__(self) -> int:
        return len(self.records)

    @property
    def config_ids(self) -> frozenset[str]:
        return frozenset(config_id for _, config_id in self.records)

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(sorted({frame_id for frame_id, _ in self.records}))


class ReplayExecutor:
    """Serves precomputed records, keyed by frame and configuration."""

    #: Provenance, equal to this backend's registry name.
    executor_id: Final = "replay"

    __slots__ = ("_record_set_id", "_records", "_source", "_strict")

    def __init__(
        self,
        records: Mapping[tuple[int, str], ReplayRecord],
        *,
        strict: bool = False,
        record_set_id: str = "",
        source: Path | None = None,
    ) -> None:
        """Args:
        records: Outcomes keyed by ``(frame_id, config_id)``.
        strict: Whether a missing record raises instead of returning a failed result.
            The default is lenient because a record set need not cover every frame of a
            long mission -- a slow configuration skips frames, so a policy can reach a
            frame nothing was precomputed for. Set it when a record set is meant to be
            exhaustive and a gap is a fixture bug.
        record_set_id: Identifier of the source set, quoted in errors and gap reports so a
            missing entry names the file it should have been in.
        source: Path the set was loaded from, for the same reason.
        """
        self._records = dict(records)
        self._strict = strict
        self._record_set_id = record_set_id
        self._source = source

    def _origin(self) -> str:
        """Human-readable description of where these records came from."""
        source = str(self._source) if self._source else ""
        parts = [part for part in (self._record_set_id, source) if part]
        return " from ".join(parts) if parts else "an unnamed record set"

    @property
    def frame_ids(self) -> tuple[int, ...]:
        return tuple(sorted({frame_id for frame_id, _ in self._records}))

    def __len__(self) -> int:
        return len(self._records)

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        record = self._records.get((request.frame_id, request.configuration.config_id))
        if record is None:
            if self._strict:
                raise SchemaValidationError(
                    f"replay gap: {self._origin()} has no entry for episode "
                    f"{request.episode_id!r}, frame {request.frame_id}, configuration "
                    f"{request.configuration.config_id!r}. Record the missing pairs, or run "
                    f"with the profile executor."
                )
            # Not an exception: a policy is allowed to reach a frame nobody precomputed.
            # It is recorded as a failed inference so the gap is visible in the metrics
            # rather than silently scoring as a success with no evidence -- and never by
            # falling back to a different backend, which would mix two provenances in one run.
            return ExecutionResult(
                success=False,
                latency_s=0.0,
                onboard_energy_j=0.0,
                failure_reason=FailureReason.NO_PREDICTION_AVAILABLE,
                metadata={
                    "episode_id": request.episode_id,
                    "frame_id": request.frame_id,
                    "config_id": request.configuration.config_id,
                    "record_set": self._origin(),
                },
            )

        return ExecutionResult(
            success=record.success,
            latency_s=record.latency_ms / _MS_PER_S,
            onboard_energy_j=record.onboard_energy_j,
            upload_mb=record.upload_mb,
            download_mb=record.download_mb,
            prediction=record.prediction if record.success else None,
            failure_reason=record.failure_reason,
            metadata={"source": "replay"},
        )


def load_replay_record_set(path: Path) -> ReplayRecordSet:
    """Load and validate a replay record set, including which episode it covers.

    ``episode_id`` was previously declared in the schema and never read, which left nothing
    able to decide *which* set belonged to an episode. It is now the resolution key.
    """
    reader = open_document(path, document_type="ReplayRecordSet", allowed_fields=_DOCUMENT_FIELDS)
    records: dict[tuple[int, str], ReplayRecord] = {}

    for entry in reader.get_object_list("records", allowed_fields=_RECORD_FIELDS):
        record = _read_record(entry)
        if record.key in records:
            raise SchemaValidationError(
                f"{path.name} -> ReplayRecordSet: duplicate record for frame "
                f"{record.frame_id} and configuration {record.config_id!r}"
            )
        records[record.key] = record

    return ReplayRecordSet(
        record_set_id=reader.get_optional_str("record_set_id") or "",
        episode_id=reader.get_str("episode_id"),
        records=records,
        source=path,
    )


def load_replay_records(path: Path) -> dict[tuple[int, str], ReplayRecord]:
    """Load only the records, discarding the set's identity.

    Kept for callers that already hold the right file and need nothing else.
    """
    return load_replay_record_set(path).records


def _read_record(reader: DocumentReader) -> ReplayRecord:
    success = reader.get_bool("success")
    failure_reason = reader.get_optional_str("failure_reason")

    if success and failure_reason is not None:
        raise SchemaValidationError(f"{reader.context}: a successful record has no failure_reason")
    if not success and failure_reason is None:
        raise SchemaValidationError(
            f"{reader.context}: a failed record must state a failure_reason"
        )

    return ReplayRecord(
        frame_id=reader.get_int("frame_id", minimum=0),
        config_id=reader.get_str("config_id"),
        success=success,
        latency_ms=reader.get_float("latency_ms", minimum=0.0),
        onboard_energy_j=reader.get_float("onboard_energy_j", minimum=0.0),
        upload_mb=reader.get_optional_float("upload_mb", default=0.0, minimum=0.0),
        download_mb=reader.get_optional_float("download_mb", default=0.0, minimum=0.0),
        failure_reason=None if failure_reason is None else _read_failure_reason(reader),
        prediction=reader.get_passthrough("prediction"),
    )


def _read_failure_reason(reader: DocumentReader) -> FailureReason:
    return reader.get_enum("failure_reason", FailureReason)


def make_replay_executor(path: Path, *, strict: bool = False) -> ReplayExecutor:
    """Build a replay executor directly from a record-set file."""
    record_set = load_replay_record_set(path)
    return ReplayExecutor(
        record_set.records,
        strict=strict,
        record_set_id=record_set.record_set_id,
        source=record_set.source,
    )
