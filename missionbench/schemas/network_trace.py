"""Deterministic network traces and the single network measurement a policy may see.

This module owns the trace *data*: parsing, validation, and the observation type. The
lookup behaviour that turns a time into an observation is a simulation concern and lives
in ``missionbench.simulator.network_trace`` behind the ``NetworkModel`` protocol, so a
future recorded-trace or live-network model can replace it without touching this schema.

The full trace is simulator-internal. A policy observes only the current
:class:`NetworkObservation`; exposing the future trace would let a policy pre-empt a
degradation it has no way of knowing about, which is precisely what the benchmark tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from missionbench.schemas.loading import SchemaValidationError, open_document

__all__ = [
    "NetworkObservation",
    "NetworkTrace",
    "NetworkTraceSegment",
    "load_network_trace",
]

_FIELDS: Final = ("trace_id", "segments")
_SEGMENT_FIELDS: Final = (
    "start_s",
    "end_s",
    "bandwidth_mbps",
    "rtt_ms",
    "packet_loss_frac",
)


@dataclass(frozen=True, slots=True)
class NetworkObservation:
    """The network measurement visible to a policy at one instant."""

    bandwidth_mbps: float
    rtt_ms: float
    packet_loss_frac: float

    @property
    def is_disconnected(self) -> bool:
        """Whether remote execution is impossible because there is no usable bandwidth."""
        return self.bandwidth_mbps <= 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "bandwidth_mbps": self.bandwidth_mbps,
            "rtt_ms": self.rtt_ms,
            "packet_loss_frac": self.packet_loss_frac,
        }


@dataclass(frozen=True, slots=True)
class NetworkTraceSegment:
    """Constant network conditions over the half-open interval ``[start_s, end_s)``."""

    start_s: float
    end_s: float
    observation: NetworkObservation

    def contains(self, time_s: float) -> bool:
        """Half-open containment, so adjacent segments never both claim a boundary time."""
        return self.start_s <= time_s < self.end_s


@dataclass(frozen=True, slots=True)
class NetworkTrace:
    """A deterministic, gap-free sequence of network segments.

    Segments are validated to be ordered and contiguous, so every time within the trace's
    span resolves to exactly one segment and a lookup can never fall into a hole.
    """

    trace_id: str
    segments: tuple[NetworkTraceSegment, ...]

    @property
    def start_s(self) -> float:
        return self.segments[0].start_s

    @property
    def end_s(self) -> float:
        return self.segments[-1].end_s


def load_network_trace(path: Path) -> NetworkTrace:
    """Load and validate a network trace file."""
    reader = open_document(path, document_type="NetworkTrace", allowed_fields=_FIELDS)
    trace_id = reader.get_str("trace_id")

    segments: list[NetworkTraceSegment] = []
    for entry in reader.get_object_list("segments", allowed_fields=_SEGMENT_FIELDS):
        start_s = entry.get_float("start_s", minimum=0.0)
        end_s = entry.get_float("end_s", exclusive_minimum=start_s)
        segments.append(
            NetworkTraceSegment(
                start_s=start_s,
                end_s=end_s,
                observation=NetworkObservation(
                    bandwidth_mbps=entry.get_float("bandwidth_mbps", minimum=0.0),
                    rtt_ms=entry.get_float("rtt_ms", minimum=0.0),
                    packet_loss_frac=entry.get_fraction("packet_loss_frac"),
                ),
            )
        )

    _check_contiguous(segments, trace_id=trace_id, context=path.name)
    return NetworkTrace(trace_id=trace_id, segments=tuple(segments))


def _check_contiguous(segments: list[NetworkTraceSegment], *, trace_id: str, context: str) -> None:
    """Reject traces with gaps, overlaps, or out-of-order segments.

    A gap would make the observation at that time undefined; an overlap would make it
    ambiguous. Both would break the determinism the benchmark depends on, so they are
    load-time errors rather than lookup-time surprises.
    """
    for index in range(1, len(segments)):
        previous, current = segments[index - 1], segments[index]
        if current.start_s != previous.end_s:
            relation = "gap" if current.start_s > previous.end_s else "overlap"
            raise SchemaValidationError(
                f"{context} -> NetworkTrace {trace_id!r}: {relation} between segment "
                f"{index - 1} (ends at {previous.end_s}) and segment {index} "
                f"(starts at {current.start_s}); segments must be ordered and contiguous"
            )
