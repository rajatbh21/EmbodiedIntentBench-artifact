"""Network observation model.

Owns the *behaviour* of turning a time into an observation; the trace data and its
validation live in ``missionbench.schemas.network_trace``. A future recorded-network or
live model implements :class:`NetworkModel` and drops in unchanged.

The model holds the whole trace. The runner asks it for the observation at the current
time and puts only that single observation into the ``RuntimeState``. A policy that could
see the remaining trace would pre-empt degradations it has no way of knowing about, which
is exactly the adaptation the benchmark is trying to measure.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Protocol

from missionbench.schemas.network_trace import NetworkObservation, NetworkTrace

__all__ = ["NetworkModel", "TraceBasedNetworkModel"]


class NetworkModel(Protocol):
    """Supplies the network conditions in force at a given time."""

    def observe(self, time_s: float) -> NetworkObservation:
        """Return the conditions at ``time_s``."""
        ...


class TraceBasedNetworkModel:
    """Reads observations from a deterministic, pre-validated trace.

    Out-of-range times are clamped to the nearest segment rather than raising. An episode
    may legitimately run past the trace's end -- a slow inference can push the clock beyond
    it -- and failing there would abort a run over a bookkeeping detail rather than a
    benchmark condition. Traces are checked at load time to span the longest deadline, so
    clamping stays a safety net rather than a routine path.
    """

    __slots__ = ("_segment_starts", "_trace")

    def __init__(self, trace: NetworkTrace) -> None:
        self._trace = trace
        # Segments are validated ordered and contiguous at load time, so their start times
        # are sorted and a binary search always lands on the owning segment.
        self._segment_starts = [segment.start_s for segment in trace.segments]

    @property
    def trace_id(self) -> str:
        return self._trace.trace_id

    def observe(self, time_s: float) -> NetworkObservation:
        index = bisect_right(self._segment_starts, time_s) - 1
        index = min(max(index, 0), len(self._trace.segments) - 1)
        return self._trace.segments[index].observation

    def change_times_s(self) -> tuple[float, ...]:
        """Return the times at which conditions change.

        Recorded so that adaptation latency -- how long a policy takes to respond to a
        network change -- can be computed later from logs. Adaptation latency is not a V1
        leaderboard metric, because "appropriately adapted" is not yet formally defined;
        this only makes sure the data to define it is not thrown away.
        """
        return tuple(
            segment.start_s
            for index, segment in enumerate(self._trace.segments)
            if index > 0 and segment.observation != self._trace.segments[index - 1].observation
        )
