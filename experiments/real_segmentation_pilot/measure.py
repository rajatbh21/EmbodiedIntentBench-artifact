"""Latency and energy measurement for the pilot -- honest by construction.

Latency. Timed with a monotonic high-resolution clock, after warm-up runs that are discarded,
one sample per frame (optionally several, summarised). The timing boundary -- what is inside
the timed region -- is the caller's to fix and document; the same boundary must be used for
every configuration. Summary statistics report the whole distribution (mean, median, p95, min,
max), never a single average that hides the tail.

Energy. There is no silent zero. An energy value must declare its provenance: ``measured``
from hardware telemetry, ``externally_supplied`` by a validated profiler, or ``estimated`` as
an explicitly assumed average power multiplied by measured time. An estimate records the power
it assumed and where that number came from, and is never described as measured.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import ceil
from typing import Any, TypeVar

__all__ = [
    "EnergyEstimate",
    "EnergyProvenance",
    "LatencyStats",
    "estimate_energy",
    "external_energy",
    "latency_stats",
    "timed",
    "warm_up",
]

_T = TypeVar("_T")


class EnergyProvenance(StrEnum):
    """Where an energy value came from. There is no default; a value must declare one."""

    MEASURED = "measured"
    EXTERNALLY_SUPPLIED = "externally_supplied"
    ESTIMATED = "estimated"


@dataclass(frozen=True, slots=True)
class EnergyEstimate:
    """An energy value with mandatory provenance.

    ``assumed_power_w`` and ``source`` are populated for an estimate and describe exactly how
    the number was produced, so a reader can weigh it. A measured or externally supplied value
    leaves ``assumed_power_w`` ``None`` and names its instrument in ``source``.
    """

    joules: float
    provenance: EnergyProvenance
    source: str
    assumed_power_w: float | None = None

    def __post_init__(self) -> None:
        if self.joules < 0.0 or self.joules != self.joules:
            raise ValueError(f"energy must be finite and non-negative, got {self.joules}")
        if not self.source.strip():
            raise ValueError("energy provenance requires a non-empty source description")

    def to_dict(self) -> dict[str, Any]:
        return {
            "joules": self.joules,
            "provenance": self.provenance.value,
            "source": self.source,
            "assumed_power_w": self.assumed_power_w,
        }


def estimate_energy(latency_s: float, *, average_power_w: float, source: str) -> EnergyEstimate:
    """Estimate energy as ``average_power_w * latency_s``, labelled ``estimated``.

    This is the honest fallback when no telemetry exists: it is explicitly an assumption, and
    the assumed power and its source are recorded so results are never mistaken for measured.
    """
    if average_power_w < 0.0:
        raise ValueError(f"average_power_w must be non-negative, got {average_power_w}")
    return EnergyEstimate(
        joules=average_power_w * latency_s,
        provenance=EnergyProvenance.ESTIMATED,
        source=source,
        assumed_power_w=average_power_w,
    )


def external_energy(joules: float, *, source: str, measured: bool) -> EnergyEstimate:
    """Wrap an energy value produced outside the pilot (telemetry or a validated profiler)."""
    provenance = EnergyProvenance.MEASURED if measured else EnergyProvenance.EXTERNALLY_SUPPLIED
    return EnergyEstimate(joules=joules, provenance=provenance, source=source)


@dataclass(frozen=True, slots=True)
class LatencyStats:
    """Per-sample latencies and their summary. Times are seconds."""

    per_sample_s: tuple[float, ...]
    mean_s: float
    median_s: float
    p95_s: float
    min_s: float
    max_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.per_sample_s),
            "mean_s": self.mean_s,
            "median_s": self.median_s,
            "p95_s": self.p95_s,
            "min_s": self.min_s,
            "max_s": self.max_s,
        }


def timed(
    fn: Callable[[], _T], *, clock: Callable[[], float] = time.perf_counter
) -> tuple[_T, float]:
    """Run ``fn`` once and return ``(result, elapsed_seconds)`` on a monotonic clock.

    ``clock`` defaults to ``time.perf_counter`` (monotonic, high resolution) and is injectable
    so tests are deterministic. A real GPU backend passes a clock that synchronises the device
    before reading the time, so launch latency is never mistaken for inference latency.
    """
    start = clock()
    result = fn()
    return result, clock() - start


def warm_up(fn: Callable[[], Any], runs: int) -> None:
    """Run ``fn`` ``runs`` times and discard the results, to reach steady state before timing."""
    for _ in range(max(0, runs)):
        fn()


def latency_stats(samples: Sequence[float]) -> LatencyStats:
    """Summarise per-sample latencies. Empty input yields zeros rather than raising."""
    ordered = sorted(samples)
    n = len(ordered)
    if n == 0:
        return LatencyStats((), 0.0, 0.0, 0.0, 0.0, 0.0)
    mean = sum(ordered) / n
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    return LatencyStats(
        per_sample_s=tuple(samples),
        mean_s=mean,
        median_s=median,
        p95_s=_percentile(ordered, 95.0),
        min_s=ordered[0],
        max_s=ordered[-1],
    )


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence."""
    n = len(ordered)
    rank = max(1, ceil(percentile / 100.0 * n))
    return ordered[min(rank, n) - 1]
