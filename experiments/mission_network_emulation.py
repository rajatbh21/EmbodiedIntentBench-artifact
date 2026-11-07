"""Deterministic virtual link/server chronology; no real network or hardware claims.

The simulator owns LinkTrace. A policy receives only LinkSnapshot at its current
absolute timestamp. Payloads use exact rational bits across half-open trace
segments, with completion rounded upward to an integer nanosecond. Outages stall
serialization; buffered work resumes when capacity returns. Server service and
propagation continue during outages. This is not TCP, packet loss, retransmission,
contention, a measured edge server, or electrical/onboard energy validation.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import pairwise
from math import isfinite

NS_PER_SECOND = 1_000_000_000


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def ceil_fraction(value):
    return -(-value.numerator // value.denominator)


@dataclass(frozen=True, slots=True)
class LinkSegment:
    start_ns: int
    end_ns: int
    upload_bps: int
    download_bps: int
    rtt_ns: int
    outage: bool = False

    def __post_init__(self):
        for name in ("start_ns", "end_ns", "upload_bps", "download_bps", "rtt_ns"):
            integer(getattr(self, name), name)
        if self.end_ns <= self.start_ns or type(self.outage) is not bool:
            raise ValueError("A positive segment duration and boolean outage are required")


@dataclass(frozen=True, slots=True)
class LinkSnapshot:
    """Current public observation, with no future boundary or trace reference."""

    timestamp_ns: int
    upload_bps: int
    download_bps: int
    rtt_ns: int
    outage: bool

    def to_dict(self):
        return dict(
            timestamp_ns=self.timestamp_ns,
            upload_bps=self.upload_bps,
            download_bps=self.download_bps,
            rtt_ns=self.rtt_ns,
            outage=self.outage,
        )


@dataclass(frozen=True, slots=True)
class LinkTrace:
    segments: tuple[LinkSegment, ...]
    _starts: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self):
        segments = tuple(self.segments)
        if not segments or any(not isinstance(s, LinkSegment) for s in segments):
            raise ValueError("A nonempty LinkSegment sequence is required")
        if any(a.end_ns != b.start_ns for a, b in pairwise(segments)):
            raise ValueError("Trace segments must be ordered, contiguous and nonoverlapping")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "_starts", tuple(s.start_ns for s in segments))

    def segment_at(self, timestamp_ns):
        integer(timestamp_ns, "timestamp_ns")
        index = bisect_right(self._starts, timestamp_ns) - 1
        if index < 0 or timestamp_ns >= self.segments[index].end_ns:
            raise ValueError("Timestamp is outside the declared trace")
        return self.segments[index]

    def snapshot(self, timestamp_ns):
        segment = self.segment_at(timestamp_ns)
        return LinkSnapshot(
            timestamp_ns,
            0 if segment.outage else segment.upload_bps,
            0 if segment.outage else segment.download_bps,
            segment.rtt_ns,
            segment.outage,
        )


@dataclass(frozen=True, slots=True)
class LinkEvent:
    phase: str
    start_ns: int
    end_ns: int
    bits: Fraction = Fraction(0)
    rate_bps: int = 0

    def to_dict(self):
        return dict(
            phase=self.phase,
            start_ns=self.start_ns,
            end_ns=self.end_ns,
            bits_numerator=self.bits.numerator,
            bits_denominator=self.bits.denominator,
            rate_bps=self.rate_bps,
        )


@dataclass(frozen=True, slots=True)
class JobResult:
    status: str
    start_ns: int
    finish_ns: int
    stop_limit_ns: int
    requested_upload_bits: int
    requested_download_bits: int
    uploaded_bits: Fraction
    downloaded_bits: Fraction
    events: tuple[LinkEvent, ...]

    @property
    def completed(self):
        return self.status == "completed"

    @property
    def prediction_eligible(self):
        """Only a full response arriving by the inclusive stop limit is usable."""
        return self.completed

    @property
    def completion_ns(self):
        return self.finish_ns if self.completed else None

    @property
    def elapsed_ns(self):
        return self.finish_ns - self.start_ns

    @property
    def phase_durations_ns(self):
        return {
            phase: sum(e.end_ns - e.start_ns for e in self.events if e.phase == phase)
            for phase in (
                "upload_tx",
                "upload_wait",
                "upload_propagation",
                "service",
                "download_tx",
                "download_wait",
                "download_propagation",
            )
        }

    def to_dict(self):
        return dict(
            status=self.status,
            completed=self.completed,
            prediction_eligible=self.prediction_eligible,
            start_ns=self.start_ns,
            finish_ns=self.finish_ns,
            completion_ns=self.completion_ns,
            stop_limit_ns=self.stop_limit_ns,
            elapsed_ns=self.elapsed_ns,
            requested_upload_bits=self.requested_upload_bits,
            requested_download_bits=self.requested_download_bits,
            uploaded_bits_numerator=self.uploaded_bits.numerator,
            uploaded_bits_denominator=self.uploaded_bits.denominator,
            downloaded_bits_numerator=self.downloaded_bits.numerator,
            downloaded_bits_denominator=self.downloaded_bits.denominator,
            phase_durations_ns=self.phase_durations_ns,
            events=[e.to_dict() for e in self.events],
        )


def emulate_job(
    trace, *, start_ns, upload_bits, download_bits, service_ns, timeout_ns, deadline_ns=None
):
    """Run one buffered virtual request with absolute timestamps and cancellation.

    Order: upload serialization, upload propagation, server service, download
    serialization, download propagation. Each RTT is sampled at that direction's
    first transmitted bit. Uplink uses floor(RTT/2), downlink ceil(RTT/2), so a
    constant odd-ns RTT is charged exactly once. Outages affect serialization,
    not an already propagated signal or server work. No payload/model output is
    returned by this function: the caller must gate its selected prediction on
    result.prediction_eligible. Bytes already sent remain chargeable on failure.

    Timeouts are relative to dispatch; deadline_ns is absolute and inclusive.
    Exact completion at either limit is accepted. If both limits are equal, a
    failure is labelled deadline. Trace exhaustion is explicit; no last segment
    is extrapolated. Dispatch before the trace begins is invalid.
    """
    if not isinstance(trace, LinkTrace):
        raise ValueError("LinkTrace required")
    for name, value in (
        ("start_ns", start_ns),
        ("service_ns", service_ns),
        ("timeout_ns", timeout_ns),
    ):
        integer(value, name)
    for name, value in (("upload_bits", upload_bits), ("download_bits", download_bits)):
        integer(value, name, 1)
    if start_ns < trace.segments[0].start_ns:
        raise ValueError("Dispatch precedes trace start")
    limit, limit_status = start_ns + timeout_ns, "timeout"
    if deadline_ns is not None:
        integer(deadline_ns, "deadline_ns")
        if deadline_ns < start_ns:
            raise ValueError("Deadline precedes dispatch")
        if deadline_ns <= limit:
            limit, limit_status = deadline_ns, "deadline"
    now, events, transferred = start_ns, [], {"upload": Fraction(0), "download": Fraction(0)}

    def result(status):
        return JobResult(
            status,
            start_ns,
            now,
            limit,
            upload_bits,
            download_bits,
            transferred["upload"],
            transferred["download"],
            tuple(events),
        )

    def transfer(direction, payload):
        nonlocal now
        remaining, first_rtt = Fraction(payload), None
        while remaining:
            if now >= limit:
                return limit_status, first_rtt
            try:
                segment = trace.segment_at(now)
            except ValueError:
                return "trace_exhausted", first_rtt
            stop = min(segment.end_ns, limit)
            rate = 0 if segment.outage else getattr(segment, direction + "_bps")
            if rate == 0:
                events.append(LinkEvent(direction + "_wait", now, stop))
                now = stop
                continue
            if first_rtt is None:
                first_rtt = segment.rtt_ns
            capacity = Fraction(rate * (stop - now), NS_PER_SECOND)
            if remaining <= capacity:
                duration = ceil_fraction(remaining * NS_PER_SECOND / rate)
                bits, finish = remaining, now + duration
            else:
                bits, finish = capacity, stop
            events.append(LinkEvent(direction + "_tx", now, finish, bits, rate))
            now, remaining = finish, remaining - bits
            transferred[direction] += bits
        return None, first_rtt

    def delay(phase, duration):
        nonlocal now
        finish = min(now + duration, limit)
        if finish > now:
            events.append(LinkEvent(phase, now, finish))
        full = finish - now == duration
        now = finish
        return full

    status, rtt = transfer("upload", upload_bits)
    if status:
        return result(status)
    if not delay("upload_propagation", rtt // 2) or not delay("service", service_ns):
        return result(limit_status)
    status, rtt = transfer("download", download_bits)
    if status:
        return result(status)
    if not delay("download_propagation", (rtt + 1) // 2):
        return result(limit_status)
    return result("completed")


def account_radio_energy(result, *, upload_power_w, download_power_w, idle_power_w):
    """Configured terminal radio joules, not measured energy or remote compute.

    Idle covers outage waits, propagation, and server-service waiting, up to the
    job's actual stop. No remote-server power, electrical measurement, or work
    after cancellation is asserted by this local accounting convention.
    """
    for power in (upload_power_w, download_power_w, idle_power_w):
        if isinstance(power, bool) or not isinstance(power, (int, float)):
            raise ValueError("Finite nonnegative configured radio powers required")
        if not isfinite(power) or power < 0:
            raise ValueError("Finite nonnegative configured radio powers required")
    phases = result.phase_durations_ns
    upload = phases["upload_tx"] / NS_PER_SECOND * upload_power_w
    download = phases["download_tx"] / NS_PER_SECOND * download_power_w
    idle_ns = result.elapsed_ns - phases["upload_tx"] - phases["download_tx"]
    idle = idle_ns / NS_PER_SECOND * idle_power_w
    return dict(
        configured_radio_j=upload + download + idle,
        upload_j=upload,
        download_j=download,
        idle_j=idle,
        measured_electrical_energy=False,
        remote_server_energy_j=None,
    )
