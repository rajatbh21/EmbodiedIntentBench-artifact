"""Integer-time capture scheduling independent of rendering and wall-clock speed."""

from __future__ import annotations

import math
from dataclasses import dataclass


def seconds_to_ns(seconds: float, *, allow_zero: bool = False) -> int:
    if not math.isfinite(seconds) or seconds < 0 or (seconds == 0 and not allow_zero):
        raise ValueError("time must be finite and positive (zero only when explicitly allowed)")
    value = round(seconds * 1_000_000_000)
    if seconds > 0 and value == 0:
        raise ValueError("positive time must be at least one nanosecond")
    return value


@dataclass(frozen=True)
class CaptureSchedule:
    """Capture times lie on the physics grid; a ready-at-capture result does not skip it."""

    epoch_ns: int
    interval_ns: int
    end_ns: int
    step_ns: int

    def __post_init__(self) -> None:
        values = (self.epoch_ns, self.interval_ns, self.end_ns, self.step_ns)
        if any(type(v) is not int for v in values):
            raise ValueError("schedule times must be integer nanoseconds")
        if self.epoch_ns < 0 or self.interval_ns <= 0 or self.end_ns <= self.epoch_ns:
            raise ValueError("invalid capture interval or mission bounds")
        if self.step_ns <= 0 or any(v % self.step_ns for v in values[:3]):
            raise ValueError("capture times must align with the physics timestep")

    def at(self, index: int) -> int:
        if type(index) is not int or index < 0:
            raise ValueError("capture index must be a nonnegative integer")
        return self.epoch_ns + index * self.interval_ns

    def complete(self, index: int, latency_ns: int) -> tuple[int, int, tuple[int, ...]]:
        """Return tick-rounded completion, next capture index, and skipped IDs.

        A final inference is allowed to finish after end_ns; no capture is started
        at/after end_ns. Overshoot remains in the record instead of being hidden.
        """
        if type(latency_ns) is not int or latency_ns < 0:
            raise ValueError("latency must be nonnegative integer nanoseconds")
        if self.at(index) >= self.end_ns:
            raise ValueError("capture is outside the mission window")
        ticks = (latency_ns + self.step_ns - 1) // self.step_ns
        completion = self.at(index) + ticks * self.step_ns
        next_index = max(
            index + 1, (completion - self.epoch_ns + self.interval_ns - 1) // self.interval_ns
        )
        skipped = tuple(i for i in range(index + 1, next_index) if self.at(i) < self.end_ns)
        return completion, next_index, skipped
