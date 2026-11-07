"""Time-indexed, object-blind reference tracking for simulated trajectory transfer."""

import bisect
import math
from itertools import pairwise


def validate(times, positions):
    if len(times) < 2 or len(times) != len(positions) or times[0] != 0:
        raise ValueError("Paired reference beginning at zero required")
    if any(not math.isfinite(t) for t in times) or any(b <= a for a, b in pairwise(times)):
        raise ValueError("Strictly increasing finite reference times required")
    if any(len(p) != 3 or any(not math.isfinite(v) for v in p) for p in positions):
        raise ValueError("Finite metric positions required")


def reference_state(t, times, positions):
    if not math.isfinite(t):
        raise ValueError("Finite control time required")
    if t < 0:
        return tuple(positions[0]), (0.0, 0.0, 0.0)
    if t >= times[-1]:
        return tuple(positions[-1]), (0.0, 0.0, 0.0)
    i = bisect.bisect_right(times, t) - 1
    dt = times[i + 1] - times[i]
    fraction = (t - times[i]) / dt
    velocity = tuple((b - a) / dt for a, b in zip(positions[i], positions[i + 1], strict=True))
    target = tuple(
        a + fraction * (b - a) for a, b in zip(positions[i], positions[i + 1], strict=True)
    )
    return target, velocity


def command(position, t, times, positions, *, speed_limit=8.0, gain=0.8):
    if (
        not math.isfinite(speed_limit)
        or not 0 < speed_limit <= 12
        or not math.isfinite(gain)
        or gain < 0
    ):
        raise ValueError("Bounded finite controller parameters required")
    if len(position) != 3 or any(not math.isfinite(v) for v in position):
        raise ValueError("Finite odometry required")
    target, velocity = reference_state(t, times, positions)
    v = tuple(
        feed + gain * (goal - actual)
        for feed, goal, actual in zip(velocity, target, position, strict=True)
    )
    norm = math.sqrt(sum(x * x for x in v))
    scale = min(1, speed_limit / norm) if norm else 1
    return tuple(x * scale for x in v)
