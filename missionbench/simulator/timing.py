"""Decision-loop timing.

V1 runs one decision per second over a nominal one-frame-per-second stream, with at most
one inference in flight. Those two rules together decide what happens when inference is
slower than the interval, and the answer has to be exact: it determines how many frames an
expensive configuration costs the mission, which is most of what the benchmark measures.

The rules, stated once here rather than scattered as constants:

- A step lasts ``max(decision_interval_s, inference_latency_s)``. The interval is a floor,
  not a budget.
- No second inference starts while one is running, so a slow configuration does not
  overlap its successor -- it delays it.
- The frame index is the wall clock quantised to the frame interval. Frames that elapse
  during a long inference are simply never seen; that dropped work is the cost being
  modelled, not an error.
"""

from __future__ import annotations

import math
from typing import Final

__all__ = [
    "DEFAULT_DECISION_INTERVAL_S",
    "DEFAULT_FRAME_INTERVAL_S",
    "elapsed_step_time_s",
    "frame_id_at",
    "frames_skipped",
]

#: One policy decision per second (V1 assumption; see docs/v1_spec.md §15).
DEFAULT_DECISION_INTERVAL_S: Final = 1.0

#: Nominal one-frame-per-second stream (V1 assumption).
DEFAULT_FRAME_INTERVAL_S: Final = 1.0


def elapsed_step_time_s(
    inference_latency_s: float,
    *,
    decision_interval_s: float = DEFAULT_DECISION_INTERVAL_S,
) -> float:
    """Return how much wall-clock time one decision step consumes.

    Inference faster than the interval does not buy a second decision: the loop still ticks
    once per interval. Inference slower than the interval stretches the step, because V1
    permits only one inference at a time.
    """
    return max(decision_interval_s, inference_latency_s)


def frame_id_at(
    time_s: float,
    *,
    frame_interval_s: float = DEFAULT_FRAME_INTERVAL_S,
) -> int:
    """Return the index of the frame available at ``time_s``.

    Floor, not round: at t=1.9 s the frame captured at 2.0 s does not exist yet.
    """
    return math.floor(time_s / frame_interval_s)


def frames_skipped(
    previous_frame_id: int,
    current_frame_id: int,
) -> int:
    """Return how many frames elapsed unprocessed between two decision steps.

    Zero when the step took one frame interval. Positive when a slow inference ran past
    frames that were never examined -- logged so the cost of an expensive configuration is
    visible in the record rather than only implied by the clock.
    """
    return max(0, current_frame_id - previous_frame_id - 1)
