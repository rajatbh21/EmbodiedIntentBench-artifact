"""Metric computation over episode records.

Responsibility
--------------
Derive per-episode metrics and cross-episode aggregates from the structured records the
simulator emits. Metrics are computed from *records*, never by reaching into live simulator
state, so a new metric can be added later without rerunning an episode as long as the step
log already carried the data.

Modules
-------
- ``episode_metrics``   -- the five hard constraints, their violation margins, resource
                           totals, and mission success.
- ``aggregate_metrics`` -- success rates and means across an episode suite. Mission Success
                           Rate is the primary metric; the other rates decompose it.

Boundaries
----------
Mission-success logic consumes a standardised ``TaskEvaluationResult`` and must never
reference a task-specific field such as ``target_f1``. Adding a task metric must not require
editing this package.
"""

from missionbench.metrics.aggregate_metrics import AggregateMetrics, aggregate_metrics
from missionbench.metrics.episode_metrics import (
    CONSTRAINT_NAMES,
    ConstraintOutcome,
    EpisodeMetrics,
    compute_episode_metrics,
)

__all__ = [
    "CONSTRAINT_NAMES",
    "AggregateMetrics",
    "ConstraintOutcome",
    "EpisodeMetrics",
    "aggregate_metrics",
    "compute_episode_metrics",
]
