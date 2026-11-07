"""Execution backends: how a selected configuration actually produces a result.

Responsibility
--------------
Turn an ``ExecutionRequest`` (episode, frame, selected configuration, current network
observation, time, seed) into an ``ExecutionResult`` (success, latency, energy,
communication volume, prediction payload, failure reason, metadata).

Modules
-------
- ``base``                      -- ``Executor`` and ``PredictionSource`` protocols,
                                   ``ExecutionRequest``/``ExecutionResult``, ``FailureReason``.
- ``profile_executor``          -- costs synthesised from a per-platform profile, including
                                   the remote latency model. The V1 default.
- ``replay_executor``           -- precomputed outcomes keyed by frame x configuration.
- ``real_segmentation_executor``-- documented stub; no ML dependency in V1.
- ``registry``                  -- name-to-backend mapping plus ``ExecutorContext`` and
                                   ``build_executor``, the single construction path.

Provenance
----------
Every backend carries a stable ``executor_id`` equal to its registry name, and
``executor_id_of`` reads it off the object. The name recorded in a result is derived from
the executor that ran, never supplied alongside it, so a result cannot misreport its
backend.

Boundaries
----------
The runner must not care which backend produced a result. Backends are selected through the
registry, never through ``if executor_type == ...`` in the loop. Behaviour comes from typed
configuration metadata -- ``strategy.placement`` decides local versus remote -- and never
from parsing a ``config_id``.

Predictions are opaque here. Their shape belongs to the task; an executor that understood
them would grow a branch per task.
"""

from missionbench.executor.base import (
    ExecutionRequest,
    ExecutionResult,
    Executor,
    FailureReason,
    Prediction,
    PredictionSource,
    executor_id_of,
)
from missionbench.executor.profile_executor import (
    DEFAULT_REMOTE_TIMEOUT_S,
    ProfileExecutor,
    remote_latency_s,
    transfer_time_s,
)
from missionbench.executor.real_segmentation_executor import (
    NOT_IMPLEMENTED_MESSAGE,
    RealSegmentationExecutor,
)
from missionbench.executor.registry import (
    ExecutorContext,
    build_executor,
    executor_registry,
)
from missionbench.executor.replay_executor import (
    ReplayExecutor,
    ReplayRecord,
    ReplayRecordSet,
    load_replay_record_set,
    load_replay_records,
    make_replay_executor,
)

__all__ = [
    "DEFAULT_REMOTE_TIMEOUT_S",
    "NOT_IMPLEMENTED_MESSAGE",
    "ExecutionRequest",
    "ExecutionResult",
    "Executor",
    "ExecutorContext",
    "FailureReason",
    "Prediction",
    "PredictionSource",
    "ProfileExecutor",
    "RealSegmentationExecutor",
    "ReplayExecutor",
    "ReplayRecord",
    "ReplayRecordSet",
    "build_executor",
    "executor_id_of",
    "executor_registry",
    "load_replay_record_set",
    "load_replay_records",
    "make_replay_executor",
    "remote_latency_s",
    "transfer_time_s",
]
