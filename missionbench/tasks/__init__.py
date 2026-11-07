"""Task plugins: everything that is specific to *what the mission is measuring*.

Responsibility
--------------
Each task supplies an ``EvidenceTracker`` (accumulates evidence from execution results and
exposes a non-leaking policy summary) and a ``TaskEvaluator`` (scores the final evidence
record against hidden ground truth and returns a standardised ``TaskEvaluationResult``). A
``TaskDefinition`` binds the two together and is resolved from the task registry by
``contract.task_id``.

Modules
-------
- ``base``     -- the four protocols plus ``TaskEvaluationResult``.
- ``registry`` -- ``task_id`` to implementation, and ``resolve_task``.
- ``human_search_segmentation`` -- the one task V1 implements.

Boundaries
----------
This is the only place mask-matching, deduplication, and target-F1 logic may live. The
simulator, metrics, and policy layers must remain unaware of it. Adding object detection
later means adding a package here plus one registry entry -- and nothing else.

The tracker is where the observation boundary is actually enforced: it holds
ground-truth-derived detail for the evaluator while exposing only prediction-derived
quantities to a policy. A value that crosses from ``final_record()`` into
``policy_summary()`` does not make a metric wrong -- it invalidates the benchmark.
"""

from missionbench.tasks.base import (
    EvidenceRecord,
    EvidenceTracker,
    GroundTruth,
    TaskDefinition,
    TaskEvaluationResult,
    TaskEvaluator,
)
from missionbench.tasks.registry import resolve_task, task_registry

__all__ = [
    "EvidenceRecord",
    "EvidenceTracker",
    "GroundTruth",
    "TaskDefinition",
    "TaskEvaluationResult",
    "TaskEvaluator",
    "resolve_task",
    "task_registry",
]
