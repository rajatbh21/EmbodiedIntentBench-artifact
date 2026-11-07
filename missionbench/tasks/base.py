"""The task interface: how evidence is accumulated and scored.

A task is the only part of the benchmark that knows what the mission is *about*. Everything
above it -- the runner, the metrics, the policies -- works through these four types and
never learns that V1 happens to be looking for people in segmentation masks.

The split that makes that possible runs through the evidence tracker. It sees every
execution result, including the ground-truth-derived detail an evaluator will later need,
but exposes two different views:

- :meth:`EvidenceTracker.policy_summary` -- prediction-derived only, safe for a policy.
- :meth:`EvidenceTracker.final_record` -- everything, consumed only by the evaluator.

Those two must never be confused. A quantity that crosses from the second into the first
does not make a metric wrong; it invalidates the benchmark, because the policy would be
scored on information no deployed system could have.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from missionbench.executor.base import ExecutionResult
from missionbench.schemas.common import ComparisonOperator
from missionbench.schemas.contract import Contract
from missionbench.schemas.episode import Episode
from missionbench.schemas.profile import ProfileCatalog
from missionbench.schemas.runtime_state import EvidenceSummary
from missionbench.schemas.task_spec import TaskSpec

__all__ = [
    "EvidenceRecord",
    "EvidenceTracker",
    "GroundTruth",
    "TaskDefinition",
    "TaskEvaluationResult",
    "TaskEvaluator",
]


@dataclass(frozen=True, slots=True)
class TaskEvaluationResult:
    """A task's verdict, in the one shape the generic metrics layer understands.

    Standardised so that mission-success logic never names a task-specific metric. Adding
    object detection later means producing this record with ``metric_name="detection_ap"``
    -- and changing nothing in ``missionbench.metrics``.

    ``details`` carries whatever else the task computed. It is reported alongside the
    headline value but takes no part in deciding success.
    """

    metric_name: str
    value: float
    operator: ComparisonOperator
    threshold: float
    success: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def against_contract(
        cls,
        contract: Contract,
        *,
        value: float,
        details: Mapping[str, Any] | None = None,
    ) -> TaskEvaluationResult:
        """Build a result by applying the contract's own operator and threshold.

        Keeps the comparison in one place: a task states what it measured, not whether that
        counts as passing.
        """
        return cls(
            metric_name=contract.quality_metric,
            value=value,
            operator=contract.quality_operator,
            threshold=contract.quality_threshold,
            success=contract.quality_satisfied(value),
            details=dict(details or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "success": self.success,
            "details": dict(self.details),
        }


@runtime_checkable
class EvidenceRecord(Protocol):
    """The complete evidence an episode gathered, as handed to the evaluator.

    Task-specific in content and deliberately opaque above the task layer: the runner
    serialises it into the episode record and passes it to the evaluator, and does not
    interpret it.
    """

    @property
    def task_id(self) -> str: ...

    @property
    def processed_frames(self) -> int: ...

    def to_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class GroundTruth(Protocol):
    """Hidden answers a task is scored against. Never reachable from a policy.

    Only ``task_id`` is required. What a set of answers is keyed on is the task's business:
    human search keys on the frame stream, since several episodes may fly the same scene
    under different conditions and must be scored identically.
    """

    @property
    def task_id(self) -> str: ...


class EvidenceTracker(Protocol):
    """Accumulates execution results into mission evidence."""

    def update(self, result: ExecutionResult) -> None:
        """Fold one execution result into the evidence gathered so far."""
        ...

    def policy_summary(self) -> EvidenceSummary:
        """Return the policy-visible summary. Must contain nothing derived from ground truth."""
        ...

    def final_record(self) -> EvidenceRecord:
        """Return the complete evidence, for the evaluator. Never shown to a policy."""
        ...


class TaskEvaluator(Protocol):
    """Scores mission evidence against hidden ground truth."""

    def evaluate(
        self,
        evidence: EvidenceRecord,
        ground_truth: GroundTruth,
        contract: Contract,
    ) -> TaskEvaluationResult: ...


class TaskDefinition(Protocol):
    """Binds a task's tracker and evaluator together, resolved by ``contract.task_id``.

    The runner asks the registry for one of these and never imports a task module, which is
    what keeps ``task_id`` from becoming a branch in the core.
    """

    @property
    def task_id(self) -> str: ...

    @property
    def task_spec(self) -> TaskSpec: ...

    def create_tracker(self) -> EvidenceTracker:
        """Return a fresh tracker. Called once per episode -- trackers are stateful."""
        ...

    def create_evaluator(self) -> TaskEvaluator: ...

    def load_ground_truth(self, directory: Path, episode: Episode) -> GroundTruth | None:
        """Find and load this episode's answers from a ground-truth directory.

        The task owns the lookup because only it knows what its answers are keyed on --
        human search keys on the frame stream, a future task might key on something else.
        Putting the rule here keeps the composition root from learning any of it.

        Returns ``None`` when no answers exist, which is legitimate: a run may exercise the
        loop and its resource accounting without scoring quality.
        """
        ...

    def create_prediction_source(
        self, ground_truth: GroundTruth | None, profiles: ProfileCatalog
    ) -> object | None:
        """Return a ``PredictionSource`` for a simulating executor, or ``None``.

        Only profile-driven execution needs this; a replay backend carries its own
        predictions. ``None`` runs the executor in resource-only mode.
        """
        ...
