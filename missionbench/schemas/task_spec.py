"""Task specification: how a task's evidence is scored.

Separated from the contract on purpose. The contract says "target_f1 must be at least
0.80"; the task specification says what makes a target count as found in the first place
(V1: a predicted mask whose IoU against a ground-truth instance is at least 0.50) and how
repeated sightings collapse into one unique target.

Consequence: adjusting the matching threshold is a one-file change, and two contracts over
the same task cannot silently disagree about what "found" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from missionbench.schemas.common import ComparisonOperator
from missionbench.schemas.loading import SchemaValidationError, open_document

__all__ = [
    "DeduplicationMethod",
    "DeduplicationRule",
    "MatchingRule",
    "TaskSpec",
    "check_contract_is_supported",
    "load_task_spec",
]


class DeduplicationMethod(StrEnum):
    """How repeated observations of one physical target collapse to a unique target."""

    #: Two predictions naming the same ground-truth track are the same target.
    GROUND_TRUTH_TRACK_ID = "ground_truth_track_id"


@dataclass(frozen=True, slots=True)
class MatchingRule:
    """When a prediction counts as matching a ground-truth instance."""

    metric: str
    operator: ComparisonOperator
    threshold: float

    def matches(self, score: float) -> bool:
        """Return whether a similarity ``score`` satisfies this rule."""
        return self.operator.compare(score, self.threshold)


@dataclass(frozen=True, slots=True)
class DeduplicationRule:
    """How the evidence tracker deduplicates targets."""

    method: DeduplicationMethod


_FIELDS: Final = (
    "task_id",
    "evidence_type",
    "target_type",
    "ground_truth_type",
    "matching_rule",
    "deduplication",
    "supported_quality_metrics",
)
_MATCHING_RULE_FIELDS: Final = ("metric", "operator", "threshold")
_DEDUPLICATION_FIELDS: Final = ("method",)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """A task definition, resolved by ``contract.task_id`` through the task registry."""

    task_id: str
    evidence_type: str
    target_type: str
    ground_truth_type: str
    matching_rule: MatchingRule
    deduplication: DeduplicationRule
    supported_quality_metrics: tuple[str, ...]

    def supports_metric(self, metric_name: str) -> bool:
        """Return whether this task can compute the named quality metric."""
        return metric_name in self.supported_quality_metrics


def load_task_spec(path: Path) -> TaskSpec:
    """Load and validate a task specification file."""
    reader = open_document(path, document_type="TaskSpec", allowed_fields=_FIELDS)
    matching = reader.get_object("matching_rule", allowed_fields=_MATCHING_RULE_FIELDS)
    deduplication = reader.get_object("deduplication", allowed_fields=_DEDUPLICATION_FIELDS)
    return TaskSpec(
        task_id=reader.get_str("task_id"),
        evidence_type=reader.get_str("evidence_type"),
        target_type=reader.get_str("target_type"),
        ground_truth_type=reader.get_str("ground_truth_type"),
        matching_rule=MatchingRule(
            metric=matching.get_str("metric"),
            operator=matching.get_enum("operator", ComparisonOperator),
            threshold=matching.get_float("threshold"),
        ),
        deduplication=DeduplicationRule(
            method=deduplication.get_enum("method", DeduplicationMethod),
        ),
        supported_quality_metrics=reader.get_str_tuple("supported_quality_metrics"),
    )


def check_contract_is_supported(task_spec: TaskSpec, *, task_id: str, quality_metric: str) -> None:
    """Raise if a contract asks this task for something it cannot provide.

    Cross-document consistency lives here rather than in either loader, because neither
    file can see the other at load time. The runner calls this once the pair is resolved.
    """
    if task_spec.task_id != task_id:
        raise SchemaValidationError(
            f"contract targets task {task_id!r} but the supplied task specification is "
            f"{task_spec.task_id!r}"
        )
    if not task_spec.supports_metric(quality_metric):
        raise SchemaValidationError(
            f"task {task_spec.task_id!r} does not support quality metric {quality_metric!r}; "
            f"supported metrics are {list(task_spec.supported_quality_metrics)}"
        )
