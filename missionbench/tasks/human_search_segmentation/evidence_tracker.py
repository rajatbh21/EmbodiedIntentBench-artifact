"""Accumulates human-search evidence across an episode.

The tracker sits exactly on the boundary between what the simulator knows and what a policy
may know, and it is the only object that holds both. It keeps every predicted instance,
including the hidden ground-truth fields the evaluator will need, and exposes them through
two methods that must never be confused:

- :meth:`policy_summary` -- derived from predictions alone.
- :meth:`final_record` -- everything, and never shown to a policy.

Two different deduplications
----------------------------
Both views count "unique targets", and they count different things on purpose.

- The **policy** sees distinct ``predicted_target_id``s: what the system believes it has
  found. It may be wrong, and the policy is given no way to discover that.
- The **evaluator** deduplicates by ``ground_truth_track_id``, per the task specification.

Collapsing them would hand the policy its own true positive count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from missionbench.executor.base import ExecutionResult
from missionbench.schemas.runtime_state import EvidenceSummary
from missionbench.tasks.human_search_segmentation.ground_truth import TASK_ID
from missionbench.tasks.human_search_segmentation.prediction import (
    EvidenceInstance,
    FramePrediction,
)

__all__ = ["HumanSearchEvidenceRecord", "HumanSearchEvidenceTracker"]


@dataclass(frozen=True, slots=True)
class HumanSearchEvidenceRecord:
    """The complete evidence one episode gathered. Evaluator input; never policy-visible.

    ``instances`` are precomputed or empirical predictions, or a mix across a suite; both
    kinds expose the same policy-visible fields, and the evaluator narrows to the concrete
    kind the ground truth calls for.
    """

    processed_frames: int
    instances: tuple[EvidenceInstance, ...]

    @property
    def task_id(self) -> str:
        return TASK_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "processed_frames": self.processed_frames,
            "instances": [instance.to_dict() for instance in self.instances],
        }


class HumanSearchEvidenceTracker:
    """Folds execution results into evidence, and serves the two views of it."""

    __slots__ = ("_confidence_total", "_instances", "_processed_frames")

    def __init__(self) -> None:
        self._instances: list[EvidenceInstance] = []
        self._processed_frames = 0
        self._confidence_total = 0.0

    def update(self, result: ExecutionResult) -> None:
        """Fold in one execution result.

        A failed execution advances nothing. It cost time and energy -- the runner accounts
        for that separately -- but it produced no observation, so counting it as a processed
        frame would tell the policy it had looked at something it never saw.
        """
        if not result.success or result.prediction is None:
            return

        # Coerce rather than type-check: a live profile run passes a FramePrediction, a
        # replay run passes the same payload after a JSON round-trip (a dict). Both are this
        # task's payload, so reading both forms is this task's responsibility. A payload from
        # a different task raises inside coerce().
        prediction = FramePrediction.coerce(result.prediction)

        self._processed_frames += 1
        for instance in prediction.instances:
            self._instances.append(instance)
            self._confidence_total += instance.confidence

    def policy_summary(self) -> EvidenceSummary:
        """Return the policy-visible summary. Prediction-derived only.

        ``predicted_unique_targets`` counts distinct predicted identities, not confirmed
        finds: a false positive inflates it and the policy cannot tell.
        """
        return EvidenceSummary(
            predicted_unique_targets=len(
                {instance.predicted_target_id for instance in self._instances}
            ),
            processed_frames=self._processed_frames,
            mean_prediction_confidence=(
                self._confidence_total / len(self._instances) if self._instances else 0.0
            ),
        )

    def final_record(self) -> HumanSearchEvidenceRecord:
        return HumanSearchEvidenceRecord(
            processed_frames=self._processed_frames,
            instances=tuple(self._instances),
        )
