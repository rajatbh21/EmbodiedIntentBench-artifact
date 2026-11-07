"""Binds the human-search tracker and evaluator into one registered task."""

from __future__ import annotations

from pathlib import Path

from missionbench.schemas.episode import Episode
from missionbench.schemas.loading import SchemaValidationError
from missionbench.schemas.profile import ProfileCatalog
from missionbench.schemas.task_spec import TaskSpec
from missionbench.tasks.human_search_segmentation.evaluator import (
    HumanSearchSegmentationEvaluator,
)
from missionbench.tasks.human_search_segmentation.evidence_tracker import (
    HumanSearchEvidenceTracker,
)
from missionbench.tasks.human_search_segmentation.ground_truth import (
    TASK_ID,
    HumanSearchGroundTruth,
    HumanSearchMaskGroundTruth,
    load_any_ground_truth,
)
from missionbench.tasks.human_search_segmentation.prediction import (
    SyntheticHumanSearchPredictions,
)

__all__ = ["HumanSearchSegmentationTask"]


class HumanSearchSegmentationTask:
    """The V1 task, resolved from the registry by ``contract.task_id``."""

    __slots__ = ("_task_spec",)

    def __init__(self, task_spec: TaskSpec) -> None:
        if task_spec.task_id != TASK_ID:
            raise SchemaValidationError(
                f"{type(self).__name__} implements {TASK_ID!r}, "
                f"but was given a specification for {task_spec.task_id!r}"
            )
        self._task_spec = task_spec

    @property
    def task_id(self) -> str:
        return TASK_ID

    @property
    def task_spec(self) -> TaskSpec:
        return self._task_spec

    def create_tracker(self) -> HumanSearchEvidenceTracker:
        """Return a fresh tracker. Trackers are stateful, so one per episode."""
        return HumanSearchEvidenceTracker()

    def create_evaluator(self) -> HumanSearchSegmentationEvaluator:
        return HumanSearchSegmentationEvaluator(self._task_spec)

    def load_ground_truth(
        self, directory: Path, episode: Episode
    ) -> HumanSearchGroundTruth | HumanSearchMaskGroundTruth | None:
        """Find the answers for this episode's *frame stream*, in either ground-truth form.

        Keyed on the stream, not the episode, so several episodes flying the same scene under
        different battery, network, or contract conditions are scored identically. Every file
        in the directory is read rather than a filename convention being assumed, so renaming
        a fixture cannot silently unhook it from its episodes. Interval and mask ground truth
        may sit side by side; each file declares which it is and is loaded accordingly.
        """
        if not directory.is_dir():
            return None
        for path in sorted(directory.glob("*.json")):
            ground_truth = load_any_ground_truth(path)
            if ground_truth.frame_stream_id == episode.frame_stream_id:
                return ground_truth
        return None

    def create_prediction_source(
        self,
        ground_truth: HumanSearchGroundTruth | HumanSearchMaskGroundTruth | None,
        profiles: ProfileCatalog,
    ) -> SyntheticHumanSearchPredictions | None:
        """Return the synthetic prediction source, or ``None`` when it does not apply.

        The synthetic source draws detections from *interval* ground truth. Mask ground truth
        belongs to empirical replay -- the predictions are recorded, not synthesised -- so
        there is nothing to generate and this returns ``None``: a profile run against a mask
        stream simply gathers no evidence rather than inventing masks it does not have.

        The source reads ground truth, which is legitimate -- it stands in for a model that
        would actually be looking at the scene -- but never lets a ground-truth quantity reach
        a policy: the hidden fields it writes are excluded from the tracker's policy summary.
        """
        if not isinstance(ground_truth, HumanSearchGroundTruth):
            return None
        return SyntheticHumanSearchPredictions(ground_truth, profiles)
