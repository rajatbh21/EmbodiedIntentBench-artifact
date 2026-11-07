"""Scores human-search evidence against hidden ground truth.

A predicted instance is a **match** when it corresponds to a real target *and* segments it
well enough to count, judged by the task specification's matching rule (V1: mask IoU at or
above 0.50). Detecting something is not the same as detecting it well enough -- which is why
a weak configuration can see a person and still fail to find them.

Counting is per **unique target**, not per instance. A mission that spots the same person on
sixty consecutive frames has found one person, so:

- a ground-truth target counts as found if *any* prediction matched it;
- a predicted identity counts as a false positive if *none* of its instances matched anything.

Both figures come out of the same pass, so precision and recall cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from missionbench.schemas.contract import Contract
from missionbench.schemas.loading import SchemaValidationError
from missionbench.schemas.task_spec import TaskSpec
from missionbench.tasks.base import TaskEvaluationResult
from missionbench.tasks.human_search_segmentation.evidence_tracker import (
    HumanSearchEvidenceRecord,
)
from missionbench.tasks.human_search_segmentation.ground_truth import (
    TARGET_CATEGORY,
    HumanSearchGroundTruth,
    HumanSearchMaskGroundTruth,
)
from missionbench.tasks.human_search_segmentation.masks import mask_iou
from missionbench.tasks.human_search_segmentation.matching import FrameMatching, match_frame
from missionbench.tasks.human_search_segmentation.prediction import MaskPredictedInstance

__all__ = ["EmpiricalQualityScores", "HumanSearchSegmentationEvaluator", "QualityScores"]

METRIC_PRECISION: Final = "target_precision"
METRIC_RECALL: Final = "target_recall"
METRIC_F1: Final = "target_f1"
#: The empirical detection-level precision metric name, distinct from the track-level
#: ``target_precision`` so the two counting units can never be confused for one another.
METRIC_DETECTION_PRECISION: Final = "detection_precision"

#: Frames per minute of examined footage at the nominal one-frame-per-second stream
#: (docs/v1_spec.md §15). Used to turn a false-positive count into a rate.
FRAMES_PER_MINUTE: Final = 60.0

#: How quality was measured, recorded in the evaluation details. Absent means the precomputed
#: scalar path (synthetic profile, or legacy scalar-IoU replay); present means IoU was
#: computed from real prediction and ground-truth masks. Together with the executor id this
#: is what distinguishes synthetic profile, legacy replay, and empirical mask replay in a
#: saved result.
QUALITY_EVAL_EMPIRICAL_MASK: Final = "empirical_mask_iou"


@dataclass(frozen=True, slots=True)
class QualityScores:
    """All three quality metrics plus the counts they came from."""

    target_precision: float
    target_recall: float
    target_f1: float
    matched_targets: int
    total_targets: int
    predicted_targets: int
    false_positive_targets: int

    def value_of(self, metric_name: str) -> float:
        try:
            return {
                METRIC_PRECISION: self.target_precision,
                METRIC_RECALL: self.target_recall,
                METRIC_F1: self.target_f1,
            }[metric_name]
        except KeyError:
            raise SchemaValidationError(
                f"human search cannot compute quality metric {metric_name!r}; "
                f"supported metrics are "
                f"{[METRIC_PRECISION, METRIC_RECALL, METRIC_F1]}"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        return {
            METRIC_PRECISION: self.target_precision,
            METRIC_RECALL: self.target_recall,
            METRIC_F1: self.target_f1,
            "matched_targets": self.matched_targets,
            "total_targets": self.total_targets,
            "predicted_targets": self.predicted_targets,
            "false_positive_targets": self.false_positive_targets,
        }


@dataclass(frozen=True, slots=True)
class EmpiricalQualityScores:
    """Unit-consistent empirical human-search metrics, two families never mixed.

    - **Mission, track-level**: ``target_recall`` = unique ground-truth tracks found / total
      valid tracks. A person found on forty frames is one find; both sides count *tracks*.
      This is the canonical empirical mission-quality metric.
    - **Frame, detection-level**: ``detection_precision`` = matched predictions / non-ignored
      predictions, alongside the raw false-positive burden. Both sides count *per-frame
      detections*.

    Track-level precision and F1 are deliberately **absent**. Computing them would require a
    prediction associated with a persistent predicted *track* across frames, and V1 empirical
    replay carries independent per-frame masks with no such identity -- a per-frame
    ``prediction_id`` is not a track. Reporting them would divide a track count by a detection
    count, which is exactly the mixed-unit bug this type exists to remove. They are surfaced
    as ``None`` with a stated reason, never as a number.
    """

    unique_targets_found: int
    total_unique_targets: int
    target_recall: float
    matched_detections: int
    total_predictions: int
    false_positive_detections: int
    detection_precision: float
    #: False positives per minute of *examined footage* (processed frames at the nominal 1 fps
    #: stream), NOT per minute of mission wall-clock. Named for its denominator; the wall-clock
    #: rate ``false_positives_per_mission_minute`` is added by the episode metrics, which know
    #: the mission's completion time.
    false_positives_per_processed_minute: float

    def value_of(self, metric_name: str) -> float:
        """Return the value a contract may be scored on -- recall, or detection precision.

        Track-level precision and F1 raise: they are unavailable in empirical mode, and a
        contract asking for one must fail loudly rather than be handed a mixed-unit number.
        """
        if metric_name == METRIC_RECALL:
            return self.target_recall
        if metric_name == METRIC_DETECTION_PRECISION:
            return self.detection_precision
        raise SchemaValidationError(
            f"empirical human search does not support quality metric {metric_name!r}. "
            f"Its canonical mission metric is {METRIC_RECALL!r}, and "
            f"{METRIC_DETECTION_PRECISION!r} is available as a detection-level diagnostic. "
            "Track-level precision and F1 are unavailable: independent per-frame masks carry "
            "no persistent predicted-track identity to compute them from."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_evaluation": QUALITY_EVAL_EMPIRICAL_MASK,
            # mission, track-level
            "unique_targets_found": self.unique_targets_found,
            "total_unique_targets": self.total_unique_targets,
            METRIC_RECALL: self.target_recall,
            # frame, detection-level
            "matched_detections": self.matched_detections,
            "total_predictions": self.total_predictions,
            "false_positive_detections": self.false_positive_detections,
            METRIC_DETECTION_PRECISION: self.detection_precision,
            "false_positives_per_processed_minute": self.false_positives_per_processed_minute,
            # explicitly unavailable -- never a mixed-unit number under a track-level name
            "track_level_precision": None,
            "target_f1": None,
            "track_level_metrics_available": False,
            "track_level_unavailable_reason": (
                "independent per-frame segmentation predictions carry no persistent "
                "predicted-track identity; track-level precision and F1 require it"
            ),
        }


class HumanSearchSegmentationEvaluator:
    """Turns evidence plus ground truth into a standardised task result.

    Two scoring paths, chosen by the ground-truth form, never by the executor. Interval
    ground truth carries a precomputed match on every instance, so scoring reads it directly
    -- unchanged from before, and byte-identical for the profile suite, producing the
    track-level :class:`QualityScores`. Per-frame *mask* ground truth carries no match:
    scoring computes IoU from the masks, matches one-to-one per frame, deduplicates the finds
    by hidden track ID, and produces the unit-consistent :class:`EmpiricalQualityScores` --
    track-level recall and detection-level precision kept apart, with no mixed-unit F1.
    """

    __slots__ = ("_task_spec",)

    def __init__(self, task_spec: TaskSpec) -> None:
        self._task_spec = task_spec

    def evaluate(
        self,
        evidence: HumanSearchEvidenceRecord,
        ground_truth: HumanSearchGroundTruth | HumanSearchMaskGroundTruth,
        contract: Contract,
    ) -> TaskEvaluationResult:
        if isinstance(ground_truth, HumanSearchMaskGroundTruth):
            empirical = self.score_empirical(evidence, ground_truth)
            return TaskEvaluationResult.against_contract(
                contract,
                value=empirical.value_of(contract.quality_metric),
                details={**empirical.to_dict(), "processed_frames": evidence.processed_frames},
            )
        scores = self.score(evidence, ground_truth)
        return TaskEvaluationResult.against_contract(
            contract,
            value=scores.value_of(contract.quality_metric),
            details={**scores.to_dict(), "processed_frames": evidence.processed_frames},
        )

    def score(
        self,
        evidence: HumanSearchEvidenceRecord,
        ground_truth: HumanSearchGroundTruth,
    ) -> QualityScores:
        """Score interval (precomputed-scalar) ground truth: profile and legacy replay.

        Mask ground truth is scored by :meth:`score_empirical`, which returns unit-consistent
        empirical metrics; passing it here is a mistake worth failing on rather than scoring
        every instance as an unmatched false positive.
        """
        if isinstance(ground_truth, HumanSearchMaskGroundTruth):
            raise SchemaValidationError(
                "mask ground truth is scored by score_empirical(), not score(); the two "
                "produce different, unit-consistent metric sets"
            )
        return self._score_precomputed(evidence, ground_truth)

    def _score_precomputed(
        self, evidence: HumanSearchEvidenceRecord, ground_truth: HumanSearchGroundTruth
    ) -> QualityScores:
        rule = self._task_spec.matching_rule

        matched_targets: set[str] = set()
        predicted_identities: set[str] = set()
        matched_identities: set[str] = set()

        for instance in evidence.instances:
            predicted_identities.add(instance.predicted_target_id)
            ground_truth_track_id = getattr(instance, "ground_truth_track_id", None)
            if ground_truth_track_id is None:
                continue
            if not rule.matches(getattr(instance, "mask_iou", 0.0)):
                # Seen, but not segmented well enough to count as found.
                continue
            matched_targets.add(ground_truth_track_id)
            matched_identities.add(instance.predicted_target_id)

        total_targets = len(ground_truth)
        false_positives = len(predicted_identities - matched_identities)

        recall = _ratio(len(matched_targets), total_targets, empty=1.0)
        precision = _ratio(len(matched_identities), len(predicted_identities), empty=1.0)
        return QualityScores(
            target_precision=precision,
            target_recall=recall,
            target_f1=_harmonic_mean(precision, recall),
            matched_targets=len(matched_targets),
            total_targets=total_targets,
            predicted_targets=len(predicted_identities),
            false_positive_targets=false_positives,
        )

    def match(
        self,
        evidence: HumanSearchEvidenceRecord,
        ground_truth: HumanSearchMaskGroundTruth,
    ) -> tuple[FrameMatching, ...]:
        """Match every frame that has predictions or scored ground truth, one-to-one.

        Exposed so the matching that produces the metrics is inspectable -- a test, or a
        debug dump, can read the exact IoU that earned each match rather than only the totals.
        Ignored ground-truth regions take no part in matching here; their effect on false
        positives is applied during scoring.
        """
        category = self._task_spec.target_type
        rule = self._task_spec.matching_rule
        predictions_by_frame = self._predictions_by_frame(evidence)

        matchings: list[FrameMatching] = []
        for frame_id in self._frame_ids(predictions_by_frame, ground_truth):
            targets = [
                target
                for target in ground_truth.instances_at(frame_id)
                if not target.ignore and target.category == category
            ]
            matchings.append(
                match_frame(frame_id, predictions_by_frame.get(frame_id, []), targets, rule=rule)
            )
        return tuple(matchings)

    def score_empirical(
        self, evidence: HumanSearchEvidenceRecord, ground_truth: HumanSearchMaskGroundTruth
    ) -> EmpiricalQualityScores:
        """Score mask ground truth into unit-consistent empirical metrics.

        Two tallies kept strictly apart. **Track-level**: a ground-truth track matched on any
        frame is one find, deduplicated by hidden track ID. **Detection-level**: each
        prediction is one detection -- matched, false positive, or (on an ignore region)
        neither. Recall divides tracks by tracks; detection precision divides detections by
        detections. Nothing divides one by the other.
        """
        category = self._task_spec.target_type or TARGET_CATEGORY
        rule = self._task_spec.matching_rule
        predictions_by_frame = self._predictions_by_frame(evidence)

        found_tracks: set[str] = set()
        matched_detections = 0
        false_positive_detections = 0
        for frame_id in self._frame_ids(predictions_by_frame, ground_truth):
            predictions = predictions_by_frame.get(frame_id, [])
            instances = ground_truth.instances_at(frame_id)
            scored = [g for g in instances if not g.ignore and g.category == category]
            ignored = [g for g in instances if g.ignore and g.category == category]

            matching = match_frame(frame_id, predictions, scored, rule=rule)
            found_tracks |= matching.matched_track_ids
            matched_detections += len(matching.matches)

            matched_ids = {match.prediction_id for match in matching.matches}
            for prediction in predictions:
                if prediction.prediction_id in matched_ids:
                    continue
                # A prediction on an ignore region is neither a find nor a fault: the region is
                # one the mission is not scored on, so it absorbs the prediction rather than
                # charging it as a false positive, and it is left out of the detection total.
                if any(rule.matches(mask_iou(prediction.mask, region.mask)) for region in ignored):
                    continue
                false_positive_detections += 1

        # Detection total excludes ignore-absorbed predictions, so precision divides matched
        # detections by scored detections -- both frame-level.
        total_predictions = matched_detections + false_positive_detections
        unique_found = len(found_tracks)
        total_tracks = len(ground_truth.target_track_ids(category))

        return EmpiricalQualityScores(
            unique_targets_found=unique_found,
            total_unique_targets=total_tracks,
            target_recall=_ratio(unique_found, total_tracks, empty=1.0),
            matched_detections=matched_detections,
            total_predictions=total_predictions,
            false_positive_detections=false_positive_detections,
            detection_precision=_ratio(matched_detections, total_predictions, empty=1.0),
            false_positives_per_processed_minute=_per_processed_minute(
                false_positive_detections, evidence.processed_frames
            ),
        )

    def _predictions_by_frame(
        self, evidence: HumanSearchEvidenceRecord
    ) -> dict[int, list[MaskPredictedInstance]]:
        """Group the target-category mask predictions by frame, rejecting the wrong kind.

        A mask ground-truth stream must be paired with an empirical replay set: a precomputed
        instance here carries no mask to compare, and silently scoring it as "found nothing"
        would hide the misconfiguration.
        """
        category = self._task_spec.target_type
        by_frame: dict[int, list[MaskPredictedInstance]] = {}
        for instance in evidence.instances:
            if not isinstance(instance, MaskPredictedInstance):
                raise SchemaValidationError(
                    "empirical mask scoring needs mask predictions, but the evidence holds a "
                    f"{type(instance).__name__}; pair a mask ground-truth stream with an "
                    "empirical replay set, not synthetic or legacy-scalar predictions"
                )
            if instance.category != category:
                continue
            by_frame.setdefault(instance.frame_id, []).append(instance)
        return by_frame

    @staticmethod
    def _frame_ids(
        predictions_by_frame: dict[int, list[MaskPredictedInstance]],
        ground_truth: HumanSearchMaskGroundTruth,
    ) -> list[int]:
        return sorted(set(predictions_by_frame) | {frame.frame_id for frame in ground_truth.frames})


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    """Divide, with an explicit answer for the empty case.

    An episode with no targets cannot miss any, and one that predicted nothing has produced
    no false positives -- both are vacuously perfect, so ``empty`` is 1.0 at both call sites.
    Making it a parameter keeps that choice visible instead of buried in a guard.
    """
    return empty if denominator == 0 else numerator / denominator


def _harmonic_mean(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _per_processed_minute(count: int, processed_frames: int) -> float:
    """A per-minute rate over *examined footage* at the nominal one-frame-per-second stream.

    Normalised by frames the detector actually ran on, not wall-clock time: this is the
    detection-error density of what was looked at. The mission wall-clock rate is a separate,
    explicitly named metric added by the episode metrics. No frames examined yields ``0.0``.
    """
    if processed_frames <= 0:
        return 0.0
    return count * FRAMES_PER_MINUTE / processed_frames
