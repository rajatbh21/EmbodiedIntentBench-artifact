"""Deterministic one-to-one matching of predicted masks to ground-truth masks.

For one frame: compute the IoU of every prediction against every ground-truth instance, keep
the pairs that clear the task's matching rule, then assign greedily -- the highest-IoU
admissible pair first, then the next, with each prediction and each ground-truth instance
used at most once. What is left over is unmatched: a prediction that matched nothing is a
false positive, a ground-truth instance that nothing matched is a miss.

Greedy, not optimal
-------------------
This is greedy assignment, not the globally optimal (Hungarian) assignment. V1 has no array
or ``scipy`` dependency, and for the handful of instances in one UAV frame greedy and optimal
disagree only under deliberately arranged overlaps -- a strong prediction next to a weak one
both eyeing two targets. The case is contrived for aerial human search; the limitation is
recorded here and its tie-breaking is pinned by a test.

Determinism
-----------
Candidates are ordered by a total key -- IoU descending, then ``prediction_id`` ascending,
then ``track_id`` ascending -- so equal-IoU ties resolve the same way on every machine and
the result never depends on set or dict iteration order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from missionbench.schemas.task_spec import MatchingRule
from missionbench.tasks.human_search_segmentation.masks import BinaryMask, mask_iou

__all__ = ["FrameMatching", "Match", "match_frame"]


class _MaskPrediction(Protocol):
    prediction_id: str
    predicted_target_id: str
    frame_id: int
    mask: BinaryMask


class _MaskTarget(Protocol):
    track_id: str
    mask: BinaryMask


@dataclass(frozen=True, slots=True)
class Match:
    """One prediction paired with the ground-truth track it was assigned to."""

    frame_id: int
    prediction_id: str
    predicted_target_id: str
    track_id: str
    #: The computed IoU that earned the match, kept so evaluation is traceable to the pixels
    #: that produced it rather than to a scalar taken on trust.
    iou: float


@dataclass(frozen=True, slots=True)
class FrameMatching:
    """The outcome of matching one frame: what paired, and what was left over."""

    frame_id: int
    matches: tuple[Match, ...]
    unmatched_prediction_ids: tuple[str, ...]

    @property
    def matched_track_ids(self) -> frozenset[str]:
        return frozenset(match.track_id for match in self.matches)


def match_frame(
    frame_id: int,
    predictions: Sequence[_MaskPrediction],
    targets: Sequence[_MaskTarget],
    *,
    rule: MatchingRule,
) -> FrameMatching:
    """Assign predictions to ground-truth targets for one frame, one-to-one.

    ``rule`` is the task specification's matching rule; the IoU threshold lives there, not as
    a constant in this module, so adjusting what "found" means is a one-file change. A
    prediction and a target may pair only if their IoU satisfies the rule, and each may pair
    at most once.
    """
    candidates: list[tuple[float, str, str, _MaskPrediction, _MaskTarget]] = []
    for prediction in predictions:
        for target in targets:
            iou = mask_iou(prediction.mask, target.mask)
            if rule.matches(iou):
                candidates.append(
                    (iou, prediction.prediction_id, target.track_id, prediction, target)
                )

    # IoU descending, then the two identifiers ascending: a stable total order, so ties do
    # not resolve by iteration order.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_predictions: set[str] = set()
    used_targets: set[str] = set()
    matches: list[Match] = []
    for iou, prediction_id, track_id, prediction, _target in candidates:
        if prediction_id in used_predictions or track_id in used_targets:
            continue
        used_predictions.add(prediction_id)
        used_targets.add(track_id)
        matches.append(
            Match(
                frame_id=frame_id,
                prediction_id=prediction_id,
                predicted_target_id=prediction.predicted_target_id,
                track_id=track_id,
                iou=iou,
            )
        )

    unmatched = tuple(
        prediction.prediction_id
        for prediction in predictions
        if prediction.prediction_id not in used_predictions
    )
    return FrameMatching(
        frame_id=frame_id, matches=tuple(matches), unmatched_prediction_ids=unmatched
    )
