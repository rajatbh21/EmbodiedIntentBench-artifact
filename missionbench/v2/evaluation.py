"""Mission-level evaluation for V2, reusing the V1 empirical metric semantics.

Scoring follows the frozen V1 rules rather than inventing parallel ones:

- **Track-level recall** (canonical): ``target_recall = unique targets found / total
  targets in the scenario``, deduplicated by hidden object id -- finding one marker on
  forty observations is one find.
- **Detection-level precision**: ``detection_precision = matched predicted components /
  total predicted components`` -- both sides count per-observation detections.
- **False-positive burden**: raw count plus the two denominator-named rates,
  ``false_positives_per_processed_minute`` (per minute of *processed observations* at
  the observation interval) and ``false_positives_per_mission_minute`` (wall-clock
  mission time).
- **No track-level F1**: predicted components carry no persistent identity, exactly as
  in V1 empirical replay, so it is reported as ``null`` with the same reason.

Every prediction is scored against the ground truth captured **with its own
observation** -- never against a re-rendered crop at completion time. Matching is the
V1 discipline: an IoU matrix, pairs above the scenario threshold, greedy one-to-one
assignment (IoU descending, then component id, then instance id). A predicted
component that matches a *distractor* is still a false positive -- a distractor is a
wrong answer, not an ignore region.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from missionbench.v2.camera import Observation
from missionbench.v2.executors import label_components

__all__ = ["MissionEvaluator", "ObservationScore"]


@dataclass(frozen=True, slots=True)
class ObservationScore:
    """What one processed observation contributed to the mission tally."""

    observation_id: int
    capture_time_s: float
    predicted_components: int
    matched_components: int
    false_positive_components: int
    matched_target_ids: tuple[str, ...]
    visible_target_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "capture_time_s": self.capture_time_s,
            "predicted_components": self.predicted_components,
            "matched_components": self.matched_components,
            "false_positive_components": self.false_positive_components,
            "matched_target_ids": list(self.matched_target_ids),
            "visible_target_ids": list(self.visible_target_ids),
        }


@dataclass
class MissionEvaluator:
    """Accumulates per-observation matching into mission-level empirical metrics."""

    matching_iou_threshold: float
    total_targets: int
    observation_interval_s: float
    min_component_px: int = 4

    found_target_ids: set[str] = field(default_factory=set)
    encountered_target_ids: set[str] = field(default_factory=set)
    matched_detections: int = 0
    total_predictions: int = 0
    false_positive_detections: int = 0
    processed_observations: int = 0
    scores: list[ObservationScore] = field(default_factory=list)

    def update(self, observation: Observation, prediction_mask: np.ndarray) -> ObservationScore:
        """Score one prediction against its capture-time ground truth."""
        if prediction_mask.shape != observation.semantic_gt.shape:
            raise ValueError(
                f"prediction shape {prediction_mask.shape} does not match the observation's "
                f"ground truth {observation.semantic_gt.shape}"
            )
        self.processed_observations += 1
        self.encountered_target_ids.update(observation.visible_target_ids)

        pred_labels, pred_count = label_components(prediction_mask)
        pred_areas = np.bincount(pred_labels.ravel(), minlength=pred_count + 1)
        component_ids = [
            c for c in range(1, pred_count + 1) if pred_areas[c] >= self.min_component_px
        ]

        instance = observation.instance_gt
        target_instances = _target_instances(observation)

        # IoU of every kept component against every visible target instance.
        candidates: list[tuple[float, int, int]] = []
        for c in component_ids:
            component = pred_labels == c
            for inst_id in target_instances:
                gt = instance == inst_id
                inter = int(np.logical_and(component, gt).sum())
                if inter == 0:
                    continue
                union = int(np.logical_or(component, gt).sum())
                iou = inter / union
                if iou >= self.matching_iou_threshold:
                    candidates.append((iou, c, inst_id))
        # Greedy one-to-one, the V1 discipline: IoU descending, ids ascending on ties.
        candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
        used_components: set[int] = set()
        used_instances: set[int] = set()
        matched_targets: list[str] = []
        for _iou, c, inst_id in candidates:
            if c in used_components or inst_id in used_instances:
                continue
            used_components.add(c)
            used_instances.add(inst_id)
            matched_targets.append(target_instances[inst_id])

        matched = len(used_components)
        false_positives = len(component_ids) - matched

        self.matched_detections += matched
        self.total_predictions += len(component_ids)
        self.false_positive_detections += false_positives
        self.found_target_ids.update(matched_targets)

        score = ObservationScore(
            observation_id=observation.observation_id,
            capture_time_s=observation.scheduled_capture_time_s,
            predicted_components=len(component_ids),
            matched_components=matched,
            false_positive_components=false_positives,
            matched_target_ids=tuple(sorted(matched_targets)),
            visible_target_ids=observation.visible_target_ids,
        )
        self.scores.append(score)
        return score

    def note_skipped_visibility(self, visible_target_ids: tuple[str, ...]) -> None:
        """Record targets that were visible during a *skipped* scheduled observation.

        No prediction exists for a skipped capture -- the executor never ran -- so this
        touches only the encountered set. The effect on the diagnostics is exactly the
        honest one: such a target counts as encountered-but-missed
        (``targets_missed_while_visible``) instead of silently disappearing. Recall is
        unaffected either way, because its denominator is the scenario's total target
        count, never the encountered set.
        """
        self.encountered_target_ids.update(visible_target_ids)

    # -- mission totals -----------------------------------------------------------------

    def quality_details(self, mission_time_s: float) -> dict[str, Any]:
        """The mission-level quality block, in the V1 empirical field vocabulary."""
        unique_found = len(self.found_target_ids)
        recall = _ratio(unique_found, self.total_targets, empty=1.0)
        precision = _ratio(self.matched_detections, self.total_predictions, empty=1.0)
        processed_minutes = self.processed_observations * self.observation_interval_s / 60.0
        mission_minutes = mission_time_s / 60.0
        encountered = len(self.encountered_target_ids)
        return {
            "quality_evaluation": "empirical_mask_iou",
            "unique_targets_found": unique_found,
            "total_unique_targets": self.total_targets,
            "target_recall": recall,
            "matched_detections": self.matched_detections,
            "total_predictions": self.total_predictions,
            "detection_precision": precision,
            "false_positive_detections": self.false_positive_detections,
            "false_positives_per_processed_minute": (
                self.false_positive_detections / processed_minutes if processed_minutes > 0 else 0.0
            ),
            "false_positives_per_mission_minute": (
                self.false_positive_detections / mission_minutes if mission_minutes > 0 else 0.0
            ),
            "target_f1": None,
            "track_level_metrics_available": False,
            "track_level_unavailable_reason": (
                "predicted components carry no persistent predicted-track identity; "
                "track-level precision and F1 require it (same rule as V1 empirical replay)"
            ),
            # -- V2 diagnostics, additional to the frozen V1 vocabulary ------------------
            "targets_encountered": encountered,
            "targets_detected": unique_found,
            "targets_missed_while_visible": encountered
            - len(self.found_target_ids & self.encountered_target_ids),
        }

    def value_of(self, metric_name: str) -> float:
        details = self.quality_details(mission_time_s=1.0)  # rates unused here
        if metric_name in ("target_recall", "detection_precision"):
            return float(details[metric_name])
        raise ValueError(
            f"V2 empirical evaluation does not support quality metric {metric_name!r}; "
            "use 'target_recall' (canonical) or 'detection_precision'"
        )


def _target_instances(observation: Observation) -> dict[int, str]:
    """Visible target instance-ids -> object ids, from the observation's own GT."""
    out: dict[int, str] = {}
    semantic = observation.semantic_gt
    instance = observation.instance_gt
    target_instance_ids = np.unique(instance[(semantic == 1)])
    ordered = sorted(observation.visible_target_ids)
    # instance ids are 1-based scenario ordinals; map via the ids visible in this frame.
    for inst_id in target_instance_ids:
        if inst_id == 0:
            continue
        # The renderer guarantees one instance id per object; recover its object id by
        # position in the scenario ordering carried on the observation.
        out[int(inst_id)] = _object_id_for_instance(int(inst_id), observation, ordered)
    return out


def _object_id_for_instance(
    inst_id: int, observation: Observation, visible_sorted: list[str]
) -> str:
    mapping = observation.provenance.get("instance_object_ids")
    if isinstance(mapping, dict) and str(inst_id) in mapping:
        return mapping[str(inst_id)]
    # Fallback: unique visible target -- unambiguous without the mapping.
    if len(visible_sorted) == 1:
        return visible_sorted[0]
    raise ValueError(
        f"cannot resolve instance id {inst_id} to an object id; the renderer must supply "
        "provenance['instance_object_ids']"
    )


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return empty if denominator == 0 else numerator / denominator
