"""Predicted instances, and the synthetic source that generates them.

Prediction payload
------------------
:class:`PredictedInstance` carries two kinds of field, and the split is the whole point:

- **Policy-visible in aggregate**: ``prediction_id``, ``frame_id``, ``predicted_target_id``,
  ``confidence``. These are what a real system knows about its own output.
- **Hidden**: ``ground_truth_track_id`` and ``mask_iou``. Only the evaluator reads them.

The hidden pair stands in for real mask comparison. A production system would store an
actual mask and the evaluator would compute IoU against ground truth; here the similarity
is precomputed. The evaluator's interface is written against ``mask_iou`` and the task
spec's matching rule, so replacing synthesis with real masks changes what fills the field,
not what reads it.

Synthetic generation
--------------------
Deterministic given ``(seed, frame_id, config_id, track_id)``. Hashing is done with
``hashlib``, not the builtin ``hash``, whose string seed is randomised per process -- using
it would make an episode irreproducible across runs, which the benchmark forbids.

Quality flows from the configuration's ``quality_tier``, so the executor stays unaware of
any of this: a better configuration detects a visible target more often, scores a higher
mask IoU, and produces fewer spurious instances.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

from missionbench.executor.base import ExecutionRequest
from missionbench.schemas.loading import SchemaValidationError
from missionbench.schemas.profile import ProfileCatalog, QualityTier
from missionbench.tasks.human_search_segmentation.ground_truth import (
    TARGET_CATEGORY,
    HumanSearchGroundTruth,
)
from missionbench.tasks.human_search_segmentation.masks import BinaryMask, decode_mask

__all__ = [
    "DEFAULT_TIER_BEHAVIOUR",
    "EvidenceInstance",
    "FramePrediction",
    "MaskPredictedInstance",
    "PredictedInstance",
    "SyntheticHumanSearchPredictions",
    "TierBehaviour",
]


class EvidenceInstance(Protocol):
    """The fields every predicted instance exposes, whatever kind of prediction it is.

    Both the precomputed :class:`PredictedInstance` and the empirical
    :class:`MaskPredictedInstance` satisfy this. The evidence tracker's *policy* summary is
    built from these fields alone -- none of them is ground-truth-derived -- so it works
    identically for either kind. The evaluator, which does see ground truth, narrows to the
    concrete type it needs.
    """

    prediction_id: str
    frame_id: int
    predicted_target_id: str
    confidence: float

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class PredictedInstance:
    """One predicted person instance."""

    prediction_id: str
    frame_id: int
    #: The system's own identity for this target, as a tracker would assign. Distinct from
    #: the ground-truth track: this is what the system *believes*, and it is the key the
    #: policy-visible unique-target count is computed from.
    predicted_target_id: str
    confidence: float
    #: Hidden. The ground-truth target this instance actually corresponds to, or ``None``
    #: for a false positive. Read only by the evaluator.
    ground_truth_track_id: str | None = None
    #: Hidden. Mask overlap against that ground-truth instance, compared against the task
    #: spec's matching rule. Zero for a false positive.
    mask_iou: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Full serialisation, for the episode record only -- this includes hidden fields."""
        return {
            "prediction_id": self.prediction_id,
            "frame_id": self.frame_id,
            "predicted_target_id": self.predicted_target_id,
            "confidence": self.confidence,
            "ground_truth_track_id": self.ground_truth_track_id,
            "mask_iou": self.mask_iou,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PredictedInstance:
        """Rebuild an instance from a serialised payload, e.g. a replay record.

        The hidden fields default when absent, which is deliberate: a replay set recorded
        for release carries only the public fields, so a replayed instance has no
        ``ground_truth_track_id`` and a zero ``mask_iou`` -- and therefore scores as a false
        positive. Replay is exact for the plumbing (latency, energy, communication,
        provenance); reproducing a quality score needs a set that kept the hidden fields.
        """
        return cls(
            prediction_id=str(payload["prediction_id"]),
            frame_id=int(payload["frame_id"]),
            predicted_target_id=str(payload["predicted_target_id"]),
            confidence=float(payload["confidence"]),
            ground_truth_track_id=payload.get("ground_truth_track_id"),
            mask_iou=float(payload.get("mask_iou", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class MaskPredictedInstance:
    """One predicted person instance carrying its actual segmentation mask.

    This is what an *empirical* replay serves: a real prediction whose overlap with ground
    truth the evaluator computes, rather than a precomputed similarity. It carries no
    ``ground_truth_track_id`` and no IoU -- a prediction may not name the target it "really"
    is, and its quality is measured, not declared. ``predicted_target_id`` is the system's own
    tracker identity (policy-visible, not ground truth); when a record omits it, the
    per-detection ``prediction_id`` stands in.
    """

    prediction_id: str
    frame_id: int
    predicted_target_id: str
    confidence: float
    category: str
    mask: BinaryMask

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.prediction_id,
            "frame_id": self.frame_id,
            "predicted_target_id": self.predicted_target_id,
            "confidence": self.confidence,
            "category": self.category,
            "mask": self.mask.to_dict(),
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, default_frame_id: int = -1
    ) -> MaskPredictedInstance:
        """Rebuild an empirical instance from a replayed payload.

        A ``ground_truth_track_id`` here is rejected outright: ground-truth identity must not
        travel inside a prediction, or replaying it would smuggle the answer key past the
        boundary the whole benchmark rests on. Any ``mask_iou`` present is ignored rather than
        trusted -- IoU in empirical mode is computed from the mask, never read off the wire.

        ``default_frame_id`` supplies the frame from the enclosing ``FramePrediction`` when an
        instance does not repeat it, so per-frame matching always knows which frame it is on.
        """
        if "ground_truth_track_id" in payload:
            raise SchemaValidationError(
                "an empirical prediction must not carry 'ground_truth_track_id'; a prediction "
                "may not claim which ground-truth target it corresponds to"
            )
        prediction_id = str(payload["prediction_id"])
        return cls(
            prediction_id=prediction_id,
            frame_id=int(payload.get("frame_id", default_frame_id)),
            predicted_target_id=str(payload.get("predicted_target_id") or prediction_id),
            confidence=float(payload["confidence"]),
            category=str(payload.get("category") or TARGET_CATEGORY),
            mask=decode_mask(payload.get("mask"), context=f"prediction {prediction_id!r} mask"),
        )


@dataclass(frozen=True, slots=True)
class FramePrediction:
    """Everything predicted for one frame. This is the ``ExecutionResult.prediction`` payload.

    Its instances are precomputed :class:`PredictedInstance`\\ s from synthetic or legacy
    replay, or empirical :class:`MaskPredictedInstance`\\ s carrying real masks -- a single
    frame does not mix the two, but a suite may run both kinds across different streams.
    """

    frame_id: int
    instances: tuple[EvidenceInstance, ...] = ()

    def __len__(self) -> int:
        return len(self.instances)

    @classmethod
    def coerce(cls, payload: object) -> FramePrediction:
        """Return ``payload`` as a ``FramePrediction``, rebuilding it from a dict if needed.

        A live profile run hands the tracker a ``FramePrediction`` object directly. A replay
        run hands it the same payload after a JSON round-trip, i.e. a plain dict. The task
        owns its payload shape, so knowing how to read both forms belongs here rather than
        in the executor or the runner. An instance dict carrying a ``mask`` is an empirical
        prediction; one without is the precomputed form.
        """
        if isinstance(payload, FramePrediction):
            return payload
        if isinstance(payload, Mapping) and "instances" in payload:
            frame_id = int(payload.get("frame_id", -1))
            return cls(
                frame_id=frame_id,
                instances=tuple(_coerce_instance(item, frame_id) for item in payload["instances"]),
            )
        # A mapping without an 'instances' list, or a non-mapping, is a payload from some
        # other task. Refuse it rather than silently reading zero instances, which would let
        # a mismatched executor score as "saw nothing" instead of failing.
        raise TypeError(
            f"cannot read a human-search prediction from {type(payload).__name__}; "
            "expected a FramePrediction or a mapping with an 'instances' list"
        )


def _coerce_instance(item: Any, frame_id: int) -> EvidenceInstance:
    if isinstance(item, Mapping) and "mask" in item:
        return MaskPredictedInstance.from_dict(item, default_frame_id=frame_id)
    return PredictedInstance.from_dict(item)


@dataclass(frozen=True, slots=True)
class TierBehaviour:
    """How a quality tier behaves: how often it sees a target, how well, and how noisily."""

    #: Probability of producing an instance for a target that is genuinely in view.
    detection_probability: float
    #: Mask IoU is drawn uniformly from this range. A range straddling the matching
    #: threshold is what makes a weak configuration produce detections that fail to match --
    #: seeing something is not the same as segmenting it well enough to count.
    mask_iou_range: tuple[float, float]
    confidence_range: tuple[float, float]
    #: Probability per processed frame of inventing an instance that matches nothing.
    false_positive_rate: float


#: Synthetic behaviour per tier. Placeholder values, chosen so that the tiers separate on
#: short-visibility targets rather than on every frame. See docs/v1_spec.md §4.12,
#: 'Ground truth and synthetic predictions'.
DEFAULT_TIER_BEHAVIOUR: Final[Mapping[QualityTier, TierBehaviour]] = {
    QualityTier.LOW: TierBehaviour(
        detection_probability=0.55,
        mask_iou_range=(0.30, 0.75),
        confidence_range=(0.45, 0.80),
        false_positive_rate=0.004,
    ),
    QualityTier.MEDIUM: TierBehaviour(
        detection_probability=0.75,
        mask_iou_range=(0.42, 0.88),
        confidence_range=(0.60, 0.90),
        false_positive_rate=0.002,
    ),
    QualityTier.HIGH: TierBehaviour(
        detection_probability=0.92,
        mask_iou_range=(0.48, 0.96),
        confidence_range=(0.70, 0.97),
        false_positive_rate=0.001,
    ),
}


def _unit(*parts: object) -> float:
    """Map arbitrary parts to a stable value in [0, 1).

    ``hashlib`` rather than the builtin ``hash``: Python randomises string hashing per
    process, so the builtin would make the same episode score differently on each run.
    """
    digest = hashlib.blake2b("|".join(str(part) for part in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big") / 2**64


def _in_range(value: float, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return low + value * (high - low)


class SyntheticHumanSearchPredictions:
    """Generates predictions from hidden ground truth and the configuration's quality tier.

    Plugs into ``ProfileExecutor`` through the ``PredictionSource`` protocol, so the executor
    needs no knowledge of this task. It reads ground truth, which is legitimate -- it is
    standing in for a model that would actually be looking at the scene -- but nothing it
    produces lets a *policy* recover a ground-truth quantity: the hidden fields it writes are
    excluded from the tracker's policy summary.
    """

    __slots__ = ("_behaviour", "_ground_truth", "_profiles")

    def __init__(
        self,
        ground_truth: HumanSearchGroundTruth,
        profiles: ProfileCatalog,
        *,
        behaviour: Mapping[QualityTier, TierBehaviour] | None = None,
    ) -> None:
        self._ground_truth = ground_truth
        self._profiles = profiles
        self._behaviour = dict(behaviour or DEFAULT_TIER_BEHAVIOUR)

    def prediction_for(self, request: ExecutionRequest) -> FramePrediction:
        config_id = request.configuration.config_id
        tier = self._profiles.get(config_id).quality_tier
        behaviour = self._behaviour[tier]

        instances: list[PredictedInstance] = []
        for target in self._ground_truth.visible_at(request.frame_id):
            roll = _unit(request.seed, "detect", request.frame_id, config_id, target.track_id)
            if roll >= behaviour.detection_probability:
                continue
            instances.append(
                PredictedInstance(
                    prediction_id=f"P{request.frame_id}_{target.track_id}",
                    frame_id=request.frame_id,
                    # A tracker re-identifies the same person across frames, so the predicted
                    # identity is stable per target. It is an opaque label, not the ground
                    # truth track ID, and reveals no ground-truth quantity by itself.
                    predicted_target_id=self._predicted_target_id(request.seed, target.track_id),
                    confidence=_in_range(
                        _unit(request.seed, "conf", request.frame_id, config_id, target.track_id),
                        behaviour.confidence_range,
                    ),
                    ground_truth_track_id=target.track_id,
                    mask_iou=_in_range(
                        _unit(request.seed, "iou", request.frame_id, config_id, target.track_id),
                        behaviour.mask_iou_range,
                    ),
                )
            )

        instances.extend(self._false_positives(request, behaviour))
        return FramePrediction(frame_id=request.frame_id, instances=tuple(instances))

    def _false_positives(
        self, request: ExecutionRequest, behaviour: TierBehaviour
    ) -> list[PredictedInstance]:
        config_id = request.configuration.config_id
        if _unit(request.seed, "fp", request.frame_id, config_id) >= behaviour.false_positive_rate:
            return []
        return [
            PredictedInstance(
                prediction_id=f"P{request.frame_id}_FP",
                frame_id=request.frame_id,
                predicted_target_id=f"PT_FP_{request.frame_id}",
                confidence=_in_range(
                    _unit(request.seed, "fpconf", request.frame_id, config_id),
                    behaviour.confidence_range,
                ),
                ground_truth_track_id=None,
                mask_iou=0.0,
            )
        ]

    @staticmethod
    def _predicted_target_id(seed: int, track_id: str) -> str:
        return "PT_" + hashlib.blake2b(f"{seed}|{track_id}".encode(), digest_size=4).hexdigest()
