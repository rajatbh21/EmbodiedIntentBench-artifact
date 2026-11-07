"""V1 task: ``HUMAN_SEARCH_SEGMENTATION``.

Find unique people along a predefined UAV path and report an instance mask set. A target
counts as found when a predicted instance matches a ground-truth instance under the task
specification's matching rule (V1: ``mask_iou >= 0.50``), deduplicated by ground-truth track
ID. Supported quality metrics: target precision, target recall, target F1.

Modules
-------
- ``ground_truth``     -- hidden targets and their visibility intervals.
- ``prediction``       -- the predicted-instance payload and the synthetic prediction source.
- ``evidence_tracker`` -- ``HumanSearchEvidenceTracker``: processed frames, accumulated
                          instances, the policy summary, and the evaluator's record.
- ``evaluator``        -- ``HumanSearchSegmentationEvaluator``: precision, recall, F1.
- ``task``             -- binds the two together for the task registry.

Boundaries
----------
The tracker's ``policy_summary()`` exposes no ground-truth-derived quantity: no true recall,
no hidden target count, no match outcome. Only ``final_record()``, consumed by the
evaluator, references matched ground-truth track IDs -- and that record never reaches a
policy.

The two "unique target" counts are deliberately different. The policy sees distinct
*predicted* identities, which a false positive inflates and which it cannot verify; the
evaluator deduplicates by *ground-truth* track ID. Collapsing them would hand the policy its
own true positive count.
"""

from missionbench.tasks.human_search_segmentation.evaluator import (
    EmpiricalQualityScores,
    HumanSearchSegmentationEvaluator,
    QualityScores,
)
from missionbench.tasks.human_search_segmentation.evidence_tracker import (
    HumanSearchEvidenceRecord,
    HumanSearchEvidenceTracker,
)
from missionbench.tasks.human_search_segmentation.ground_truth import (
    TARGET_CATEGORY,
    TASK_ID,
    FrameGroundTruth,
    HumanSearchGroundTruth,
    HumanSearchMaskGroundTruth,
    MaskTarget,
    TargetTrack,
    load_any_ground_truth,
    load_human_search_ground_truth,
    load_human_search_mask_ground_truth,
)
from missionbench.tasks.human_search_segmentation.masks import (
    BinaryMask,
    decode_mask,
    mask_iou,
)
from missionbench.tasks.human_search_segmentation.matching import (
    FrameMatching,
    Match,
    match_frame,
)
from missionbench.tasks.human_search_segmentation.prediction import (
    DEFAULT_TIER_BEHAVIOUR,
    EvidenceInstance,
    FramePrediction,
    MaskPredictedInstance,
    PredictedInstance,
    SyntheticHumanSearchPredictions,
    TierBehaviour,
)
from missionbench.tasks.human_search_segmentation.task import HumanSearchSegmentationTask

__all__ = [
    "DEFAULT_TIER_BEHAVIOUR",
    "TARGET_CATEGORY",
    "TASK_ID",
    "BinaryMask",
    "EmpiricalQualityScores",
    "EvidenceInstance",
    "FrameGroundTruth",
    "FrameMatching",
    "FramePrediction",
    "HumanSearchEvidenceRecord",
    "HumanSearchEvidenceTracker",
    "HumanSearchGroundTruth",
    "HumanSearchMaskGroundTruth",
    "HumanSearchSegmentationEvaluator",
    "HumanSearchSegmentationTask",
    "MaskPredictedInstance",
    "MaskTarget",
    "Match",
    "PredictedInstance",
    "QualityScores",
    "SyntheticHumanSearchPredictions",
    "TargetTrack",
    "TierBehaviour",
    "decode_mask",
    "load_any_ground_truth",
    "load_human_search_ground_truth",
    "load_human_search_mask_ground_truth",
    "mask_iou",
    "match_frame",
]
