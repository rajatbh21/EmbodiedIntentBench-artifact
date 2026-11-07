"""Hidden ground truth for human search: who was where, and when.

**Never reachable from a policy.** Only the evaluator and the synthetic prediction source
read this, and the prediction source only uses it to decide what a model would plausibly
have seen -- it never puts a ground-truth quantity into anything the tracker will summarise
for a policy.

Visibility is stored as a frame interval rather than a list, because "this person is in
view from frame 40 to frame 100" is both what the fixture means and far easier to read than
sixty frame numbers. The interval is the only thing that makes latency matter: a target
visible for three frames is missed outright by a configuration slow enough to skip them,
which is precisely the cost the benchmark is trying to measure.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from missionbench.schemas.loading import (
    SchemaValidationError,
    open_document,
    read_json_object,
)
from missionbench.tasks.human_search_segmentation.masks import BinaryMask, decode_mask

__all__ = [
    "TASK_ID",
    "FrameGroundTruth",
    "HumanSearchGroundTruth",
    "HumanSearchMaskGroundTruth",
    "MaskTarget",
    "TargetTrack",
    "load_any_ground_truth",
    "load_human_search_ground_truth",
    "load_human_search_mask_ground_truth",
]

#: The one task V1 implements. Declared here rather than repeated as a literal, and matched
#: against ``contract.task_id`` through the task registry.
TASK_ID: Final = "HUMAN_SEARCH_SEGMENTATION"

_DOCUMENT_FIELDS: Final = ("ground_truth_id", "frame_stream_id", "task_id", "targets")
_TARGET_FIELDS: Final = ("track_id", "first_frame_id", "last_frame_id", "note")

_MASK_DOCUMENT_FIELDS: Final = ("ground_truth_id", "frame_stream_id", "task_id", "frames")
_FRAME_FIELDS: Final = ("frame_id", "instances")
_MASK_INSTANCE_FIELDS: Final = ("track_id", "category", "mask", "ignore")

#: The category of the thing a human-search mission looks for. Ground-truth instances of any
#: other category, and predictions of any other category, are ignored by scoring.
TARGET_CATEGORY: Final = "person"


@dataclass(frozen=True, slots=True)
class TargetTrack:
    """One person, visible over an inclusive frame interval."""

    track_id: str
    first_frame_id: int
    last_frame_id: int
    note: str = ""

    def is_visible_at(self, frame_id: int) -> bool:
        return self.first_frame_id <= frame_id <= self.last_frame_id

    @property
    def visible_frame_count(self) -> int:
        return self.last_frame_id - self.first_frame_id + 1


@dataclass(frozen=True, slots=True)
class HumanSearchGroundTruth:
    """Every target that appears in one frame stream.

    Keyed on the stream rather than the episode because ground truth describes the *scene*.
    Several episodes may fly the same stream under different battery, network, or contract
    conditions, and they must all be scored against the same answers.
    """

    frame_stream_id: str
    targets: tuple[TargetTrack, ...]
    ground_truth_id: str = ""

    @property
    def task_id(self) -> str:
        return TASK_ID

    @property
    def track_ids(self) -> frozenset[str]:
        return frozenset(target.track_id for target in self.targets)

    def __len__(self) -> int:
        return len(self.targets)

    def visible_at(self, frame_id: int) -> Iterator[TargetTrack]:
        """Yield the targets in view at ``frame_id``, in declaration order."""
        return (target for target in self.targets if target.is_visible_at(frame_id))


def load_human_search_ground_truth(path: Path) -> HumanSearchGroundTruth:
    """Load and validate a ground-truth file."""
    reader = open_document(
        path, document_type="HumanSearchGroundTruth", allowed_fields=_DOCUMENT_FIELDS
    )

    declared_task_id = reader.get_str("task_id")
    if declared_task_id != TASK_ID:
        raise SchemaValidationError(
            f"{reader.context}: this loader reads {TASK_ID!r} ground truth, "
            f"but the file declares {declared_task_id!r}"
        )

    targets: list[TargetTrack] = []
    seen: set[str] = set()
    for entry in reader.get_object_list("targets", allowed_fields=_TARGET_FIELDS):
        first = entry.get_int("first_frame_id", minimum=0)
        last = entry.get_int("last_frame_id", minimum=first)
        track_id = entry.get_str("track_id")
        if track_id in seen:
            raise SchemaValidationError(
                f"{reader.context}: duplicate target track_id {track_id!r}; "
                "deduplication is keyed on it, so it must be unique"
            )
        seen.add(track_id)
        targets.append(
            TargetTrack(
                track_id=track_id,
                first_frame_id=first,
                last_frame_id=last,
                note=entry.get_optional_str("note") or "",
            )
        )

    return HumanSearchGroundTruth(
        frame_stream_id=reader.get_str("frame_stream_id"),
        targets=tuple(targets),
        ground_truth_id=reader.get_optional_str("ground_truth_id") or "",
    )


# --- empirical, mask-based ground truth ------------------------------------------------------
#
# The interval form above stands in for real mask comparison: it says *when* a target is in
# view and lets synthesis precompute an IoU. The mask form below carries the actual answer --
# a segmentation mask per target per frame -- so the evaluator can compute IoU against a
# prediction rather than trust a scalar. Both are hidden from the policy; both key on the
# frame stream, since several episodes may fly the same scene.


@dataclass(frozen=True, slots=True)
class MaskTarget:
    """One ground-truth person instance in one frame: who, and exactly which pixels.

    ``ignore`` marks a region that must neither be counted as a target to find nor penalise a
    prediction that overlaps it -- a crowd, an out-of-scope object, an annotation the dataset
    flagged unreliable. It is excluded from the target total and from matching entirely.
    """

    track_id: str
    category: str
    mask: BinaryMask
    ignore: bool = False


@dataclass(frozen=True, slots=True)
class FrameGroundTruth:
    """Every ground-truth instance visible in one frame."""

    frame_id: int
    instances: tuple[MaskTarget, ...]


@dataclass(frozen=True, slots=True)
class HumanSearchMaskGroundTruth:
    """Per-frame mask ground truth for one frame stream.

    The same physical person carries one ``track_id`` across every frame they appear in, so
    "found in forty frames" collapses to one unique target. Keyed on the stream rather than
    the episode for the same reason the interval form is: the scene is what has answers.
    """

    frame_stream_id: str
    frames: tuple[FrameGroundTruth, ...]
    ground_truth_id: str = ""

    @property
    def task_id(self) -> str:
        return TASK_ID

    def instances_at(self, frame_id: int) -> tuple[MaskTarget, ...]:
        """Return the ground-truth instances in ``frame_id``, or an empty tuple if none."""
        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame.instances
        return ()

    def target_track_ids(self, category: str = TARGET_CATEGORY) -> frozenset[str]:
        """Distinct non-ignored track IDs of ``category`` across the whole stream.

        This is the mission's target total: the set of unique people a perfect run would find.
        Ignored instances and other categories never enter it.
        """
        return frozenset(
            instance.track_id
            for frame in self.frames
            for instance in frame.instances
            if not instance.ignore and instance.category == category
        )

    def __len__(self) -> int:
        return len(self.target_track_ids())


def load_human_search_mask_ground_truth(path: Path) -> HumanSearchMaskGroundTruth:
    """Load and validate a per-frame mask ground-truth file."""
    reader = open_document(
        path, document_type="HumanSearchMaskGroundTruth", allowed_fields=_MASK_DOCUMENT_FIELDS
    )

    declared_task_id = reader.get_str("task_id")
    if declared_task_id != TASK_ID:
        raise SchemaValidationError(
            f"{reader.context}: this loader reads {TASK_ID!r} ground truth, "
            f"but the file declares {declared_task_id!r}"
        )

    frames: list[FrameGroundTruth] = []
    seen_frames: set[int] = set()
    for frame_reader in reader.get_object_list("frames", allowed_fields=_FRAME_FIELDS):
        frame_id = frame_reader.get_int("frame_id", minimum=0)
        if frame_id in seen_frames:
            raise SchemaValidationError(
                f"{frame_reader.context}: duplicate frame_id {frame_id}; each frame may appear once"
            )
        seen_frames.add(frame_id)

        instances: list[MaskTarget] = []
        seen_tracks: set[str] = set()
        for instance_reader in frame_reader.get_object_list(
            "instances", allowed_fields=_MASK_INSTANCE_FIELDS
        ):
            track_id = instance_reader.get_str("track_id")
            if track_id in seen_tracks:
                raise SchemaValidationError(
                    f"{instance_reader.context}: track_id {track_id!r} appears twice in frame "
                    f"{frame_id}; one instance per track per frame"
                )
            seen_tracks.add(track_id)
            instances.append(
                MaskTarget(
                    track_id=track_id,
                    category=instance_reader.get_str("category"),
                    mask=decode_mask(
                        instance_reader.get_passthrough("mask"),
                        context=f"{instance_reader.context}.mask",
                    ),
                    ignore=instance_reader.get_bool("ignore")
                    if _has_key(instance_reader, "ignore")
                    else False,
                )
            )
        frames.append(FrameGroundTruth(frame_id=frame_id, instances=tuple(instances)))

    return HumanSearchMaskGroundTruth(
        frame_stream_id=reader.get_str("frame_stream_id"),
        frames=tuple(frames),
        ground_truth_id=reader.get_optional_str("ground_truth_id") or "",
    )


def load_any_ground_truth(path: Path) -> HumanSearchGroundTruth | HumanSearchMaskGroundTruth:
    """Load either ground-truth form, choosing by the field that discriminates them.

    A file declaring ``frames`` is per-frame mask ground truth; one declaring ``targets`` is
    the interval form. The discriminator is read first, then the chosen loader validates
    strictly -- schema version, task ID, and every field. A file with neither key, or both,
    is rejected rather than guessed at.
    """
    raw = read_json_object(path)
    has_frames = "frames" in raw
    has_targets = "targets" in raw
    if has_frames and not has_targets:
        return load_human_search_mask_ground_truth(path)
    if has_targets and not has_frames:
        return load_human_search_ground_truth(path)
    raise SchemaValidationError(
        f"{path.name}: ground truth must declare exactly one of 'targets' (visibility "
        "intervals) or 'frames' (per-frame masks)"
    )


def _has_key(reader, key: str) -> bool:
    """Whether an optional key is present at all, distinct from present-but-false."""
    return reader.get_passthrough(key) is not None
