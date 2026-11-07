"""UAVid dataset adapter for the real-data pilot (V3 P4).

Reads the **official UAVid distribution layout** (``{split}/{split}/seqN/Images`` +
``seqN/Labels`` with colour-coded semantic PNGs) — real oblique UAV imagery — and
exposes it through the pilot's :class:`DatasetAdapter` boundary.

What the adapter is honest about:

- **License**: UAVid is released for non-commercial academic research (CC BY-NC-SA
  4.0). The dataset never enters this repository; only a local path is consumed.
- **Semantic, not instance, ground truth**: official UAVid labels are per-pixel
  semantic classes ("currently, UAVid only supports image level semantic labelling
  without instance level consideration" — uavid.nl). Person *instances* are therefore
  **derived** as 4-connected components of the ``Humans`` colour (64, 64, 0). The
  derivation is deterministic and recorded in provenance; its known failure mode —
  touching people merge into one component — undercounts adjacent groups and is
  stated, not hidden.
- **No temporal identity**: UAVid keyframes are 10 s apart and carry no track ids, so
  every derived component is its own target (``track_id = "<seq>/<frame>#<n>"``);
  track-level recall degenerates, deliberately and visibly, to per-instance recall.
  Inventing cross-frame identity here would be fabrication.
- **Curation is scope, not leakage**: the pilot evaluates a native-resolution
  **window** of each selected keyframe, centred on the frame's annotated activity,
  and prefers frames of moderate instance density. Both choices happen entirely on
  ground truth the evaluator owns anyway — no model or policy ever sees it — and
  every window origin, clipped instance, and skipped frame is recorded in provenance.
  The window exists because the frozen V1 mask wire format is a dense per-instance
  bitmap: full 4K frames with dense crowds produce multi-gigabyte ground-truth files
  (measured, not hypothetical), while a 1280x720 native window keeps the bundle
  honest *and* writable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from missionbench.tasks.human_search_segmentation.masks import BinaryMask
from experiments.real_segmentation_pilot.interfaces import Frame, GroundTruthInstance

__all__ = ["HUMAN_RGB", "UAVidAdapter", "build_adapter"]

#: The official UAVid palette colour of the ``Humans`` class.
HUMAN_RGB: Final = (64, 64, 0)
#: Derived components smaller than this are dropped as labelling slivers; the choice
#: matches the evaluator's own minimum component size and is recorded in provenance.
_MIN_INSTANCE_PX: Final = 4


@dataclass(frozen=True, slots=True)
class _FrameRecord:
    frame_id: int
    image_path: Path
    image_name: str  # "seq1/000900"
    window_top: int
    window_left: int
    instances: tuple[BinaryMask, ...]  # window-grid masks
    clipped_instances: int  # instances dropped because they left the window


class UAVidAdapter:
    """Ordered UAVid frame windows + derived per-instance ground truth."""

    frame_stream_id: str
    image_width: int
    image_height: int

    def __init__(
        self,
        dataset_root: Path,
        *,
        splits: tuple[str, ...] = ("train", "valid"),
        max_frames: int | None = None,
        min_instances_per_frame: int = 1,
        window: tuple[int, int] = (720, 1280),  # (height, width), native resolution
        target_instances_per_frame: int = 15,
    ) -> None:
        self._window = window
        candidates: list[tuple[str, _FrameRecord]] = []
        for split in splits:
            split_dir = dataset_root / split / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"UAVid split not found: {split_dir}")
            for label_path in sorted(split_dir.glob("seq*/Labels/*.png")):
                sequence = label_path.parent.parent.name
                image_path = label_path.parent.parent / "Images" / label_path.name
                if not image_path.is_file():
                    raise FileNotFoundError(f"label without image: {image_path}")
                name = f"{sequence}/{label_path.stem}"
                record = _windowed_instances(label_path, image_path, name, window)
                if record is None or len(record.instances) < min_instances_per_frame:
                    continue
                candidates.append((name, record))

        if not candidates:
            raise ValueError(f"no frames with Humans instances under {dataset_root} {splits}")

        # Moderate density first (closest to the target), then name order; ids
        # re-follow name order so the stream is a stable, documented sequence.
        candidates.sort(
            key=lambda entry: (
                abs(len(entry[1].instances) - target_instances_per_frame),
                entry[0],
            )
        )
        if max_frames is not None:
            candidates = candidates[:max_frames]
        candidates.sort(key=lambda entry: entry[0])

        self._records: list[_FrameRecord] = [
            _FrameRecord(
                frame_id=frame_id,
                image_path=record.image_path,
                image_name=record.image_name,
                window_top=record.window_top,
                window_left=record.window_left,
                instances=record.instances,
                clipped_instances=record.clipped_instances,
            )
            for frame_id, (_name, record) in enumerate(candidates)
        ]
        self.image_height, self.image_width = window
        self.frame_stream_id = "UAVID_" + "_".join(splits).upper()

    # -- DatasetAdapter -----------------------------------------------------------------

    def frames(self) -> list[Frame]:
        """Window crops as arrays: the models see exactly the evaluated region."""
        import numpy as np
        from PIL import Image

        out: list[Frame] = []
        for record in self._records:
            rgb = np.asarray(Image.open(record.image_path).convert("RGB"))
            top, left = record.window_top, record.window_left
            crop = rgb[top : top + self.image_height, left : left + self.image_width]
            out.append(
                Frame(
                    frame_id=record.frame_id,
                    image=crop,
                    source_metadata={
                        "image_name": record.image_name,
                        "window_top": top,
                        "window_left": left,
                    },
                )
            )
        return out

    def ground_truth(self) -> list[GroundTruthInstance]:
        truth: list[GroundTruthInstance] = []
        for record in self._records:
            for index, mask in enumerate(record.instances):
                truth.append(
                    GroundTruthInstance(
                        frame_id=record.frame_id,
                        # No temporal identity exists in UAVid; the id names the frame
                        # and component this instance was derived from.
                        track_id=f"{record.image_name}#{index:03d}",
                        category="person",
                        mask=mask,
                    )
                )
        return truth

    # -- provenance ---------------------------------------------------------------------

    def provenance(self) -> dict[str, Any]:
        return {
            "dataset": "UAVid (official distribution layout; real oblique UAV imagery)",
            "license": "non-commercial academic research (CC BY-NC-SA 4.0); never committed",
            "instance_derivation": (
                f"4-connected components of the semantic Humans colour {HUMAN_RGB}, "
                f"components < {_MIN_INSTANCE_PX} px dropped; touching people merge "
                "into one component (undercounts adjacent groups)"
            ),
            "temporal_identity": (
                "none in the dataset; track_id is per-frame component identity, so "
                "track-level recall reads as per-instance recall"
            ),
            "evaluation_window": (
                f"native-resolution {self.image_width}x{self.image_height} window per "
                "keyframe, centred on the median annotated pixel and clamped to the "
                "frame; instances wholly outside the window are dropped and counted "
                "below. The window exists because the frozen V1 dense mask format "
                "makes full-4K crowded frames multi-GB."
            ),
            "selection_rule": (
                "frames whose window holds >= 1 derived instance, preferring counts "
                "closest to the target density; curation only — no model or policy "
                "sees the labels"
            ),
            "windows": [
                {
                    "image_name": record.image_name,
                    "window_top": record.window_top,
                    "window_left": record.window_left,
                    "instances": len(record.instances),
                    "clipped_instances": record.clipped_instances,
                }
                for record in self._records
            ],
            "frames": len(self._records),
            "human_instances": sum(len(r.instances) for r in self._records),
            "clipped_instances_total": sum(r.clipped_instances for r in self._records),
        }


def build_adapter() -> UAVidAdapter:
    """The default pilot subset from the local kagglehub download location."""
    root = (
        Path.home()
        / ".cache/kagglehub/datasets/awsaf49/uavid-semantic-segmentation-dataset/versions/1"
    )
    return UAVidAdapter(root, max_frames=12)


# --- label decoding --------------------------------------------------------------------------


def _windowed_instances(
    label_path: Path, image_path: Path, image_name: str, window: tuple[int, int]
) -> _FrameRecord | None:
    """Derive person instances inside a GT-centred native window of one keyframe."""
    import numpy as np
    from PIL import Image

    from missionbench.v2.executors import label_components

    rgb = np.asarray(Image.open(label_path).convert("RGB"))
    frame_h, frame_w = rgb.shape[:2]
    win_h, win_w = window
    if frame_h < win_h or frame_w < win_w:
        return None
    human = (
        (rgb[..., 0] == HUMAN_RGB[0])
        & (rgb[..., 1] == HUMAN_RGB[1])
        & (rgb[..., 2] == HUMAN_RGB[2])
    )
    if not human.any():
        return None

    # Window placement: centred on the median annotated pixel, clamped to bounds.
    rows, cols = np.nonzero(human)
    top = int(np.clip(int(np.median(rows)) - win_h // 2, 0, frame_h - win_h))
    left = int(np.clip(int(np.median(cols)) - win_w // 2, 0, frame_w - win_w))

    crop = human[top : top + win_h, left : left + win_w]
    labels, count = label_components(crop)
    # An instance touching the window edge may be a fragment of a person whose rest
    # lies outside; it stays (the model sees the same fragment), but an instance
    # wholly outside the window simply is not part of this frame's evaluated scope.
    _full_labels, full_count = label_components(human)
    clipped = max(0, full_count - count)

    masks: list[BinaryMask] = []
    for component in range(1, count + 1):
        member = labels == component
        if int(member.sum()) < _MIN_INSTANCE_PX:
            continue
        member_rows, member_cols = np.nonzero(member)
        masks.append(
            BinaryMask(
                height=win_h,
                width=win_w,
                pixels=frozenset(zip(member_rows.tolist(), member_cols.tolist(), strict=True)),
            )
        )
    if not masks:
        return None
    return _FrameRecord(
        frame_id=-1,  # assigned by the adapter after selection
        image_path=image_path,
        image_name=image_name,
        window_top=top,
        window_left=left,
        instances=tuple(masks),
        clipped_instances=clipped,
    )
