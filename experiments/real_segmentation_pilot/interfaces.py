"""The dataset and model boundaries the pilot is written against.

Two protocols keep dataset- and model-specific code out of the benchmark core (and out of the
generic bundle builder). A real pilot supplies concrete implementations; the tests here supply
tiny in-memory ones. Neither the runner nor the converters know which they are talking to.

Ground truth and predictions are deliberately separate types: a prediction carries a mask, a
category, and a confidence, and **never** a ground-truth track id or any match outcome. That
separation is the whole reason the policy can be scored honestly, so it is enforced by the
type, not by convention.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from missionbench.tasks.human_search_segmentation.masks import BinaryMask

__all__ = [
    "DatasetAdapter",
    "Frame",
    "GroundTruthInstance",
    "InMemoryDatasetAdapter",
    "PredictedInstance",
    "RealSegmentationBackend",
    "SegmentationModel",
    "StubSegmentationModel",
]


@dataclass(frozen=True, slots=True)
class Frame:
    """One ordered frame handed to a model. ``image`` is opaque to the pilot tooling.

    For a real backend ``image`` is whatever that model consumes (a path, an array); for the
    stub it is unused. Frame order is the ``frame_id`` order, fixed and identical across every
    configuration.
    """

    frame_id: int
    image: Any = None
    source_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GroundTruthInstance:
    """One hidden ground-truth person instance in one frame."""

    frame_id: int
    track_id: str
    category: str
    mask: BinaryMask
    ignore: bool = False


@dataclass(frozen=True, slots=True)
class PredictedInstance:
    """One model-produced instance. Carries no ground-truth identity, ever."""

    category: str
    confidence: float
    mask: BinaryMask


@runtime_checkable
class DatasetAdapter(Protocol):
    """Turns a real dataset into ordered frames and hidden ground truth.

    A concrete adapter lives outside the benchmark core and knows one dataset's on-disk layout.
    It must not convert bounding boxes into masks, and must expose track ids only when the
    dataset genuinely provides them.
    """

    frame_stream_id: str
    image_width: int
    image_height: int

    def frames(self) -> Iterable[Frame]: ...

    def ground_truth(self) -> Iterable[GroundTruthInstance]: ...


class SegmentationModel(Protocol):
    """One executable model-strategy configuration."""

    config_id: str
    model_id: str

    def predict(self, frame: Frame) -> Sequence[PredictedInstance]: ...


@dataclass(frozen=True, slots=True)
class InMemoryDatasetAdapter:
    """A dataset held in memory. For tests and reference only -- not a real dataset."""

    frame_stream_id: str
    image_width: int
    image_height: int
    _frames: tuple[Frame, ...]
    _ground_truth: tuple[GroundTruthInstance, ...]

    def frames(self) -> Iterable[Frame]:
        return sorted(self._frames, key=lambda frame: frame.frame_id)

    def ground_truth(self) -> Iterable[GroundTruthInstance]:
        return self._ground_truth


@dataclass(frozen=True, slots=True)
class StubSegmentationModel:
    """A deterministic stand-in that returns pre-set predictions per frame.

    **Not a model.** It runs no inference and its masks are supplied by the caller. It exists
    only to exercise the pilot pipeline (timing, conversion, bundle build) in tests, and must
    never be presented as producing a real empirical result.
    """

    config_id: str
    model_id: str
    _by_frame: Mapping[int, Sequence[PredictedInstance]]

    def predict(self, frame: Frame) -> Sequence[PredictedInstance]:
        return tuple(self._by_frame.get(frame.frame_id, ()))


class RealSegmentationBackend:
    """Placeholder for a real segmentation model. **Construction raises.**

    A real backend belongs here: it would import an optional model stack (see
    ``requirements.txt``), load a checkpoint, run inference, and return
    :class:`PredictedInstance` masks in the model's output resolution for the runner to resize
    onto the ground-truth grid. None of that is implemented, and it must not be faked, so
    selecting it fails immediately with instructions rather than pretending to run.
    """

    config_id: str
    model_id: str

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise NotImplementedError(
            "RealSegmentationBackend is not implemented. Provide a concrete SegmentationModel: "
            "install the pilot extras (experiments/real_segmentation_pilot/requirements.txt), "
            "load a checkpoint, and return PredictedInstance masks from predict(). This "
            "environment has no model stack, weights, or GPU, so no real run is possible here."
        )

    def predict(self, frame: Frame) -> Sequence[PredictedInstance]:  # pragma: no cover
        raise NotImplementedError
