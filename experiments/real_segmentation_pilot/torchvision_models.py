"""Tiled torchvision segmentation models for the real-data pilot (V3 P4).

Concrete :class:`~experiments.real_segmentation_pilot.interfaces.SegmentationModel`
implementations around the same pretrained torchvision checkpoints the V2.1 executors
use (LRASPP-MobileNetV3 light / DeepLabV3-ResNet50 strong, official DEFAULT weights,
person class index read from weight metadata — never hardcoded).

Aerial frames are large (UAVid: 4096x2160) while people project to tens of pixels, so
whole-frame downscaling destroys the objects being sought. The models therefore run
**tiled**: fixed-size crops on a regular grid (edge tiles overlap inward), the per-tile
person probabilities are stitched back onto the full-resolution grid, and connected
components of the thresholded map become predicted instances. Tiling is a standard
inference strategy, not ground-truth knowledge: the tile grid is fixed per model
configuration and never depends on annotations.

Honesty rules carried over from V2.1: pretrained COCO/VOC checkpoints were not trained
on aerial imagery — low recall here is a documented domain observation, never to be
"fixed" by peeking at labels; latency is measured wall-clock around the full tiled
forward pass (with device synchronisation); energy is estimated by the pilot's
power-assumption protocol and labelled as such.

Heavy imports (torch/torchvision) happen lazily inside the builders so that importing
this module stays cheap and the stub-based CI tests never touch them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

from missionbench.tasks.human_search_segmentation.masks import BinaryMask
from experiments.real_segmentation_pilot.interfaces import Frame, PredictedInstance

__all__ = ["TiledTorchvisionModel", "build_deeplab_strong", "build_lraspp_light"]

_PERSON_CATEGORY: Final = "person"
#: Predicted components smaller than this are discarded as sensor noise. Matches the
#: default the V2 evaluator uses for its own component filter.
_MIN_COMPONENT_PX: Final = 4


@dataclass
class TiledTorchvisionModel:
    """One pretrained torchvision semantic-segmentation checkpoint, run tiled."""

    config_id: str
    model_id: str
    tile_px: int
    probability_threshold: float
    device_preference: str = "auto"
    #: Physical-size prior, model-side: at UAVid's ground sampling distance a person
    #: projects to at most a few thousand pixels, so components beyond this are
    #: certainly not people (out-of-domain mega-blobs — whole roofs/roads). Applied to
    #: the model's own output only; no annotation is consulted. Without it a single
    #: out-of-domain frame can emit gigabytes of pixel-list masks.
    max_component_px: int = 20_000
    #: Upper bound on instances per frame, kept by descending confidence — a plain
    #: robustness cap on degenerate out-of-domain outputs.
    max_instances_per_frame: int = 500
    _backend: Any = None  # (model, preprocess, person_index, device), built lazily

    def predict(self, frame: Frame) -> Sequence[PredictedInstance]:
        import numpy as np

        model, preprocess, person_index, device, torch = self._ensure_backend()
        rgb = self._frame_rgb(frame)
        height, width = rgb.shape[:2]
        probability = np.zeros((height, width), dtype=np.float32)

        for top, left in self._tile_origins(height, width):
            tile = rgb[top : top + self.tile_px, left : left + self.tile_px]
            with torch.inference_mode():
                batch = preprocess(torch.from_numpy(tile.copy()).permute(2, 0, 1)).unsqueeze(0)
                logits = model(batch.to(device))["out"]
                person_prob = logits.softmax(dim=1)[0, person_index]
                resized = torch.nn.functional.interpolate(
                    person_prob[None, None],
                    size=(tile.shape[0], tile.shape[1]),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            patch = resized.cpu().numpy()
            region = probability[top : top + self.tile_px, left : left + self.tile_px]
            np.maximum(region, patch, out=region)  # overlap resolution: max probability

        if device.type != "cpu":
            torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()

        return self._instances_from(probability)

    # -- internals ----------------------------------------------------------------------

    def _ensure_backend(self):
        if self._backend is None:
            import torch
            from torchvision.models import segmentation as seg

            if self.model_id == "lraspp_mobilenet_v3_large":
                weights = seg.LRASPP_MobileNet_V3_Large_Weights.DEFAULT
                model = seg.lraspp_mobilenet_v3_large(weights=weights)
            elif self.model_id == "deeplabv3_resnet50":
                weights = seg.DeepLabV3_ResNet50_Weights.DEFAULT
                model = seg.deeplabv3_resnet50(weights=weights)
            else:
                raise ValueError(f"unknown model_id {self.model_id!r}")
            categories = weights.meta["categories"]
            person_index = categories.index(_PERSON_CATEGORY)  # from metadata, never hardcoded
            device = self._resolve_device(torch)
            model.eval().to(device)
            self._backend = (model, weights.transforms(), person_index, device, torch)
        return self._backend

    def _resolve_device(self, torch):
        if self.device_preference != "auto":
            return torch.device(self.device_preference)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _frame_rgb(self, frame: Frame):
        import numpy as np
        from PIL import Image

        if isinstance(frame.image, np.ndarray):
            return frame.image
        return np.asarray(Image.open(frame.image).convert("RGB"))

    def _tile_origins(self, height: int, width: int) -> list[tuple[int, int]]:
        """A fixed grid covering the frame; edge tiles shift inward to stay full-size."""

        def starts(extent: int) -> list[int]:
            if extent <= self.tile_px:
                return [0]
            positions = list(range(0, extent - self.tile_px, self.tile_px))
            positions.append(extent - self.tile_px)
            return positions

        return [(top, left) for top in starts(height) for left in starts(width)]

    def _instances_from(self, probability) -> list[PredictedInstance]:
        import numpy as np

        from missionbench.v2.executors import label_components

        mask = probability >= self.probability_threshold
        labels, count = label_components(mask)
        areas = np.bincount(labels.ravel(), minlength=count + 1)
        instances: list[PredictedInstance] = []
        for component in range(1, count + 1):
            area = int(areas[component])
            if area < _MIN_COMPONENT_PX or area > self.max_component_px:
                continue
            member = labels == component
            rows, cols = np.nonzero(member)
            pixels = frozenset(zip(rows.tolist(), cols.tolist(), strict=True))
            confidence = float(probability[member].mean())
            instances.append(
                PredictedInstance(
                    category=_PERSON_CATEGORY,
                    confidence=round(confidence, 4),
                    mask=BinaryMask(
                        height=int(probability.shape[0]),
                        width=int(probability.shape[1]),
                        pixels=pixels,
                    ),
                )
            )
        instances.sort(key=lambda inst: (-inst.confidence, inst.mask.area))
        return instances[: self.max_instances_per_frame]


def build_lraspp_light() -> TiledTorchvisionModel:
    return TiledTorchvisionModel(
        config_id="CFG_LOCAL_LIGHT",
        model_id="lraspp_mobilenet_v3_large",
        tile_px=512,
        probability_threshold=0.5,
    )


def build_deeplab_strong() -> TiledTorchvisionModel:
    return TiledTorchvisionModel(
        config_id="CFG_LOCAL_STRONG",
        model_id="deeplabv3_resnet50",
        tile_px=512,
        probability_threshold=0.5,
    )
