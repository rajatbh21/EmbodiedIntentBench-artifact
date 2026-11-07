"""Official Mask R-CNN weights behind the RGB-only mask executor interface.

Instance probabilities are unioned for the existing connected-component evaluator;
this does not introduce box scoring or claim persistent instance tracking. Optional
Torch imports occur only when constructing a real backend.
"""

from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np

from missionbench.schemas.loading import SchemaValidationError
from missionbench.v2.real_models import (
    BackendInfo,
    TorchSemanticSegmentationExecutor,
    _resolve_device,
)

MODELS = ("maskrcnn_resnet50_fpn", "maskrcnn_resnet50_fpn_v2")
_CACHE = {}


def validate_parameters(params):
    if params.get("model_id") not in MODELS:
        raise SchemaValidationError(f"instance model_id must be one of {MODELS}")
    score = params.get("score_threshold", 0.5)
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 < score < 1:
        raise SchemaValidationError("score_threshold must be in (0, 1)")
    if params.get("device") not in ("cpu", "cuda", "mps", "auto"):
        raise SchemaValidationError("invalid instance device")
    if params.get("dtype", "float32") != "float32":
        raise SchemaValidationError("instance backend currently requires float32")
    warmup = params.get("warmup_runs", 1)
    if type(warmup) is not int or warmup < 0:
        raise SchemaValidationError("warmup_runs must be a nonnegative integer")

    tile_keys = {"tile_width_px", "tile_height_px", "tile_overlap_frac"}
    if tile_keys & params.keys():
        if not tile_keys <= params.keys():
            raise SchemaValidationError("all tile dimensions and overlap are required")
        for axis in ("width", "height"):
            value = params[f"tile_{axis}_px"]
            if type(value) is not int or not 1 <= value <= params[f"input_{axis}_px"]:
                raise SchemaValidationError("tile dimensions must fit the configured input")
        overlap = params["tile_overlap_frac"]
        if (
            isinstance(overlap, bool)
            or not isinstance(overlap, (int, float))
            or not math.isfinite(overlap)
            or not 0 <= overlap <= 0.5
        ):
            raise SchemaValidationError("tile overlap must be in [0, 0.5]")
        windows = tile_windows(
            params["input_width_px"],
            params["input_height_px"],
            params["tile_width_px"],
            params["tile_height_px"],
            overlap,
        )
        if len(windows) > 64:
            raise SchemaValidationError("at most 64 tiles per inference are supported")


def tile_windows(width, height, tile_width, tile_height, overlap):
    """Cover the full input including right/bottom edges, in fixed raster order."""

    def starts(size, tile):
        stride = max(1, int(tile * (1 - overlap)))
        values = list(range(0, size - tile + 1, stride))
        if values[-1] != size - tile:
            values.append(size - tile)
        return values

    return [
        (x, y, x + tile_width, y + tile_height)
        for y in starts(height, tile_height)
        for x in starts(width, tile_width)
    ]


def tiled_probabilities(rgb, windows, infer):
    """Union only model probabilities in image coordinates; no GT-driven cropping.

    The mask union preserves existing connected-component semantics; this is not
    SAHI's box NMS implementation or persistent instance identity tracking.
    """
    merged = np.zeros(rgb.shape[:2], dtype=np.float32)
    elapsed = 0.0
    for x0, y0, x1, y1 in windows:
        probabilities, latency = infer(rgb[y0:y1, x0:x1].copy())
        if (
            probabilities.shape != (y1 - y0, x1 - x0)
            or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("invalid tile probability output")
        np.maximum(merged[y0:y1, x0:x1], probabilities, out=merged[y0:y1, x0:x1])
        elapsed += latency
    return merged, elapsed


def union_person_probabilities(masks, labels, scores, target_index, threshold, shape):
    """No detections means an empty mask; never substitute GT or boxes."""
    masks, labels, scores = np.asarray(masks), np.asarray(labels), np.asarray(scores)
    if masks.shape != (len(labels), *shape) or scores.shape != labels.shape:
        raise ValueError("invalid instance backend output shapes")
    if not np.isfinite(masks).all() or not np.isfinite(scores).all():
        raise ValueError("non-finite instance prediction")
    keep = (labels == target_index) & (scores >= threshold)
    return masks[keep].max(axis=0) if keep.any() else np.zeros(shape, dtype=np.float32)


class InstanceBackend:
    def __init__(self, params):
        import torch
        import torchvision
        from torchvision.models import get_model, get_model_weights

        validate_parameters(params)
        start = time.perf_counter()
        self.torch = torch
        enum = get_model_weights(params["model_id"])
        weights = (
            enum.DEFAULT
            if params["weights_id"] in ("DEFAULT", "official_default")
            else enum[params["weights_id"]]
        )
        categories = weights.meta["categories"]
        self.target = categories.index(params["task_class"])
        self.device = _resolve_device(torch, params["device"])
        if params["dtype"] != "float32":
            raise SchemaValidationError("instance backend currently requires float32")
        w, h = params["input_width_px"], params["input_height_px"]
        self.windows = (
            tile_windows(
                w, h, params["tile_width_px"], params["tile_height_px"], params["tile_overlap_frac"]
            )
            if "tile_width_px" in params
            else None
        )
        # Preserve aspect ratio after the explicit executor resize; do not silently
        # upscale every configuration to torchvision's default short side of 800.
        self.model = (
            get_model(params["model_id"], weights=weights, min_size=min(w, h), max_size=max(w, h))
            .eval()
            .to(self.device)
        )
        self.transform = weights.transforms()
        self.score_threshold = float(params.get("score_threshold", 0.5))
        self.info = BackendInfo(
            framework="torch/torchvision",
            framework_version=f"{torch.__version__}/{torchvision.__version__}",
            model_id=params["model_id"],
            weights_id=params["weights_id"],
            weights_identifier=str(weights),
            target_class=params["task_class"],
            target_class_index=self.target,
            category_count=len(categories),
            device_requested=params["device"],
            device_used=self.device,
            dtype_requested="float32",
            dtype_used="float32",
            input_width_px=w,
            input_height_px=h,
            preprocessing_id=f"{weights}.transforms(); internal min={min(w, h)},max={max(w, h)}",
            model_load_s=time.perf_counter() - start,
            warmup_runs=int(params.get("warmup_runs", 1)),
        )
        for _ in range(self.info.warmup_runs):
            self.infer(np.zeros((h, w, 3), dtype=np.uint8))

    def sync(self):
        if self.device == "cuda":
            self.torch.cuda.synchronize()
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def infer(self, rgb):
        if self.windows is not None:
            return tiled_probabilities(rgb, self.windows, self._infer_single)
        return self._infer_single(rgb)

    def _infer_single(self, rgb):
        torch = self.torch
        tensor = self.transform(torch.from_numpy(rgb.copy()).permute(2, 0, 1)).to(self.device)
        self.sync()
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.model([tensor])[0]
        self.sync()
        elapsed = time.perf_counter() - start
        mask = union_person_probabilities(
            result["masks"][:, 0].cpu().numpy(),
            result["labels"].cpu().numpy(),
            result["scores"].cpu().numpy(),
            self.target,
            self.score_threshold,
            rgb.shape[:2],
        )
        return mask, elapsed


def backend_for(params):
    key = tuple(sorted(params.items()))
    if key not in _CACHE:
        _CACHE[key] = InstanceBackend(params)
    return _CACHE[key]


class TorchInstanceSegmentationExecutor(TorchSemanticSegmentationExecutor):
    def __init__(self, spec, *, backend=None):
        validate_parameters(spec.parameters)
        super().__init__(spec, backend=backend, backend_factory=backend_for)

    def run(self, rgb):
        result = super().run(rgb)
        return replace(
            result,
            diagnostics={
                **result.diagnostics,
                "executor_kind": "torch_instance_segmentation",
                "score_threshold": float(self._spec.parameters.get("score_threshold", 0.5)),
                "tiling": {
                    key: value
                    for key, value in self._spec.parameters.items()
                    if key.startswith("tile_")
                },
                "mask_semantics": (
                    "union of predicted person instance masks; connected-component scoring"
                ),
            },
        )
