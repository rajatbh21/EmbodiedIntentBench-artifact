"""V2.1 real model-strategies: pretrained torchvision semantic segmentation executors.

This is the seam where an *actual neural network* enters the visual closed loop. A
``torch_semantic_segmentation`` executor loads official pretrained weights once, runs a
real forward pass on each RGB crop the camera renders, converts the target-class
probabilities into a binary mask aligned with the observation, and reports **measured**
wall-clock latency alongside the configured (simulated) energy.

Split of responsibilities:

- :class:`TorchSegmentationBackend` -- model construction, official weight loading,
  device/dtype placement, warmup, the forward pass, and target-class resolution **from
  the weight metadata** (never a hardcoded index).
- :class:`TorchSemanticSegmentationExecutor` -- the ``ImageExecutor``: resize the crop
  to the configured model input, invoke the backend, threshold, postprocess, resize the
  binary mask back (nearest-neighbour), package the result with full provenance.
- :func:`backend_for` -- a process-wide cache keyed by (model, weights, device, dtype,
  input size), so weights load once per compatible strategy, never per frame.

Dependency honesty: ``torch``/``torchvision`` are imported lazily inside the backend.
The default install, the V1 core, V2.0 scenarios, and the default test suite never
touch them; a missing dependency fails with the exact install command. Unit tests
inject a fake backend through the executor's ``backend`` argument instead of loading a
real model.

Latency honesty: ``latency_mode: "measured"`` drives the mission clock with the real
end-to-end executor latency (which makes missions machine-dependent -- said so in the
result); ``"configured"`` keeps the deterministic profile value and records the
measurement as a diagnostic. Energy stays simulated either way -- elapsed time is not a
power meter.

Domain honesty: the shipped V2 targets are synthetic rescue markers. A COCO/VOC-trained
person model has never seen them; poor or empty masks on the synthetic scenario are a
**domain mismatch**, not an implementation failure and not a perception result.

Future extension (documented, out of scope here): a prompt-based strategy
(detector -> person boxes -> SAM/SAM2 -> instance masks) slots in as another executor
kind behind the same ``run(rgb) -> ImageExecutionResult`` surface; SAM alone is not an
autonomous detector and must never be fed GT boxes in benchmark mode.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np

from missionbench.schemas.loading import SchemaValidationError
from missionbench.v2.executors import (
    ImageExecutionResult,
    _drop_small_components,
)
from missionbench.v2.scenario import ExecutorConfigSpec
from missionbench.v2.world import resize_mask_nearest

__all__ = [
    "BackendInfo",
    "TorchSegmentationBackend",
    "TorchSemanticSegmentationExecutor",
    "backend_for",
    "check_real_model_availability",
    "clear_backend_cache",
]

#: device: "auto" resolves in this documented order.
_DEVICE_PREFERENCE: Final = ("cuda", "mps", "cpu")

_MISSING_DEPS_MESSAGE: Final = (
    "a torch_semantic_segmentation executor needs the optional real-model extras. "
    "Install them with: pip install -e '.[v2-real-models]' "
    "(torch and torchvision; the default install stays lightweight on purpose)"
)


@dataclass(frozen=True, slots=True)
class BackendInfo:
    """What a backend actually is and actually did at load time. Pure provenance."""

    framework: str
    framework_version: str
    model_id: str
    weights_id: str
    weights_identifier: str
    target_class: str
    target_class_index: int
    category_count: int
    device_requested: str
    device_used: str
    dtype_requested: str
    dtype_used: str
    input_width_px: int
    input_height_px: int
    preprocessing_id: str
    model_load_s: float
    warmup_runs: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "framework": self.framework,
            "framework_version": self.framework_version,
            "model_id": self.model_id,
            "weights_id": self.weights_id,
            "weights_identifier": self.weights_identifier,
            "target_class": self.target_class,
            "target_class_index": self.target_class_index,
            "category_count": self.category_count,
            "device_requested": self.device_requested,
            "device_used": self.device_used,
            "dtype_requested": self.dtype_requested,
            "dtype_used": self.dtype_used,
            "input_width_px": self.input_width_px,
            "input_height_px": self.input_height_px,
            "preprocessing_id": self.preprocessing_id,
            "model_load_s": self.model_load_s,
            "warmup_runs": self.warmup_runs,
        }


class SegmentationBackend(Protocol):
    """The narrow surface an executor needs: probabilities for the target class."""

    info: BackendInfo

    def infer(self, rgb_input: np.ndarray) -> tuple[np.ndarray, float]:
        """Run one forward pass on an already-resized uint8 RGB array.

        Returns ``(probability_map, forward_latency_s)`` where the map is float32 with
        the same height/width as the input and holds the target class's probability.
        """
        ...


class TorchSegmentationBackend:
    """A torchvision semantic-segmentation model with official pretrained weights."""

    def __init__(
        self,
        *,
        model_id: str,
        weights_id: str,
        target_class: str,
        device: str,
        dtype: str,
        input_width_px: int,
        input_height_px: int,
        warmup_runs: int = 1,
    ) -> None:
        try:
            import torch
            import torchvision
            from torchvision.models import get_model, get_model_weights
        except ImportError as error:
            raise SchemaValidationError(_MISSING_DEPS_MESSAGE) from error

        self._torch = torch
        started = time.perf_counter()

        # -- weights: the official enum, never a guessed URL or index -------------------
        try:
            weights_enum = get_model_weights(model_id)
        except ValueError as error:
            raise SchemaValidationError(
                f"unknown torchvision model_id {model_id!r}: {error}"
            ) from None
        if weights_id in ("official_default", "DEFAULT"):
            weights = weights_enum.DEFAULT
        else:
            try:
                weights = weights_enum[weights_id]
            except KeyError:
                raise SchemaValidationError(
                    f"model {model_id!r} has no weights {weights_id!r}; available: "
                    f"{[w.name for w in weights_enum]}"
                ) from None

        # -- target class: resolved from the weight metadata ---------------------------
        categories = list(weights.meta.get("categories") or [])
        if target_class not in categories:
            raise SchemaValidationError(
                f"weights {weights} have no category {target_class!r}; categories: {categories}"
            )
        self._class_index = categories.index(target_class)

        # -- device / dtype: validated, and the *actual* choice recorded ---------------
        device_used = _resolve_device(torch, device)
        if dtype == "float16" and device_used == "cpu":
            raise SchemaValidationError(
                "dtype float16 is not supported on cpu; use float32, or a cuda/mps device"
            )
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        # -- model: constructed once, weights loaded once ------------------------------
        model = get_model(model_id, weights=weights)
        model.eval()
        model.to(device=device_used, dtype=torch_dtype)
        self._model = model
        self._device = device_used
        self._dtype = torch_dtype

        # -- preprocessing: the official weight transform, with its internal resize
        #    disabled when the preset allows it (we resize explicitly to the configured
        #    model input); recorded either way. Never hand-guessed normalisation.
        try:
            self._transform = weights.transforms(resize_size=None)
            preprocessing_id = f"{weights}.transforms(resize_size=None)"
        except TypeError:  # older preset without the knob: its own resize applies
            self._transform = weights.transforms()
            preprocessing_id = f"{weights}.transforms()"

        load_s = time.perf_counter() - started

        self.info = BackendInfo(
            framework="torch/torchvision",
            framework_version=f"{torch.__version__}/{torchvision.__version__}",
            model_id=model_id,
            weights_id=weights_id,
            weights_identifier=str(weights),
            target_class=target_class,
            target_class_index=self._class_index,
            category_count=len(categories),
            device_requested=device,
            device_used=device_used,
            dtype_requested=dtype,
            dtype_used=str(torch_dtype).removeprefix("torch."),
            input_width_px=input_width_px,
            input_height_px=input_height_px,
            preprocessing_id=preprocessing_id,
            model_load_s=load_s,
            warmup_runs=max(0, warmup_runs),
        )

        # -- warmup: outside every mission timing ---------------------------------------
        blank = np.zeros((input_height_px, input_width_px, 3), dtype=np.uint8)
        for _ in range(max(0, warmup_runs)):
            self.infer(blank)

    def infer(self, rgb_input: np.ndarray) -> tuple[np.ndarray, float]:
        torch = self._torch
        tensor = self._transform(torch.from_numpy(rgb_input.copy()).permute(2, 0, 1))
        batch = tensor.unsqueeze(0).to(device=self._device, dtype=self._dtype)

        self._synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self._model(batch)["out"]
        self._synchronize()
        forward_s = time.perf_counter() - started

        probabilities = torch.softmax(logits.float(), dim=1)[0, self._class_index]
        return probabilities.cpu().numpy().astype(np.float32), forward_s

    def _synchronize(self) -> None:
        if self._device == "cuda":
            self._torch.cuda.synchronize()
        elif self._device == "mps":
            self._torch.mps.synchronize()


def _resolve_device(torch, requested: str) -> str:
    if requested != "auto":
        if requested == "cuda" and not torch.cuda.is_available():
            raise SchemaValidationError("device 'cuda' requested but CUDA is not available")
        if requested == "mps" and not torch.backends.mps.is_available():
            raise SchemaValidationError("device 'mps' requested but MPS is not available")
        if requested not in ("cpu", "cuda", "mps"):
            raise SchemaValidationError(f"unknown device {requested!r}")
        return requested
    for candidate in _DEVICE_PREFERENCE:
        if candidate == "cuda" and torch.cuda.is_available():
            return "cuda"
        if candidate == "mps" and torch.backends.mps.is_available():
            return "mps"
    return "cpu"


# --- backend cache ---------------------------------------------------------------------------

_BACKEND_CACHE: dict[tuple, SegmentationBackend] = {}


def backend_for(parameters: Mapping[str, Any]) -> SegmentationBackend:
    """Return the cached backend for a model-strategy, loading weights at most once.

    The cache key is everything that changes the loaded artefact: model, weights,
    device, dtype, and input size. Two configs differing only in threshold or
    postprocessing share one backend.
    """
    key = (
        parameters["model_id"],
        parameters["weights_id"],
        parameters["device"],
        parameters["dtype"],
        int(parameters["input_width_px"]),
        int(parameters["input_height_px"]),
        parameters["task_class"],
    )
    if key not in _BACKEND_CACHE:
        _BACKEND_CACHE[key] = TorchSegmentationBackend(
            model_id=str(parameters["model_id"]),
            weights_id=str(parameters["weights_id"]),
            target_class=str(parameters["task_class"]),
            device=str(parameters["device"]),
            dtype=str(parameters["dtype"]),
            input_width_px=int(parameters["input_width_px"]),
            input_height_px=int(parameters["input_height_px"]),
            warmup_runs=int(parameters.get("warmup_runs", 1)),
        )
    return _BACKEND_CACHE[key]


def clear_backend_cache() -> None:
    _BACKEND_CACHE.clear()


# --- the executor ----------------------------------------------------------------------------


class TorchSemanticSegmentationExecutor:
    """The ``ImageExecutor`` over a real segmentation backend.

    Receives the RGB crop and nothing else. Ground truth, object metadata, and scenario
    hidden state are structurally out of reach -- ``run(rgb)`` is the whole surface.
    """

    def __init__(
        self,
        spec: ExecutorConfigSpec,
        *,
        backend: SegmentationBackend | None = None,
        backend_factory: Callable[[Mapping[str, Any]], SegmentationBackend] | None = None,
    ) -> None:
        self.config_id = spec.config_id
        self.model_strategy_id = spec.model_strategy_id
        self._spec = spec
        params = spec.parameters
        self._input_w = int(params["input_width_px"])
        self._input_h = int(params["input_height_px"])
        self._threshold = float(params["probability_threshold"])
        self._latency_mode = str(params["latency_mode"])
        self._min_component_px = int(params.get("min_component_px", 0))
        #: Injectable for tests; the real path loads (or reuses) the cached torch backend.
        #: The default factory is looked up late (module global), so tests may patch it.
        if backend is not None:
            self._backend = backend
        else:
            self._backend = (backend_factory or backend_for)(params)

    def run(self, rgb: np.ndarray) -> ImageExecutionResult:
        started = time.perf_counter()
        observation_h, observation_w = rgb.shape[:2]

        resized = _resize_rgb_bilinear(rgb, self._input_w, self._input_h)
        probabilities, forward_s = self._backend.infer(resized)
        mask_input = probabilities >= self._threshold
        if self._min_component_px > 0:
            mask_input = _drop_small_components(mask_input, self._min_component_px)
        mask = resize_mask_nearest(mask_input, observation_w, observation_h)
        end_to_end_s = time.perf_counter() - started

        measured_drives_clock = self._latency_mode == "measured"
        mission_latency_s = end_to_end_s if measured_drives_clock else self._spec.mission_latency_s

        info = self._backend.info
        return ImageExecutionResult(
            config_id=self.config_id,
            model_strategy_id=self.model_strategy_id,
            success=True,
            prediction_mask=mask,
            mission_latency_s=mission_latency_s,
            measured_wall_clock_s=end_to_end_s,
            energy_j=self._spec.energy_j_per_call,
            communication_mb=self._spec.communication_mb_per_call,
            measurement_provenance={
                "mission_latency_s": (
                    "measured (end-to-end executor latency drives the mission clock; "
                    "machine-dependent)"
                    if measured_drives_clock
                    else "simulated (configured per executor; measured latency recorded "
                    "as diagnostic)"
                ),
                "measured_wall_clock_s": "measured (end-to-end: preprocess + transfer + "
                "forward + conversion + postprocess)",
                "model_forward_latency_s": "measured (forward pass only, device-synchronised)",
                "energy_j": "simulated (configured per call; elapsed time is not a power meter)",
                "communication_mb": "configured (local executor; no real transmission)",
            },
            diagnostics={
                "executor_kind": "torch_semantic_segmentation",
                **info.to_dict(),
                "observation_width_px": observation_w,
                "observation_height_px": observation_h,
                "resize_method": "bilinear (RGB in), nearest (mask out)",
                "probability_threshold": self._threshold,
                "postprocessing": {"min_component_px": self._min_component_px},
                "latency_mode": self._latency_mode,
                "model_forward_latency_s": round(forward_s, 6),
                "end_to_end_executor_latency_s": round(end_to_end_s, 6),
                "predicted_positive_px": int(mask.sum()),
            },
        )


def _resize_rgb_bilinear(rgb: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    from PIL import Image

    if rgb.shape[0] == out_h and rgb.shape[1] == out_w:
        return rgb
    return np.array(Image.fromarray(rgb).resize((out_w, out_h), Image.BILINEAR), dtype=np.uint8)


# --- availability check (CLI) ----------------------------------------------------------------


def check_real_model_availability(
    parameter_sets: list[Mapping[str, Any]], *, load: bool = False
) -> list[dict[str, Any]]:
    """Report whether each real-model config is runnable, without a mission.

    With ``load=False`` this verifies the import, the model id, the weights, and the
    target-class resolution from weight metadata -- no weight download. With
    ``load=True`` it constructs each backend (downloading/caching weights) and reports
    the load diagnostics.
    """
    reports: list[dict[str, Any]] = []
    for params in parameter_sets:
        report: dict[str, Any] = {"model_id": params.get("model_id")}
        try:
            import torch
            import torchvision
            from torchvision.models import get_model_weights

            report["framework"] = (
                f"torch {torch.__version__} / torchvision {torchvision.__version__}"
            )
            weights_enum = get_model_weights(str(params["model_id"]))
            weights = weights_enum.DEFAULT
            categories = list(weights.meta.get("categories") or [])
            target = str(params["task_class"])
            report["weights"] = str(weights)
            report["target_class_index"] = (
                categories.index(target) if target in categories else None
            )
            if report["target_class_index"] is None:
                report["status"] = f"error: no category {target!r} in weight metadata"
                reports.append(report)
                continue
            if load:
                backend = backend_for(params)
                report["loaded"] = backend.info.to_dict()
            report["status"] = "ok"
        except SchemaValidationError as error:
            report["status"] = f"error: {error}"
        except ImportError:
            report["status"] = f"error: {_MISSING_DEPS_MESSAGE}"
        reports.append(report)
    return reports
