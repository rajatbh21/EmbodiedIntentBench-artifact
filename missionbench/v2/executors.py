"""Image-based executors: lightweight model-strategies whose output depends on RGB.

These are deliberately *not* strong perception models. They exist to make the closed
loop real: the prediction is computed from image content (a policy cannot score without
the executor actually looking), and the two configurations trade quality against
mission latency the way a small and a large model would.

- ``fast_weak``: aggressive downsampling and a coarse RGB threshold, no cleanup. Low
  configured mission latency, ragged masks, more false positives.
- ``slow_strong``: full resolution, a more selective chroma rule, morphological
  open/close and small-component removal. Higher configured mission latency, cleaner
  masks.

Honest measurement labels: ``mission_latency_s`` is the **configured, simulated**
latency the mission experiences; ``measured_wall_clock_s`` is what the Python actually
took (diagnostic only, never fed to the simulation); ``energy_j`` and
``communication_mb`` are configured values, labelled ``simulated``/``configured``.
Executors receive the RGB crop and their configuration -- never ground truth, object
metadata, or the world.

Extension point: a future heavy backend (MobileSAM / SAM2 / detector+SAM, FP16/INT8,
remote) implements the same ``run(rgb) -> ImageExecutionResult`` surface, registers a
new ``kind`` in the scenario schema, and lives in an optional module so the core never
imports PyTorch. The real-segmentation pilot's ``SegmentationModel`` protocol
(``experiments/real_segmentation_pilot``) remains the ingestion-side twin of this
interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Final, Protocol

import numpy as np

from missionbench.schemas.loading import SchemaValidationError
from missionbench.v2.scenario import ExecutorConfigSpec

__all__ = ["ImageExecutionResult", "ImageExecutor", "build_executors"]


@dataclass(frozen=True, slots=True)
class ImageExecutionResult:
    """What one execution produced and cost, with every quantity's provenance labelled."""

    config_id: str
    model_strategy_id: str
    success: bool
    prediction_mask: np.ndarray  # bool (H, W), same shape as the input RGB
    #: The latency the mission experiences. Configured/simulated -- drives the clock.
    mission_latency_s: float
    #: What this Python implementation actually took. Diagnostic only.
    measured_wall_clock_s: float
    energy_j: float
    communication_mb: float
    measurement_provenance: dict[str, str]
    #: Radio energy for transmitting/receiving this call's payloads (V3 P1). Zero for
    #: local executors, so their result shape and existing accounting are unchanged.
    communication_energy_j: float = 0.0
    #: Backend-specific provenance (V2.1 real models record framework, weights, device,
    #: dtype, per-call latencies, etc. here). ``None`` for the V2.0 heuristic executors,
    #: so their result shape is unchanged.
    diagnostics: dict[str, Any] | None = None

    def to_summary(self) -> dict[str, Any]:
        if self.diagnostics is not None:
            return {**self._base_summary(), "diagnostics": dict(self.diagnostics)}
        return self._base_summary()

    def _base_summary(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "model_strategy_id": self.model_strategy_id,
            "success": self.success,
            "predicted_pixels": int(self.prediction_mask.sum()),
            "mission_latency_s": self.mission_latency_s,
            "measured_wall_clock_s": round(self.measured_wall_clock_s, 6),
            "energy_j": self.energy_j,
            "communication_mb": self.communication_mb,
            "communication_energy_j": self.communication_energy_j,
            "measurement_provenance": dict(self.measurement_provenance),
        }


class ImageExecutor(Protocol):
    """One selectable model-strategy over an RGB crop."""

    config_id: str
    model_strategy_id: str

    def run(self, rgb: np.ndarray) -> ImageExecutionResult: ...


_PROVENANCE: Final = {
    "mission_latency_s": "simulated (configured per executor)",
    "measured_wall_clock_s": "measured (diagnostic only; not used by the simulation)",
    "energy_j": "simulated (configured per call)",
    "communication_mb": "configured (local executor; no real transmission)",
}


class _ThresholdExecutor:
    """Shared machinery: segment 'high-visibility marker' colours from RGB."""

    def __init__(self, spec: ExecutorConfigSpec) -> None:
        self.config_id = spec.config_id
        self.model_strategy_id = spec.model_strategy_id
        self._spec = spec
        params = spec.parameters
        #: Marker hue gate, tunable per scenario but with sane defaults for the
        #: shipped orange rescue markers.
        self._min_red = int(params.get("min_red", 150))
        self._max_blue = int(params.get("max_blue", 110))
        self._min_red_minus_green = int(params.get("min_red_minus_green", 40))

    def _chroma_mask(self, rgb: np.ndarray) -> np.ndarray:
        r = rgb[..., 0].astype(np.int16)
        g = rgb[..., 1].astype(np.int16)
        b = rgb[..., 2].astype(np.int16)
        return (r >= self._min_red) & (b <= self._max_blue) & ((r - g) >= self._min_red_minus_green)

    def _result(self, mask: np.ndarray, started: float) -> ImageExecutionResult:
        return ImageExecutionResult(
            config_id=self.config_id,
            model_strategy_id=self.model_strategy_id,
            success=True,
            prediction_mask=mask,
            mission_latency_s=self._spec.mission_latency_s,
            measured_wall_clock_s=time.perf_counter() - started,
            energy_j=self._spec.energy_j_per_call,
            communication_mb=self._spec.communication_mb_per_call,
            measurement_provenance=dict(_PROVENANCE),
        )


class FastWeakExecutor(_ThresholdExecutor):
    """Downsample hard, threshold coarsely, upsample. Fast, ragged, and greedy."""

    def run(self, rgb: np.ndarray) -> ImageExecutionResult:
        started = time.perf_counter()
        scale = int(self._spec.parameters.get("downsample", 4))
        small = rgb[::scale, ::scale]
        # A looser gate than the strong config: the weak model 'sees orange' more
        # eagerly, which is what produces its extra false positives on distractors.
        r = small[..., 0].astype(np.int16)
        g = small[..., 1].astype(np.int16)
        b = small[..., 2].astype(np.int16)
        coarse = (
            (r >= self._min_red - 30)
            & (b <= self._max_blue + 30)
            & ((r - g) >= self._min_red_minus_green - 20)
        )
        mask = np.repeat(np.repeat(coarse, scale, axis=0), scale, axis=1)[
            : rgb.shape[0], : rgb.shape[1]
        ]
        if mask.shape != rgb.shape[:2]:  # short edge when scale does not divide the size
            padded = np.zeros(rgb.shape[:2], dtype=bool)
            padded[: mask.shape[0], : mask.shape[1]] = mask
            mask = padded
        return self._result(mask, started)


class SlowStrongExecutor(_ThresholdExecutor):
    """Full resolution, selective chroma rule, morphology, small-component removal."""

    def run(self, rgb: np.ndarray) -> ImageExecutionResult:
        started = time.perf_counter()
        mask = self._chroma_mask(rgb)
        mask = _binary_open_close(mask)
        min_area = int(self._spec.parameters.get("min_component_px", 12))
        mask = _drop_small_components(mask, min_area)
        return self._result(mask, started)


def build_executors(specs: tuple[ExecutorConfigSpec, ...]) -> dict[str, ImageExecutor]:
    """Instantiate the executor for every configured model-strategy, keyed by config_id.

    The torch kind resolves through a function-level import so that scenarios using only
    the V2.0 heuristic executors never touch the optional real-model module (whose heavy
    dependencies are themselves imported lazily, with an actionable error when absent).
    """
    kinds: dict[str, Any] = {"fast_weak": FastWeakExecutor, "slow_strong": SlowStrongExecutor}
    if any(spec.kind == "torch_semantic_segmentation" for spec in specs):
        from missionbench.v2.real_models import TorchSemanticSegmentationExecutor

        kinds["torch_semantic_segmentation"] = TorchSemanticSegmentationExecutor
    if any(spec.kind == "torch_cabinet_segmentation" for spec in specs):
        from missionbench.v2.cabinet_models import TorchCabinetSegmentationExecutor

        kinds["torch_cabinet_segmentation"] = TorchCabinetSegmentationExecutor
    if any(spec.kind == "torch_instance_segmentation" for spec in specs):
        from missionbench.v2.instance_models import TorchInstanceSegmentationExecutor

        kinds["torch_instance_segmentation"] = TorchInstanceSegmentationExecutor
    registry: dict[str, ImageExecutor] = {}
    for spec in specs:
        if spec.kind == "simulated_remote":
            continue  # remotes are wired second, so they can reference local fallbacks
        try:
            registry[spec.config_id] = kinds[spec.kind](spec)
        except KeyError:  # pragma: no cover - schema already validates kinds
            raise SchemaValidationError(f"unknown executor kind {spec.kind!r}") from None
    remote_specs = [spec for spec in specs if spec.kind == "simulated_remote"]
    if remote_specs:
        from missionbench.v2.remote import build_remote_executor

        for spec in remote_specs:
            registry[spec.config_id] = build_remote_executor(spec, registry)
    return registry


# --- tiny numpy morphology (no scipy) --------------------------------------------------------


def _shift_or(mask: np.ndarray) -> np.ndarray:
    """3x3 dilation via shifted ORs."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _shift_and(mask: np.ndarray) -> np.ndarray:
    """3x3 (cross) erosion via shifted ANDs."""
    out = mask.copy()
    shifted = np.zeros_like(mask)
    shifted[1:, :] = mask[:-1, :]
    out &= shifted
    shifted[:] = False
    shifted[:-1, :] = mask[1:, :]
    out &= shifted
    shifted[:] = False
    shifted[:, 1:] = mask[:, :-1]
    out &= shifted
    shifted[:] = False
    shifted[:, :-1] = mask[:, 1:]
    out &= shifted
    return out


def _binary_open_close(mask: np.ndarray) -> np.ndarray:
    """One opening (drop speckle) then one closing (fill pinholes)."""
    opened = _shift_or(_shift_and(mask))
    return ~_shift_or(_shift_and(~opened))


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labelling with a two-pass union-find. Returns (labels, count)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    next_label = 1
    for y in range(h):
        row = mask[y]
        for x in range(w):
            if not row[x]:
                continue
            up = labels[y - 1, x] if y > 0 else 0
            left = labels[y, x - 1] if x > 0 else 0
            if up == 0 and left == 0:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            elif up != 0 and left != 0:
                ru, rl = find(up), find(left)
                labels[y, x] = min(ru, rl)
                parent[max(ru, rl)] = min(ru, rl)
            else:
                labels[y, x] = up or left

    if next_label == 1:
        return labels, 0
    # Flatten the union-find and renumber densely for stable, deterministic ids.
    roots = np.array([find(i) for i in range(next_label)], dtype=np.int32)
    remap = np.zeros(next_label, dtype=np.int32)
    count = 0
    for i in range(1, next_label):
        if roots[i] == i:
            count += 1
            remap[i] = count
    labels = remap[roots[labels]]
    return labels, count


def _drop_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    labels, count = label_components(mask)
    if count == 0:
        return mask
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = areas >= min_area_px
    keep[0] = False
    return keep[labels]
