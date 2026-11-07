"""Fixed RGB-only specialists with native segmentation outputs and explicit passes.

Tiling and luminance correction are configuration choices, not learned aerial
expertise. CABiNet is the separately trained UAVid semantic model. No labels,
scene positions or evaluator state enter any inference method.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from missionbench.v2.executors import build_executors
from experiments.lake_fast_perception import Segmenter
from experiments.lake_missions.assets import sha
from experiments.lake_rotation_study import CONTROL, ENSEMBLE, ROTATED

KIND = "mission_routing_rgb_pipeline_v2"
YOLO_FULL = "CFG_YOLO11S_960_ROT0"
YOLO_NANO = "CFG_YOLO11N_640_ROT0"
TILED = "CFG_YOLO11S_TILED_640"
TILED_ROTATED = "CFG_YOLO11S_TILED_640_ROT90"
TILED_GAMMA = "CFG_YOLO11S_TILED_640_GAMMA"
AERIAL = "CFG_CABINET_UAVID_1280"
NEW = (TILED, TILED_ROTATED, TILED_GAMMA, AERIAL)
POOL = (CONTROL, ROTATED, ENSEMBLE, YOLO_FULL, YOLO_NANO, *NEW)


def candidates(existing, cabinet, weights: Path):
    by_id = {s.config_id: s for s in existing}
    result = [by_id[c] for c in POOL if c not in NEW]
    control = by_id[CONTROL]
    for name, turn, gamma in ((TILED, 0, 1.0), (TILED_ROTATED, 1, 1.0), (TILED_GAMMA, 0, 0.5)):
        result.append(
            replace(
                control,
                config_id=name,
                model_strategy_id=name.lower(),
                kind=KIND,
                parameters=dict(
                    pipeline="fixed_tiled_yolo",
                    checkpoint=str((weights / "yolo11s-seg.pt").resolve()),
                    checkpoint_sha256=sha(weights / "yolo11s-seg.pt"),
                    tile_input_size=640,
                    grid=2,
                    overlap=0.15,
                    turns=turn,
                    gamma=gamma,
                    score_threshold=0.5,
                    device="cpu",
                ),
                mission_latency_s=1.0,
                energy_j_per_call=1.0,
            )
        )
    result.append(
        replace(
            cabinet,
            config_id=AERIAL,
            model_strategy_id="cabinet_uavid_small_1280",
            parameters={**cabinet.parameters, "input_width_px": 1280, "input_height_px": 960},
            mission_latency_s=1.0,
            energy_j_per_call=1.0,
        )
    )
    return tuple(result)


def tiles(shape, overlap=0.15):
    h, w = shape
    if h < 2 or w < 2 or not 0 <= overlap < 0.5:
        raise ValueError("Invalid image dimensions or tile overlap")
    # Every pixel is covered; overlap is a fixed fraction of a half-image tile.
    dh, dw = round(h * overlap / 2), round(w * overlap / 2)
    return [
        (
            max(0, y * h // 2 - dh),
            min(h, (y + 1) * h // 2 + dh),
            max(0, x * w // 2 - dw),
            min(w, (x + 1) * w // 2 + dw),
        )
        for y in range(2)
        for x in range(2)
    ]


def tiled_mask(rgb, predictor, *, turns=0, gamma=1.0):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or turns not in (0, 1):
        raise ValueError("Expected RGB and fixed 0/90-degree view")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Positive finite gamma required")
    image = rgb
    if gamma != 1.0:
        lut = np.round((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
        image = lut[rgb]
    result = np.zeros(rgb.shape[:2], dtype=bool)
    for top, bottom, left, right in tiles(rgb.shape[:2]):
        prediction, _ = predictor.predict(image[top:bottom, left:right].copy(), turns)
        if prediction.dtype != np.bool_ or prediction.shape != (bottom - top, right - left):
            raise ValueError("Tiled prediction must use original sensor coordinates")
        result[top:bottom, left:right] |= prediction
    return result


class InferencePool:
    def __init__(self, specs):
        self.specs = {s.config_id: s for s in specs}
        self.executors = {}
        self.yolos = {}
        for s in specs:
            if s.config_id == ENSEMBLE:
                continue
            if s.config_id == ROTATED:
                base = self.specs[CONTROL]
                actual = replace(
                    base,
                    parameters={**base.parameters, "input_width_px": 960, "input_height_px": 1280},
                )
                self.executors[s.config_id] = build_executors([actual])[actual.config_id]
            elif s.config_id in (YOLO_FULL, YOLO_NANO):
                self.yolos[s.config_id] = Segmenter(
                    Path(s.parameters["checkpoint"]), s.parameters["imgsz"]
                )
            elif s.kind == KIND:
                if "tiles" not in self.yolos:
                    self.yolos["tiles"] = Segmenter(Path(s.parameters["checkpoint"]), 640)
            else:
                self.executors[s.config_id] = build_executors([s])[s.config_id]

    def infer(self, rgb, configs=None):
        selected = tuple(configs) if configs is not None else tuple(self.specs)
        masks, times, calls = {}, {}, {}
        for name in selected:
            if name == ENSEMBLE:
                continue
            started = time.perf_counter()
            spec = self.specs[name]
            if spec.kind == KIND:
                masks[name] = tiled_mask(
                    rgb,
                    self.yolos["tiles"],
                    turns=spec.parameters["turns"],
                    gamma=spec.parameters["gamma"],
                )
                calls[name] = 4
            elif name in (YOLO_FULL, YOLO_NANO):
                masks[name], _ = self.yolos[name].predict(rgb, 0)
                calls[name] = 1
            else:
                frame = np.rot90(rgb, 1).copy() if name == ROTATED else rgb.copy()
                mask = self.executors[name].run(frame).prediction_mask
                masks[name] = np.rot90(mask, -1).copy() if name == ROTATED else mask
                calls[name] = 1
            times[name] = time.perf_counter() - started
        if ENSEMBLE in selected:
            started = time.perf_counter()
            masks[ENSEMBLE] = np.logical_or(masks[CONTROL], masks[ROTATED])
            times[ENSEMBLE] = times[CONTROL] + times[ROTATED] + time.perf_counter() - started
            calls[ENSEMBLE] = 0  # Derived from two actual calls already recorded above.
        return masks, times, calls
