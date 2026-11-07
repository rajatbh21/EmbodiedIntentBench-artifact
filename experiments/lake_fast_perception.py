"""RGB-only CPU segmentation study on complete, immutable lake recordings.

Optional Ultralytics dependencies and checkpoints live outside the core package.
No GT, crops selected by GT, or adaptive view selection enter neural inference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.run import freeze
from experiments.lake_missions.scenario import load
from experiments.lake_rotation_study import CONTROL, union_masks
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key

KIND = "lake_ultralytics_segmentation"
FAMILIES = (("YOLO11N_640", "yolo11n-seg.pt", 640), ("YOLO11S_960", "yolo11s-seg.pt", 960))


def candidates(control, weights):
    specs = []
    for family, filename, size in FAMILIES:
        for views in ("0", "90", "0,90"):
            specs.append(
                replace(
                    control,
                    config_id=f"CFG_{family}_ROT{views.replace(',', '_')}",
                    kind=KIND,
                    model_strategy_id=f"{family.lower()}_views_{views}",
                    mission_latency_s=1.0,  # Placeholder; measured profiles replace before replay.
                    energy_j_per_call=30.0 * len(views.split(",")),
                    parameters=dict(
                        checkpoint=str((weights / filename).resolve()),
                        checkpoint_sha256=sha(weights / filename),
                        imgsz=size,
                        fixed_views_ccw=views,
                        score_threshold=0.5,
                        nms_iou=0.7,
                        max_det=300,
                        retina_masks=True,
                        rect=True,
                        device="cpu",
                        dtype="float32",
                        task_class="person",
                    ),
                )
            )
    return specs


def rgb_to_bgr_view(rgb, quarter_turns):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected uint8 RGB image")
    if quarter_turns not in (0, 1):
        raise ValueError("Only predeclared 0/90 views supported")
    # Ultralytics numpy inputs use BGR. Copy also removes negative strides.
    return np.rot90(rgb, quarter_turns)[..., ::-1].copy()


def restore_union(masks, classes, scores, target, threshold, rotated_shape, turns):
    """Union native-resolution predicted instance masks; no box rasterization."""
    classes, scores = np.asarray(classes), np.asarray(scores)
    if classes.ndim != 1 or scores.shape != classes.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid detection metadata")
    if masks is None:
        if len(classes):
            raise ValueError("Detections without segmentation masks")
        union = np.zeros(rotated_shape, dtype=bool)
    else:
        masks = np.asarray(masks)
        if masks.shape != (len(classes), *rotated_shape) or not np.isfinite(masks).all():
            raise ValueError("Native mask geometry required; refuse silent resize")
        if not np.isin(masks, (0, 1)).all():
            raise ValueError("Expected binary native masks from retina_masks")
        keep = (classes == target) & (scores >= threshold)
        union = np.any(masks[keep].astype(bool), axis=0)
    return np.rot90(union, -turns).copy()


class Segmenter:
    def __init__(self, checkpoint, size):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(str(checkpoint), task="segment")
        self.target = next(k for k, name in self.model.names.items() if name == "person")
        self.size = size
        # Predictor setup can reset Torch threads; fix again after initial warmup.
        self.predict(np.zeros((720, 960, 3), dtype=np.uint8), 0)
        torch.set_num_threads(4)
        self.predict(np.zeros((720, 960, 3), dtype=np.uint8), 0)
        self.predict(np.zeros((720, 960, 3), dtype=np.uint8), 1)

    def predict(self, rgb, turns):
        started = time.perf_counter()
        bgr = rgb_to_bgr_view(rgb, turns)
        result = self.model.predict(
            source=bgr,
            device="cpu",
            imgsz=self.size,
            conf=0.5,
            iou=0.7,
            classes=[self.target],
            max_det=300,
            retina_masks=True,
            rect=True,
            half=False,
            augment=False,
            verbose=False,
            save=False,
            stream=False,
        )[0]
        masks = None if result.masks is None else result.masks.data.cpu().numpy()
        mask = restore_union(
            masks,
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            self.target,
            0.5,
            bgr.shape[:2],
            turns,
        )
        return mask, time.perf_counter() - started


def initialize(source, prior, weights, output):
    freeze(source)
    suite = read(source / "suite.json")
    scenario = load(source / "scenario.json", suite["environments"]["day"], Path(suite["assets"]))
    control = next(s for s in scenario.executors if s.config_id == CONTROL)
    output.mkdir(parents=True, exist_ok=False)
    save(
        output / "protocol.json",
        dict(
            schema_version="1.0",
            source=str(source.resolve()),
            prior=str(prior.resolve()),
            source_lock_sha256=sha(source / "lock.json"),
            implementation_sha256=sha(Path(__file__)),
            configurations=[asdict(s) for s in candidates(control, weights)],
            scope="Complete-flight development on known failures; not held-out evaluation",
            prior_summary_sha256=sha(prior / "summary.json"),
            prior_costs_sha256=sha(prior / "costs.json"),
            policies="All six new fixed configurations, three old view controls, two view alternators",
            policy_rule="Fixed choice or alternate original/90 by decision count; no GT",
            expected_mission_runs=66,
            sampling="Every canonical capture, every candidate; no output-based promotion",
            input="Whole RGB only; fixed views; native person mask union; no GT input",
            ensemble="Always run both views; OR their restored masks; charge both passes",
            host_threads=4,
            warmup="Both orientations after predictor initialization, excluded",
            environments={
                env: dict(
                    recording_sha256=sha(source / env / "recording/recording.json"),
                    captures=len(read(source / env / "recording/recording.json")["frames"]),
                )
                for env in ("day", "dusk_haze")
            },
            runtime_versions={
                p: importlib.metadata.version(p)
                for p in ("ultralytics", "torch", "torchvision", "numpy", "opencv-python-headless")
            },
            upstream=dict(
                documentation="https://docs.ultralytics.com/models/yolo11/",
                inference="https://docs.ultralytics.com/modes/predict/",
                checkpoint_base="https://github.com/ultralytics/assets/releases/download/v8.3.0/",
                license="Ultralytics AGPL-3.0 / Enterprise; external optional dependency",
            ),
            timing_rule="Maximum environment p95 per view; ensemble sum plus union p95; ceil 10 ms",
            energy="Assumed 30 J per view, 60 J per ensemble; not measured",
            resources="Inherited 2400 MB, 0.3 s / 2 J load-switch assumptions; not measured",
        ),
    )


def infer(output, environment):
    import torch

    p = read(output / "protocol.json")
    assert p["implementation_sha256"] == sha(Path(__file__))
    source = Path(p["source"])
    freeze(source)
    assert p["source_lock_sha256"] == sha(source / "lock.json")
    suite = read(source / "suite.json")
    scenario = load(
        source / "scenario.json", suite["environments"][environment], Path(suite["assets"])
    )
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    from missionbench.v2.scenario import ExecutorConfigSpec

    specs = [ExecutorConfigSpec(**s) for s in p["configurations"]]
    segmenters = {}
    for family, filename, size in FAMILIES:
        spec = next(s for s in specs if s.config_id == f"CFG_{family}_ROT0")
        path = Path(spec.parameters["checkpoint"])
        assert sha(path) == spec.parameters["checkpoint_sha256"] and path.name == filename
        segmenters[family] = Segmenter(path, size)
    folder = output / environment / "predictions"
    folder.mkdir(parents=True, exist_ok=False)
    models = {s.config_id: dict(signature=model_signature(s), predictions={}) for s in specs}
    backend = ReplayBackend(scenario, source / environment / "recording")
    assert (
        backend.provenance["recording_sha256"] == p["environments"][environment]["recording_sha256"]
    )
    rows = []
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            rgb = backend.capture().rgb
            key = rgb_key(rgb)
            # Alternate family/view order deterministically to reduce timing order bias.
            families = list(segmenters) if index % 2 == 0 else list(reversed(segmenters))
            for family in families:
                masks, timings = {}, {}
                for turns in (0, 1) if index % 2 == 0 else (1, 0):
                    masks[turns], timings[turns] = segmenters[family].predict(rgb, turns)
                start = time.perf_counter()
                masks[2] = union_masks(masks[0], masks[1])
                union_s = time.perf_counter() - start
                timings[2] = timings[0] + timings[1] + union_s
                for view, suffix in enumerate(("0", "90", "0_90")):
                    config = f"CFG_{family}_ROT{suffix}"
                    mask = masks[view]
                    path = folder / f"{index:04d}_{config}.npz"
                    np.savez_compressed(path, prediction_mask=mask)
                    item = dict(
                        file=path.name,
                        sha256=sha(path),
                        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                    )
                    table = models[config]["predictions"]
                    if key in table and table[key]["mask_sha256"] != item["mask_sha256"]:
                        raise ValueError(
                            "Stateless cache requires identical output for identical RGB"
                        )
                    table[key] = item
                    rows.append(
                        dict(
                            capture_id=index,
                            timestamp_ns=timestamp,
                            capture_time_s=(timestamp - scenario.schedule.epoch_ns) / 1e9,
                            config_id=config,
                            rgb_sha256=key,
                            measured_wall_clock_s=timings[view],
                            union_s=union_s if view == 2 else 0.0,
                            timing_scope="sum of same-frame measured constituent calls plus union"
                            if view == 2
                            else "RGB conversion through restored native mask",
                        )
                    )
            if index % 10 == 0:
                print(
                    f"{environment}: {index + 1}/{len(backend.frames)} complete captures",
                    flush=True,
                )
                save(folder / "progress.json", dict(captures=index + 1))
    finally:
        backend.close()
    assert len(rows) == p["environments"][environment]["captures"] * len(specs)
    save(folder / "inference.json", rows)
    save(
        folder / "manifest.json",
        dict(
            cache_schema_version="0.1",
            recording_identity=scenario.recording_identity,
            recording_sha256=backend.provenance["recording_sha256"],
            execution_mode="precomputed_real_model_predictions",
            capture_count=len(backend.frames),
            models=models,
            protocol_sha256=sha(output / "protocol.json"),
            inference_sha256=sha(folder / "inference.json"),
        ),
    )
    save(
        folder / "completion.json",
        dict(
            complete=True,
            actual_neural_calls=len(backend.frames) * 4,
            derived_ensembles=len(backend.frames) * 2,
            manifest_sha256=sha(folder / "manifest.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init")
    for name in ("source", "prior", "weights", "output"):
        init.add_argument(name, type=Path)
    run = commands.add_parser("infer")
    run.add_argument("output", type=Path)
    run.add_argument("--environment", choices=("day", "dusk_haze"), required=True)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.source, args.prior, args.weights, args.output)
    else:
        infer(args.output, args.environment)
