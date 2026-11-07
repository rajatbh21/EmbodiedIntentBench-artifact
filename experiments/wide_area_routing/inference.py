"""Freeze geometric task windows and infer every model on their original RGB.

This is a bounded sector-inspection pilot, not dense whole-flight inference.
Window membership uses authored pass times only, never visibility or predictions.
The complete recording identity and original capture indices remain unchanged.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_missions.assets import sha
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key

FLIGHTS = ("shoreline_loop", "wide_area_search", "sector_revisit")
DEFAULT_MODELS = Path("results/routing_development_matrix_v1_20261017/east_shore_day/rapid_survey/scenario.json")
DEFAULT_PLAN = Path("results/mission_joint_constraints_v1_20261027/plan.json")
DEFAULT_COHORT = Path("results/wide_area_missions_v2_20261028")


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def freeze_json(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f"Frozen specification changed: {path}")
    else:
        atomic_json(path, value)


def selected_captures(recording, design, half_window_s=8.0):
    if not np.isfinite(half_window_s) or half_window_s <= 0:
        raise ValueError("Positive finite window required")
    schedule = recording["capture_schedule"]
    expected = list(range(schedule["epoch_ns"], schedule["end_ns"], schedule["interval_ns"]))
    if [frame["timestamp_ns"] for frame in recording["frames"]] != expected:
        raise ValueError("Canonical recording capture clock is incomplete")
    windows = [{"sector_id": s["sector_id"], "nominal_pass_time_s": s["nominal_pass_time_s"],
                "start_s": s["nominal_pass_time_s"] - half_window_s,
                "end_s": s["nominal_pass_time_s"] + half_window_s} for s in design["sectors"]]
    if not windows or len({w["sector_id"] for w in windows}) != len(windows):
        raise ValueError("Unique authored sectors required")
    selected = []
    for index, frame in enumerate(recording["frames"]):
        relative = (frame["timestamp_ns"] - schedule["epoch_ns"]) / 1e9
        sectors = [w["sector_id"] for w in windows if w["start_s"] <= relative <= w["end_s"]]
        if sectors:
            selected.append({"capture_id": index, "timestamp_ns": frame["timestamp_ns"],
                             "mission_time_s": relative, "sector_ids": sectors,
                             "source_file": frame["file"], "source_sha256": frame["sha256"]})
    if not selected:
        raise ValueError("No canonical captures in authored windows")
    return windows, selected


def source_hashes():
    paths = [Path(__file__).resolve(), Path("experiments/mission_routing_v2/models.py"),
             Path("experiments/lake_fast_perception.py"), Path("missionbench/v2/executors.py"),
             Path("experiments/v4_benchmark/calibrated_router.py"),
             Path("experiments/v4_benchmark/probe_cache.py")]
    return {str(p.resolve()): sha(p) for p in paths}


def verified_model_specs(scenario_path, parent_plan):
    specs = tuple(ExecutorConfigSpec(**s) for s in read(scenario_path)["executor_configs"])
    signatures = {s.config_id: model_signature(s) for s in specs}
    if len(specs) != 9 or signatures != read(parent_plan)["model_signatures"]:
        raise ValueError("Model pool differs from the frozen parent signatures")
    assets = {}
    for spec in specs:
        params = spec.parameters
        if "checkpoint" in params:
            path = Path(params["checkpoint"])
            if sha(path) != params["checkpoint_sha256"]:
                raise ValueError("Pretrained checkpoint changed")
            assets[str(path.resolve())] = sha(path)
        if "bundle_manifest" in params:
            manifest = Path(params["bundle_manifest"])
            if sha(manifest) != params["bundle_sha256"]:
                raise ValueError("CABiNet bundle manifest changed")
            assets[str(manifest.resolve())] = sha(manifest)
            for relative, expected in read(manifest)["files"].items():
                path = manifest.parent / relative
                if sha(path) != expected:
                    raise ValueError("CABiNet bundle file changed")
                assets[str(path.resolve())] = expected
    torch_home = Path(os.environ.get("TORCH_HOME", ".cache/torch"))
    checkpoint = torch_home / "hub/checkpoints/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth"
    actual = sha(checkpoint)
    if not actual.startswith("73cbd019"):
        raise ValueError("Cached official Mask R-CNN checkpoint is absent or invalid")
    assets[str(checkpoint.resolve())] = actual
    return specs, assets


def prepare(output, cohort=DEFAULT_COHORT, scenario_path=DEFAULT_MODELS, parent_plan=DEFAULT_PLAN):
    specs, assets = verified_model_specs(scenario_path, parent_plan)
    output, cohort = Path(output).resolve(), Path(cohort).resolve()
    folder = output / "predictions"
    folder.mkdir(parents=True, exist_ok=True)
    spec_document = {"schema_version": "1.0", "source_files": source_hashes(),
                     "models_source": str(Path(scenario_path).resolve()),
                     "models_source_sha256": sha(scenario_path), "parent_plan": str(Path(parent_plan).resolve()),
                     "parent_plan_sha256": sha(parent_plan), "model_assets": assets,
                     "executor_configs": [asdict(s) for s in specs],
                     "model_signatures": {s.config_id: model_signature(s) for s in specs},
                     "window_half_width_s": 8.0, "selection_uses": "Authored nominal pass times and canonical capture clock only",
                     "model_inputs": "Original RGB array only", "expected_neural_calls_per_frame": 17,
                     "timing_scope": "Measured host wall times are diagnostics, not mission execution costs",
                     "workers": 2, "torch_threads_per_worker": 4}
    freeze_json(folder / "specification.json", spec_document)
    selections = {}
    for flight in FLIGHTS:
        flight_dir = cohort / flight
        recording_path = flight_dir / "recording/recording.json"
        recording = read(recording_path)
        provenance = recording["provenance"]
        if not provenance["full_declared_mission"] or not provenance["physical_route_passed"]:
            raise ValueError("Only complete validated physical flights may be inferred")
        design_path = flight_dir / "mission_design.json"
        windows, frames = selected_captures(recording, read(design_path))
        selection = {"flight": flight, "recording_path": str(recording_path),
                     "recording_identity": recording["recording_identity"], "recording_sha256": sha(recording_path),
                     "mission_design_path": str(design_path), "mission_design_sha256": sha(design_path),
                     "capture_schedule": recording["capture_schedule"], "full_recording_capture_count": len(recording["frames"]),
                     "selection_rule": "Closed union of authored sector nominal pass time +/- 8 seconds",
                     "selection_independent_of_gt_and_predictions": True, "windows": windows,
                     "capture_count": len(frames), "frames": frames,
                     "specification_sha256": sha(folder / "specification.json")}
        destination = folder / flight
        destination.mkdir(exist_ok=True)
        freeze_json(destination / "selection.json", selection)
        selections[flight] = {"path": str(destination / "selection.json"),
                              "sha256": sha(destination / "selection.json"), "capture_count": len(frames)}
    freeze_json(folder / "selection_index.json", selections)
    return selections


def verify_frozen(folder):
    folder = Path(folder)
    spec = read(folder.parent / "specification.json")
    if spec["source_files"] != source_hashes():
        raise ValueError("Inference implementation changed since selection was frozen")
    for category in ("source_files", "model_assets"):
        for path, expected in spec[category].items():
            if sha(Path(path)) != expected:
                raise ValueError(f"Frozen {category} mismatch: {path}")
    for name in ("models_source", "parent_plan"):
        if sha(Path(spec[name])) != spec[name + "_sha256"]:
            raise ValueError("Frozen model specification source changed")
    selection = read(folder / "selection.json")
    index = read(folder.parent / "selection_index.json")[selection["flight"]]
    if sha(folder / "selection.json") != index["sha256"]:
        raise ValueError("Frozen frame selection changed")
    if selection["specification_sha256"] != sha(folder.parent / "specification.json"):
        raise ValueError("Frozen inference specification changed")
    for name in ("recording", "mission_design"):
        if sha(Path(selection[name + "_path"])) != selection[name + "_sha256"]:
            raise ValueError("Frozen physical source changed")
    return spec, selection


def verified_journal(folder, frame, config_ids):
    path = folder / f"frame_{frame['capture_id']:05d}.json"
    if not path.exists():
        return None
    checksum = path.with_suffix(".sha256")
    if not checksum.exists():
        # An interrupted atomic journal commit is not a completed frame.
        return None
    if checksum.read_text().strip() != sha(path):
        raise ValueError("Completed frame journal checksum mismatch")
    item = read(path)
    if item["source"] != frame or set(item["predictions"]) != set(config_ids):
        raise ValueError("Completed frame journal binding mismatch")
    for prediction in item["predictions"].values():
        mask_path = (folder / prediction["file"]).resolve()
        if mask_path.parent != folder.resolve() or sha(mask_path) != prediction["sha256"]:
            raise ValueError("Cached prediction bytes changed")
        with np.load(mask_path, allow_pickle=False) as data:
            mask = data["prediction_mask"]
        if mask.dtype != np.bool_ or list(mask.shape) != item["rgb_shape"][:2]:
            raise ValueError("Cached prediction geometry changed")
        if hashlib.sha256(mask.tobytes()).hexdigest() != prediction["mask_sha256"]:
            raise ValueError("Cached prediction mask changed")
    return item


def infer_flight(folder):
    import torch
    from experiments.mission_routing_v2.models import InferencePool

    folder = Path(folder)
    spec, selection = verify_frozen(folder)
    configs = tuple(ExecutorConfigSpec(**s) for s in spec["executor_configs"])
    ids = tuple(s.config_id for s in configs)
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    pool = None
    journals = []
    begun = time.perf_counter()
    for number, frame in enumerate(selection["frames"]):
        recording_dir = Path(selection["recording_path"]).parent
        source = (recording_dir / frame["source_file"]).resolve()
        if source.parent != recording_dir.resolve() or sha(source) != frame["source_sha256"]:
            raise ValueError("Original acquisition frame changed")
        # Deliberately never access object_ids or other evaluator-only NPZ entries.
        with np.load(source, allow_pickle=False) as data:
            rgb = data["rgb"].copy()
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected native uint8 RGB")
        key = rgb_key(rgb)
        journal = verified_journal(folder, frame, ids)
        if journal is not None:
            if journal["rgb_key"] != key:
                raise ValueError("Cached RGB binding changed")
            journals.append(journal)
            continue
        if pool is None:
            pool = InferencePool(configs)
        masks, times, calls = pool.infer(rgb)
        if set(masks) != set(ids) or sum(calls.values()) != 17:
            raise ValueError("Missing model outputs or unexpected call accounting")
        predictions = {}
        for config in ids:
            mask = masks[config]
            if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
                raise ValueError("Invalid native segmentation output")
            path = folder / f"prediction_{frame['capture_id']:05d}_{config}.npz"
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, prediction_mask=mask)
            temporary.replace(path)
            predictions[config] = {"file": path.name, "sha256": sha(path),
                                   "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                                   "measured_wall_clock_s": times[config], "actual_neural_calls": calls[config]}
        journal = {"source": frame, "rgb_key": key, "rgb_shape": list(rgb.shape), "predictions": predictions}
        journal_path = folder / f"frame_{frame['capture_id']:05d}.json"
        atomic_json(journal_path, journal)
        temporary = journal_path.with_suffix(".sha256.tmp")
        temporary.write_text(sha(journal_path) + "\n")
        temporary.replace(journal_path.with_suffix(".sha256"))
        journals.append(journal)
        atomic_json(folder / "progress.json", {"captures": number + 1, "expected": len(selection["frames"]),
                                               "elapsed_wall_clock_s": time.perf_counter() - begun})
        print(f"{selection['flight']}: {number + 1}/{len(selection['frames'])}", flush=True)
    models = {c: {"signature": spec["model_signatures"][c], "predictions": {}} for c in ids}
    rows = []
    for journal in journals:
        for config, prediction in journal["predictions"].items():
            previous = models[config]["predictions"].get(journal["rgb_key"])
            if previous and previous["mask_sha256"] != prediction["mask_sha256"]:
                raise ValueError("Identical RGB produced inconsistent native masks")
            models[config]["predictions"][journal["rgb_key"]] = prediction
            rows.append({"capture_id": journal["source"]["capture_id"], "config_id": config,
                         "timestamp_ns": journal["source"]["timestamp_ns"], "rgb_key": journal["rgb_key"],
                         "measured_wall_clock_s": prediction["measured_wall_clock_s"],
                         "actual_neural_calls": prediction["actual_neural_calls"]})
    verify_frozen(folder)
    atomic_json(folder / "inference.json", rows)
    atomic_json(folder / "manifest.json", {"cache_schema_version": "0.1", "models": models,
                "recording_identity": selection["recording_identity"], "recording_sha256": selection["recording_sha256"],
                "selection_sha256": sha(folder / "selection.json"),
                "specification_sha256": sha(folder.parent / "specification.json"),
                "inference_sha256": sha(folder / "inference.json"), "capture_count": len(journals),
                "full_recording_capture_count": selection["full_recording_capture_count"],
                "scope": "Authored sector task windows; not dense whole-flight inference"})
    atomic_json(folder / "completion.json", {"complete": True, "manifest_sha256": sha(folder / "manifest.json"),
                "actual_neural_calls": sum(r["actual_neural_calls"] for r in rows),
                "frame_model_outputs": len(rows), "captures": len(journals)})
    return {"flight": selection["flight"], "captures": len(journals), "outputs": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    selections = prepare(args.output, args.cohort)
    print(json.dumps({f: s["capture_count"] for f, s in selections.items()}), flush=True)
    if not args.prepare_only:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as workers:
            futures = [workers.submit(infer_flight, args.output / "predictions" / f) for f in FLIGHTS]
            for future in futures:
                print(future.result(), flush=True)


if __name__ == "__main__":
    main()
