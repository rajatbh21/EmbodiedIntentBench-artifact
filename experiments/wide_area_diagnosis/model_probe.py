"""Posthoc development probe of fixed RGB scale/context interventions.

The 48-frame geometric set and its 12-frame first stage are frozen before any
new inference. A larger model input adds no new sensor information; uniform
tiling also changes context. Neither intervention establishes domain causality.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import resource
import time

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import build_executors
from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_missions.assets import sha
from experiments.mission_routing_v2.models import tiles
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import EvaluationFrame
from experiments.wide_area_routing.inference import atomic_json, freeze_json, read

REPO = Path(__file__).resolve().parents[2]
MODELS = ("CFG_MASKRCNN_1280", "CFG_YOLO11S_960_ROT0", "CFG_CABINET_UAVID_1280")
ARMS = ("full_2x", "uniform_tiles_2x2")


def geometric_selection(selection, design):
    """Two full authored sector windows; two nearest captures per window first."""
    result = []
    for sector in (design["sectors"][0], design["sectors"][4]):
        candidates = [f for f in selection["frames"] if sector["sector_id"] in f["sector_ids"]]
        if len(candidates) != 8:
            raise ValueError("Expected exactly eight frozen captures in each sector window")
        first = sorted(candidates, key=lambda f: (
            abs(f["mission_time_s"] - sector["nominal_pass_time_s"]), f["capture_id"]))[:2]
        first_ids = {f["capture_id"] for f in first}
        for frame in candidates:
            result.append({**frame, "flight": selection["flight"],
                           "stage_one": frame["capture_id"] in first_ids,
                           "source_path": str(Path(selection["recording_path"]).parent / frame["source_file"])})
    if len({r["capture_id"] for r in result}) != 16:
        raise ValueError("Sector windows must be disjoint")
    return sorted(result, key=lambda r: r["capture_id"])


def promoted(baseline_scores, intervention_scores):
    if len(baseline_scores) != len(intervention_scores):
        raise ValueError("Paired frames required")
    tp_gain = sum(b["matched_components"] - a["matched_components"]
                  for a, b in zip(baseline_scores, intervention_scores))
    extra_fp = sum(b["false_positive_components"] - a["false_positive_components"]
                   for a, b in zip(baseline_scores, intervention_scores))
    return {"net_paired_true_positive_gain": tp_gain, "extra_false_positive_components": extra_fp,
            "expand_to_all_48": tp_gain > 0 and extra_fp <= tp_gain}


def implementation_hashes():
    names = ["experiments/wide_area_diagnosis/model_probe.py",
             "experiments/wide_area_routing/inference.py",
             "experiments/mission_routing_v2/models.py", "experiments/lake_fast_perception.py",
             "missionbench/v2/executors.py", "missionbench/v2/instance_models.py",
             "missionbench/v2/real_models.py", "missionbench/v2/cabinet_models.py",
             "missionbench/v2/evaluation.py", "missionbench/v2/world.py"]
    return {str(REPO / name): sha(REPO / name) for name in names}


def prepare(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    source_spec = read(study / "predictions/specification.json")
    cohort = Path(read(study / "plan.json")["cohort"])
    specs = [s for s in source_spec["executor_configs"] if s["config_id"] in MODELS]
    files = {str(study / "plan.json"): sha(study / "plan.json"),
             str(study / "predictions/specification.json"): sha(study / "predictions/specification.json")}
    frames, targets, baseline = [], {}, {}
    for flight in ("shoreline_loop", "wide_area_search", "sector_revisit"):
        selection_path = study / "predictions" / flight / "selection.json"
        design_path = cohort / flight / "mission_design.json"
        scenario_path = cohort / flight / "scenario.json"
        recording_path = cohort / flight / "recording/recording.json"
        manifest_path = study / "predictions" / flight / "manifest.json"
        complete_path = manifest_path.with_name("completion.json")
        for path in (selection_path, design_path, scenario_path, recording_path, manifest_path, complete_path):
            files[str(path)] = sha(path)
        selection = read(selection_path)
        if sha(recording_path) != selection["recording_sha256"]:
            raise ValueError("Original recording changed")
        complete, manifest = read(complete_path), read(manifest_path)
        if not complete["complete"] or complete["manifest_sha256"] != sha(manifest_path):
            raise ValueError("Original inference cache incomplete")
        for spec in specs:
            if manifest["models"][spec["config_id"]]["signature"] != model_signature(ExecutorConfigSpec(**spec)):
                raise ValueError("Original model signature mismatch")
        selected = geometric_selection(selection, read(design_path))
        frames.extend(selected)
        for frame in selected:
            if sha(Path(frame["source_path"])) != frame["source_sha256"]:
                raise ValueError("Original frame changed")
            files[frame["source_path"]] = frame["source_sha256"]
        targets[flight] = {str(o["label"]): o["object_id"] for o in read(scenario_path)["world"]["objects"]}
        baseline[flight] = {"directory": str(manifest_path.parent), "manifest": str(manifest_path),
                            "manifest_sha256": sha(manifest_path)}
    if len(frames) != 48 or sum(f["stage_one"] for f in frames) != 12:
        raise ValueError("Expected exactly 12 initial and 48 total geometric frames")
    plan = dict(schema_version="wide_area_image_intervention_development_v1", study=str(study),
                purpose="Posthoc development diagnosis; original outcomes were already known",
                holdout_claim=False, mission_repair_claim=False, frames=frames, targets=targets,
                executor_specs=specs, baseline=baseline, source_files=files,
                implementation_sha256=implementation_hashes(), model_assets=source_spec["model_assets"],
                variants={"baseline": "Frozen original full-image native mask cache",
                          "full_2x": "Twice original model input width and height; original entire RGB retained",
                          "uniform_tiles_2x2": "Every fixed 2x2 tile; 0.15 whole-image overlap; original model input canvas; OR native tile masks in source coordinates"},
                selection="Sectors 1 and 5 of each flight: all eight original window captures. First stage chooses two nearest nominal pass times per window, ties earlier original capture.",
                expansion="For each model/arm independently: sum paired matched-component differences >0 AND sum paired false-component differences <= that net TP gain. Then evaluate all 48 geometric frames; preserve every negative initial arm.",
                thresholds=dict(score=0.5, probability=0.5, matching_iou=0.3, min_component_px=4),
                accounting="Model initialization/warmup excluded from scored calls; measured wall times diagnostic only. Baseline times, if cited, belong to the historical run and are not paired timing controls. CPU RSS is cumulative process high-water mark, not isolated arm memory.",
                limitations=["Larger tensors add no camera information", "Tiling changes effective scale and visual context",
                             "Families differ in architecture, training data, and weights; this is not causal domain attribution",
                             "Twelve screening frames and dependent temporal followup are development evidence, not significance or held-out mission repair"],
                workers=2, torch_threads_per_worker=4)
    output.mkdir(parents=True, exist_ok=True)
    freeze_json(output / "plan.json", plan)
    return plan


def verify(plan):
    if implementation_hashes() != plan["implementation_sha256"]:
        raise ValueError("Probe implementation changed after lock")
    for category in ("source_files", "implementation_sha256", "model_assets"):
        for path, expected in plan[category].items():
            if sha(Path(path)) != expected:
                raise ValueError(f"Frozen {category} changed: {path}")


class InstrumentedModel:
    """Unmodified model executor with passive effective-tensor instrumentation."""
    def __init__(self, spec, doubled=False):
        self.tensor_shapes = []
        self.source_config = spec.config_id
        self.yolo = spec.config_id == "CFG_YOLO11S_960_ROT0"
        factor = 2 if doubled else 1
        if self.yolo:
            from experiments.lake_fast_perception import Segmenter
            self.runner = Segmenter(Path(spec.parameters["checkpoint"]), spec.parameters["imgsz"] * factor)
            network = self.runner.model.predictor.model.model
            self.metadata = dict(requested_imgsz=spec.parameters["imgsz"] * factor,
                                 network_stride=int(self.runner.model.predictor.model.stride),
                                 channel_order="RGB to BGR for API, restored to RGB by installed predictor",
                                 mask_source="retina_masks native segmentation tensor; never boxes")
        else:
            params = dict(spec.parameters)
            params["input_width_px"] *= factor
            params["input_height_px"] *= factor
            actual = replace(spec, parameters=params)
            self.runner = build_executors((actual,))[actual.config_id]
            network = self.runner._backend.model
            self.metadata = dict(requested_input_wh=[params["input_width_px"], params["input_height_px"]],
                                 channel_order="RGB", preprocessing=self.runner._backend.info.preprocessing_id)
            if hasattr(network, "transform"):
                self.metadata.update(transform_min_size=list(network.transform.min_size),
                                     transform_max_size=network.transform.max_size,
                                     padding_multiple=network.transform.size_divisible)
                network = network.backbone
        def hook(module, inputs):
            tensor = inputs[0]
            self.tensor_shapes.append(list(tensor.shape))
        self.hook = network.register_forward_pre_hook(hook)

    def predict(self, rgb):
        start = len(self.tensor_shapes)
        mask = self.runner.predict(rgb, 0)[0] if self.yolo else self.runner.run(rgb).prediction_mask
        shapes = self.tensor_shapes[start:]
        if len(shapes) != 1 or len(shapes[0]) != 4:
            raise ValueError("Expected one instrumented effective model tensor per call")
        return mask, shapes[0]


def infer_arm(rgb, runner, arm):
    if arm not in ARMS:
        raise ValueError("Undeclared image intervention")
    result = np.zeros(rgb.shape[:2], dtype=bool)
    windows = [(0, rgb.shape[0], 0, rgb.shape[1])] if arm == "full_2x" else tiles(rgb.shape[:2], 0.15)
    geometry = []
    started = time.perf_counter()
    for top, bottom, left, right in windows:
        patch = rgb[top:bottom, left:right].copy()
        mask, shape = runner.predict(patch)
        if mask.dtype != np.bool_ or mask.shape != patch.shape[:2]:
            raise ValueError("Model output must retain source tile geometry")
        result[top:bottom, left:right] |= mask
        geometry.append(dict(source_box_xyxy=[left, top, right, bottom],
                             source_rgb_shape=list(patch.shape), effective_tensor_shape=shape))
    return result, dict(measured_wall_clock_s=time.perf_counter() - started,
                        scored_neural_calls=len(windows), tensor_geometry=geometry,
                        process_high_water_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        model_runtime=runner.metadata)


def frame_key(frame):
    return frame["flight"] + f"_{frame['capture_id']:05d}"


def cached_prediction(output, model, arm, frame, plan_sha):
    journal = output / model / arm / (frame_key(frame) + ".json")
    if not journal.exists():
        return None
    item = read(journal)
    if item["plan_sha256"] != plan_sha or item["source_sha256"] != frame["source_sha256"]:
        raise ValueError("Prediction journal binding changed")
    if item["journal_content_sha256"] != hashlib.sha256(json.dumps(
            {k: v for k, v in item.items() if k != "journal_content_sha256"}, sort_keys=True).encode()).hexdigest():
        raise ValueError("Prediction journal checksum changed")
    path = journal.parent / item["prediction_file"]
    if path.parent != journal.parent or sha(path) != item["prediction_sha256"]:
        raise ValueError("Prediction file changed")
    with np.load(path, allow_pickle=False) as arrays:
        mask = arrays["prediction_mask"].copy()
    if mask.dtype != np.bool_ or hashlib.sha256(mask.tobytes()).hexdigest() != item["mask_sha256"]:
        raise ValueError("Native prediction changed")
    return mask, item


def score_frame(plan, frame, mask):
    with np.load(frame["source_path"], allow_pickle=False) as arrays:
        labels = arrays["object_ids"]
    target_ids = plan["targets"][frame["flight"]]
    present = set(map(int, np.unique(labels))) - {0}
    truth = EvaluationFrame(frame["capture_id"], frame["mission_time_s"], (labels > 0).astype(np.uint8),
                            labels, tuple(sorted(target_ids[str(label)] for label in present)),
                            {"instance_object_ids": target_ids})
    return MissionEvaluator(0.3, len(target_ids), 2.0, min_component_px=4).update(truth, mask).to_dict()


def model_probe(output, model_name):
    import torch
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    output = Path(output)
    plan, plan_sha = read(output / "plan.json"), sha(output / "plan.json")
    verify(plan)
    spec = ExecutorConfigSpec(**next(s for s in plan["executor_specs"] if s["config_id"] == model_name))
    masks = {flight: VerifiedMasks(Path(item["directory"]),
                    read(item["manifest"])["models"][model_name]) for flight, item in plan["baseline"].items()}
    baseline_scores, baseline_masks = {}, {}
    for frame in plan["frames"]:
        with np.load(frame["source_path"], allow_pickle=False) as arrays:
            rgb = arrays["rgb"]
        key = frame_key(frame)
        mask = masks[frame["flight"]][rgb_key(rgb)]
        baseline_scores[key], baseline_masks[key] = score_frame(plan, frame, mask), mask
    directory = output / model_name
    directory.mkdir(exist_ok=True)
    freeze_json(directory / "baseline.json", dict(plan_sha256=plan_sha, cached=True, scores=baseline_scores))
    runners, results = {}, {}
    first = [f for f in plan["frames"] if f["stage_one"]]
    for arm in ARMS:
        arm_dir = directory / arm
        arm_dir.mkdir(exist_ok=True)
        arm_scores = {}
        for stage in (1, 2):
            if stage == 2:
                decision = promoted([baseline_scores[frame_key(f)] for f in first],
                                    [arm_scores[frame_key(f)] for f in first])
                freeze_json(arm_dir / "screening.json", dict(plan_sha256=plan_sha, **decision))
                if not decision["expand_to_all_48"]:
                    break
            frames = first if stage == 1 else [f for f in plan["frames"] if not f["stage_one"]]
            for number, frame in enumerate(frames):
                key = frame_key(frame)
                cached = cached_prediction(output, model_name, arm, frame, plan_sha)
                if cached:
                    mask, journal = cached
                else:
                    if arm not in runners:
                        runners[arm] = InstrumentedModel(spec, doubled=arm == "full_2x")
                    with np.load(frame["source_path"], allow_pickle=False) as arrays:
                        rgb = arrays["rgb"].copy()
                    mask, timing = infer_arm(rgb, runners[arm], arm)
                    path = arm_dir / (key + ".npz")
                    temporary = path.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, prediction_mask=mask)
                    temporary.replace(path)
                    journal = dict(plan_sha256=plan_sha, source_path=frame["source_path"],
                                   source_sha256=frame["source_sha256"], rgb_key=rgb_key(rgb),
                                   prediction_file=path.name, prediction_sha256=sha(path),
                                   mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                                   stage=stage, **timing)
                    journal["journal_content_sha256"] = hashlib.sha256(
                        json.dumps(journal, sort_keys=True).encode()).hexdigest()
                    atomic_json(arm_dir / (key + ".json"), journal)
                arm_scores[key] = score_frame(plan, frame, mask)
                print(f"{model_name} {arm} stage{stage}: {number + 1}/{len(frames)}", flush=True)
        decision = read(arm_dir / "screening.json")
        evaluated = [f for f in plan["frames"] if frame_key(f) in arm_scores]
        paired = [{"frame": frame, "baseline": baseline_scores[frame_key(frame)],
                   "intervention": arm_scores[frame_key(frame)]} for frame in evaluated]
        results[arm] = dict(screening=decision, evaluated_frames=len(evaluated), paired_scores=paired,
                            total_paired_effect=promoted([p["baseline"] for p in paired],
                                                        [p["intervention"] for p in paired]))
        atomic_json(arm_dir / "summary.json", results[arm])
    verify(plan)
    report = dict(model=model_name, plan_sha256=plan_sha, baseline_cached=True, arms=results,
                  complete=True, scope="Posthoc development image diagnostic; no mission repair or holdout claim")
    atomic_json(directory / "completion.json", report)
    return {model_name: {arm: {"frames": r["evaluated_frames"], **r["screening"]}
                         for arm, r in results.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=REPO / "results/wide_area_routing_pilot_v1_20261028")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    plan = prepare(args.study, args.output)
    print(json.dumps({"plan": str(args.output / "plan.json"), "plan_sha256": sha(args.output / "plan.json"),
                      "first_stage_frames": sum(f["stage_one"] for f in plan["frames"]),
                      "full_followup_frames": len(plan["frames"])}), flush=True)
    if not args.prepare_only:
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn")) as workers:
            futures = [workers.submit(model_probe, args.output, name) for name in MODELS]
            for future in futures:
                print(json.dumps(future.result()), flush=True)


if __name__ == "__main__":
    main()
