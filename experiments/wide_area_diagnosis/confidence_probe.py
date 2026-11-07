"""A fixed detector-confidence intervention, never a relaxed mission criterion.

The original 48 geometry-selected development frames are retained. Detector
confidence changes from 0.5 to 0.25; mask probability, IoU, component area,
repeat evidence and false-positive limits remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import time

import numpy as np

from missionbench.v2.executors import build_executors
from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_fast_perception import Segmenter, rgb_to_bgr_view, restore_union
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.mission_runtime_profile import read, save, sha
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.wide_area_diagnosis.model_probe import verify, score_frame, frame_key

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'results/wide_area_image_probe_v1_20261028'
OUTPUT = ROOT / 'results/wide_area_confidence_probe_v1_20261028'
MODELS = ('CFG_YOLO11S_960_ROT0', 'CFG_MASKRCNN_1280')


def low_confidence_yolo(runner, rgb):
    bgr = rgb_to_bgr_view(rgb, 0)
    result = runner.model.predict(source=bgr, device='cpu', imgsz=runner.size,
        conf=.25, iou=.7, classes=[runner.target], max_det=300, retina_masks=True,
        rect=True, half=False, augment=False, verbose=False, save=False, stream=False)[0]
    masks = None if result.masks is None else result.masks.data.cpu().numpy()
    return restore_union(masks, result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                         runner.target, .25, bgr.shape[:2], 0)


def windows(frames, scores):
    groups = {}
    for frame in frames:
        for sector in frame['sector_ids']:
            key = frame['flight'] + '/' + sector
            group = groups.setdefault(key, dict(evidence={}, false_components=0, frames=0))
            score = scores[frame_key(frame)]
            group['frames'] += 1
            group['false_components'] += score['false_positive_components']
            for target in score['matched_target_ids']:
                if target.startswith(frame['flight'] + '_' + sector + '_'):
                    group['evidence'].setdefault(target, []).append(frame['mission_time_s'])
    for group in groups.values():
        group['repeated_people'] = sum(len(set(times)) >= 2 and max(times)-min(times) >= 2
                                       for times in group['evidence'].values())
        group['sector_pass'] = group['repeated_people'] >= 2 and group['false_components'] <= 2
    return groups


def improvement_gate(baseline, candidate):
    return [key for key in baseline if candidate[key]['repeated_people'] > baseline[key]['repeated_people']
            and candidate[key]['false_components'] <= 2]


def prepare(output=OUTPUT, parent=PARENT):
    output, parent = Path(output), Path(parent)
    if output.exists():
        raise FileExistsError('Preserve prior confidence intervention outputs')
    source = read(parent / 'plan.json')
    verify(source)
    if len(source['frames']) != 48 or len({frame_key(f) for f in source['frames']}) != 48:
        raise ValueError('Exactly 48 distinct declared frames required')
    counts = {}
    for frame in source['frames']:
        for sector in frame['sector_ids']:
            key = (frame['flight'], sector)
            counts[key] = counts.get(key, 0) + 1
    if len(counts) != 6 or set(counts.values()) != {8}:
        raise ValueError('Six complete eight-frame windows required')
    specs = [replace(ExecutorConfigSpec(**s), parameters={**s['parameters'], 'score_threshold': .25})
             for s in source['executor_specs'] if s['config_id'] in MODELS]
    output.mkdir(parents=True)
    plan = dict(schema_version='wide_area_detector_confidence_development_v1',
        parent_plan=str(parent / 'plan.json'), parent_plan_sha256=sha(parent / 'plan.json'),
        implementation_sha256=sha(__file__), models=list(MODELS),
        candidate_specs=[asdict(s) for s in specs],
        candidate_model_signatures={s.config_id: model_signature(s) for s in specs},
        selected_frames=source['frames'], selection='The same 48 geometric full-window frames; no new filtering',
        detector_confidence=dict(baseline=.5, candidate=.25),
        unchanged=dict(mask_probability=.5, matching_iou=.3, min_component_px=4,
                       min_hits=2, min_span_s=2, required_people=2, false_component_limit=2),
        decision='Additional repeated true people in at least one full window without violating its false-component cap authorizes paid testing only; report every window including regressions',
        development_only=True, evaluator_only_scoring=True, no_gt_in_model=True,
        timing='Actual candidate wall times under concurrent host load are descriptive; historical baseline cache is not a paired latency measurement. No operational cost profile is claimed.',
        note='Confidence is an instance detector class-score setting. CABiNet semantic classification is not assigned an artificial equivalent cutoff.')
    save(output / 'plan.json', plan)
    return plan


def execute(output=OUTPUT):
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    output = Path(output)
    plan = read(output / 'plan.json')
    if sha(__file__) != plan['implementation_sha256'] or sha(plan['parent_plan']) != plan['parent_plan_sha256']:
        raise ValueError('Confidence experiment changed after lock')
    parent = read(plan['parent_plan']); verify(parent)
    results = {}
    for name in MODELS:
        spec = ExecutorConfigSpec(**next(s for s in parent['executor_specs'] if s['config_id'] == name))
        runner = (Segmenter(Path(spec.parameters['checkpoint']), spec.parameters['imgsz'])
                  if name == MODELS[0] else build_executors((replace(spec, parameters={
                      **spec.parameters, 'score_threshold': .25}),))[name])
        cached = {flight: VerifiedMasks(Path(info['directory']), read(info['manifest'])['models'][name])
                  for flight, info in parent['baseline'].items()}
        folder = output / name; folder.mkdir(exist_ok=False)
        baseline, candidate, receipts = {}, {}, []
        for number, frame in enumerate(parent['frames'], 1):
            if sha(frame['source_path']) != frame['source_sha256']:
                raise ValueError('Original image changed')
            with np.load(frame['source_path'], allow_pickle=False) as data:
                rgb = data['rgb'].copy()
            key = frame_key(frame)
            before = cached[frame['flight']][rgb_key(rgb)]
            started = time.perf_counter()
            after = low_confidence_yolo(runner, rgb) if name == MODELS[0] else runner.run(rgb).prediction_mask
            elapsed = time.perf_counter() - started
            if after.dtype != np.bool_ or after.shape != rgb.shape[:2]:
                raise ValueError('Native prediction geometry changed')
            prediction = folder / (key + '.npz')
            np.savez_compressed(prediction, prediction_mask=after)
            baseline[key] = score_frame(parent, frame, before)
            candidate[key] = score_frame(parent, frame, after)
            receipt = dict(frame=frame, prediction_file=prediction.name, prediction_sha256=sha(prediction),
                mask_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
                plan_sha256=sha(output / 'plan.json'), measured_wall_clock_s=elapsed, scored_calls=1)
            save(folder / (key + '.json'), receipt); receipts.append(receipt)
            if number % 8 == 0:
                print(f'{name}: {number}/48 confidence pairs', flush=True)
        old, new = windows(parent['frames'], baseline), windows(parent['frames'], candidate)
        results[name] = dict(baseline_scores=baseline, candidate_scores=candidate,
            baseline_windows=old, candidate_windows=new,
            additional_supported_windows_without_fp_violation=improvement_gate(old, new),
            timing_p50_s=float(np.median([r['measured_wall_clock_s'] for r in receipts])),
            timing_p95_s=float(np.quantile([r['measured_wall_clock_s'] for r in receipts], .95)),
            receipts=receipts)
        save(folder / 'summary.json', results[name])
    verify(parent)
    save(output / 'completion.json', dict(complete=True, plan_sha256=sha(output / 'plan.json'),
        models=results, scored_neural_calls=96, mission_repair_established=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'execute'])
    p.add_argument('--output', type=Path, default=OUTPUT)
    args = p.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
    else:
        execute(args.output)
