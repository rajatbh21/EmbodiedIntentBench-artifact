"""Three-family native-RGB probe of predeclared physical development clips.

Acquisition, inference and evaluator data are separate. Inference opens only the
RGB NPZ entry and receives no labels, poses or object coordinates. Evaluation
uses all original world labels, then applies the unchanged focal-sector rule.
No output-dependent frame selection, threshold changes or full-mission claims.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.scenario import ExecutorConfigSpec
from experiments.lake_tracking import camera_ray
from experiments.mission_runtime_profile import read, sha
from experiments.v4_benchmark.backends import load_frame
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.wide_area_diagnosis.acquisition import verify as verify_acquisition
from experiments.wide_area_missions.runtime import digest, normalize_specification
from experiments.wide_area_routing.diagnostics import phase_evidence
from experiments.wide_area_routing.inference import (
    DEFAULT_MODELS, DEFAULT_PLAN, atomic_json, freeze_json,
    source_hashes as inference_sources, verified_journal, verified_model_specs,
)

MODELS=('CFG_MASKRCNN_1280','CFG_YOLO11S_960_ROT0','CFG_CABINET_UAVID_1280')


def sources():
    paths=inference_sources()
    for name in ('experiments/wide_area_diagnosis/physical_probe.py',
                 'experiments/wide_area_routing/diagnostics.py',
                 'experiments/v4_benchmark/backends.py','experiments/v4_benchmark/runner.py',
                 'experiments/lake_tracking.py','experiments/lake_missions/objectives.py',
                 'missionbench/v2/evaluation.py','missionbench/v2/real_models.py',
                 'missionbench/v2/cabinet_models.py'):
        paths[str(Path(name).resolve())]=sha(name)
    return paths


def prepare(cohort, output):
    cohort,output=Path(cohort).resolve(),Path(output).resolve()
    plan=verify_acquisition(cohort)
    specs,assets=verified_model_specs(DEFAULT_MODELS,DEFAULT_PLAN)
    selected=tuple(next(spec for spec in specs if spec.config_id==name) for name in MODELS)
    output.mkdir(parents=True,exist_ok=True)
    protocol=dict(schema_version='physical_native_probe_v1',development_only=True,
        prepared_utc=datetime.now(timezone.utc).isoformat(),cohort=str(cohort),
        acquisition_plan_sha256=sha(cohort/'plan.json'),model_source=str(DEFAULT_MODELS.resolve()),
        model_source_sha256=sha(DEFAULT_MODELS),parent_model_plan_sha256=sha(DEFAULT_PLAN),
        executor_configs=[asdict(spec) for spec in selected],
        model_signatures={spec.config_id:model_signature(spec) for spec in selected},
        model_assets_sha256=assets,implementation_sha256=sources(),
        cases=plan['cases'],expected_scored_captures=sum(len(case['evaluation_capture_ids']) for case in plan['cases']),
        expected_neural_calls_per_capture=3,
        neural_call_scope='Scored RGB forwards; initialization warmups are excluded and are not mission tasks.',
        input='Original native RGB only; no object labels, poses or evaluator coordinates passed to inference.',
        selection='Every acquisition-plan evaluation_capture_id, excluding all predeclared settle captures.',
        inference_workers=1,torch_intraop_threads=4,torch_interop_threads=1,
        timing='Actual host inference wall time including wrapper preprocessing; warmup excluded. Concurrent work may affect timing; not mission service costs.',
        objective='All-world native mask IoU>=0.3 with min component4px; focal three targets require two hits >=2s apart for at least2/3 targets and <=2 false components.',
        no_threshold_changes=True,no_full_mission_claim=True,
        causal_scope=plan['causal_scope'])
    path=output/'protocol.json'
    if path.exists():
        protocol['prepared_utc']=read(path)['prepared_utc']
    freeze_json(path,protocol)
    return protocol


def verify_protocol(output):
    output=Path(output)
    protocol=read(output/'protocol.json')
    if protocol['implementation_sha256']!=sources():
        raise ValueError('Probe implementation changed after lock')
    cohort=Path(protocol['cohort'])
    verify_acquisition(cohort)
    if sha(cohort/'plan.json')!=protocol['acquisition_plan_sha256']:
        raise ValueError('Acquisition plan changed')
    for path,expected in protocol['model_assets_sha256'].items():
        if sha(path)!=expected:
            raise ValueError('Model asset changed')
    if sha(protocol['model_source'])!=protocol['model_source_sha256'] or sha(DEFAULT_PLAN)!=protocol['parent_model_plan_sha256']:
        raise ValueError('Model source plan changed')
    return protocol


def selected_frames(case, recording, specification):
    """Validate complete physical identity before returning geometry-only IDs."""
    source=recording['provenance']['source']
    if source['specification']!=specification or recording['recording_identity']!=digest(source):
        raise ValueError('Recording identity differs from the locked physical specification')
    if recording['world']!=specification['world']:
        raise ValueError('Recording world differs from the physical specification')
    if not recording['provenance']['physical_route_passed'] or not recording['provenance']['full_declared_mission']:
        raise ValueError('Complete physically valid acquisition required')
    clock=recording['capture_schedule'];interval=clock['interval_ns'];epoch=clock['epoch_ns']
    if interval!=2_000_000_000 or clock['end_ns']-epoch!=round(case['duration_s']*1e9):
        raise ValueError('Changed original physical capture clock')
    if [frame['timestamp_ns'] for frame in recording['frames']]!=list(range(epoch,clock['end_ns'],interval)):
        raise ValueError('Incomplete physical capture clock')
    expected=[i for i in range(len(recording['frames'])) if i*interval/1e9<case['motion_s']]
    if expected!=case['evaluation_capture_ids'] or case['excluded_settle_capture_ids']!=[
            i for i in range(len(recording['frames'])) if i not in expected]:
        raise ValueError('Motion-only selection changed or settle captures were included')
    objects=specification['world']['objects']
    if len(objects)!=case['full_world_objects'] or not set(case['evaluator_target_ids'])<={o['object_id'] for o in objects}:
        raise ValueError('Focal or full-world target scope changed')
    result=[]
    for index in expected:
        frame=recording['frames'][index]
        if frame['body_pose']['timestamp_ns']!=frame['timestamp_ns']:
            raise ValueError('Pose and RGB clocks differ')
        result.append(dict(capture_id=index,timestamp_ns=frame['timestamp_ns'],mission_time_s=index*interval/1e9,
            source_file=frame['file'],source_sha256=frame['sha256'],body_pose_sha256=digest(frame['body_pose'])))
    return result


def bind_acquisition(output, protocol):
    cohort=Path(protocol['cohort']);output=Path(output)
    completion=read(cohort/'completion.json')
    if not completion['complete'] or completion['plan_sha256']!=sha(cohort/'plan.json'):
        raise ValueError('Wait for complete, bound acquisition attempts')
    indexed={case['case_id']:case for case in completion['cases']}
    if len(indexed)!=len(completion['cases']) or set(indexed)!={case['case_id'] for case in protocol['cases']}:
        raise ValueError('Acquisition completion omits or repeats a declared case')
    bindings=[]
    for case in protocol['cases']:
        receipt=indexed[case['case_id']];folder=cohort/case['case_id']/'recording'
        if not receipt['physical_passed']:
            bindings.append(dict(case_id=case['case_id'],status='acquisition_failed',receipt=receipt))
            continue
        path=folder/'recording.json'
        if sha(path)!=receipt['recording_sha256']:
            raise ValueError('Recording changed after acquisition completion')
        recording=read(path);spec=normalize_specification(cohort/case['scenario'])
        frames=selected_frames(case,recording,spec)
        for frame in frames:
            source=(folder/frame['source_file']).resolve()
            if source.parent!=folder.resolve() or sha(source)!=frame['source_sha256']:
                raise ValueError('Physical frame bytes changed')
        bindings.append(dict(case_id=case['case_id'],status='valid',recording_path=str(path),
            recording_sha256=sha(path),recording_identity=recording['recording_identity'],
            physical_validation_sha256=sha(folder/'physical_route_validation.json'),
            frames=frames,full_capture_count=len(recording['frames'])))
    document=dict(protocol_sha256=sha(output/'protocol.json'),completion_sha256=sha(cohort/'completion.json'),cases=bindings)
    freeze_json(output/'acquisition_bindings.json',document)
    return bindings


def infer_frame(folder, frame, source, pool):
    """Only RGB is opened here. The predictor API accepts exactly one RGB array."""
    folder=Path(folder);source=Path(source)
    if sha(source)!=frame['source_sha256']:
        raise ValueError('Physical frame changed before inference')
    with np.load(source,allow_pickle=False) as data:
        rgb=data['rgb'].copy()
    if rgb.dtype!=np.uint8 or rgb.shape!=(720,960,3):
        raise ValueError('Original native960x720 RGB required')
    prior=verified_journal(folder,frame,MODELS)
    if prior is not None:
        if prior['rgb_key']!=rgb_key(rgb):
            raise ValueError('Cached RGB identity differs')
        return prior
    masks,times,calls=pool.infer(rgb)
    if set(masks)!=set(MODELS) or set(calls)!=set(MODELS) or any(calls[name]!=1 for name in MODELS):
        raise ValueError('Exactly three native single-family forwards required')
    predictions={}
    for name in MODELS:
        mask=masks[name]
        if mask.dtype!=np.bool_ or mask.shape!=rgb.shape[:2] or not math.isfinite(times[name]) or times[name]<0:
            raise ValueError('Invalid native model geometry or timing')
        path=folder/f'prediction_{frame["capture_id"]:05d}_{name}.npz'
        temporary=path.with_suffix('.tmp.npz')
        np.savez_compressed(temporary,prediction_mask=mask)
        temporary.replace(path)
        predictions[name]=dict(file=path.name,sha256=sha(path),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                               measured_wall_clock_s=times[name],actual_neural_calls=calls[name])
    journal=dict(source=frame,rgb_key=rgb_key(rgb),rgb_shape=list(rgb.shape),predictions=predictions)
    path=folder/f'frame_{frame["capture_id"]:05d}.json'
    atomic_json(path,journal)
    path.with_suffix('.sha256').write_text(sha(path)+'\n')
    return journal


def phase(case):
    return dict(phase_id=case['focal_sector'],start_s=0.,end_s=case['motion_s'],deadline_s=case['motion_s'],
        evaluator_target_ids=case['evaluator_target_ids'],min_hits=2,min_evidence_span_s=2.,
        required_recall=2/3,max_false_components=2)


def pixel_measurements(frame, case, world, capture_id):
    """Evaluator geometry includes invisible focal targets; summaries label conditioning."""
    objects={o['object_id']:o for o in world['objects']}
    h,w=frame.rgb.shape[:2]
    origin,_=camera_ray(((w-1)/2,(h-1)/2),(h,w),frame.body_pose,world['camera_pitch_rad'])
    rows=[]
    for target in case['evaluator_target_ids']:
        obj=objects[target];ys,xs=np.where(frame.object_ids==obj['label'])
        rows.append(dict(capture_id=capture_id,target_id=target,visible_pixels=len(xs),
            bbox_width_px=int(xs.max()-xs.min()+1) if len(xs) else 0,
            bbox_height_px=int(ys.max()-ys.min()+1) if len(ys) else 0,
            bbox_longest_side_px=int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)) if len(xs) else 0,
            slant_range_to_authored_object_origin_m=float(np.linalg.norm(origin-np.asarray(obj['position_m']))),
            camera_height_above_authored_object_origin_m=float(origin[2]-obj['position_m'][2])))
    return rows


def score_case(output, case, binding, protocol):
    folder=Path(output)/'cases'/case['case_id'];recording_path=Path(binding['recording_path'])
    if sha(recording_path)!=binding['recording_sha256']:
        raise ValueError('Recording changed before evaluation')
    recording=read(recording_path);world=recording['world']
    scenario=SimpleNamespace(world=world,targets={o['label']:o['object_id'] for o in world['objects']})
    scorers={name:MissionEvaluator(.3,len(scenario.targets),2.) for name in (*MODELS,'perfect_gt')}
    frames,measurements,timings=[],[],{name:[] for name in MODELS}
    input_hashes={str(recording_path):sha(recording_path),str(Path(output)/'protocol.json'):sha(Path(output)/'protocol.json')}
    for selected in binding['frames']:
        index=selected['capture_id'];metadata=recording['frames'][index];source=recording_path.parent/metadata['file']
        if sha(source)!=selected['source_sha256'] or digest(metadata['body_pose'])!=selected['body_pose_sha256']:
            raise ValueError('Acquisition pixels or captured pose changed')
        frame=load_frame(source,metadata);truth=evaluation_frame(frame,index,selected['mission_time_s'],scenario)
        journal=verified_journal(folder,selected,MODELS)
        if journal is None or journal['rgb_key']!=rgb_key(frame.rgb):
            raise ValueError('Every selected RGB requires completed native inference')
        scores={'perfect_gt':scorers['perfect_gt'].update(truth,truth.semantic_gt==1).to_dict()}
        input_hashes[str(source)]=sha(source)
        for name in MODELS:
            entry=journal['predictions'][name];path=folder/entry['file']
            with np.load(path,allow_pickle=False) as arrays:
                mask=arrays['prediction_mask']
            scores[name]=scorers[name].update(truth,mask).to_dict()
            timings[name].append(entry['measured_wall_clock_s']);input_hashes[str(path)]=sha(path)
        frames.append(dict(capture_id=index,capture_time_s=selected['mission_time_s'],
                           body_pose=frame.body_pose,scores=scores))
        measurements.extend(pixel_measurements(frame,case,world,index))
    models={}
    for name,scorer in scorers.items():
        focal=phase_evidence([phase(case)],[frame['scores'][name] for frame in frames])[0]
        models[name]=dict(quality_all_world=scorer.quality_details(case['motion_s']),focal_sector=focal,
            raw_phase_pass=focal['zero_cost_phase_requirement_met'],
            latency_relaxed=True,paid_routing_or_full_mission_success=False,
            measured_wall_time_s=(dict(count=len(timings[name]),p50=float(np.median(timings[name])),
                p95=float(np.percentile(timings[name],95)),total=float(sum(timings[name]))) if name in timings else None))
    visible=[row for row in measurements if row['visible_pixels']>0]
    report=dict(schema_version='physical_native_probe_case_v1',development_only=True,case=case,
        recording_identity=recording['recording_identity'],physical_passed=True,
        scored_capture_ids=[row['capture_id'] for row in binding['frames']],
        excluded_settle_capture_ids=case['excluded_settle_capture_ids'],all_world_target_count=len(scenario.targets),
        models=models,frames=frames,target_measurements=measurements,
        geometry_summary=dict(total_focal_target_opportunities=len(measurements),visible_target_observations=len(visible),
            visible_longest_side_p10_p50_p90_px=np.percentile([row['bbox_longest_side_px'] for row in visible],[10,50,90]).tolist() if visible else [],
            camera_to_object_origin_range_p10_p50_p90_m=np.percentile([row['slant_range_to_authored_object_origin_m'] for row in measurements],[10,50,90]).tolist(),
            range_definition='Actual captured camera position to authored object mesh origin, not terrain AGL or a measured body-center range.'),
        implementation_sha256=protocol['implementation_sha256'],model_signatures=protocol['model_signatures'],input_sha256=input_hashes,
        caveat='Native actual inference; raw sensing evidence only. No paid routing, downstream reporting, or full mission evaluation.')
    freeze_json(folder/'score.json',report)
    return report


def execute(output):
    output=Path(output).resolve();protocol=verify_protocol(output)
    bindings=bind_acquisition(output,protocol)
    checkpoint=next(Path(path) for path in protocol['model_assets_sha256']
                    if Path(path).name=='maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth')
    os.environ['TORCH_HOME']=str(checkpoint.parent.parent.parent)
    import torch
    from experiments.mission_routing_v2.models import InferencePool

    if Path(torch.hub.get_dir()).resolve()!=checkpoint.parent.parent.resolve():
        raise ValueError('Torch must load the bound local checkpoint cache')
    torch.set_num_threads(4)
    if torch.get_num_interop_threads()!=1:
        torch.set_num_interop_threads(1)
    pool=None;reports=[];begun=time.perf_counter()
    for case,binding in zip(protocol['cases'],bindings):
        folder=output/'cases'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
        if binding['status']!='valid':
            failure=dict(case_id=case['case_id'],status='acquisition_failed',physical_result=binding['receipt'],
                         model_results=None,not_a_model_failure=True)
            freeze_json(folder/'acquisition_failure.json',failure);reports.append(failure)
            continue
        if pool is None:
            pool=InferencePool(tuple(ExecutorConfigSpec(**value) for value in protocol['executor_configs']))
            torch.set_num_threads(4)
        for number,frame in enumerate(binding['frames'],1):
            source=Path(binding['recording_path']).parent/frame['source_file']
            infer_frame(folder,frame,source,pool)
            print(f"{case['case_id']}: inferred {number}/{len(binding['frames'])}",flush=True)
        report=score_case(output,case,binding,protocol)
        reports.append(dict(case_id=case['case_id'],status='completed',score_sha256=sha(folder/'score.json'),
            native_phase_pass={name:report['models'][name]['raw_phase_pass'] for name in MODELS},
            perfect_gt_phase_pass=report['models']['perfect_gt']['raw_phase_pass']))
        print(f"Scored {case['case_id']}: {reports[-1]['native_phase_pass']}",flush=True)
    verify_protocol(output)
    bind_acquisition(output,protocol)
    freeze_json(output/'completion.json',dict(complete=True,protocol_sha256=sha(output/'protocol.json'),
        acquisition_bindings_sha256=sha(output/'acquisition_bindings.json'),cases=reports,
        actual_neural_calls=sum(len(b['frames'])*3 for b in bindings if b['status']=='valid'),
        neural_call_scope=protocol['neural_call_scope'],
        frameworks={name:importlib.metadata.version(name) for name in ('torch','torchvision','ultralytics','numpy')},
        timing_scope=protocol['timing'],no_full_mission_claim=True))
    print(f'Completed {len(reports)} physical probe cases in {time.perf_counter()-begun:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    parser.add_argument('--cohort',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.cohort is None:
            parser.error('--cohort is required for prepare')
        p=prepare(args.cohort,args.output)
        print(f"Locked {p['expected_scored_captures']} captures x3 native families")
    else:
        execute(args.output)
