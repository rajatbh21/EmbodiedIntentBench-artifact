"""Posthoc evaluator-only attrition and downstream positive controls.

The original experiment is read-only. Native masks retain their exact geometry;
no false components are removed. Zero-perception-latency controls use every
original eligible capture and the unchanged paid downstream pipeline. Perfect
GT masks are diagnostic inputs, never an executable model or router result.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_missions.objectives import score as phase_score
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_pipeline_routing.runtime import CompletedPrediction, PublicPose
from experiments.mission_pipeline_routing.study import public_context
from experiments.mission_runtime_profile import read, save, sha
from experiments.mission_task_deadlines.contract import DeadlineProfile
from experiments.mission_task_deadlines.pipeline import DeadlinePipeline
from experiments.mission_task_deadlines.study import DOWNSTREAM
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.wide_area_routing.adapter import load_bank, load_case
from experiments.wide_area_routing.diagnostics import phase_evidence, verify_saved
from experiments.wide_area_routing.study import verify


def summary_geometry(rows):
    """Successive evaluator-identity projections; never used for association."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['target_id'], []).append(row)
    pairs = []
    for target, observations in grouped.items():
        observations.sort(key=lambda row: row['capture_time_s'])
        for first, second in zip(observations, observations[1:]):
            pairs.append(dict(target_id=target,
                first_capture_s=first['capture_time_s'], second_capture_s=second['capture_time_s'],
                gap_s=second['capture_time_s']-first['capture_time_s'],
                displacement_m=math.dist(first['position_m'], second['position_m'])))
    distances = [row['displacement_m'] for row in pairs]
    return dict(pairs=pairs, matched_projection_count=len(rows), pair_count=len(pairs),
        max_displacement_m=max(distances, default=None),
        displacement_p50_p90_p99_m=np.percentile(distances, [50,90,99]).tolist() if distances else [],
        pairs_outside_1m_gate=sum(row['displacement_m']>1 for row in pairs),
        pairs_outside_8s_gap=sum(row['gap_s']>8 for row in pairs),
        within_gap_but_outside_distance=sum(row['gap_s']<=8 and row['displacement_m']>1 for row in pairs),
        caveat='Pairs use evaluator identity only after predicted association; thresholds and assignments are unchanged.')


def semantic_checks(contract, phases, reports):
    """Name relaxed control success separately from paid mission success."""
    requested = contract.document()['reporting']
    timely = {r['target_id'] for r in reports['reports']
              if r['target_id'] is not None and r['completion_time_s']<=requested['deadline_s']}
    recall = len(timely)/len(contract.document()['evaluator_target_ids'])
    checks = dict(all_sector_requirements=all(p['success'] for p in phases),
        report_recall_by_deadline=recall>=requested['min_report_recall'],
        false_report_limit=reports['false_reports']<=requested['max_false_reports'],
        duplicate_report_limit=reports['duplicate_correct_reports']<=requested['max_duplicate_reports'])
    return dict(checks=checks, evidence_and_reporting_pass=all(checks.values()),
        report_recall_by_deadline=recall,
        paid_mission_feasibility_established=False)


def downstream_control(case, bank, terrain, source_id, masks=None, *, schedule=None):
    """Original captures, optional fixed admission schedule, unchanged downstream.

    ``schedule`` maps original capture index to an already-admitted perception
    completion. With no schedule the perception stage is an explicit free,
    instantaneous diagnostic input. No policy is selected or retrained here.
    """
    scenario, contract = case['scenario'], case['contract']
    profile = DeadlineProfile('diagnostic_input_original_downstream',8_000_000_000,150_000_000,30_000_000)
    pipeline = DeadlinePipeline(profile=profile, terrain=terrain, step_ns=scenario.schedule.step_ns)
    scorer = MissionEvaluator(scenario.matching_iou_threshold,len(scenario.targets),scenario.observation_interval_s)
    reporter = RunnerReportEvaluator(scenario.matching_iou_threshold,scenario.targets.values(),2)
    events, geometry = [], []
    counts, reasons = Counter(), Counter()
    for index in bank['opportunity_indices']:
        if schedule is not None and index not in schedule:
            continue
        frame = bank['frames'][index]
        capture = index*scenario.schedule.interval_ns
        available = capture if schedule is None else schedule[index]
        truth = evaluation_frame(frame,index,capture/1e9,scenario)
        mask = truth.semantic_gt==1 if masks is None else masks[bank['payloads'][index].rgb_sha256]
        pose = PublicPose(tuple(frame.body_pose['position_m']),tuple(frame.body_pose['quaternion_xyzw']),
                          scenario.world.get('camera_pitch_rad',math.pi/2))
        result = pipeline.execute(DOWNSTREAM,CompletedPrediction(source_id,capture,available,mask,pose),
            public_context(contract,capture,available,1.0,0.0))
        counts[result['status']] += 1
        for decision in result['verification_decisions']:
            reasons[decision['reason']] += 1
        score = scorer.update(truth,mask).to_dict()
        reporter.update(truth,mask,result['emitted_reports'])
        for candidate in result['candidates']:
            target = reporter.evaluator.ledger.get((capture/1e9,candidate['current_component_id']))
            if target is not None and candidate['predicted_position_m'] is not None:
                geometry.append(dict(target_id=target,capture_time_s=capture/1e9,
                                     position_m=candidate['predicted_position_m']))
        events.append(dict(observation_id=index,capture_time_s=capture/1e9,
            perception_completed_ns=available,completion_time_s=result['completed_ns']/1e9,
            execution=dict(success=True),score=score,emitted_reports=result['emitted_reports'],pipeline=result))
    final_time = max(scenario.duration_s,max((e['completion_time_s'] for e in events),default=0))
    reports = reporter.summary(final_time)
    phases = phase_score(contract.document()['mission'],events,dict(
        final_battery_frac=1.0,constraints=dict(deadline_success=True,memory_constraint_success=True,
        privacy_constraint_success=True),status='completed',router_errors=0,inference_failures=0,
        invalid_actions=0))['phases']
    return dict(source_id=source_id, evaluator_only=True, deployable_result=False,
        perfect_gt_input=masks is None, original_capture_clock=True,
        relaxed='Perception service/admission, resources and policy selection; fixed original downstream service and deadlines.',
        schedule_mode='all_eligible_zero_perception_latency' if schedule is None else 'original_paid_admissions',
        unchanged_downstream=asdict(DOWNSTREAM),processed_captures=len(events),
        pipeline_status_counts=dict(counts),verification_reasons=dict(reasons),
        quality=scorer.quality_details(final_time),report_quality=reports,
        phases=phases,semantic_control=semantic_checks(contract,phases,reports),
        projection_diagnostic=summary_geometry(geometry),events=events)


def fixed_retention(events, native_scores, phases):
    """Expose original fixed-model aliasing without inventing adaptive choices."""
    scores = {row['observation_id']:row for row in native_scores}
    attempted = {row['observation_id'] for row in events}
    admitted = {row['observation_id'] for row in events if row['perception']['admitted']}
    positive = {index for index,row in scores.items() if row['matched_target_ids']}
    actual_phases = phase_evidence(phases,[scores[i] for i in sorted(admitted)])
    potential_phases = phase_evidence(phases,native_scores)
    return dict(native_positive_capture_ids=sorted(positive),native_positive_captures=len(positive),
        attempted_positive_captures=len(positive&attempted),admitted_positive_captures=len(positive&admitted),
        busy_skipped_positive_capture_ids=sorted(positive-attempted),
        canceled_positive_capture_ids=sorted((positive&attempted)-admitted),
        full64_native_matched_target_observations=sum(len(row['matched_target_ids']) for row in native_scores),
        admitted_matched_target_observations=sum(len(scores[i]['matched_target_ids']) for i in admitted),
        raw_evidence_sectors=sum(p['evidence_requirement_met'] for p in potential_phases),
        admitted_evidence_sectors=sum(p['evidence_requirement_met'] for p in actual_phases),
        full64_native_phases=potential_phases,
        caveat='Matched-positive means any canonical target match, not a prediction selected using labels.')


def cadence_offsets(native_scores, phases, stride=4):
    """Report every relative phase of a fixed-stride cadence, never a winner.

    Eight contiguous 2s-grid captures constitute each original inspection
    window. A four-capture stride approximates the existing heavy local action's
    roughly seven-second occupancy. This mask/evidence-only diagnostic omits
    paid start-up, network and downstream service and cannot establish a policy
    score. All offsets are enumerated independently of prediction outcomes.
    """
    if type(stride) is not int or stride<1:
        raise ValueError('Positive integer stride required')
    windows=[]
    for row in sorted(native_scores,key=lambda row:row['observation_id']):
        if not windows or row['observation_id']!=windows[-1][-1]['observation_id']+1:
            windows.append([])
        windows[-1].append(row)
    offsets=[]
    for offset in range(stride):
        selected=[row for window in windows for row in window[offset::stride]]
        evidence=phase_evidence(phases,selected)
        offsets.append(dict(offset=offset,capture_ids=[row['observation_id'] for row in selected],
            positive_captures=sum(bool(row['matched_target_ids']) for row in selected),
            matched_target_observations=sum(len(row['matched_target_ids']) for row in selected),
            supported_target_phase_pairs=sum(row['supported_target_count'] for row in evidence),
            evidence_sectors=sum(row['evidence_requirement_met'] for row in evidence),
            phases_with_fp_cap=sum(row['zero_cost_phase_requirement_met'] for row in evidence)))
    return dict(evaluator_only=True,stride_captures=stride,all_offsets_reported=True,
        selected_best_offset=None,offsets=offsets,
        caveat='Posthoc evidence-only cadence sensitivity, not a paid routing result or a tuned schedule.')


def receipt_waterfall(root, plan):
    rows = read(root/'run_index.json')
    diagnostics = {flight:read(root/'model_diagnostics'/f'{flight}.json') for flight in plan['cases']}
    for diagnostic in diagnostics.values():
        verify_saved(diagnostic)
    totals, reports, reasons = Counter(), Counter(), Counter()
    cases = []
    for row in rows:
        folder = root/row['path']
        for filename,digest in row['artifacts_sha256'].items():
            if sha(folder/filename)!=digest:
                raise ValueError('Changed frozen receipt')
        events,result,outcome = (read(folder/f'{name}.json') for name in ('events','results','outcome'))
        per_run = Counter(eligible=row['eligible_opportunity_count'],attempted=len(events),
            admitted=sum(e['perception']['admitted'] for e in events),busy_skipped=row['skipped_opportunity_count'])
        for event in events:
            per_run['matched_target_observations']+=len(event['score']['matched_target_ids'])
            per_run['false_components']+=event['score']['false_positive_components']
            if event['source_config_id']:
                native = next(x for x in diagnostics[row['flight']]['models'][event['source_config_id']]['frame_scores']
                              if x['observation_id']==event['observation_id'])
                per_run['selected_native_matched_before_admission']+=len(native['matched_target_ids'])
            pipe = event['pipeline']
            if pipe:
                per_run['pipeline_'+pipe['status']]+=1
                for stage in pipe['stages']:
                    per_run[stage['task']+'_'+stage['status']]+=1
                reasons.update(decision['reason'] for decision in pipe['verification_decisions'])
        per_run['sector_evidence_passes']=sum(p['evidence_recall']>=p['required_recall'] for p in outcome['phases'])
        per_run['sector_fp_limit_failures']=sum(p['false_components']>p['max_false_components'] for p in outcome['phases'])
        per_run['supported_target_phase_pairs']=sum(len(p['supported_targets']) for p in outcome['phases'])
        per_run['total_reports']=result['report_quality']['total_reports']
        per_run['correct_reports']=result['report_quality']['correct_reports']
        totals.update(per_run)
        reports.update({key:result['report_quality'][key] for key in (
            'total_reports','correct_reports','false_reports','duplicate_correct_reports',
            'mixed_identity_reports','last_frame_correct_but_invalid_support')})
        configs = {e['source_config_id'] for e in events if e['source_config_id'] is not None}
        retention = None
        if row['policy'].startswith('fixed_') and len(configs)==1:
            case=load_case(root,plan,row)
            retention=fixed_retention(events,diagnostics[row['flight']]['models'][next(iter(configs))]['frame_scores'],
                                      case['contract'].document()['mission']['phases'])
        cases.append(dict(run_id=row['run_id'],flight=row['flight'],policy=row['policy'],
            link=row['link'],budget_s=row['perception_budget_s'],battery=row['battery'],
            counts=dict(per_run),fixed_model_retention=retention))
    return dict(evaluator_only=True,posthoc=True,totals=dict(totals),report_totals=dict(reports),
        verification_reasons=dict(reasons),runs=cases,
        evidence='Existing native canonical scores and verified paid receipts; no new inference.',
        caveat='Pooled replay counts repeat the same three recordings and are not independent observations.')


def build(root, output, mode='all'):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output==root or root in output.parents:
        raise ValueError('Posthoc diagnosis must be stored outside the frozen study')
    output.mkdir(parents=True,exist_ok=True)
    plan=verify(root)
    before={str(root/name):sha(root/name) for name in ('plan.json','run_index.json','completion.json','independent_audit.json')}
    completion=read(root/'completion.json')
    if not completion['complete'] or completion['run_index_sha256']!=sha(root/'run_index.json'):
        raise ValueError('Completed bound study required')
    if mode in ('all','receipts'):
        waterfall=receipt_waterfall(root,plan)
        save(output/'receipt_waterfall.json',waterfall)
        print('Receipt waterfall:',waterfall['totals'],flush=True)
    if mode in ('all','perfect','native'):
        terrain,terrain_metadata=load_public_terrain(Path(plan['assets']))
        for flight in plan['cases']:
            item=next(i for i in plan['grid'] if i['flight']==flight and i['battery']==.95)
            case=load_case(root,plan,item);bank=load_bank(root,plan,flight)
            folder=root/'predictions'/flight;manifest=read(folder/'manifest.json')
            if sha(folder/'manifest.json')!=completion['prediction_manifest_sha256'][flight]:
                raise ValueError('Native prediction manifest changed')
            if manifest['recording_sha256']!=bank['provenance']['recording_sha256']:
                raise ValueError('Predictions do not belong to the physical recording')
            native_diagnostic=read(root/'model_diagnostics'/f'{flight}.json')
            verify_saved(native_diagnostic)
            heavy='CFG_MASKRCNN_1280_ROT0_90'
            save(output/'cadence'/f'{flight}.json',cadence_offsets(
                native_diagnostic['models'][heavy]['frame_scores'],
                case['contract'].document()['mission']['phases']))
            controls=[]
            if mode in ('all','perfect'):
                controls.append(('perfect_gt',None))
            if mode in ('all','native'):
                controls.extend((model_id,VerifiedMasks(folder,model)) for model_id,model in manifest['models'].items())
            for source_id,masks in controls:
                result=downstream_control(case,bank,terrain,source_id,masks)
                result.update(flight=flight,terrain=terrain_metadata,recording=bank['provenance'],
                              plan_sha256=sha(root/'plan.json'),manifest_sha256=sha(folder/'manifest.json'))
                path=output/'controls'/flight/f'{source_id}.json'
                save(path,result)
                print(f"{flight} / {source_id}: sectors {sum(p['success'] for p in result['phases'])}/8; "
                      f"correct reports {result['report_quality']['unique_targets_reported']}/24; "
                      f"semantic pass {result['semantic_control']['evidence_and_reporting_pass']}",flush=True)
    if not all(sha(path)==digest for path,digest in before.items()):
        raise ValueError('Frozen study changed during diagnosis')
    inventory={str(path.relative_to(output)):sha(path) for path in output.rglob('*.json') if path.name!='provenance.json'}
    save(output/'provenance.json',dict(schema_version='wide_area_pipeline_diagnosis_v1',
        inspected_utc=datetime.now(timezone.utc).isoformat(),posthoc=True,evaluator_only=True,
        new_inference=False,original_study_modified=False,original_outcomes_known_before_analysis=True,
        source_sha256={str(Path(__file__).resolve()):sha(__file__)},frozen_input_sha256=before,
        artifact_sha256=inventory,
        caveat='Diagnostic controls relax perception and resource costs. They are not deployable policies or held-out validation.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('all','receipts','perfect','native'),default='all')
    args=parser.parse_args()
    build(args.study,args.output,args.mode)
