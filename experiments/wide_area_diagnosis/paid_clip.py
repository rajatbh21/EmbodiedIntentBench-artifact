"""Exploratory paid routing on fixed local-inspection development clips.

All-world segmentation/report correctness is unchanged. A separate evaluator
counts focal-sector recall; correct nonfocal reports are disclosed, never turned
into false reports. This is not the original eight-sector mission benchmark.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from missionbench.schemas.common import ComparisonOperator
from missionbench.schemas.contract import Contract, PrivacyLevel
from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_causal_router import MissionCosts
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import canonical, digest, immutable
from experiments.lake_missions.objectives import public_phases, score as phase_score
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_network_emulation import LinkSegment, LinkTrace
from experiments.mission_network_policies import ActionDescriptor, PolicyCosts
from experiments.mission_network_study_v1.executors import prepare_payload
from experiments.mission_pipeline_routing.runtime import CompletedPrediction, PublicPose
from experiments.mission_pipeline_routing.study import public_context
from experiments.mission_runtime_profile import read, save, sha
from experiments.mission_task_deadlines.contract import DeadlineProfile
from experiments.mission_task_deadlines.pipeline import DeadlinePipeline
from experiments.mission_task_deadlines.study import DOWNSTREAM, require, task_metrics
from experiments.perception_router_suite.policy import make_policy as original_policy
from experiments.v4_benchmark.backends import load_frame
from experiments.v4_benchmark.resources import ResourceConfig
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.v4_gazebo_pilot.timing import CaptureSchedule, seconds_to_ns
from experiments.wide_area_diagnosis.physical_probe import MODELS, bind_acquisition, verify_protocol
from experiments.wide_area_missions.runtime import normalize_specification
from experiments.wide_area_routing.adapter import WideReplayScenario
from experiments.wide_area_routing.inference import freeze_json, verified_journal
from experiments.wide_area_routing.runner import advance_opportunities, latency_summary, validate_bank
from experiments.wide_area_routing.runner import source_hashes as runtime_sources
from experiments.wide_area_routing.study import verify as verify_parent

PARENT=Path('results/wide_area_routing_pilot_v1_20261028')
FIXED=dict(fixed_local_yolo='CFG_YOLO11S_960_ROT0',fixed_local_mask='CFG_MASKRCNN_1280',
           fixed_local_cabinet='CFG_CABINET_UAVID_1280',fixed_remote_mask='REMOTE__CFG_MASKRCNN_1280')
POLICIES=(*FIXED,'deadline_aware','accuracy_latency')


class LocalInspectionIntent:
    """Versioned interval/deadline contract; public view never includes IDs.

    Acquisition ends before the delivery allowance. Unlike the older full
    mission contract, a phase delivery deadline may extend beyond recording end.
    """
    def __init__(self,case):
        require(0<case['motion_s']<=case['duration_s'],'Invalid motion/recording interval')
        focal=case['evaluator_target_ids']
        require(len(focal)==3 and len(set(focal))==3,'Three unique focal targets required')
        deadline=case['duration_s']+10.
        self._json=canonical(dict(schema_version='local_inspection_intent_v1',duration_s=case['duration_s'],
            mission=dict(mission_id='local_inspection_'+case['case_id'],title='Local inspection development clip',
                user_request='Confirm at least two of three focal-sector people from repeated observations and report them before the delivery deadline.',
                initial_battery_frac=.95,min_final_battery_frac=.2,
                phases=[dict(phase_id=case['focal_sector'],instruction='Inspect the scheduled sector along the fixed pass.',
                    start_s=0.,end_s=case['motion_s'],deadline_s=deadline,min_hits=2,min_evidence_span_s=2.,
                    required_recall=2/3,max_false_components=2,weight=1,evaluator_target_ids=focal)]),
            limits=dict(deadline_s=deadline,communication_budget_mb=10000.,min_final_battery_frac=.2,privacy_level='remote_allowed'),
            reporting=dict(mode='local_focal_recall_world_correctness',deadline_s=deadline,min_report_recall=2/3,
                           max_false_reports=2,max_duplicate_reports=2),evaluator_target_ids=focal))

    def document(self):return json.loads(self._json)
    def public_document(self):
        doc=self.document();del doc['evaluator_target_ids']
        doc['mission']['phases']=public_phases(doc['mission'])
        return doc
    @property
    def sha256(self):return hashlib.sha256(self._json.encode()).hexdigest()
    @property
    def public_sha256(self):return digest(self.public_document())


def local_outcome(contract,events,result):
    doc=contract.document();focal=set(doc['evaluator_target_ids'])
    phases=phase_score(doc['mission'],events,result)['phases']
    reports=result['report_quality']['reports'];deadline=doc['reporting']['deadline_s']
    correct=[r['target_id'] for r in reports if r['target_id'] is not None]
    timely={r['target_id'] for r in reports if r['target_id'] is not None and r['completion_time_s']<=deadline}
    focal_timely=timely&focal
    false=len(reports)-len(correct);duplicates=len(correct)-len(set(correct))
    metrics=task_metrics(events)
    allowed={'completed','task_deadline_miss','transport_timeout','mission_deadline_exceeded','abstained','timeout','deadline'}
    valid=all(e['perception']['status'] in allowed and
        (e['execution']['success'] is e['perception']['admitted']) and
        (e['perception']['admitted'] or (e['pipeline'] is None and not e['emitted_reports']
            and not e['score']['matched_target_ids'] and e['score']['false_positive_components']==0)) and
        (e['pipeline'] is None or e['pipeline']['status'] in ('completed','task_deadline_miss','battery_depleted'))
        for e in events)
    valid &= (result['status']=='completed' and result['error'] is None and result['mission_completed']
        and result['termination_reason']=='mission_window_complete' and result['final_time_s']>=doc['duration_s']
        and result['processed_observations']==len(events) and result['router_errors']==result['invalid_actions']==0
        and result['inference_failures']==sum(not e['execution']['success'] for e in events))
    checks=dict(execution=bool(valid),sector_evidence=all(p['success'] for p in phases),
        focal_report_recall=len(focal_timely)/3>=doc['reporting']['min_report_recall'],
        world_false_report_limit=false<=2,world_duplicate_report_limit=duplicates<=2,
        **result['constraints'])
    return dict(schema_version='local_inspection_paid_outcome_v1',local_inspection_success=all(checks.values()),
        full_wide_mission_evaluated=False,checks=checks,failed_constraints=[k for k,v in checks.items() if not v],
        phases=phases,reporting=dict(recall_by_deadline=len(focal_timely)/3,focal_denominator=3,
            timely_focal_reports=sorted(focal_timely),correct_nonfocal_reports=sum(t not in focal for t in correct),
            correct_nonfocal_unique_targets=sorted(set(correct)-focal),world_false_reports=false,world_duplicates=duplicates),
        task_metrics=metrics,abstentions=sum(e['selected_action_id'] is None for e in events),
        semantics='Focal recall; original all-world consistent report correctness. Correct nonfocal reports neither help focal recall nor count as false.')


class FixedAction:
    def __init__(self,name,delegate):self.name,self.delegate=name,delegate
    def select(self,**inputs):
        result=self.delegate.select(**inputs)
        action=next(row for row in result['candidates'] if row['action_id']==FIXED[self.name])
        result.update(policy=self.name,action_id=action['action_id'] if action['legal'] else None,
                      reason='fixed_control_ignores_resource_forecast' if action['legal'] else 'no_legal_fixed_action')
        return result


def make_policy(name,actions,costs,profile,quality,risk,learned_profile=None):
    require(name in POLICIES,'Undeclared local-inspection policy')
    delegate=original_policy('deadline_aware' if name in FIXED else name,actions,costs,profile,quality,risk)
    return FixedAction(name,delegate) if name in FIXED else delegate


def link_trace(name,duration):
    end=round((duration+30)*1e9)
    if name=='good':return LinkTrace((LinkSegment(0,end,50_000_000,100_000_000,30_000_000),))
    require(name=='outage_4_10','Undeclared local link trace')
    return LinkTrace(tuple(LinkSegment(round(a*1e9),min(end,round(b*1e9)),50_000_000,100_000_000,30_000_000,outage)
        for a,b,outage in ((0,4,False),(4,10,True),(10,duration+30,False))))


def source_hashes():
    return {**runtime_sources(),str(Path(__file__).resolve()):sha(__file__),
            str(Path('experiments/wide_area_diagnosis/physical_probe.py').resolve()):sha('experiments/wide_area_diagnosis/physical_probe.py')}


def prepare(root,probe):
    root,probe=Path(root).resolve(),Path(probe).resolve()
    require(not root.exists(),'Preserve prior paid study')
    parent=verify_parent(PARENT);native=verify_protocol(probe)
    root.mkdir(parents=True)
    actions=[a for a in parent['actions'] if a['source_config_id'] in MODELS]
    require(len(actions)==6,'Exactly six local/remote actions required')
    profiles={}
    for name,path in parent['profile_paths'].items():
        doc=read(PARENT/path);doc['models']={key:value for key,value in doc['models'].items() if key in MODELS}
        dest=Path('inputs')/f'{name}.json';save(root/dest,doc);profiles[name]=str(dest)
    grid=[]
    for case in native['cases']:
        for policy in POLICIES:
            for budget in (2,8):
                for link in ('good','outage_4_10'):
                    run_id=f"{case['case_id']}/{policy}/budget_{budget}/{link}"
                    grid.append(dict(flight=case['case_id'],policy=policy,perception_budget_s=budget,link=link,battery=.95,run_id=run_id))
    plan=dict(schema_version='paid_local_inspection_development_v1',development_only=True,
        prepared_utc=datetime.now(timezone.utc).isoformat(),probe=str(probe),native_protocol_sha256=sha(probe/'protocol.json'),
        parent_plan_sha256=sha(PARENT/'plan.json'),source_hashes=source_hashes(),
        profiles=profiles,profile_sha256={name:sha(root/path) for name,path in profiles.items()},
        cases={case['case_id']:case for case in native['cases']},grid=grid,expected_runs=len(grid),
        actions=actions,policy_costs=parent['policy_costs'],resources=dict(battery_capacity_wh=120,flight_power_w=120),
        assets=parent['assets'],raw_phase_trigger='At least one native raw sector pass; every predeclared source case retained regardless of outcomes.',
        factors=dict(policies=list(POLICIES),perception_budget_s=[2,8],links=['good','outage_4_10'],
            initial_battery_frac=[.95],source_case_ids=[case['case_id'] for case in native['cases']]),
        network=dict(upload_bps=50_000_000,download_bps=100_000_000,rtt_ms=30,
            outage_interval_s=[4,10],outage_interval='half-open, identical absolute clip clock for every speed'),
        outcome_awareness='Exploratory follow-up: native raw scores inspected before paid-plan lock; no fitted policy, model filtering, threshold change, or selected-case omission.',
        policy_changes='Fixed-action adapters only; unchanged adaptive factories, source-matched subset of prior quality/risk heads; no fitting or host-time recalibration.',
        deadline_semantics='Motion-only acquisition. Local sector/report deadline is recorded duration+10s. Per-invocation perception budget2/8s remains independent.',
        scope='Local inspection task with focal3recall and all-world24correctness, not full eight-sector mission success.',
        accounting='Charge max(original recorded duration,last completion) at120W plus unchanged configured service/load/codec/radio energy. No extra captures or free settle evidence.',
        reporting='Correct nonfocal reports neither help focal recall nor count as false. World-inconsistent reports and duplicate correct reports retain their caps.',
        trained_heads='Frozen old development heads; three-identity subset only; no test-label or reward update.',
        interpretation='Exploratory matched three-site development study; no population inference or held-out claim.')
    freeze_json(root/'plan.json',plan)
    return plan


def verify(root):
    root=Path(root);plan=read(root/'plan.json')
    require(plan['source_hashes']==source_hashes(),'Paid diagnostic implementation changed')
    require(sha(PARENT/'plan.json')==plan['parent_plan_sha256'],'Cost parent changed')
    for name,path in plan['profiles'].items():require(sha(root/path)==plan['profile_sha256'][name],'Frozen head changed')
    require(sha(Path(plan['probe'])/'protocol.json')==plan['native_protocol_sha256'],'Native protocol changed')
    verify_protocol(plan['probe'])
    return plan


def load_case(root,plan,item):
    case=plan['cases'][item['flight']];contract=LocalInspectionIntent(case)
    probe=Path(plan['probe']);native=read(probe/'protocol.json');cohort=Path(native['cohort'])
    recording=read(cohort/case['case_id']/'recording/recording.json')
    actions=tuple(ActionDescriptor(**a) for a in plan['actions'])
    outer=Contract('LOCAL_INSPECTION','HUMAN_SEARCH_SEGMENTATION','instance_mask_set','target_recall',ComparisonOperator('>='),0.,
                   case['duration_s']+10,10000.,.2,PrivacyLevel.REMOTE_ALLOWED)
    resources=ResourceConfig.parse(dict(memory_capacity_mb=4096,models={a.action_id:dict(memory_mb=a.memory_mb,
        load_latency_s=a.load_latency_s,load_energy_j=a.load_energy_j) for a in actions}),[a.action_id for a in actions])
    scenario=WideReplayScenario(case['duration_s'],2.,120.,120.,.95,.05,.75,.3,recording['world'],
        {o['label']:o['object_id'] for o in recording['world']['objects']},outer,recording['recording_identity'],
        CaptureSchedule(**recording['capture_schedule'],step_ns=10_000_000),resources)
    return dict(scenario=scenario,contract=contract,actions=actions,
        costs=PolicyCosts(MissionCosts.from_scenario(scenario),**plan['policy_costs']),
        trace=link_trace(item['link'],case['duration_s']),
        profiles={name:read(Path(root)/path) for name,path in plan['profiles'].items()},learned_profile=None)


def bind_predictions(root,plan):
    probe=Path(plan['probe']);completion=read(probe/'completion.json')
    require(completion['complete'] and completion['protocol_sha256']==sha(probe/'protocol.json'),'Complete native study required')
    require(any(any(row.get('native_phase_pass',{}).values()) for row in completion['cases']),'No native raw sector passes; paid trigger not met')
    native=verify_protocol(probe);bindings=bind_acquisition(probe,native)
    require(len(completion['cases'])==len(plan['cases']) and
            {r['case_id'] for r in completion['cases']}==set(plan['cases']),
            'Native completion must cover the complete predeclared source design')
    require(completion['acquisition_bindings_sha256']==sha(probe/'acquisition_bindings.json'),
            'Native acquisition binding changed')
    for row in completion['cases']:
        score_path=probe/'cases'/row['case_id']/'score.json'
        require(row['status']=='completed' and row['score_sha256']==sha(score_path),'Native result changed')
        score=read(score_path)
        require(row['native_phase_pass']=={name:score['models'][name]['raw_phase_pass'] for name in MODELS},
                'Native trigger differs from bound scores')
    require(all(b['status']=='valid' for b in bindings),'All predeclared physical cases required')
    for binding in bindings:
        folder=Path(root)/'predictions'/binding['case_id'];folder.mkdir(parents=True,exist_ok=True)
        original=probe/'cases'/binding['case_id'];models={name:dict(signature=native['model_signatures'][name],predictions={}) for name in MODELS}
        for frame in binding['frames']:
            journal=verified_journal(original,frame,MODELS);require(journal is not None,'Missing native cache')
            for name in MODELS:
                pred=journal['predictions'][name];target=folder/pred['file']
                if target.exists():require(sha(target)==pred['sha256'],'Copied cache changed')
                else:shutil.copyfile(original/pred['file'],target)
                prior=models[name]['predictions'].get(journal['rgb_key'])
                require(prior is None or prior['mask_sha256']==pred['mask_sha256'],'Duplicate RGB disagrees')
                models[name]['predictions'][journal['rgb_key']]=pred
        freeze_json(folder/'manifest.json',dict(models=models,recording_identity=binding['recording_identity'],
            recording_sha256=binding['recording_sha256'],native_protocol_sha256=sha(probe/'protocol.json')))
    freeze_json(Path(root)/'native_bindings.json',dict(native_completion_sha256=sha(probe/'completion.json'),
        acquisition_bindings_sha256=sha(probe/'acquisition_bindings.json'),bindings=bindings))
    return bindings


def load_bank(binding):
    path=Path(binding['recording_path']);require(sha(path)==binding['recording_sha256'],'Physical recording changed')
    recording=read(path);frames={};payloads={}
    for selected in binding['frames']:
        index=selected['capture_id'];metadata=recording['frames'][index];source=path.parent/metadata['file']
        require(sha(source)==selected['source_sha256'] and digest(metadata['body_pose'])==selected['body_pose_sha256'],'Physical pixels or pose changed')
        frame=load_frame(source,metadata)
        frame.rgb.flags.writeable=False;frame.object_ids.flags.writeable=False
        frames[index]=frame;payloads[index]=prepare_payload(frame.rgb)
    indices=[frame['capture_id'] for frame in binding['frames']]
    return dict(frames=frames,payloads=payloads,opportunity_indices=indices,
        provenance=dict(recording_identity=binding['recording_identity'],recording_sha256=binding['recording_sha256'],
                        capture_schedule=recording['capture_schedule'],opportunity_indices=indices))


def _initialize_worker(root,plan,binding):
    global _STATE
    terrain,_=load_public_terrain(Path(plan['assets']))
    _STATE=(root,plan,terrain,load_bank(binding),
            read(Path(root)/'predictions'/binding['case_id']/'manifest.json'))


def _run_worker(item):
    root,plan,terrain,bank,manifest=_STATE
    row=run_case(root,plan,item,terrain,bank,manifest)
    save(Path(root)/row['path']/'row.json',row)
    return row


def execute(root,workers=2):
    root=Path(root).resolve();plan=verify(root)
    require(workers in (1,2,3),'One to three replay workers required')
    bindings=bind_predictions(root,plan);rows=[];started=time.perf_counter()
    for binding in bindings:
        pending=[]
        for item in (i for i in plan['grid'] if i['flight']==binding['case_id']):
            folder=root/'runs'/item['run_id'];row_path=folder/'row.json'
            if row_path.exists():
                row=read(row_path)
                require(all(row[k]==v for k,v in item.items()),'Existing run differs from declared cell')
                require(read(folder/'protocol.json')['plan_sha256']==sha(root/'plan.json'),'Existing run belongs to another plan')
                for name,expected in row['artifacts_sha256'].items():
                    require(sha(folder/name)==expected,'Changed completed run')
                rows.append(row)
            else:pending.append(item)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize_worker,initargs=(root,plan,binding)) as pool:
            futures={pool.submit(_run_worker,item):item for item in pending}
            for future in as_completed(futures):
                row=future.result();rows.append(row)
                save(root/'progress.json',dict(completed_runs=len(rows),expected_runs=plan['expected_runs']))
                print(f'{len(rows)}/{plan["expected_runs"]} {row["run_id"]}: local_success={row["local_inspection_success"]}, focal_reports={row["focal_correct_reports"]}/3, raw_false={row["raw_false_components"]}',flush=True)
    require(len(rows)==plan['expected_runs'] and len({r['run_id'] for r in rows})==plan['expected_runs'],
            'Incomplete or duplicate paired replay grid')
    verify(root)
    rows.sort(key=lambda r:r['run_id']);freeze_json(root/'run_index.json',rows)
    freeze_json(root/'completion.json',dict(complete=True,runs=len(rows),
        plan_sha256=sha(root/'plan.json'),run_index_sha256=sha(root/'run_index.json'),
        native_bindings_sha256=sha(root/'native_bindings.json'),
        prediction_manifest_sha256={c:sha(root/'predictions'/c/'manifest.json') for c in plan['cases']},
        elapsed_host_s=time.perf_counter()-started,local_inspection_successes=sum(r['local_inspection_success'] for r in rows),
        full_wide_mission_evaluated=False))
    return rows


# The paid run loop below is copied from the independently audited wide-area
# runner. Only factories/intent evaluation and the named local result scope differ.
def run_case(root, plan, item, terrain, bank, manifest):
    from experiments.mission_task_deadlines.perception import execute_perception

    root = Path(root)
    require(item in plan["grid"], "Run is outside the declared study grid")
    case = load_case(root, plan, item)
    scenario, contract = case["scenario"], case["contract"]
    actions, costs, trace = case["actions"], case["costs"], case["trace"]
    profiles, learned_profile = case["profiles"], case["learned_profile"]
    require(manifest["recording_identity"] == scenario.recording_identity, "Wrong prediction flight")
    require(
        manifest["recording_sha256"] == bank["provenance"]["recording_sha256"],
        "Predictions and RGB bank bind different recording bytes",
    )
    predictions_dir = root / "predictions" / item["flight"]
    require(read(predictions_dir / "manifest.json") == manifest, "Prediction manifest changed")
    require(read(root / "plan.json") == plan, "Study plan changed")
    indices = validate_bank(scenario, bank)
    descriptors = {a.action_id: a for a in actions}
    for action in actions:
        model = manifest["models"][action.source_config_id]
        require(model["signature"] == action.source_signature, "Prediction model identity changed")
        require(
            all(bank["payloads"][i].rgb_sha256 in model["predictions"] for i in indices),
            "Incomplete prediction bank for declared opportunities",
        )
    budget = item["perception_budget_s"]
    require(type(budget) in (int, float) and budget in (2, 8), "Undeclared perception budget")
    profile = DeadlineProfile(f"perception_{budget:g}s", seconds_to_ns(budget), 150_000_000, 30_000_000)
    policy = make_policy(
        item["policy"],
        actions,
        costs,
        profile,
        profiles["recall"],
        profiles["risk"],
        learned_profile=learned_profile,
    )
    pipeline = DeadlinePipeline(
        profile=profile,
        terrain=terrain,
        step_ns=scenario.schedule.step_ns,
        battery_capacity_j=scenario.battery_capacity_wh * 3600,
        flight_power_w=scenario.flight_power_w,
    )
    scorer = MissionEvaluator(
        scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
    )
    reporter = RunnerReportEvaluator(scenario.matching_iou_threshold, scenario.targets.values(), 2)
    masks = {
        a.action_id: VerifiedMasks(
            predictions_dir,
            manifest["models"][a.source_config_id],
        )
        for a in actions
    }
    frames, payloads, provenance = bank["frames"], bank["payloads"], bank["provenance"]
    require(
        provenance["recording_identity"] == scenario.recording_identity, "Wrong shared RGB bank"
    )
    destination = root / "runs" / item["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    save(
        destination / "protocol.json",
        dict(
            schema_version="local_inspection_paid_routing_v1",
            case=item,
            implementation_sha256=source_hashes(),
            opportunity_indices=list(indices),
            observation_clock="Original recording capture indices; only predeclared opportunities are eligible",
            flight_energy_scope="Entire elapsed flight including initial transit, inactive gaps and tail",
            plan_sha256=sha(root / "plan.json"),
            contract_sha256=contract.sha256,
            public_contract_sha256=contract.public_sha256,
            recording=provenance,
            prediction_manifest_sha256=sha(
                predictions_dir / "manifest.json"
            ),
            deadline_profile=profile.to_dict(),
            downstream_selection=asdict(DOWNSTREAM),
        ),
    )
    save(destination / "intent.json", contract.document())
    schedule = scenario.schedule
    capacity = scenario.battery_capacity_wh * 3600
    nonflight = communication = peak_memory = 0.0
    current_action = None
    now_ns = position = 0
    events, skipped_ids = [], []
    termination = "mission_window_complete"

    def battery(at_ns, extra_j=0.0):
        energy = at_ns / 1e9 * scenario.flight_power_w + nonflight + extra_j
        return max(0.0, scenario.initial_battery_frac - energy / capacity)

    while position < len(indices):
        index = indices[position]
        if battery(now_ns) <= 0:
            termination = "battery_depleted"
            break
        capture_ns = schedule.at(index) - schedule.epoch_ns
        require(capture_ns >= now_ns, "Capture occurred while preceding work was busy")
        if battery(capture_ns) <= 0:
            now_ns, termination = capture_ns, "battery_depleted"
            break
        if capture_ns > seconds_to_ns(scenario.contract.deadline_s):
            now_ns, termination = capture_ns, "deadline_exceeded"
            break
        frame, prepared = frames[index], payloads[index]
        public_intent = contract.public_document()
        decision_ns = capture_ns + seconds_to_ns(costs.encode_s, allow_zero=True)
        observed_battery = battery(decision_ns, costs.encode_s * costs.codec_power_w)
        selection = policy.select(
            release_ns=capture_ns,
            current_action_id=current_action,
            battery_frac=observed_battery,
            cumulative_communication_mb=communication,
            link=trace.snapshot(decision_ns),
            payload_bytes=len(prepared.payload),
            response_bytes=prepared.response_bytes,
            intent=public_intent,
            rgb=frame.rgb,
        )
        action_id = selection["action_id"]
        action = descriptors[action_id] if action_id is not None else None
        perception = execute_perception(
            action=action,
            rgb=frame.rgb,
            prepared=prepared,
            predictions=masks[action_id] if action_id is not None else None,
            release_ns=capture_ns,
            profile=profile,
            mission_deadline_ns=seconds_to_ns(scenario.contract.deadline_s),
            current_action_id=current_action,
            trace=trace,
            costs=costs,
            router_latency_ns=seconds_to_ns(scenario.router_latency_s, allow_zero=True),
            router_energy_j=scenario.router_energy_j,
            step_ns=schedule.step_ns,
        )
        nonflight += perception.nonflight_energy_j
        communication += perception.communication_mb
        current_action = perception.resident_action_id
        now_ns = perception.completed_ns
        output = None
        reports = []
        downstream_skip_reason = None
        if perception.mask is not None and battery(now_ns) > 0:
            resource = scenario.resources.for_config(action_id)
            context = public_context(
                contract, capture_ns, now_ns, battery(now_ns), resource.memory_mb
            )
            output = pipeline.execute(
                DOWNSTREAM,
                CompletedPrediction(
                    action.source_config_id,
                    capture_ns,
                    now_ns,
                    perception.mask,
                    PublicPose(
                        tuple(frame.body_pose["position_m"]),
                        tuple(frame.body_pose["quaternion_xyzw"]),
                        scenario.world.get("camera_pitch_rad", math.pi / 2),
                    ),
                ),
                context,
            )
            nonflight += output["charged_energy_j"]
            now_ns = output["completed_ns"]
            reports = output["emitted_reports"]
            peak_memory = max(
                peak_memory,
                action.memory_mb,
                *(s["costs"]["resident_memory_mb"] for s in output["stages"] if s["dispatched"]),
            )
        elif perception.mask is not None:
            downstream_skip_reason = "battery_depleted_before_association_release"
            peak_memory = max(peak_memory, action.memory_mb + 68)
        elif action is not None:
            downstream_skip_reason = "perception_output_not_admitted"
            peak_memory = max(peak_memory, action.memory_mb + 68)
        completion, next_position, skipped = advance_opportunities(schedule, indices, position, now_ns)
        require(completion == now_ns, "Inconsistent usable completion ticks")
        truth = evaluation_frame(frame, index, capture_ns / 1e9, scenario)
        prediction = (
            perception.mask
            if perception.mask is not None
            else np.zeros(frame.rgb.shape[:2], dtype=np.bool_)
        )
        score = scorer.update(truth, prediction)
        reporter.update(truth, prediction, reports)
        for missed in skipped:
            missed_truth = evaluation_frame(
                frames[missed], missed, (schedule.at(missed) - schedule.epoch_ns) / 1e9, scenario
            )
            scorer.note_skipped_visibility(missed_truth.visible_target_ids)
            reporter.note_visibility(missed_truth)
        events.append(
            dict(
                observation_id=index,
                capture_time_s=capture_ns / 1e9,
                capture_timestamp_ns=frame.timestamp_ns,
                completion_time_s=now_ns / 1e9,
                perception_completed_ns=perception.completed_ns,
                selected_action_id=action_id,
                source_config_id=action.source_config_id if action else None,
                source_rgb_key=prepared.rgb_sha256,
                selected_mask_sha256=hashlib.sha256(perception.mask.tobytes()).hexdigest()
                if perception.mask is not None
                else None,
                capture_body_pose=frame.body_pose,
                public_link=trace.snapshot(decision_ns).to_dict(),
                battery_at_decision=observed_battery,
                router_decision=selection,
                perception=perception.event,
                pipeline=output,
                downstream_skip_reason=downstream_skip_reason,
                execution=dict(
                    success=perception.mask is not None,
                    config_id=action_id,
                    predicted_pixels=int(prediction.sum()),
                ),
                score=score.to_dict(),
                emitted_reports=reports,
                cumulative_nonflight_energy_j=nonflight,
                cumulative_energy_j=now_ns / 1e9 * scenario.flight_power_w + nonflight,
                cumulative_communication_mb=communication,
                battery_after=battery(now_ns),
                skipped_observation_ids=list(skipped),
            )
        )
        skipped_ids.extend(skipped)
        position = next_position
    if termination == "mission_window_complete":
        require(len(events) + len(skipped_ids) == len(indices), "Missing declared opportunity accounting")
        now_ns = max(now_ns, seconds_to_ns(scenario.duration_s))
    if battery(now_ns) <= 0:
        termination = "battery_depleted"
    quality, report_quality = scorer.quality_details(now_ns / 1e9), reporter.summary(now_ns / 1e9)
    flags = dict(
        memory_constraint_success=peak_memory <= 4096,
        deadline_success=now_ns / 1e9 <= scenario.contract.deadline_s,
        battery_constraint_success=battery(now_ns) >= scenario.contract.min_final_battery_frac,
        communication_constraint_success=communication <= scenario.contract.communication_budget_mb,
        privacy_constraint_success=True,
    )
    result = dict(
        schema_version="local_inspection_task_deadline_execution_v1",
        contract=asdict(scenario.contract),
        status="completed",
        error=None,
        termination_reason=termination,
        mission_completed=termination == "mission_window_complete",
        final_time_s=now_ns / 1e9,
        final_battery_frac=battery(now_ns),
        cumulative_energy_j=now_ns / 1e9 * scenario.flight_power_w + nonflight,
        cumulative_nonflight_energy_j=nonflight,
        flight_energy_j=now_ns / 1e9 * scenario.flight_power_w,
        cumulative_communication_mb=communication,
        peak_configured_memory_mb=peak_memory,
        processed_observations=len(events),
        eligible_opportunities=len(indices),
        inactive_capture_count=(schedule.end_ns - schedule.epoch_ns + schedule.interval_ns - 1) // schedule.interval_ns - len(indices),
        unprocessed_opportunity_ids=list(indices[position:]),
        perception_latency=latency_summary(events),
        selected_action_counts=dict(Counter(e["selected_action_id"] for e in events if e["selected_action_id"] is not None)),
        skipped_observation_ids=skipped_ids,
        router_errors=0,
        invalid_actions=0,
        privacy_violations=0,
        inference_failures=sum(not e["execution"]["success"] for e in events),
        constraints=flags,
        quality=quality,
        report_quality=report_quality,
        recording_identity=scenario.recording_identity,
    )
    outcome = local_outcome(contract, events, result)
    for name, value in (("events", events), ("results", result), ("outcome", outcome)):
        save(destination / (name + ".json"), value)
    return dict(
        **item,
        path=str(destination.relative_to(root)),
        local_inspection_success=outcome["local_inspection_success"],
        full_wide_mission_evaluated=False,
        raw_false_components=outcome["phases"][0]["false_components"],
        evidence_recall=outcome["phases"][0]["evidence_recall"],
        world_false_reports=outcome["reporting"]["world_false_reports"],
        world_duplicate_reports=outcome["reporting"]["world_duplicates"],
        correct_nonfocal_reports=outcome["reporting"]["correct_nonfocal_reports"],
        focal_correct_reports=len(outcome["reporting"]["timely_focal_reports"]),
        successful_phases=sum(p["success"] for p in outcome["phases"]),
        report_recall=outcome["reporting"]["recall_by_deadline"],
        final_battery_frac=result["final_battery_frac"],
        communication_mb=communication,
        decision_count=len(events),
        skipped_opportunity_count=len(skipped_ids),
        eligible_opportunity_count=len(indices),
        selected_action_counts=result["selected_action_counts"],
        perception_latency=result["perception_latency"],
        final_time_s=result["final_time_s"],
        flight_energy_j=result["flight_energy_j"],
        nonflight_energy_j=nonflight,
        total_energy_j=result["cumulative_energy_j"],
        mission_completed=result["mission_completed"],
        termination_reason=termination,
        task_metrics=outcome["task_metrics"],
        abstentions=outcome["abstentions"],
        failed_constraints=outcome["failed_constraints"],
        artifacts_sha256={
            name + ".json": sha(destination / (name + ".json"))
            for name in ("protocol", "intent", "events", "results", "outcome")
        },
    )


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','verify','execute'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',type=Path,default=Path('results/wide_area_physical_native_probe_v1_20261028'))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.output,args.probe)
    elif args.mode=='verify':verify(args.output)
    else:execute(args.output,args.workers)


if __name__=='__main__':main()
