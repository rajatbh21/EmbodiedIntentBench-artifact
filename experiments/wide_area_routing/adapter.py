"""Explicit binding from a wide physical recording to a perception-only replay.

No legacy scenario identity is substituted. Acquisition files and their identity
are retained, while the separate intent/resource contract is declared here.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from missionbench.schemas.common import ComparisonOperator
from missionbench.schemas.contract import Contract, PrivacyLevel
from experiments.lake_causal_router import MissionCosts
from experiments.lake_intent_contract import IntentContract
from experiments.mission_network_emulation import LinkSegment, LinkTrace
from experiments.mission_network_policies import ActionDescriptor, PolicyCosts
from experiments.mission_network_study_v1.executors import prepare_payload
from experiments.mission_runtime_profile import read, sha
from experiments.v4_benchmark.backends import load_frame
from experiments.v4_benchmark.resources import ResourceConfig
from experiments.v4_gazebo_pilot.timing import CaptureSchedule, seconds_to_ns


def opportunity_indices(recording, design, half_window_s=8.0):
    """Authored pass times only; never inspect RGB, visibility, or predictions."""
    if not math.isfinite(half_window_s) or half_window_s <= 0:
        raise ValueError('Positive finite window required')
    clock=recording['capture_schedule']
    expected=list(range(clock['epoch_ns'],clock['end_ns'],clock['interval_ns']))
    if [f['timestamp_ns'] for f in recording['frames']] != expected:
        raise ValueError('Incomplete canonical recording clock')
    epoch = clock['epoch_ns']
    centers = [s['nominal_pass_time_s'] for s in design['sectors']]
    return [i for i, f in enumerate(recording['frames'])
            if any(abs((f['timestamp_ns']-epoch)/1e9-t) <= half_window_s
                   for t in centers)]


def checked_recording(root, plan, flight):
    root=Path(root);case=plan['cases'][flight]
    path=Path(plan['cohort'])/flight/'recording/recording.json'
    if sha(path)!=case['recording_sha256']:
        raise ValueError('Physical recording changed')
    rec=read(path);clock=rec['capture_schedule'];provenance=rec['provenance']
    if rec['recording_identity']!=case['recording_identity'] or clock!=case['capture_schedule']:
        raise ValueError('Recording identity or acquisition clock was substituted')
    duration=(clock['end_ns']-clock['epoch_ns'])/1e9
    if duration!=case['duration_s'] or duration!=provenance['recorded_duration_s'] or not (
        provenance['full_declared_mission'] and provenance['physical_route_passed']):
        raise ValueError('Complete validated physical flight required')
    doc=read(root/case['scenario_path']);normalized=copy.deepcopy(rec['world'])
    for obj in normalized['objects']:
        obj.get('mesh',{}).pop('asset',None)
    if normalized!=doc['world'] or doc['duration_s']!=duration:
        raise ValueError('Replay world differs from the acquired physical world')
    design=read(root/case['mission_design_path'])
    if opportunity_indices(rec,design)!=case['opportunity_indices']:
        raise ValueError('Changed geometry-only observation opportunities')
    return rec


def make_intent(flight, duration, design, battery):
    sectors = design['sectors']
    centers = [s['nominal_pass_time_s'] for s in sectors]
    boundaries = [0.0, *[(a+b)/2 for a,b in zip(centers,centers[1:])], duration]
    targets = [t for s in sectors for t in s['evaluator_target_ids']]
    phases = [dict(phase_id=s['sector_id'],
        instruction='Obtain repeated person evidence in this scheduled inspection sector.',
        start_s=boundaries[i], end_s=boundaries[i+1], deadline_s=boundaries[i+1],
        min_hits=2, min_evidence_span_s=2.0, required_recall=2/3,
        max_false_components=2, weight=1, evaluator_target_ids=s['evaluator_target_ids'])
        for i,s in enumerate(sectors)]
    return IntentContract(dict(schema_version='lake_intent_v1', duration_s=duration,
        mission=dict(mission_id='scheduled_sector_survey_'+flight,
            title='Scheduled sector survey',
            user_request='Inspect eight scheduled sectors on the fixed route. Confirm at least two of the three people in each sector using two observations separated by at least two seconds, and report at least half of all people before the flight ends.',
            initial_battery_frac=battery, min_final_battery_frac=.2, phases=phases),
        limits=dict(deadline_s=duration+10, communication_budget_mb=10000.,
                    min_final_battery_frac=.2, privacy_level='remote_allowed'),
        reporting=dict(mode='report_required', target_scope='all_scenario_targets',
            deadline_s=duration, min_report_recall=.5, max_false_reports=2,
            max_duplicate_reports=2), evaluator_target_ids=targets))


def network_trace(name, duration_s):
    """Repeat the old 140 s intermittent pattern, independent of sector timing."""
    end = seconds_to_ns(duration_s+30)
    if name == 'good':
        return LinkTrace((LinkSegment(0,end,50_000_000,100_000_000,30_000_000),))
    if name != 'intermittent':
        raise ValueError('Unknown declared link')
    parts=[]
    for cycle in range(int((duration_s+30)//140)+1):
        for a,b,outage in [(0,45,False),(45,65,True),(65,95,False),
                            (95,105,True),(105,140,False)]:
            start=seconds_to_ns(cycle*140+a,allow_zero=True)
            stop=min(end,seconds_to_ns(cycle*140+b))
            if start < stop:
                parts.append(LinkSegment(start,stop,50_000_000,100_000_000,30_000_000,outage))
    return LinkTrace(tuple(parts))


@dataclass(frozen=True)
class WideReplayScenario:
    duration_s: float
    observation_interval_s: float
    battery_capacity_wh: float
    flight_power_w: float
    initial_battery_frac: float
    router_latency_s: float
    router_energy_j: float
    matching_iou_threshold: float
    world: dict
    targets: dict
    contract: Contract
    recording_identity: str
    schedule: CaptureSchedule
    resources: ResourceConfig


def load_case(root, plan, item):
    root=Path(root)
    case=plan['cases'][item['flight']]
    checked_recording(root,plan,item['flight'])
    doc=read(root/case['scenario_path'])
    design=read(root/case['mission_design_path'])
    contract=make_intent(item['flight'],case['duration_s'],design,item['battery'])
    if 'intent_paths' in case:
        frozen=read(root/case['intent_paths'][str(item['battery'])])
        if contract.document()!=frozen:
            raise ValueError('Runtime intent differs from the frozen intent document')
    limits=contract.document()['limits']
    outer=Contract('WIDE_SCHEDULED_SURVEY','HUMAN_SEARCH_SEGMENTATION','instance_mask_set',
        'target_recall',ComparisonOperator('>='),0.,limits['deadline_s'],
        limits['communication_budget_mb'],limits['min_final_battery_frac'],PrivacyLevel.REMOTE_ALLOWED)
    actions=tuple(ActionDescriptor(**a) for a in plan['actions'])
    resources=ResourceConfig.parse(dict(memory_capacity_mb=4096, models={a.action_id:dict(
        memory_mb=a.memory_mb,load_latency_s=a.load_latency_s,load_energy_j=a.load_energy_j)
        for a in actions}),[a.action_id for a in actions])
    clock=case['capture_schedule']
    scenario=WideReplayScenario(case['duration_s'],clock['interval_ns']/1e9,
        plan['resources']['battery_capacity_wh'],plan['resources']['flight_power_w'],
        item['battery'],.05,.75,.3,copy.deepcopy(doc['world']),
        {o['label']:o['object_id'] for o in doc['world']['objects']},outer,
        case['recording_identity'],CaptureSchedule(**clock,step_ns=10_000_000),resources)
    if set(contract.document()['evaluator_target_ids']) != set(scenario.targets.values()):
        raise ValueError('Intent target scope differs from the physical recording')
    profiles={k:read(root/v) for k,v in plan['profile_paths'].items()}
    return dict(scenario=scenario,contract=contract,actions=actions,
        costs=PolicyCosts(MissionCosts.from_scenario(scenario),**plan['policy_costs']),
        trace=network_trace(item['link'],scenario.duration_s),profiles=profiles,
        learned_profile=read(root/plan['learned_profile_path']))


def load_bank(root, plan, flight):
    """Evaluator owns labels; only frame.rgb enters the router and model."""
    root=Path(root);case=plan['cases'][flight]
    folder=Path(plan['cohort'])/flight/'recording'
    rec=checked_recording(root,plan,flight)
    frames,payloads={},{}
    for i in case['opportunity_indices']:
        metadata=rec['frames'][i];path=folder/metadata['file']
        if path.parent.resolve()!=folder.resolve() or sha(path)!=metadata['sha256']:
            raise ValueError('Invalid physical frame path or bytes')
        if metadata['timestamp_ns'] != case['capture_schedule']['epoch_ns']+i*case['capture_schedule']['interval_ns']:
            raise ValueError('Physical capture clock changed')
        frame=load_frame(path,metadata)
        frame.rgb.setflags(write=False);frame.object_ids.setflags(write=False)
        frames[i]=frame;payloads[i]=prepare_payload(frame.rgb)
    return dict(frames=frames,payloads=payloads,
        opportunity_indices=case['opportunity_indices'],provenance=dict(
            recording_identity=case['recording_identity'],recording_sha256=case['recording_sha256'],
            capture_schedule=case['capture_schedule'],opportunity_indices=case['opportunity_indices'],
            observation_schedule='Predeclared pass time +/-8s, on the original 2s capture grid',
            physical_flight_unchanged=True))
