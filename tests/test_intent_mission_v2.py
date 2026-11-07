"""Urgency-only updates preserve past work without waiving request or quality checks."""
import copy
from dataclasses import replace
import pytest
from experiments.intent_mission_v2.protocol import MissionIntent
from experiments.dynamic_intent.protocol import WindowIntent,current_update,NS
from experiments.intent_mission_v2.runtime import outcome
from experiments.intent_mission_v2.policy import make_policy
from tests.test_wide_area_paid_clip import event,result
from tests.test_mission_task_deadline_policy import ingredients,inputs
from tests.test_strong_routing_calibration_v2 import fixture

CASE=dict(case_id='synthetic',motion_s=36,duration_s=42,focal_sector='s',evaluator_target_ids=['a','b','c'])

def sample():
    events=[]
    for t in (10,14):
        e=event(t);e['window_id']=0;events.append(e)
    r=result(events);r['final_time_s']=42
    r['report_quality']['reports']=[dict(window_id=0,target_id=t,completion_time_s=14.2,evidence=[dict(capture_s=10),dict(capture_s=14)]) for t in ('a','b')]
    return events,r

def test_urgency_alone_keeps_evidence_across_update():
    e,r=sample();o=outcome(MissionIntent(CASE,.95),e,r)
    assert o['dynamic_mission_success']
    assert len(o['windows'])==1 and o['reporting']['recall_by_deadline']==2/3
    assert MissionIntent(CASE,.95).document()['mission']['phases'][0]['max_false_components']==2

def test_periodic_resurvey_is_distinct_and_still_fails_missing_windows():
    e,r=sample()
    from experiments.dynamic_intent.scoring import dynamic_outcome
    assert not dynamic_outcome(WindowIntent(CASE,.95),e,r)['dynamic_mission_success']

@pytest.mark.parametrize('failure',['late_report','one_capture','false_components','false_reports','duplicates','battery'])
def test_global_mission_constraints_are_still_enforced(failure):
    e,r=sample()
    if failure=='late_report':
        for q in r['report_quality']['reports']:q['completion_time_s']=44.01
    elif failure=='one_capture':e.pop()
    elif failure=='false_components':e[0]['score']['false_positive_components']=3
    elif failure=='false_reports':r['report_quality']['reports'] += [dict(window_id=0,target_id=None,completion_time_s=20,evidence=[]) for _ in range(3)]
    elif failure=='duplicates':r['report_quality']['reports'] += copy.deepcopy(r['report_quality']['reports'][:1])*3
    elif failure=='battery':r['constraints']['battery_constraint_success']=False
    assert not outcome(MissionIntent(CASE,.95),e,r)['dynamic_mission_success']

def test_no_future_update_or_private_target_id_in_router_view():
    c=MissionIntent(CASE,.95)
    assert c.at(current_update('no_change',0))==c.at(current_update('urgency_recovery',0))
    assert 'evaluator_target_ids' not in str(c.at(current_update('urgency_recovery',12*NS)))

@pytest.mark.parametrize('family',['quality','risk'])
def test_matched_router_pair_differs_only_in_update_visibility(family):
    a=ingredients();x=inputs();x['intent']=MissionIntent(CASE,.95).at(current_update('urgency_recovery',0))
    pair=[make_policy(family+'_'+v,a['actions'],a['costs'],a['profile'],None,None,fixture(a)) for v in ('current','initial')]
    first=[p.select(**x) for p in pair]
    assert first[0]['action_id']==first[1]['action_id']
    x['release_ns']=12*NS;x['link']=replace(x['link'],timestamp_ns=12*NS+round(a['costs'].encode_s*NS));x['intent']['current_intent']=current_update('urgency_recovery',12*NS)
    after=[p.select(**x) for p in pair]
    assert [q['observed_intent']['perception_budget_ns'] for q in after]==[2*NS,8*NS]
