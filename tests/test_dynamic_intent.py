"""Intent boundaries, paid in-flight work, matched visibility and fresh evidence."""
import copy
from dataclasses import replace
import pytest
from experiments.dynamic_intent.protocol import current_update, deadline_profile, WindowIntent, NS
from experiments.dynamic_intent.policy import UpdatingPolicy
from experiments.dynamic_intent.scoring import dynamic_outcome
from experiments.mission_task_deadlines.perception import execute_perception
from tests.test_mission_task_deadline_perception import setup
from tests.test_mission_task_deadline_policy import ingredients, inputs
from tests.test_strong_routing_calibration_v2 import fixture


def contract():
    return WindowIntent(dict(case_id='synthetic',motion_s=36,duration_s=42,
        focal_sector='sector_1',evaluator_target_ids=['a','b','c']),.95)

@pytest.mark.parametrize('t,version,budget',[(0,0,8),(12*NS-1,0,8),(12*NS,1,2),(24*NS-1,1,2),(24*NS,2,8)])
def test_exact_update_boundaries(t,version,budget):
    u=current_update('urgency_recovery',t)
    assert (u['version'],u['perception_budget_ns'])==(version,budget*NS)

def test_current_public_view_cannot_leak_future_schedule_or_private_ids():
    c=contract()
    assert c.at(current_update('no_change',0))==c.at(current_update('urgency_recovery',0))
    assert 'evaluator_target_ids' not in str(c.at(current_update('urgency_recovery',13*NS)))
    assert set(c.at(current_update('urgency_recovery',13*NS))['current_intent'])=={'version','effective_from_ns','perception_budget_ns'}

def test_request_started_before_update_keeps_deadline_and_full_paid_work():
    args=setup();args['release_ns']=10*NS
    args['action']=replace(args['action'],latency_s=3,energy_j=45)
    args['profile']=deadline_profile(current_update('urgency_increase',args['release_ns']))
    old=execute_perception(**args)
    assert old.completed_ns>12*NS and old.event['admitted']
    assert old.event['deadline_ns']==18*NS and old.event['budget_ns']==8*NS
    assert old.nonflight_energy_j>45
    args['release_ns']=14*NS;args['profile']=deadline_profile(current_update('urgency_increase',14*NS))
    new=execute_perception(**args)
    assert not new.event['admitted'] and new.event['budget_ns']==2*NS
    assert new.nonflight_energy_j==old.nonflight_energy_j

def test_only_observed_budget_differs_between_matched_routers():
    args=ingredients();data=inputs();routers=[UpdatingPolicy(n,args['actions'],args['costs'],args['profile'],fixture(args)) for n in ('quality_current','quality_initial')]
    data['intent']['current_intent']=current_update('urgency_recovery',0)
    before=[r.select(**data) for r in routers]
    assert before[0]['action_id']==before[1]['action_id']
    data['intent']['current_intent']=current_update('urgency_recovery',12*NS)
    after=[r.select(**data) for r in routers]
    assert [r.delegate.forecaster.profile.perception_ns for r in routers]==[2*NS,8*NS]
    assert after[0]['observed_intent']['version']==1 and after[1]['observed_intent']['version']==0
    data['intent']['current_intent']=current_update('urgency_recovery',24*NS)
    assert routers[0].select(**data)['observed_intent']['version']==2

def outcome_fixture():
    from tests.test_wide_area_paid_clip import event,result
    events=[];reports=[]
    for w in range(3):
        for t in (w*12,w*12+2):
            e=event(t);e['window_id']=w;events.append(e)
        for target in ('a','b'):
            reports.append(dict(window_id=w,target_id=target,completion_time_s=w*12+2.2,
                evidence=[dict(capture_s=w*12),dict(capture_s=w*12+2)]))
    r=result(events);r['final_time_s']=42;r['report_quality']['reports']=reports
    return events,r

def test_all_windows_must_pass_and_repeated_requested_reports_are_not_global_duplicates():
    events,r=outcome_fixture();out=dynamic_outcome(contract(),events,r)
    assert out['dynamic_mission_success'] and out['window_successes']==3
    assert out['reporting']['world_duplicates']==0
    assert [w['first_valid_report_wait_s'] for w in out['windows']]==pytest.approx([2.2]*3)
    r['report_quality']['reports']=[v for v in r['report_quality']['reports'] if v['window_id']!=1]
    out=dynamic_outcome(contract(),events,r)
    assert not out['dynamic_mission_success'] and out['window_successes']==2
    assert out['windows'][1]['first_valid_report_wait_s'] is None

def test_old_evidence_cannot_satisfy_new_window_and_prior_success_is_not_rewritten():
    events,r=outcome_fixture()
    for v in r['report_quality']['reports']:
        if v['window_id']==1:v['evidence']=[dict(capture_s=0),dict(capture_s=2)]
    out=dynamic_outcome(contract(),events,r)
    assert out['windows'][0]['success'] and not out['windows'][1]['success'] and out['windows'][2]['success']

def test_complete_runtime_applies_updates_without_free_residency_or_battery_reset(tmp_path,monkeypatch):
    from tests.test_wide_area_paid_clip import local_fixture
    from tests.test_wide_area_runner import outputs
    from experiments.dynamic_intent import runtime
    from experiments.wide_area_diagnosis import paid_clip as paid
    from experiments.mission_runtime_profile import save
    settings,state=local_fixture(tmp_path,monkeypatch)
    targets=state['contract'].document()['evaluator_target_ids']
    state['contract']=WindowIntent(dict(case_id='test',motion_s=36,duration_s=42,
        focal_sector='s',evaluator_target_ids=targets),state['scenario'].initial_battery_frac)
    state['scenario']=replace(state['scenario'],duration_s=42)
    settings['bank']['provenance']['capture_schedule']={n:getattr(state['scenario'].schedule,n) for n in ('epoch_ns','interval_ns','end_ns')}
    settings['manifest']['recording_identity']=state['scenario'].recording_identity
    settings['bank']['provenance']['recording_identity']=state['scenario'].recording_identity
    save(settings['root']/'predictions/synthetic_flight/manifest.json',settings['manifest'])
    settings['item']['intent_schedule']='urgency_recovery'
    save(settings['root']/'plan.json',settings['plan'])
    monkeypatch.setattr(runtime,'load_case',lambda *_:state)
    monkeypatch.setattr(runtime,'make_policy',paid.make_policy)
    row=runtime.run_case(**settings);saved=outputs(settings,row)
    assert [e['applied_intent']['perception_budget_ns'] for e in saved['events']]==[8*NS,2*NS,8*NS]
    assert row['flight_energy_j']==420 and row['nonflight_energy_j']>0
    assert saved['events'][1]['perception']['deadline_missed']
    assert not saved['events'][1]['perception']['reset_slot']
    assert saved['events'][2]['battery_after']<saved['events'][1]['battery_after']
