"""Local-inspection scope, privacy, scheduling, and paid cancellation contracts."""
import copy
from dataclasses import replace

import pytest

from experiments.mission_runtime_profile import read, save
from experiments.wide_area_diagnosis import paid_clip as paid
from tests.test_wide_area_runner import prepared_case, outputs


def intent():
    return paid.LocalInspectionIntent(dict(case_id='site/offset/speed',motion_s=10.,duration_s=16.,
        focal_sector='sector_1',evaluator_target_ids=['a','b','c']))


def event(t, targets=('a','b'), false=0, finish=None):
    return dict(capture_time_s=t,completion_time_s=t+.2 if finish is None else finish,
        selected_action_id='native',execution=dict(success=True),
        perception=dict(status='completed',admitted=True,task='perception',task_deadline_met=True,
            deadline_missed=False,on_time=True,lateness_ns=0),
        pipeline=None,emitted_reports=[],score=dict(matched_target_ids=list(targets),false_positive_components=false))


def result(events, reports=()):
    return dict(status='completed',error=None,mission_completed=True,termination_reason='mission_window_complete',
        final_time_s=16.,processed_observations=len(events),router_errors=0,invalid_actions=0,inference_failures=0,
        final_battery_frac=.9,constraints=dict(deadline_success=True,memory_constraint_success=True,
            privacy_constraint_success=True,battery_constraint_success=True,communication_constraint_success=True),
        report_quality=dict(reports=[dict(target_id=t,completion_time_s=12.) for t in reports]))


def test_public_intent_redacts_focal_ids_and_allows_later_delivery():
    contract=intent();public=contract.public_document()
    assert 'evaluator_target_ids' not in public
    assert 'evaluator_target_ids' not in public['mission']['phases'][0]
    assert public['mission']['phases'][0]['end_s']==10
    assert public['mission']['phases'][0]['deadline_s']==26>public['duration_s']
    public['mission']['phases'][0]['required_recall']=0
    assert contract.public_document()['mission']['phases'][0]['required_recall']==2/3


def test_focal_recall_does_not_relabel_correct_nonfocal_reports_as_false():
    events=[event(0),event(2)];state=result(events,['a','b','other1','other2','other3'])
    out=paid.local_outcome(intent(),events,state)
    assert out['local_inspection_success'] and not out['full_wide_mission_evaluated']
    assert out['reporting']['recall_by_deadline']==2/3
    assert out['reporting']['world_false_reports']==0
    assert out['reporting']['correct_nonfocal_reports']==3
    assert out['reporting']['correct_nonfocal_unique_targets']==['other1','other2','other3']


@pytest.mark.parametrize('reports,failed',[
    (['a','other1','other2'],'focal_report_recall'),
    (['a','b',None,None,None],'world_false_report_limit'),
    (['a','b','other','other','other','other'],'world_duplicate_report_limit'),
])
def test_explicit_local_reporting_constraints(reports,failed):
    events=[event(0),event(2)];out=paid.local_outcome(intent(),events,result(events,reports))
    assert not out['local_inspection_success'] and failed in out['failed_constraints']


@pytest.mark.parametrize('events',[
    [event(0),event(1.99)], [event(0),event(10)], [event(0),event(2,finish=26.01)],
    [event(0,targets=('a',)),event(2,targets=('a',))],
])
def test_repeated_evidence_needs_two_distinct_in_window_hits_two_seconds_apart(events):
    out=paid.local_outcome(intent(),events,result(events,['a','b']))
    assert not out['checks']['sector_evidence']


def test_phase_false_components_are_separate_from_confirmed_false_reports():
    events=[event(0,false=2),event(2,false=1)]
    out=paid.local_outcome(intent(),events,result(events,['a','b']))
    assert out['phases'][0]['evidence_recall']==2/3
    assert out['phases'][0]['false_components']==3
    assert not out['checks']['sector_evidence'] and out['checks']['world_false_report_limit']


def test_delivery_after_exit_before_outer_deadline_is_eligible():
    events=[event(6,finish=18),event(8,finish=24)]
    out=paid.local_outcome(intent(),events,result(events,['a','b']))
    assert out['local_inspection_success']


def test_absolute_outage_window_is_half_open_and_does_not_scale_with_speed():
    for duration in (20,42):
        trace=paid.link_trace('outage_4_10',duration)
        assert not trace.snapshot(3_999_999_999).outage
        assert trace.snapshot(4_000_000_000).outage
        assert trace.snapshot(9_999_999_999).outage
        assert not trace.snapshot(10_000_000_000).outage
        assert not trace.snapshot((duration+10)*1_000_000_000).outage


def local_fixture(tmp_path,monkeypatch,budget=8):
    settings,case=prepared_case(tmp_path,monkeypatch,budget=budget)
    targets=list(case['scenario'].targets.values())[:3]
    while len(targets)<3:targets.append('unobserved_'+str(len(targets)))
    contract=paid.LocalInspectionIntent(dict(case_id='synthetic',motion_s=26.,duration_s=30.,
        focal_sector='sector_1',evaluator_target_ids=targets))
    case=dict(case,contract=contract)
    monkeypatch.setattr(paid,'load_case',lambda *_:case)
    settings['item']['policy']='fixed_local_yolo'
    save(settings['root']/'plan.json',settings['plan'])
    return settings,case


def test_paid_loop_preserves_full_flight_energy_and_public_policy_boundary(tmp_path,monkeypatch):
    settings,case=local_fixture(tmp_path,monkeypatch)
    original=paid.make_policy;seen=[]
    def factory(*args,**kwargs):
        policy=original(*args,**kwargs)
        class Spy:
            def select(self,**inputs):
                seen.append(copy.deepcopy(inputs['intent']))
                assert set(inputs)=={'release_ns','current_action_id','battery_frac',
                    'cumulative_communication_mb','link','payload_bytes','response_bytes','intent','rgb'}
                return policy.select(**inputs)
        return Spy()
    monkeypatch.setattr(paid,'make_policy',factory)
    row=paid.run_case(**settings);saved=outputs(settings,row)
    assert row['flight_energy_j']==300
    assert [e['observation_id'] for e in saved['events']]==[2,8,12]
    assert saved['results']['skipped_observation_ids']==[3,9]
    assert saved['results']['final_time_s']==30
    assert seen and all('evaluator_target_ids' not in str(d) for d in seen)
    assert 'mission_success' not in row and 'local_inspection_success' in row
    assert not row['full_wide_mission_evaluated']


def test_late_native_masks_not_read_and_have_no_downstream_evidence(tmp_path,monkeypatch):
    settings,_=local_fixture(tmp_path,monkeypatch,budget=2)
    (settings['root']/'predictions/synthetic_flight/mask.npz').write_bytes(b'invalid if read')
    row=paid.run_case(**settings);saved=outputs(settings,row)
    assert all(e['perception']['status']=='task_deadline_miss' for e in saved['events'])
    assert all(not e['perception']['prediction_cache_read'] and e['pipeline'] is None for e in saved['events'])
    assert row['total_energy_j']>300 and not row['local_inspection_success']
    assert saved['outcome']['checks']['execution']


@pytest.mark.parametrize('empty',[False,True])
def test_perfect_and_empty_masks_through_paid_downstream_local_endpoint(tmp_path,monkeypatch,empty):
    import hashlib
    import numpy as np
    from experiments.mission_runtime_profile import sha
    settings,case=local_fixture(tmp_path,monkeypatch)
    # Fixed pose and exactly 8-second admitted separation preserve association's
    # public gate. This is a synthetic evaluator control, not a model result.
    settings['bank']['opportunity_indices']=[8,12]
    settings['bank']['provenance']['opportunity_indices']=[8,12]
    settings['bank']['frames']={i:f for i,f in settings['bank']['frames'].items() if i in (8,12)}
    settings['bank']['payloads']={i:f for i,f in settings['bank']['payloads'].items() if i in (8,12)}
    path=settings['root']/'predictions/synthetic_flight/mask.npz'
    mask=settings['bank']['frames'][8].object_ids>0
    if empty:mask=np.zeros_like(mask)
    np.savez_compressed(path,prediction_mask=mask)
    for model in settings['manifest']['models'].values():
        for entry in model['predictions'].values():entry.update(sha256=sha(path),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    save(path.parent/'manifest.json',settings['manifest'])
    row=paid.run_case(**settings);saved=outputs(settings,row)
    assert row['local_inspection_success'] is (not empty)
    assert row['focal_correct_reports']==(0 if empty else 3)
    assert row['world_false_reports']==0 and row['raw_false_components']==0
    assert all(e['pipeline']['status']=='completed' for e in saved['events'])
    assert row['flight_energy_j']==300 and row['nonflight_energy_j']>0
