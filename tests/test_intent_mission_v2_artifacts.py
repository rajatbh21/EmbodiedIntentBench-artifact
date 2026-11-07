"""Independent receipt checks and intentional corruption on completed v2 runs."""
from pathlib import Path
import pytest
from experiments.mission_runtime_profile import read
from experiments.intent_mission_v2.audit import check_run

ROOT=Path('results/intent_mission_v2_20261031')
pytestmark=pytest.mark.skipif(not (ROOT/'completion.json').exists(),reason='Completed local v2 experiment required')


def receipt():
    p=read(ROOT/'plan.json');r=next(r for r in read(ROOT/'rows.json') if r['policy']=='quality_current' and r['intent_schedule']=='urgency_recovery' and r['link']=='degraded' and r['energy_condition']=='ample')
    item=next(i for i in p['grid'] if i['run_id']==r['run_id']);folder=ROOT/r['path']
    e,result,o=[read(folder/n) for n in ('events.json','results.json','outcome.json')]
    return p,item,e,result,o

@pytest.mark.parametrize('defect',['budget','future','battery','free_compute','mission','window','wait','new_window'])
def test_corrupted_receipt_is_rejected(defect):
    p,item,e,r,o=receipt();assert check_run(p,item,e,r,o)>0
    if defect=='budget':e[0]['perception']['budget_ns']=2*10**9
    elif defect=='future':e[0]['applied_intent']['version']=2
    elif defect=='battery':e[-1]['battery_after']=item['battery']
    elif defect=='free_compute':e[0]['perception']['costs']['compute_energy_j']=0
    elif defect=='mission':o['dynamic_mission_success']=not o['dynamic_mission_success']
    elif defect=='window':o['windows'][0]['success']=not o['windows'][0]['success']
    elif defect=='wait':o['windows'][0]['first_valid_report_wait_s']=0
    else:e[-1]['window_id']=2
    with pytest.raises(AssertionError):check_run(p,item,e,r,o)


def test_actual_pipeline_is_not_reloaded_at_urgency_boundaries():
    _,_,events,_,_=receipt();seen=set();versions=set()
    for e in events:
        versions.add(e['applied_intent']['version'])
        if e['pipeline'] is None:continue
        for s in e['pipeline']['stages']:
            if not s['dispatched']:continue
            assert s['reset_slot']==(s['task'] not in seen)
            if s['task'] in seen:assert s['costs']['load_energy_j']==0
            seen.add(s['task'])
    assert versions=={0,1,2} and seen=={'association','verification'}


def test_locked_grid_keeps_all_splits_policies_and_failures():
    p=read(ROOT/'plan.json');rows=read(ROOT/'rows.json')
    assert len(rows)==648 and {r['run_id'] for r in rows}=={r['run_id'] for r in p['grid']}
    assert sum(r['split']=='development' for r in rows)==216
    assert sum(r['split']=='transfer' for r in rows)==432
    assert len({r['policy'] for r in rows})==9
