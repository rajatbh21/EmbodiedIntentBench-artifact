"""Negative controls on real completed receipts; skipped without local experiment assets."""
import copy
from pathlib import Path
import pytest
from experiments.mission_runtime_profile import read
from experiments.dynamic_intent.audit import check_run

ROOT=Path('results/dynamic_intent_v1_20261031')
pytestmark=pytest.mark.skipif(not (ROOT/'completion.json').exists(),reason='Completed local dynamic-intent artifacts required')

@pytest.mark.parametrize('defect',['budget','future_version','battery_reset','free_compute','mission_success','window_success','zero_wait'])
def test_tampered_real_receipt_is_rejected(defect):
    plan=read(ROOT/'plan.json');row=next(r for r in read(ROOT/'rows.json') if r['policy']=='quality_current' and r['intent_schedule']=='urgency_recovery' and r['link']=='degraded' and r['energy_condition']=='ample')
    item=next(r for r in plan['grid'] if r['run_id']==row['run_id']);folder=ROOT/row['path']
    events,result,outcome=[read(folder/n) for n in ('events.json','results.json','outcome.json')]
    assert check_run(plan,item,events,result,outcome)>0
    if defect=='budget':events[0]['perception']['budget_ns']=2*10**9
    elif defect=='future_version':events[0]['applied_intent']['version']=1
    elif defect=='battery_reset':events[-1]['battery_after']=item['battery']
    elif defect=='free_compute':events[0]['perception']['costs']['compute_energy_j']=0
    elif defect=='mission_success':outcome['dynamic_mission_success']=True
    elif defect=='window_success':outcome['windows'][0]['success']=not outcome['windows'][0]['success']
    else:outcome['windows'][0]['first_valid_report_wait_s']=0
    with pytest.raises(AssertionError):check_run(plan,item,events,result,outcome)
