"""Independent receipt/endpoint reconstruction; does not import runner or scorer.

This checks clocks, released intent, accounting, fresh report support and all
window endpoints from saved receipts. It is not independent neural inference or
mask-to-ground-truth matching; those retain the frozen native cache/scorers.
"""
from pathlib import Path
import argparse
import math
from experiments.mission_runtime_profile import read, save, sha


def check_run(plan, item, events, result, outcome):
    assert events and result['mission_completed']
    updates=plan['intent_schedules'][item['intent_schedule']]
    last_finish=0;energy=bytes_mb=0.;resident=None;seen=[];skipped=[]
    actions={a['action_id']:a for a in plan['actions']}
    for e in events:
        t=round(e['capture_time_s']*1e9);p=e['perception']
        version=max(i for i,(s,_) in enumerate(updates) if s*1e9<=t)
        start,budget=updates[version]
        expected=dict(version=version,effective_from_ns=start*10**9,perception_budget_ns=budget*10**9)
        assert e['applied_intent']==expected
        assert t>=last_finish and p['release_ns']==t and p['budget_ns']==budget*10**9
        assert p['deadline_ns']==t+budget*10**9
        observed=e['router_decision']['observed_intent']
        assert observed==(dict(version=0,effective_from_ns=0,perception_budget_ns=8*10**9) if item['policy']=='quality_initial' else expected)
        assert e['window_id']==min(2,int(e['capture_time_s']//12))
        finish=p['usable_finish_ns'];step=p['step_ns']
        assert finish==((p['raw_finish_ns']+step-1)//step)*step
        assert not p['admitted'] or (finish<=p['deadline_ns'] and e['selected_mask_sha256'])
        assert e['execution']['success']==p['admitted']
        action=actions.get(e['selected_action_id'])
        assert p['reset_slot']==(action is not None and action['action_id']!=resident)
        if action is not None:
            if action['placement']=='local':
                assert p['raw_finish_ns']==p['dispatch_ns']+round(action['latency_s']*1e9)
                assert math.isclose(p['costs']['compute_energy_j'],action['energy_j'],abs_tol=1e-8)
            resident=action['action_id']
        charges=p['costs']
        subtotal=sum(charges[k] for k in ('preparation_energy_j','routing_energy_j','load_energy_j','compute_energy_j','decode_energy_j','radio_energy_j'))
        assert math.isclose(subtotal,charges['charged_energy_j'],abs_tol=1e-8)
        energy+=subtotal;bytes_mb+=p['communication_mb']
        if e['pipeline'] is not None:
            stages=e['pipeline']['stages']
            stage_j=sum(s['costs']['charged_energy_j'] for s in stages if s['dispatched'])
            assert math.isclose(stage_j,e['pipeline']['charged_energy_j'],abs_tol=1e-8)
            energy+=stage_j
        else:assert not e['emitted_reports']
        if not p['admitted']:
            assert e['pipeline'] is None and not e['score']['matched_target_ids'] and not e['score']['false_positive_components']
        last_finish=round(e['completion_time_s']*1e9)
        assert last_finish>=finish
        assert math.isclose(e['cumulative_nonflight_energy_j'],energy,abs_tol=1e-7)
        assert math.isclose(e['cumulative_communication_mb'],bytes_mb,abs_tol=1e-7)
        expected_battery=max(0.,item['battery']-(last_finish/1e9*120+energy)/(120*3600))
        assert math.isclose(e['battery_after'],expected_battery,abs_tol=1e-12)
        seen.append(e['observation_id']);skipped.extend(e['skipped_observation_ids'])
    assert sorted(seen+skipped)==list(range(18)) and len(set(seen+skipped))==18
    assert result['final_time_s']==max(42,last_finish/1e9)
    assert math.isclose(result['cumulative_energy_j'],result['final_time_s']*120+energy,abs_tol=1e-7)
    assert math.isclose(result['final_battery_frac'],item['battery']-result['cumulative_energy_j']/432000,abs_tol=1e-12)
    windows=[];focal=set(plan['cases'][item['flight']]['evaluator_target_ids'])
    for i,(a,b) in enumerate(plan['report_windows']):
        ev=[e for e in events if a<=e['capture_time_s']<b]
        eligible=[e for e in ev if e['execution']['success'] and e['completion_time_s']<=b+8]
        supported=[]
        for target in focal:
            times=sorted({e['capture_time_s'] for e in eligible if target in e['score']['matched_target_ids']})
            if len(times)>=2 and times[-1]-times[0]>=2:supported.append(target)
        false_components=sum(e['score']['false_positive_components'] for e in ev)
        reports=[r for r in result['report_quality']['reports'] if r['window_id']==i]
        correct=[r['target_id'] for r in reports if r['target_id'] is not None]
        valid=[]
        for r in reports:
            if r['target_id'] in focal and r['completion_time_s']<=b+8 and all(a<=x['capture_s']<b for x in r['evidence']):valid.append(r)
        targets={r['target_id'] for r in valid}
        checks=dict(repeated_evidence=len(supported)>=2 and false_components<=2,
            fresh_report_recall=len(targets)>=2,false_report_limit=len(reports)-len(correct)<=2,
            duplicate_report_limit=len(correct)-len(set(correct))<=2)
        w=outcome['windows'][i]
        assert checks==w['checks'] and all(checks.values())==w['success']
        first=min((r['completion_time_s'] for r in valid),default=None)
        assert w['first_valid_report_wait_s']==(None if first is None else first-a)
        assert w['deadline_misses']==sum(e['perception']['deadline_missed'] for e in ev)
        windows.append(all(checks.values()))
    assert outcome['dynamic_mission_success']==(all(windows) and all(result['constraints'].values()))
    return len(events)


def audit(root):
    root=Path(root);plan=read(root/'plan.json');rows=read(root/'rows.json')
    assert len(rows)==len(plan['grid']) and {r['run_id'] for r in rows}=={r['run_id'] for r in plan['grid']}
    by_id={r['run_id']:r for r in rows};traces={};decisions=0
    for item in plan['grid']:
        row=by_id[item['run_id']];folder=root/row['path']
        assert all(row[k]==v for k,v in item.items())
        for name,h in row['artifacts_sha256'].items():assert sha(folder/name)==h
        assert read(folder/'protocol.json')['plan_sha256']==sha(root/'plan.json')
        e=read(folder/'events.json');r=read(folder/'results.json');o=read(folder/'outcome.json')
        decisions+=check_run(plan,item,e,r,o)
        assert row['dynamic_mission_success']==o['dynamic_mission_success']
        assert row['window_successes']==o['window_successes']
        traces[item['run_id']]=e
    def signature(events,until):
        return [(e['observation_id'],e['selected_action_id'],e['completion_time_s'],e['selected_mask_sha256'],
            e['cumulative_nonflight_energy_j'],e['emitted_reports']) for e in events if e['capture_time_s']<until]
    pairs=0
    for item in plan['grid']:
        if item['policy']=='quality_current':
            other=item['run_id'].removesuffix('quality_current')+'quality_initial'
            until=36 if item['intent_schedule']=='no_change' else 12
            assert signature(traces[item['run_id']],until)==signature(traces[other],until)
            pairs+=1
    out=dict(passed=True,runs=len(rows),decisions=decisions,matched_prefix_checks=pairs,
        plan_sha256=sha(root/'plan.json'),rows_sha256=sha(root/'rows.json'),auditor_sha256=sha(__file__),
        scope=__doc__)
    save(root/'audit.json',out);return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();print(audit(a.root))
