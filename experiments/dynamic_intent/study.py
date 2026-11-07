"""Frozen development pilot for changes in urgency during fixed flights.

Reuses six previously evaluated physical recordings and real-model output banks.
This is a new dynamic-intent protocol, not fresh held-out or real-drone evidence.
"""
from pathlib import Path
import argparse
import copy
import json
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from experiments.mission_runtime_profile import read, save, sha
from experiments.strong_routing_20261030 import study as parent, round2
from experiments.wide_area_diagnosis import paid_clip as paid
from experiments.dynamic_intent.protocol import SCHEDULES, WINDOWS
from experiments.dynamic_intent.policy import POLICIES

PARENT=Path('results/strong_routing_round2_comparison_v1_20261030')
DEFAULT=Path('results/dynamic_intent_v1_20261031')

def source_hashes():
    return {**round2.source_hashes(),**{str(p.resolve()):sha(p) for p in (Path(__file__).parent/n for n in ('protocol.py','policy.py','scoring.py','runtime.py','study.py','audit.py'))}}

def grid(cases):
    return [dict(flight=k,policy=p,intent_schedule=s,link=link,energy_condition=energy,
        battery=parent.initial_battery(case,energy),
        run_id=f'{k}/{s}/{link}/{energy}/{p}')
        for k,case in sorted(cases.items()) for s in SCHEDULES
        for link in ('good','degraded') for energy in ('ample','residual_250j') for p in POLICIES]

def sensor_gate(plan, bindings):
    rows=[]
    for binding in bindings:
        bank=paid.load_bank(binding);case=plan['cases'][binding['case_id']];windows=[]
        state=parent.load_case(PARENT,plan,dict(flight=case['case_id'],link='good',energy_condition='ample',battery=.95))
        scenario=state['scenario']
        for i,(a,b) in enumerate(WINDOWS):
            times={t:[] for t in case['evaluator_target_ids']}
            for index,frame in bank['frames'].items():
                t=(scenario.schedule.at(index)-scenario.schedule.epoch_ns)/1e9
                if a<=t<b:
                    visible=paid.evaluation_frame(frame,index,t,scenario).visible_target_ids
                    for target in times:
                        if target in visible:times[target].append(t)
            supported=[t for t,ts in times.items() if len(ts)>=2 and ts[-1]-ts[0]>=2]
            windows.append(dict(window_id=i,evidence_capture_times=times,supported_targets=supported,
                sensor_possible=len(supported)>=2))
        rows.append(dict(flight=case['case_id'],windows=windows,sensor_possible=all(w['sensor_possible'] for w in windows)))
    return dict(cases=rows,scope='Optimistic visibility bound only, ignoring model errors and paid scheduling. Pass is not proof of paid feasibility. All six cases are retained; impossible windows are disclosed, not silently dropped.')

def prepare(root):
    root=Path(root);plan=copy.deepcopy(round2.verify(PARENT))
    if root.exists():raise ValueError('Preserve existing study; use a new output directory')
    bindings=read(PARENT/'native_bindings.json')['bindings']
    gate=sensor_gate(plan,bindings)
    root.mkdir(parents=True)
    for path in [*plan['profiles'].values(),plan['calibration_path']]:
        dst=root/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PARENT/path,dst)
    (root/'predictions').symlink_to((PARENT/'predictions').resolve(),target_is_directory=True)
    save(root/'sensor_gate.json',gate)
    save(root/'native_bindings.json',dict(bindings=bindings))
    plan.update(schema_version='dynamic_intent_fixed_flight_v1',prepared_utc=datetime.now(timezone.utc).isoformat(),
        parent_root=str(PARENT.resolve()),parent_plan_sha256=sha(PARENT/'plan.json'),
        source_hashes=source_hashes(),grid=grid(plan['cases']),expected_runs=len(grid(plan['cases'])),
        intent_schedules={k:[list(x) for x in v] for k,v in SCHEDULES.items()},report_windows=[list(w) for w in WINDOWS],
        sensor_gate_sha256=sha(root/'sensor_gate.json'),bindings_sha256=sha(root/'native_bindings.json'),
        prediction_manifests={b['case_id']:sha(root/'predictions'/b['case_id']/'manifest.json') for b in bindings},
        development_only=True,outcome_awareness='All source recording/model outcomes were previously observed. No policy/threshold fitting on dynamic outcomes.',
        scope='Three periodic local-sector reports within a 42 s recorded flight. Not an eight-sector mission, fresh acquisition, real network or natural-language intent parser test.',
        semantics=dict(request_binding='Intent budget is bound at input release. Existing work finishes under its original budget; no cancellation at intent updates.',
            notification='Update effective on absolute clock. Only current version and budget are exposed at next available decision; future updates are private.',
            freshness='Every schedule has the same three status windows. Fresh repeated observations and reports required in each, with equal 8 s delivery allowance.',
            state='Perception residency, battery, bytes and clock persist. Each status window starts fresh downstream state and pays its normal load costs, including no_change.',
            duplicates='Report identities and duplicate counting are local to requested status windows.',
            controls='quality_current and quality_initial share every score, cost and guard; only their observed intent budget differs.',
            energy='Inherited declared energy values; no new electrical measurement.',
            interpretation='Schedule change does not by itself prove useful adaptation. All outcomes retained.'))
    save(root/'plan.json',plan)
    return dict(expected_runs=plan['expected_runs'],sensor_possible=sum(r['sensor_possible'] for r in gate['cases']),cases=len(gate['cases']))

def verify(root):
    root=Path(root);p=read(root/'plan.json')
    assert p['source_hashes']==source_hashes(),'Frozen implementation changed'
    assert p['parent_plan_sha256']==sha(PARENT/'plan.json')
    assert p['grid']==grid(p['cases']) and p['expected_runs']==504
    assert p['intent_schedules']=={k:[list(x) for x in v] for k,v in SCHEDULES.items()}
    assert p['report_windows']==[list(w) for w in WINDOWS]
    assert p['sensor_gate_sha256']==sha(root/'sensor_gate.json')
    assert p['bindings_sha256']==sha(root/'native_bindings.json')
    for k,h in p['prediction_manifests'].items():assert sha(root/'predictions'/k/'manifest.json')==h
    for k,path in p['profiles'].items():assert sha(root/path)==p['profile_sha256'][k]
    assert sha(root/p['calibration_path'])==p['calibration_sha256']
    return p

_STATE=None

def initialize(root, plan, binding):
    global _STATE
    terrain,_=paid.load_public_terrain(Path(plan['assets']))
    _STATE=(root,plan,terrain,paid.load_bank(binding),read(Path(root)/'predictions'/binding['case_id']/'manifest.json'))

def run(item):
    from experiments.dynamic_intent.runtime import run_case
    root,plan,terrain,bank,manifest=_STATE
    row=run_case(root,plan,item,terrain,bank,manifest)
    save(Path(root)/row['path']/'row.json',row)
    return row

def execute(root,workers=2):
    root=Path(root);plan=verify(root)
    if workers not in (1,2,3):raise ValueError('Use one to three worker processes')
    rows=[]
    for binding in read(root/'native_bindings.json')['bindings']:
        pending=[]
        for item in (r for r in plan['grid'] if r['flight']==binding['case_id']):
            path=root/'runs'/item['run_id']/'row.json'
            if path.exists():
                row=read(path)
                assert all(row[k]==v for k,v in item.items())
                assert read(path.parent/'protocol.json')['plan_sha256']==sha(root/'plan.json')
                for f,h in row['artifacts_sha256'].items():assert sha(path.parent/f)==h
                rows.append(row)
            else:pending.append(item)
        if pending:
            with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(root,plan,binding)) as pool:
                for row in pool.map(run,pending):
                    rows.append(row)
                    if len(rows)%14==0:print(f'{len(rows)}/{len(plan["grid"])}',flush=True)
    assert len(rows)==len(plan['grid'])
    save(root/'rows.json',sorted(rows,key=lambda r:r['run_id']))
    save(root/'completion.json',dict(complete=True,runs=len(rows),plan_sha256=sha(root/'plan.json'),rows_sha256=sha(root/'rows.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','execute','verify'));p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.mode=='prepare':print(json.dumps(prepare(a.output)))
    elif a.mode=='verify':print(verify(a.output)['expected_runs'])
    else:execute(a.output,a.workers)
