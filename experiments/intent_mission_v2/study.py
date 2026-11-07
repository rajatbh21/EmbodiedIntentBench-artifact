"""Prespecified cumulative mission experiment; known clips, no fresh-test claim."""
import argparse
import copy
import json
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
from experiments.mission_runtime_profile import read,save,sha
from experiments.dynamic_intent import study as previous
from experiments.strong_routing_20261030 import study as parent
from experiments.wide_area_diagnosis import paid_clip as paid
from experiments.intent_mission_v2.protocol import SCHEDULES,WINDOWS
from experiments.intent_mission_v2.policy import POLICIES

DEFAULT=Path('results/intent_mission_v2_20261031')
SOURCE=Path('results/dynamic_intent_v1_20261031')

def source_hashes():
    return {**previous.source_hashes(),**{str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py') if p.name not in ('report.py',)}}

def split(case):return 'development' if case.startswith('shoreline_loop/') else 'transfer'

def grid(cases):
    return [dict(flight=k,policy=p,intent_schedule=s,link=link,energy_condition=energy,
        battery=parent.initial_battery(case,energy),split=split(k),run_id=f'{k}/{s}/{link}/{energy}/{p}')
        for k,case in sorted(cases.items()) for s in SCHEDULES
        for link in ('good','degraded') for energy in ('ample','residual_250j') for p in POLICIES]

def prepare(root):
    root=Path(root);p=copy.deepcopy(previous.verify(SOURCE))
    if root.exists():raise ValueError('Preserve prior experiments')
    root.mkdir(parents=True)
    for path in [*p['profiles'].values(),p['calibration_path']]:
        dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/path,dest)
    shutil.copy2(SOURCE/'native_bindings.json',root/'native_bindings.json')
    (root/'predictions').symlink_to((SOURCE/'predictions').resolve(),target_is_directory=True)
    p.update(schema_version='cumulative_dynamic_intent_study_v2',prepared_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=source_hashes(),grid=grid(p['cases']),expected_runs=len(grid(p['cases'])),
        source_plan_sha256=sha(SOURCE/'plan.json'),report_windows=[list(w) for w in WINDOWS],
        intent_schedules={k:[list(x) for x in v] for k,v in SCHEDULES.items()},
        scope='One cumulative local-sector inspection with dynamic per-request urgency; not a periodic resurvey or full eight-sector mission.',
        outcome_awareness='All six source clips were previously evaluated. Shoreline clips are development; the remaining four clips are a locked transfer replay, not unseen or prospectively acquired data.',
        stopping_rule='Run all 648 cells. No change to policies, costs, thresholds, cases or schedule after the lock. Retain all failures.',
        changes=['One 36 s mission evidence window instead of three fresh 12 s reports',
            'Persistent tracking, verification and report deduplication across urgency updates',
            'One global cap of 2 raw false components, 2 false reports and 2 duplicate reports',
            'Two matched risk-policy variants added using unchanged pooled development priors'],
        semantics=dict(request_binding='Budget at input release; pending work keeps its original deadline and costs.',
            state='Continuous evidence, tracking, report deduplication, residency, battery and clock.',
            quality='Confirm and report >=2/3 targets with >=2 captures separated by >=2s; deadline44s; global raw and report false caps2.',
            notification='Only current update is exposed; no future schedule or evaluator identity.',
            interpretation='This is a different mission contract, motivated by urgency-only user changes. v1 outcomes are not relabeled.'))
    p.pop('sensor_gate_sha256',None)
    save(root/'plan.json',p)
    return p

def verify(root):
    root=Path(root);p=read(root/'plan.json')
    assert p['source_hashes']==source_hashes(),'Frozen implementation changed'
    assert p['source_plan_sha256']==sha(SOURCE/'plan.json')
    assert p['grid']==grid(p['cases']) and p['expected_runs']==648
    assert p['report_windows']==[list(w) for w in WINDOWS]
    assert p['intent_schedules']=={k:[list(x) for x in v] for k,v in SCHEDULES.items()}
    assert p['bindings_sha256']==sha(root/'native_bindings.json')
    for k,h in p['prediction_manifests'].items():assert sha(root/'predictions'/k/'manifest.json')==h
    for k,path in p['profiles'].items():assert sha(root/path)==p['profile_sha256'][k]
    assert sha(root/p['calibration_path'])==p['calibration_sha256']
    return p

_STATE=None

def initialize(root,plan,binding):
    global _STATE
    terrain,_=paid.load_public_terrain(Path(plan['assets']))
    _STATE=(root,plan,terrain,paid.load_bank(binding),read(Path(root)/'predictions'/binding['case_id']/'manifest.json'))

def run(item):
    from experiments.intent_mission_v2.runtime import run_case
    root,p,terrain,bank,manifest=_STATE
    row=run_case(root,p,item,terrain,bank,manifest)
    save(Path(root)/row['path']/'row.json',row)
    return row

def execute(root,selected='all',workers=2):
    root=Path(root);p=verify(root);rows=[]
    for b in read(root/'native_bindings.json')['bindings']:
        pending=[]
        for item in (i for i in p['grid'] if i['flight']==b['case_id'] and (selected=='all' or i['split']==selected)):
            path=root/'runs'/item['run_id']/'row.json'
            if path.exists():
                row=read(path);assert all(row[k]==v for k,v in item.items())
                assert read(path.parent/'protocol.json')['plan_sha256']==sha(root/'plan.json')
                for f,h in row['artifacts_sha256'].items():assert sha(path.parent/f)==h
                rows.append(row)
            else:pending.append(item)
        if pending:
            with ProcessPoolExecutor(max_workers=workers,initializer=initialize,initargs=(root,p,b)) as pool:
                for row in pool.map(run,pending):
                    rows.append(row)
                    if len(rows)%36==0:print(f'{selected}: {len(rows)} completed',flush=True)
    save(root/f'rows_{selected}.json',sorted(rows,key=lambda r:r['run_id']))
    if selected=='all':
        assert len(rows)==648
        save(root/'rows.json',sorted(rows,key=lambda r:r['run_id']))
        save(root/'completion.json',dict(complete=True,runs=len(rows),plan_sha256=sha(root/'plan.json'),rows_sha256=sha(root/'rows.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','verify','execute'));p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--split',choices=('all','development','transfer'),default='all');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.mode=='prepare':print(prepare(a.output)['expected_runs'])
    elif a.mode=='verify':print(verify(a.output)['expected_runs'])
    else:execute(a.output,a.split,a.workers)
