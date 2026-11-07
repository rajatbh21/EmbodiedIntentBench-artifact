"""Final posthoc sensitivity: all consecutive 6s outage positions, unchanged clips.

The frozen paid run loop's exact bytecode is reused with two explicit dependency
bindings: a link-trace case factory and an augmented source receipt. No global
runtime monkeypatch, policy update, target filtering, or capture change occurs.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import shutil
import time
from types import FunctionType

from experiments.mission_network_emulation import LinkSegment, LinkTrace
from experiments.mission_runtime_profile import read, save, sha
from experiments.mission_task_deadlines.study import require
from experiments.wide_area_diagnosis import paid_clip as paid
from experiments.wide_area_routing.inference import freeze_json

PARENT=Path('results/wide_area_viewpoint_paid_clip_v1_20261028')
LINKS=('good',*(f'outage_{start}_{start+6}' for start in range(0,36,6)))


def link_trace(name,duration):
    require(name in LINKS,'Undeclared exhaustive outage position')
    if name=='good':return paid.link_trace('good',duration)
    _,start,end=name.split('_');start,end=int(start),int(end)
    trace_end=round((duration+30)*1e9)
    require(trace_end>end*1_000_000_000,'Trace must extend beyond the full outage')
    intervals=[(0,start,False),(start,end,True),(end,duration+30,False)]
    return LinkTrace(tuple(LinkSegment(round(a*1e9),round(b*1e9),50_000_000,100_000_000,30_000_000,outage)
        for a,b,outage in intervals if b>a))


def load_case(root,plan,item):
    # The original factory only names its two original traces. Build that exact
    # scenario/intent/profile state, then replace the independent link trace.
    require(item['link'] in LINKS,'Run uses an undeclared outage position')
    case=paid.load_case(root,plan,dict(item,link='good'))
    case['trace']=link_trace(item['link'],case['scenario'].duration_s)
    return case


def source_hashes():
    return {**paid.source_hashes(),str(Path(__file__).resolve()):sha(__file__)}


def bound_run_loop():
    # Keep every original runner global by identity except the two declared
    # dependencies. FunctionType creates an isolated namespace, not a mutation
    # of the frozen module or other simultaneously running studies.
    namespace=dict(paid.run_case.__globals__)
    namespace.update(load_case=load_case,source_hashes=source_hashes)
    return FunctionType(paid.run_case.__code__,namespace,paid.run_case.__name__,
                        paid.run_case.__defaults__,paid.run_case.__closure__)


def run_case(root,plan,item,terrain,bank,manifest):
    return bound_run_loop()(root,plan,item,terrain,bank,manifest)


def make_grid(cases):
    return [dict(flight=case,policy=policy,perception_budget_s=budget,link=link,battery=.95,
                 run_id=f'{case}/{policy}/budget_{budget}/{link}')
            for case in cases for policy in paid.POLICIES for budget in (2,8) for link in LINKS]


def prepare(root):
    root=Path(root).resolve();require(not root.exists(),'Preserve prior network sensitivity study')
    parent=paid.verify(PARENT);completion=read(PARENT/'completion.json')
    require(completion['complete'] and completion['runs']==72 and
        completion['plan_sha256']==sha(PARENT/'plan.json') and
        completion['run_index_sha256']==sha(PARENT/'run_index.json'),'Completed parent study required')
    require(len(parent['cases'])==3 and all(c['motion_s']==36 for c in parent['cases'].values()),
            'All three original 36-second viewpoint passes required')
    plan=copy.deepcopy(parent)
    plan.update(schema_version='local_inspection_network_position_sensitivity_v1',
        prepared_utc=datetime.now(timezone.utc).isoformat(),source_hashes=source_hashes(),
        sensitivity_parent=str(PARENT.resolve()),sensitivity_parent_plan_sha256=sha(PARENT/'plan.json'),
        sensitivity_parent_completion_sha256=sha(PARENT/'completion.json'),
        grid=make_grid(parent['cases']),expected_runs=252,
        only_intervention='Link outages occupy each of six exhaustive nonoverlapping6s bins in[0,36), plus unchanged good control.',
        adaptation='Exact frozen paid run-loop bytecode; only link case factory and source receipt dependencies rebound in isolated namespace.',
        outcome_awareness='Final posthoc sensitivity after viewing prior native and paid results, including null[4,10) success contrasts. All six positions and three sites retained; no further tuning.',
        interpretation='Exhaustive synthetic outage-position sensitivity on three development sites; no representative real-network, held-out, or full-wide-mission claim.')
    plan['factors']['links']=list(LINKS)
    plan['network'].pop('outage_interval_s',None)
    plan['network'].update(outage_intervals_s=[[start,start+6] for start in range(0,36,6)],
        outage_interval='Half-open; all consecutive6s positions covering the36s motion window; same clip clock for every site')
    root.mkdir(parents=True)
    for path in plan['profiles'].values():
        dest=root/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/path,dest)
    freeze_json(root/'plan.json',plan)
    return plan


def verify(root):
    root=Path(root);plan=read(root/'plan.json');parent=paid.verify(PARENT)
    require(plan['source_hashes']==source_hashes(),'Frozen network adapter or paid runtime changed')
    require(plan['sensitivity_parent_plan_sha256']==sha(PARENT/'plan.json') and
            plan['sensitivity_parent_completion_sha256']==sha(PARENT/'completion.json'),'Frozen paid parent changed')
    for key in ('cases','actions','policy_costs','resources','assets','profiles','profile_sha256',
                'probe','native_protocol_sha256','parent_plan_sha256','deadline_semantics','scope','accounting','reporting'):
        require(plan[key]==parent[key],'Non-network intervention: '+key)
    require(plan['grid']==make_grid(plan['cases']) and plan['expected_runs']==len(plan['grid'])==252,
            'Exhaustive network grid changed')
    for name,path in plan['profiles'].items():require(sha(root/path)==plan['profile_sha256'][name],'Frozen profile changed')
    return plan


def _initialize(root,plan,binding):
    global _STATE
    terrain,_=paid.load_public_terrain(Path(plan['assets']))
    _STATE=(root,plan,terrain,paid.load_bank(binding),read(Path(root)/'predictions'/binding['case_id']/'manifest.json'))


def _run(item):
    root,plan,terrain,bank,manifest=_STATE
    row=run_case(root,plan,item,terrain,bank,manifest)
    save(Path(root)/row['path']/'row.json',row)
    return row


def good_control_parity(root,rows):
    checked=[]
    for row in rows:
        if row['link']!='good':continue
        prior=PARENT/'runs'/row['run_id'];current=Path(root)/row['path']
        receipt={name:sha(current/(name+'.json')) for name in ('intent','events','results','outcome')}
        require(all(value==sha(prior/(name+'.json')) for name,value in receipt.items()),
                'Good-control semantics differ from the frozen parent')
        checked.append(dict(run_id=row['run_id'],identical_sha256=receipt))
    require(len(checked)==36,'All36 paired good controls required')
    return dict(passed=True,checked_runs=36,exact_artifacts_per_run=4,rows=checked,
        scope='Byte-identical intent, events, execution and outcome; protocol differs only by declared plan/source bindings.')


def execute(root,workers=2):
    root=Path(root).resolve();plan=verify(root);bindings=paid.bind_predictions(root,plan)
    require(workers in (1,2,3),'One to three replay workers required')
    rows=[];started=time.perf_counter()
    for binding in bindings:
        pending=[]
        for item in (i for i in plan['grid'] if i['flight']==binding['case_id']):
            folder=root/'runs'/item['run_id'];path=folder/'row.json'
            if path.exists():
                row=read(path)
                require(all(row[k]==v for k,v in item.items()),'Changed replay cell')
                require(read(folder/'protocol.json')['plan_sha256']==sha(root/'plan.json'),'Wrong existing run plan')
                for name,value in row['artifacts_sha256'].items():require(sha(folder/name)==value,'Changed completed replay')
                rows.append(row)
            else:pending.append(item)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize,initargs=(root,plan,binding)) as pool:
            for future in as_completed([pool.submit(_run,item) for item in pending]):
                row=future.result();rows.append(row)
                save(root/'progress.json',dict(completed_runs=len(rows),expected_runs=252))
                print(f'{len(rows)}/252 {row["run_id"]}: success={row["local_inspection_success"]}',flush=True)
    require(len(rows)==len({r['run_id'] for r in rows})==252,'Incomplete network grid')
    verify(root);rows.sort(key=lambda r:r['run_id'])
    freeze_json(root/'good_control_parity.json',good_control_parity(root,rows))
    freeze_json(root/'run_index.json',rows)
    freeze_json(root/'completion.json',dict(complete=True,runs=252,
        plan_sha256=sha(root/'plan.json'),run_index_sha256=sha(root/'run_index.json'),
        native_bindings_sha256=sha(root/'native_bindings.json'),good_control_parity_sha256=sha(root/'good_control_parity.json'),
        prediction_manifest_sha256={c:sha(root/'predictions'/c/'manifest.json') for c in plan['cases']},
        elapsed_host_s=time.perf_counter()-started,local_inspection_successes=sum(r['local_inspection_success'] for r in rows),
        full_wide_mission_evaluated=False))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','verify','execute'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.output)
    elif args.mode=='verify':verify(args.output)
    else:execute(args.output,args.workers)
