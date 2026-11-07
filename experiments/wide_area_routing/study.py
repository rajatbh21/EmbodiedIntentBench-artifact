"""A locked first integration pilot on three wide-area physical recordings.

The flight is complete; perception requests occur in eight authored inspection
windows. All policies share the same opportunities, actual neural predictions,
paid deadline executor and resource assumptions. This is development evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import itertools
import multiprocessing
from pathlib import Path
import shutil
import time

from experiments.mission_runtime_profile import read,save,sha,implementation_inventory
from experiments.perception_router_suite.policy import make_policy
from experiments.wide_area_routing.adapter import (
    opportunity_indices,make_intent,load_bank,load_case,network_trace,checked_recording)

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'results/wide_area_routing_pilot_v1_20261028'
COHORT=ROOT/'results/wide_area_missions_v2_20261028'
PARENT=ROOT/'results/mission_joint_constraints_v1_20261027'
LEARNED=ROOT/'results/perception_router_suite_v1_20261028/training/profile.json'
FLIGHTS=('shoreline_loop','wide_area_search','sector_revisit')
POLICIES=('fixed_local_fast','fixed_local_heavy','fixed_remote_union','deadline_aware',
          'resource_threshold','accuracy_latency','supervised_utility','linucb_frozen')
_STATE=None


def require(value,message):
    if not value:raise ValueError(message)


def prepare(root=OUTPUT,cohort=COHORT):
    from experiments.wide_area_missions.cohort import verify as verify_cohort
    from experiments.wide_area_routing.runner import run_case
    from experiments.mission_task_deadlines.contract import DeadlineProfile
    from experiments.wide_area_routing.inference import verify_frozen
    from experiments.mission_task_deadlines.perception import execute_perception
    root=Path(root).resolve();cohort=Path(cohort).resolve()
    require(not (root/'plan.json').exists(),'Preserve the existing frozen replay plan')
    acquisition=verify_cohort(cohort)
    parent=read(PARENT/'plan.json');learned=read(LEARNED)
    actions=parent['actions_by_scale']['1.0']
    require(learned['action_signatures']=={a['action_id']:a['source_signature'] for a in actions},
            'Learned profiles and frozen actions differ')
    root.mkdir(parents=True,exist_ok=True)
    frozen={};source_inputs={}
    def copy(source,relative):
        source=Path(source);destination=root/relative
        require(not destination.exists(),'Cannot overwrite frozen input '+relative)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination)
        source_inputs[str(source.resolve())]=sha(source);frozen[relative]=sha(destination)
        return relative
    profile_paths={name:copy(PARENT/path,'inputs/'+name+'.json')
                   for name,path in parent['profile_paths'].items()}
    learned_path=copy(LEARNED,'inputs/learned_profile.json')
    copy(cohort/'protocol.json','inputs/acquisition_protocol.json')
    sensor_path=ROOT/'results/wide_area_sensor_audit_20261028.json'
    if sensor_path.exists():copy(sensor_path,'inputs/pre_neural_sensor_diagnostic.json')
    cases={}
    for flight in FLIGHTS:
        folder=cohort/flight;rec=read(folder/'recording/recording.json')
        design=read(folder/'mission_design.json')
        spec_path=root/'predictions'/flight/'selection.json'
        require(spec_path.exists(),'Freeze RGB inference selection before replay plan')
        verify_frozen(spec_path.parent)
        selection=read(spec_path)
        indices=opportunity_indices(rec,design)
        require(indices==[f['capture_id'] for f in selection['frames']],
                'Independent replay and inference schedules disagree')
        require(selection['recording_identity']==rec['recording_identity'] and
                selection['recording_sha256']==sha(folder/'recording/recording.json'),
                'Inference selection belongs to a different physical flight')
        frozen[str(spec_path.relative_to(root))]=sha(spec_path)
        clock=rec['capture_schedule'];duration=(clock['end_ns']-clock['epoch_ns'])/1e9
        cases[flight]=dict(scenario_path=copy(folder/'scenario.json',f'inputs/{flight}/scenario.json'),
            mission_design_path=copy(folder/'mission_design.json',f'inputs/{flight}/mission_design.json'),
            recording_sha256=sha(folder/'recording/recording.json'),
            recording_identity=rec['recording_identity'],capture_schedule=clock,
            duration_s=duration,opportunity_indices=indices,
            total_physical_captures=len(rec['frames']),scheduled_task_opportunities=len(indices),
            intent_paths={})
        for battery in (.8,.95):
            path=f'inputs/{flight}/intent_{battery}.json'
            save(root/path,make_intent(flight,duration,design,battery).document())
            frozen[path]=sha(root/path);cases[flight]['intent_paths'][str(battery)]=path
    factors=dict(flights=list(FLIGHTS),policies=list(POLICIES),batteries=[.8,.95],
                 links=['good','intermittent'],perception_budgets_s=[2,8])
    grid=[]
    for flight,policy,battery,link,budget in itertools.product(
        FLIGHTS,POLICIES,factors['batteries'],factors['links'],factors['perception_budgets_s']):
        grid.append(dict(flight=flight,policy=policy,battery=battery,link=link,
            perception_budget_s=budget,
            run_id=f'{flight}/{link}/charge_{battery}/budget_{budget}s/{policy}'))
    plan=dict(schema_version='wide_area_scheduled_routing_pilot_v1',
        prepared_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),cohort=str(cohort),
        cases=cases,factors=factors,grid=grid,expected_runs=len(grid),
        parent_plan_sha256=sha(PARENT/'plan.json'),actions=actions,
        policy_costs=parent['policy_costs'],profile_paths=profile_paths,
        learned_profile_path=learned_path,assets=acquisition['asset_directory'],
        resources=dict(battery_capacity_wh=120.,flight_power_w=120.,
            min_final_battery_frac=.2,communication_budget_mb=10000.),
        task_budgets=dict(perception_s=[2,8],association_ms=150,verification_ms=30),
        inference_specification_sha256=sha(root/'predictions/specification.json'),
        source_inputs_sha256=source_inputs,frozen_files=frozen,
        opportunity_rule='Original canonical captures satisfying abs(mission_time - authored_nominal_sector_pass) <= 8s; no snapping.',
        mission_criteria='All eight sectors: at least two of three targets with two IoU>=0.3 matches separated by>=2s, at most two false components per sector; report>=12/24 people by flight end, <=2 false reports and <=2 duplicates; retain20% battery reserve.',
        interpretation='Scheduled inspection of known mission sectors; not continuous unknown-target area search. Three routes on one known terrain, deterministic development integration pilot.',
        timing='Per-call deadline begins at capture. Full flight, transit, late work and endpoint hold remain charged. Mask inference is actually run on CPU; replay service costs reuse earlier frozen host-p95 sensitivity parameters.',
        network='Virtual link only: good50/100Mbps,30msRTT; intermittent repeats the previous140s trace (outages45-65s and95-105s). Policies see current snapshot only.',
        battery='120Wh pack and120W flight are configured assumptions, not measured wide-flight energy. Charge80/95%; reserve20%. Battery and10GB traffic limits may not bind.',
        forecast='Existing adaptive policies conservatively reserve cheap inference every2s including transit. This is retained unchanged, rather than silently modifying existing router mechanisms.',
        design_chronology='Geometry-only windows fixed before neural inference. Sensor-GT visibility/matching diagnostics inspected before the replay plan; neural outcomes were not queried to choose windows, thresholds, costs or policies.',
        learned_models_updated=False,policy_receives_ground_truth=False,
        future_link_visible_to_policy=False,paths_changed_by_policy=False)
    # Instantiate every policy before freezing, without evaluating any RGB/output.
    for policy in POLICIES:
        item=dict(grid[0],policy=policy)
        inputs=load_case(root,plan,item)
        make_policy(policy,inputs['actions'],inputs['costs'],
            DeadlineProfile('pilot_2s',2_000_000_000,150_000_000,30_000_000),
            inputs['profiles']['recall'],inputs['profiles']['risk'],inputs['learned_profile'])
    plan['implementation_sha256']=implementation_inventory()
    save(root/'plan.json',plan)
    verify(root)
    print(f'Frozen {len(grid)} paired replays over {sum(len(c["opportunity_indices"]) for c in cases.values())} geometry-selected captures',flush=True)
    return plan


def verify(root):
    from experiments.wide_area_missions.cohort import verify as verify_acquisition
    root=Path(root);plan=read(root/'plan.json')
    verify_acquisition(Path(plan['cohort']))
    for field in ('implementation_sha256','source_inputs_sha256'):
        for path,digest in plan[field].items():
            require(sha(Path(path))==digest,'Changed frozen source/input '+path)
    for path,digest in plan['frozen_files'].items():
        require(sha(root/path)==digest,'Changed frozen replay input '+path)
    require(sha(root/'predictions/specification.json')==plan['inference_specification_sha256'],
            'Changed inference specification')
    for flight in plan['cases']:checked_recording(root,plan,flight)
    return plan


def _initialize(root,plan,flight):
    global _STATE
    from experiments.lake_tracking_runtime import load_public_terrain
    terrain,_=load_public_terrain(Path(plan['assets']))
    bank=load_bank(root,plan,flight)
    manifest=read(Path(root)/'predictions'/flight/'manifest.json')
    _STATE=(root,plan,terrain,bank,manifest)


def _run(item):
    from experiments.wide_area_routing.runner import run_case
    root,plan,terrain,bank,manifest=_STATE
    row=run_case(root,plan,item,terrain,bank,manifest)
    save(Path(root)/row['path']/'row.json',row)
    return row


def execute(root=OUTPUT,workers=2):
    root=Path(root).resolve();plan=verify(root);rows=[]
    require(workers in (1,2,3,4),'Use one to four replay workers')
    started=time.perf_counter()
    for flight in plan['cases']:
        pred=root/'predictions'/flight
        require((pred/'completion.json').exists(),'Actual inference incomplete: '+flight)
        completion=read(pred/'completion.json')
        require(completion['complete'] and completion['manifest_sha256']==sha(pred/'manifest.json'),
                'Incomplete or changed prediction manifest')
        manifest=read(pred/'manifest.json')
        require(manifest['recording_sha256']==plan['cases'][flight]['recording_sha256'] and
            manifest['capture_count']==len(plan['cases'][flight]['opportunity_indices']),
            'Wrong prediction recording or observation set')
        for action in plan['actions']:
            require(manifest['models'][action['source_config_id']]['signature']==action['source_signature'],
                    'Wrong neural prediction semantics')
        pending=[]
        for item in (i for i in plan['grid'] if i['flight']==flight):
            row_path=root/'runs'/item['run_id']/'row.json'
            if row_path.exists():
                row=read(row_path)
                require(read(row_path.parent/'protocol.json')['plan_sha256']==sha(root/'plan.json'),
                        'Existing run belongs to another plan')
                for filename,digest in row['artifacts_sha256'].items():
                    require(sha(row_path.parent/filename)==digest,'Changed completed run')
                rows.append(row)
            else:pending.append(item)
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize,initargs=(root,plan,flight)) as pool:
            futures={pool.submit(_run,item):item for item in pending}
            for future in as_completed(futures):
                row=future.result();rows.append(row)
                save(root/'progress.json',dict(completed_runs=len(rows),expected_runs=plan['expected_runs']))
                print(f'{len(rows)}/{plan["expected_runs"]} {row["run_id"]}: phases={row["successful_phases"]}/8 success={row["mission_success"]}',flush=True)
    require(len(rows)==plan['expected_runs'],'Incomplete replay grid')
    rows.sort(key=lambda r:r['run_id']);save(root/'run_index.json',rows)
    save(root/'completion.json',dict(complete=True,runs=len(rows),
        elapsed_host_s=time.perf_counter()-started,plan_sha256=sha(root/'plan.json'),
        run_index_sha256=sha(root/'run_index.json'),
        prediction_manifest_sha256={f:sha(root/'predictions'/f/'manifest.json') for f in plan['cases']}))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','verify','execute'])
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    if a.mode=='prepare':prepare(a.output)
    elif a.mode=='verify':verify(a.output)
    else:execute(a.output,a.workers)


if __name__=='__main__':main()
