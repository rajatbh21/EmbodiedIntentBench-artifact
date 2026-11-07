"""Second and final prospective round: pooled development priors, fresh clips.

The complete first evaluation is retained as negative transfer evidence and then
promoted to development. No outcome-based filtering or threshold tuning occurs.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import math
from datetime import datetime, timezone
from pathlib import Path
import shutil
from types import FunctionType
import numpy as np

from experiments.mission_runtime_profile import read, save, sha
from experiments.mission_task_deadlines.study import require
from experiments.strong_routing_20261030 import study, policy, calibration_v2, remote_control, mission_risk
from experiments.wide_area_diagnosis import paid_clip as paid
from experiments.wide_area_diagnosis.physical_probe import verify_protocol
from experiments.wide_area_routing.inference import freeze_json

PARENT = Path('results/strong_routing_comparison_v2_20261030')
POLICIES = (*study.POLICIES, 'fixed_remote_yolo', 'previous_calibrated_evidence_rate', 'mission_risk_budget')


def make_grid(cases):
    return [dict(flight=key, policy=name, perception_budget_s=budget, link=link,
            energy_condition=energy, battery=study.initial_battery(case, energy),
            run_id=f'{key}/{name}/budget_{budget}/{link}/{energy}')
            for key,case in sorted(cases.items()) for name in POLICIES for budget in (2,8)
            for link in study.LINKS for energy in study.BATTERIES]


class NamedPolicy:
    def __init__(self, name, delegate):
        self.name, self.delegate = name, delegate
    def select(self, **inputs):
        result = self.delegate.select(**inputs)
        result['policy'] = self.name
        return result


def make_policy(name, actions, costs, profile, quality, risk, learned_profile=None):
    if name in paid.POLICIES:
        return paid.make_policy(name, actions, costs, profile, quality, risk)
    if name == 'fixed_remote_yolo':
        return remote_control.make_policy(name, actions, costs, profile, quality, risk)
    if name == 'previous_calibrated_evidence_rate':
        return NamedPolicy(name, policy.make_policy('calibrated_evidence_rate', actions, costs,
            profile, quality, risk, learned_profile['previous']))
    if name == 'mission_risk_budget':
        return mission_risk.make_policy(name, actions, costs, profile, quality, risk, learned_profile['current'])
    require(name in POLICIES, 'Undeclared second-round policy')
    return calibration_v2.make_policy(name, actions, costs, profile, quality, risk, learned_profile['current'])


def load_case(root, plan, item):
    result = study.load_case(root, plan, item)
    result['learned_profile'] = dict(current=result['learned_profile'],
        previous=read(Path(root)/plan['previous_calibration_path']))
    return result


def source_hashes():
    return {**study.source_hashes(), str(Path(__file__).resolve()):sha(__file__),
        str(Path(calibration_v2.__file__).resolve()):sha(calibration_v2.__file__),
        str(Path(remote_control.__file__).resolve()):sha(remote_control.__file__),
        str(Path(mission_risk.__file__).resolve()):sha(mission_risk.__file__)}


def split_integrity(native, calibration):
    """Check pixels, not merely episode names, before prospective inference."""
    by_hash={value:path for path,value in calibration['input_sha256'].items()}
    training=[]
    for frame in calibration['training_frames']:
        path=Path(by_hash[frame['source_sha256']])
        require(sha(path)==frame['source_sha256'],'Training source changed')
        with np.load(path,allow_pickle=False) as data:
            pixel_hash=hashlib.sha256(data['rgb'].tobytes()).hexdigest()
        training.append(dict(source=str(path), source_sha256=sha(path), rgb_sha256=pixel_hash))
    evaluation=[];cohort=Path(native['cohort'])
    for case in native['cases']:
        path=cohort/case['case_id']/'recording/recording.json';recording=read(path)
        require(recording['provenance']['physical_route_passed'],'Physical evaluation recording required')
        for index in case['evaluation_capture_ids']:
            frame=recording['frames'][index];source=path.parent/frame['file']
            require(sha(source)==frame['sha256'],'Evaluation source changed')
            with np.load(source,allow_pickle=False) as data:
                pixel_hash=hashlib.sha256(data['rgb'].tobytes()).hexdigest()
            evaluation.append(dict(source=str(source),source_sha256=sha(source),rgb_sha256=pixel_hash))
    shared_rgb={r['rgb_sha256'] for r in training}&{r['rgb_sha256'] for r in evaluation}
    shared_npz={r['source_sha256'] for r in training}&{r['source_sha256'] for r in evaluation}
    require(len(training)==162 and len(evaluation)==108 and not shared_rgb and not shared_npz,
            'Training/evaluation pixel overlap or missing input')
    return dict(passed=True,training_frames=training,evaluation_frames=evaluation,
        shared_rgb_sha256=sorted(shared_rgb),shared_source_sha256=sorted(shared_npz),
        scope='Exact RGB/capture overlap only. The map, layouts and visual assets remain known.')


def prepare(root, probe, calibration):
    root, probe, calibration = (Path(p).resolve() for p in (root,probe,calibration))
    require(not root.exists(), 'Preserve earlier comparisons')
    require(not (probe/'cases').exists() and not (probe/'completion.json').exists(),
            'Second round must be locked before its native inference')
    parent, native = study.verify(PARENT), verify_protocol(probe)
    learned=calibration_v2.verify_calibration(read(calibration))
    cases={c['case_id']:c for c in native['cases']}
    require(len(cases)==6 and all(c['focal_sector'] in ('sector_3','sector_7') and
        c['motion_s']==36 and c['duration_s']==42 for c in cases.values()), 'All six fresh round2 sectors required')
    root.mkdir(parents=True)
    freeze_json(root/'split_integrity.json',split_integrity(native,learned))
    for path in parent['profiles'].values():
        target=root/path;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/path,target)
    previous='inputs/previous_oblique_calibration.json'
    current='inputs/pooled_oblique_calibration.json'
    shutil.copyfile(PARENT/parent['calibration_path'],root/previous)
    shutil.copyfile(calibration,root/current)
    grid=make_grid(cases)
    plan=copy.deepcopy(parent)
    plan.update(schema_version='prospective_fixed_flight_routing_round2_v1',
        prepared_utc=datetime.now(timezone.utc).isoformat(),source_hashes=source_hashes(),
        previous_round_plan_sha256=sha(PARENT/'plan.json'), cases=cases,grid=grid,expected_runs=len(grid),
        split_integrity_sha256=sha(root/'split_integrity.json'),
        probe=str(probe),native_protocol_sha256=sha(probe/'protocol.json'),
        calibration_path=current,calibration_sha256=sha(root/current),
        previous_calibration_path=previous,previous_calibration_sha256=sha(root/previous),
        outcome_awareness='Sequential development: first6 prospective clips became development after their full native outcomes were observed. All162 historical+first-round frames pooled. No second-round native outputs existed at lock.',
        policy_changes='Same original prior formulas, scores, models and common guard; enlarged complete development set. First-round prior retained byte-identically on the same new clips. Fixed remote YOLO included. Separate mission_risk_budget additionally filters expected full-phase false-component count against the public intent cap.',
        trained_heads='Pooled162-frame constant priors for the three new objectives; old54-frame calibrated evidence policy retained; historical old heads remain for stale baselines.',
        mission_risk_rule='Expected FP = pooled per-capture FP rate times ceil(full phase duration / candidate capture cadence); filter expected FP <= public max_false_components then maximize pooled recall. Causal expectation heuristic, not a probability guarantee or observed-error feedback.',
        stopping_rule='Complete rounds1 and2 and consolidate all outcomes regardless performance. Do not tune policies, thresholds, sites or resource grid using round2 outputs.',
        endpoint_sensitivity=dict(registered_before_new_inference=True,
            rule='Separately rescore original execution by removing only raw-component cap from sector evidence; keep repeated evidence, timely correct reporting, false/duplicate report caps and all resource checks.',
            rationale='Previously registered GT fragment diagnostic; primary strict contract remains unchanged.',
            scope='Same original-intent executions; not a new public-intent experiment.'),
        interpretation='Second sequential prospective recording round on known Lake assets. All previous failures retained. No unseen-map, population significance or full-wide-mission claim.')
    plan['factors']['policies']=list(POLICIES);plan['factors']['source_case_ids']=list(cases)
    freeze_json(root/'plan.json',plan)
    return plan


def verify(root):
    root=Path(root);plan=read(root/'plan.json');parent=study.verify(PARENT)
    require(plan['source_hashes']==source_hashes(),'Frozen second-round implementation changed')
    require(plan['previous_round_plan_sha256']==sha(PARENT/'plan.json'),'First-round plan changed')
    require(sha(Path(plan['probe'])/'protocol.json')==plan['native_protocol_sha256'],'Native protocol changed')
    verify_protocol(plan['probe'])
    require(plan['grid']==make_grid(plan['cases']) and plan['expected_runs']==864,'Second-round grid changed')
    require(sha(root/'split_integrity.json')==plan['split_integrity_sha256'],'Split integrity receipt changed')
    for name,path in plan['profiles'].items():
        require(sha(root/path)==plan['profile_sha256'][name],'Historical head changed')
    require(sha(root/plan['calibration_path'])==plan['calibration_sha256'],'Pooled calibration changed')
    calibration_v2.verify_calibration(read(root/plan['calibration_path']))
    require(sha(root/plan['previous_calibration_path'])==plan['previous_calibration_sha256']==parent['calibration_sha256'],
            'Original prior baseline changed')
    for key in ('actions','policy_costs','resources','network','battery_design','accounting','reporting'):
        require(plan[key]==parent[key],'Undeclared intervention: '+key)
    return plan


def run_case(root,plan,item,terrain,bank,manifest):
    namespace=dict(paid.run_case.__globals__)
    namespace.update(load_case=load_case,make_policy=make_policy,source_hashes=source_hashes)
    return FunctionType(paid.run_case.__code__,namespace,paid.run_case.__name__)(root,plan,item,terrain,bank,manifest)


def _run(item):
    root,plan,terrain,bank,manifest=study._STATE
    row=run_case(root,plan,item,terrain,bank,manifest)
    save(Path(root)/row['path']/'row.json',row)
    return row


def execute(root,workers=3):
    namespace=dict(study.execute.__globals__)
    namespace.update(verify=verify,_run=_run)
    return FunctionType(study.execute.__code__,namespace,study.execute.__name__)(root,workers)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','verify','execute'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe',type=Path)
    parser.add_argument('--calibration',type=Path)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.output,args.probe,args.calibration)
    elif args.mode=='verify':verify(args.output)
    else:execute(args.output,args.workers)
