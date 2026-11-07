"""Separate placement-attribution control registered before new score inspection.

Native inference had already started at registration. This is a supplemental
fixed-remote-YOLO control, not a retroactive change to the primary 648-cell plan.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
from types import FunctionType

from experiments.mission_runtime_profile import read, save, sha
from experiments.mission_task_deadlines.study import require
from experiments.strong_routing_20261030 import study
from experiments.wide_area_diagnosis import paid_clip as paid
from experiments.wide_area_routing.inference import freeze_json

PRIMARY = Path('results/strong_routing_comparison_v2_20261030')
NAME = 'fixed_remote_yolo'
ACTION = 'REMOTE__CFG_YOLO11S_960_ROT0'


class RemoteYolo:
    def __init__(self, delegate):
        self.delegate = delegate

    def select(self, **observation):
        result = self.delegate.select(**observation)
        action = next(row for row in result['candidates'] if row['action_id'] == ACTION)
        result.update(policy=NAME, action_id=ACTION if action['legal'] else None,
            reason='fixed_control_ignores_resource_forecast' if action['legal'] else 'no_legal_fixed_action')
        return result


def make_policy(name, actions, costs, profile, quality, risk, learned_profile=None):
    require(name == NAME, 'Only the supplemental fixed remote YOLO is declared')
    return RemoteYolo(paid.original_policy('deadline_aware', actions, costs, profile, quality, risk))


def source_hashes():
    return {**study.source_hashes(), str(Path(__file__).resolve()): sha(__file__)}


def make_grid(parent):
    return [dict(row, policy=NAME, run_id=row['run_id'].replace('/fixed_local_yolo/', f'/{NAME}/'))
            for row in parent['grid'] if row['policy'] == 'fixed_local_yolo']


def prepare(root):
    root = Path(root).resolve()
    require(not root.exists(), 'Preserve previous supplemental control')
    parent = study.verify(PRIMARY)
    plan = copy.deepcopy(parent)
    plan.update(schema_version='supplemental_fixed_remote_yolo_control_v1',
        prepared_utc=datetime.now(timezone.utc).isoformat(), source_hashes=source_hashes(),
        primary_plan_path=str((PRIMARY / 'plan.json').resolve()), primary_plan_sha256=sha(PRIMARY / 'plan.json'),
        grid=make_grid(parent), expected_runs=72,
        registration='Registered after native inference started, before new native scores or paid outcomes were inspected by the study designer. Primary plan unchanged.',
        purpose='Isolate same-source placement: compare local YOLO, fixed remote YOLO and causal placement fallback.',
        native_progress_at_registration=dict(
            existing_frame_journals=len(list((Path(parent['probe'])/'cases').rglob('frame_*.json'))),
            existing_score_files=len(list((Path(parent['probe'])/'cases').rglob('score.json'))),
            completion_present=(Path(parent['probe'])/'completion.json').exists()))
    plan['factors']['policies'] = [NAME]
    root.mkdir(parents=True)
    for path in [*plan['profiles'].values(), plan['calibration_path']]:
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PRIMARY / path, destination)
    freeze_json(root / 'plan.json', plan)
    return plan


def verify(root):
    root = Path(root)
    plan, parent = read(root / 'plan.json'), study.verify(PRIMARY)
    require(plan['source_hashes'] == source_hashes(), 'Supplemental implementation changed')
    require(plan['primary_plan_sha256'] == sha(PRIMARY / 'plan.json'), 'Primary plan changed')
    require(plan['grid'] == make_grid(parent) and plan['expected_runs'] == 72, 'Control grid changed')
    for key in ('cases', 'actions', 'policy_costs', 'profiles', 'profile_sha256', 'assets',
                'probe', 'native_protocol_sha256', 'calibration_path', 'calibration_sha256'):
        require(plan[key] == parent[key], 'Non-policy intervention: ' + key)
    for name, path in plan['profiles'].items():
        require(sha(root / path) == plan['profile_sha256'][name], 'Profile changed')
    require(sha(root / plan['calibration_path']) == plan['calibration_sha256'], 'Calibration changed')
    return plan


def run_case(root, plan, item, terrain, bank, manifest):
    namespace = dict(paid.run_case.__globals__)
    namespace.update(load_case=study.load_case, make_policy=make_policy, source_hashes=source_hashes)
    loop = FunctionType(paid.run_case.__code__, namespace, paid.run_case.__name__)
    return loop(root, plan, item, terrain, bank, manifest)


def _run(item):
    root, plan, terrain, bank, manifest = study._STATE
    row = run_case(root, plan, item, terrain, bank, manifest)
    save(Path(root) / row['path'] / 'row.json', row)
    return row


def execute(root, workers=3):
    namespace = dict(study.execute.__globals__)
    namespace.update(verify=verify, _run=_run)
    loop = FunctionType(study.execute.__code__, namespace, study.execute.__name__)
    return loop(root, workers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output)
    elif args.mode == 'verify':
        verify(args.output)
    else:
        execute(args.output, args.workers)
