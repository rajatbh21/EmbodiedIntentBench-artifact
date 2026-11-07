"""Registered endpoint sensitivity; never changes policies or original scores."""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from experiments.mission_runtime_profile import read, sha
from experiments.mission_task_deadlines.study import require
from experiments.wide_area_routing.inference import freeze_json

ROOTS = ('results/strong_routing_comparison_v2_20261030',
         'results/strong_routing_remote_yolo_control_v1_20261030')
CONTROL = Path('results/strong_routing_geometry_controls_v1_20261030')


def rescore(outcome):
    checks = dict(outcome['checks'])
    require(outcome['local_inspection_success'] == all(checks.values()), 'Original outcome inconsistent')
    require(bool(outcome['phases']), 'Nonempty evidence phases required')
    checks['sector_evidence'] = all(p['evidence_recall'] >= p['required_recall'] for p in outcome['phases'])
    passed = all(checks.values())
    require(not outcome['local_inspection_success'] or passed, 'Removing one predicate cannot remove success')
    return dict(original_success=outcome['local_inspection_success'],
        report_centered_sensitivity_success=passed, checks=checks,
        raw_component_cap_only_failure=passed and not outcome['local_inspection_success'])


def prepare(root):
    root = Path(root)
    require(not root.exists(), 'Preserve previous endpoint registration')
    native = Path(read(Path(ROOTS[0])/'plan.json')['probe'])
    plan = dict(schema_version='report_centered_endpoint_sensitivity_v1',
        registered_utc=datetime.now(timezone.utc).isoformat(),
        roots=list(ROOTS), primary_plan_sha256={p:sha(Path(p)/'plan.json') for p in ROOTS},
        implementation_sha256={str(Path(__file__).resolve()):sha(__file__)},
        motivating_control_sha256={str(CONTROL/name):sha(CONTROL/name) for name in
            ('completion.json', 'shoreline_sector2_gt_fragment_audit.json')},
        registration='After native inference started, before new native score or paid outcome inspection; motivated exclusively by perfect-GT controls.',
        native_progress_at_registration=dict(existing_frame_journals=len(list((native/'cases').rglob('frame_*.json'))),
            existing_score_files=len(list((native/'cases').rglob('score.json')))),
        intervention='Replace only the raw-component part of the sector_evidence predicate with repeated-evidence recall. Retain every other original check.',
        retained='Repeated focal evidence, timely consistent focal reports, world false/duplicate report caps, task admission, execution and all resource predicates.',
        scope='Endpoint sensitivity of executions generated under the ORIGINAL public intent. No new-intent policy experiment, source-model improvement, or change to primary success.',
        reason='A true partially visible person produces five unmatched GT mask fragments but zero false-positive pixels; all six perfect-GT controls produce three correct reports without false/duplicate reports.',
        expected_runs=720, output_dependent_selection=False)
    root.mkdir(parents=True)
    freeze_json(root/'plan.json', plan)
    return plan


def execute(root):
    root = Path(root)
    plan = read(root/'plan.json')
    for path, expected in {**plan['implementation_sha256'], **plan['motivating_control_sha256']}.items():
        require(sha(path)==expected, 'Sensitivity dependency changed')
    rows, sources, grouped = [], {}, defaultdict(lambda:dict(runs=0, original=0, sensitivity=0, recovered=0))
    for source in plan['roots']:
        base=Path(source)
        require(sha(base/'plan.json')==plan['primary_plan_sha256'][source], 'Registered execution plan changed')
        completion=read(base/'completion.json')
        require(completion['complete'] and completion['run_index_sha256']==sha(base/'run_index.json'), 'Incomplete source')
        sources[str(base/'run_index.json')]=sha(base/'run_index.json')
        for row in read(base/'run_index.json'):
            path=base/row['path']/'outcome.json'
            require(sha(path)==row['artifacts_sha256']['outcome.json'], 'Outcome changed')
            value=rescore(read(path))
            require(row['local_inspection_success']==value['original_success'], 'Index disagrees')
            rows.append(dict(source=source, run_id=row['run_id'], policy=row['policy'], flight=row['flight'],
                budget_s=row['perception_budget_s'], link=row['link'], energy_condition=row['energy_condition'],
                outcome_sha256=sha(path), **value))
            g=grouped[row['policy']];g['runs']+=1;g['original']+=int(value['original_success'])
            g['sensitivity']+=int(value['report_centered_sensitivity_success'])
            g['recovered']+=int(value['raw_component_cap_only_failure'])
    require(len(rows)==plan['expected_runs']==720, 'All original and supplemental cells required')
    freeze_json(root/'results.json', dict(plan_sha256=sha(root/'plan.json'), source_sha256=sources,
        scope=plan['scope'], complete=True, runs=len(rows), by_policy=dict(grouped), rows=rows))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','execute'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (prepare if args.mode=='prepare' else execute)(args.output)
