"""Perfect-mask evaluator controls on every fresh fixed-path inspection clip.

No native prediction cache is opened. This diagnostic admits each capture with
zero perception latency and preserves the existing downstream association,
verification, and reporting pipeline. It is not a deployable router result.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from experiments.mission_runtime_profile import read, save, sha
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.strong_routing_20261030.acquisition import OUTPUT as COHORT, verify
from experiments.v4_benchmark.backends import load_frame
from experiments.v4_gazebo_pilot.timing import CaptureSchedule
from experiments.wide_area_diagnosis.paid_clip import LocalInspectionIntent
from experiments.wide_area_diagnosis.pipeline import downstream_control
from experiments.wide_area_diagnosis.physical_probe import phase
from experiments.wide_area_routing.diagnostics import phase_evidence
from experiments.wide_area_routing.inference import freeze_json

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/strong_routing_geometry_controls_v1_20261030'


def focal_semantics(contract, phases, report_quality):
    """Use the focal denominator while retaining all-world false/duplicate costs."""
    document = contract.document()
    focal = set(document['evaluator_target_ids'])
    deadline = document['reporting']['deadline_s']
    correct = [r['target_id'] for r in report_quality['reports'] if r['target_id'] is not None]
    timely = {r['target_id'] for r in report_quality['reports']
              if r['target_id'] in focal and r['completion_time_s'] <= deadline}
    checks = dict(sector_requirements=all(p['success'] for p in phases),
        focal_report_recall=len(timely) / len(focal) >= document['reporting']['min_report_recall'],
        false_report_limit=report_quality['false_reports'] <= document['reporting']['max_false_reports'],
        duplicate_report_limit=report_quality['duplicate_correct_reports'] <= document['reporting']['max_duplicate_reports'])
    return dict(checks=checks, evidence_and_reporting_pass=all(checks.values()),
        timely_focal_reports=sorted(timely), focal_report_recall=len(timely) / len(focal),
        correct_nonfocal_reports=sum(t not in focal for t in correct),
        paid_mission_feasibility_established=False)


def sources():
    names = ('experiments/strong_routing_20261030/geometry_control.py',
        'experiments/wide_area_diagnosis/pipeline.py', 'experiments/wide_area_diagnosis/paid_clip.py',
        'experiments/wide_area_diagnosis/physical_probe.py', 'experiments/wide_area_routing/diagnostics.py',
        'experiments/lake_missions/objectives.py', 'experiments/lake_tracking_evaluation.py',
        'experiments/lake_tracking_runtime.py', 'experiments/mission_task_deadlines/pipeline.py',
        'experiments/mission_pipeline_routing/runtime.py', 'experiments/v4_benchmark/runner.py',
        'experiments/v4_benchmark/reporting.py', 'missionbench/v2/evaluation.py')
    return {str(ROOT / name): sha(ROOT / name) for name in names}


def execute(output=OUTPUT, cohort=COHORT):
    output, cohort = Path(output).resolve(), Path(cohort).resolve()
    if output.exists():
        raise FileExistsError('Preserve previous geometry controls')
    acquisition = verify(cohort)
    completion = read(cohort / 'completion.json')
    if not completion['complete'] or completion['valid_cases'] != 6:
        raise ValueError('All six predeclared physically valid recordings are required')
    output.mkdir(parents=True)
    plan = dict(schema_version='fresh_inspection_perfect_mask_controls_v1',
        acquisition_plan=str(cohort / 'plan.json'), acquisition_plan_sha256=sha(cohort / 'plan.json'),
        acquisition_completion_sha256=sha(cohort / 'completion.json'), implementation_sha256=sources(),
        cases=[c['case_id'] for c in acquisition['cases']], native_prediction_inputs=[],
        relaxed='Perception and resource costs; instantaneous admission of every predeclared motion capture.',
        unchanged='All-world mask-component matching, focal repeated-evidence rule, downstream tracking/verification/reporting.',
        scope='Evaluator-only positive control; no deployable or full eight-sector mission claim.')
    freeze_json(output / 'plan.json', plan)
    rows = []
    for case in acquisition['cases']:
        path = cohort / case['case_id'] / 'recording/recording.json'
        recording = read(path)
        world = recording['world']
        scenario = SimpleNamespace(world=world, targets={o['label']: o['object_id'] for o in world['objects']},
            matching_iou_threshold=.3, observation_interval_s=2., duration_s=case['duration_s'],
            schedule=CaptureSchedule(**recording['capture_schedule'], step_ns=10_000_000))
        frames = {}
        inputs = {str(path): sha(path), str(cohort / 'plan.json'): sha(cohort / 'plan.json')}
        for index in case['evaluation_capture_ids']:
            metadata = recording['frames'][index]
            source = path.parent / metadata['file']
            if sha(source) != metadata['sha256']:
                raise ValueError('Recorded frame changed')
            frames[index] = load_frame(source, metadata)
            inputs[str(source)] = sha(source)
        terrain, _ = load_public_terrain(Path(recording['provenance']['source']['specification']['asset_directory']))
        contract = LocalInspectionIntent(case)
        control = downstream_control(dict(scenario=scenario, contract=contract),
            dict(frames=frames, opportunity_indices=case['evaluation_capture_ids']), terrain, 'PERFECT_GT_EVALUATOR_ONLY')
        control['semantic_control'] = focal_semantics(contract, control['phases'], control['report_quality'])
        raw = phase_evidence([phase(case)], [e['score'] for e in control['events']])[0]
        control.update(case_id=case['case_id'], raw_focal_sector=raw,
            input_sha256=inputs, implementation_sha256=plan['implementation_sha256'])
        destination = output / 'cases' / case['case_id'] / 'control.json'
        destination.parent.mkdir(parents=True)
        freeze_json(destination, control)
        rows.append(dict(case_id=case['case_id'], raw_sector_pass=raw['zero_cost_phase_requirement_met'],
            repeated_targets=raw['supported_target_count'], raw_false_components=raw['false_components'],
            downstream_semantics_pass=control['semantic_control']['evidence_and_reporting_pass'],
            focal_reports=len(control['semantic_control']['timely_focal_reports']),
            correct_nonfocal_reports=control['semantic_control']['correct_nonfocal_reports'],
            false_reports=control['report_quality']['false_reports'],
            duplicates=control['report_quality']['duplicate_correct_reports'],
            pipeline_status_counts=control['pipeline_status_counts'],
            artifact=str(destination.relative_to(output)), artifact_sha256=sha(destination)))
        print(rows[-1], flush=True)
    verify(cohort)
    if sources() != plan['implementation_sha256']:
        raise ValueError('Control implementation changed while executing')
    freeze_json(output / 'completion.json', dict(complete=True, plan_sha256=sha(output / 'plan.json'), cases=rows,
        raw_passing_cases=sum(r['raw_sector_pass'] for r in rows),
        downstream_passing_cases=sum(r['downstream_semantics_pass'] for r in rows),
        paid_mission_feasibility_established=False, no_native_predictions_read=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--cohort', type=Path, default=COHORT)
    args = parser.parse_args()
    execute(args.output, args.cohort)
