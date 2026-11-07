"""Paired camera-mount sensitivity on all three low/slow development sites.

This posthoc control changes the camera mount from nadir to the old corridor
pitch. It retains every original world object and every declared motion frame.
It is not a held-out benchmark, a look-at camera, or an isolated size test.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from experiments.mission_runtime_profile import read, save, sha
from experiments.wide_area_diagnosis.acquisition import verify, FLIGHTS
from experiments.wide_area_missions.runtime import normalize_specification

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / 'results/wide_area_physical_diagnosis_v2_20261028'
OUTPUT = ROOT / 'results/wide_area_viewpoint_diagnosis_v1_20261028'


def select_cases(cases):
    selected = [copy.deepcopy(case) for case in cases
                if case['offset_above_safe_envelope_m'] == 14.
                and case['planned_peak_speed_mps'] == 1.5]
    if len(selected) != 3 or {c['flight'] for c in selected} != set(FLIGHTS):
        raise ValueError('Exactly one predeclared low/slow case from every site is required')
    return selected


def camera_variant(specification):
    if not math.isclose(specification['world']['camera_pitch_rad'], math.pi / 2):
        raise ValueError('Paired baseline must use the original nadir mount')
    variant = copy.deepcopy(specification)
    variant['world']['camera_pitch_rad'] = 1.2
    return variant


def prepare(output=OUTPUT, baseline=BASELINE):
    output, baseline = Path(output).resolve(), Path(baseline).resolve()
    if output.exists():
        raise FileExistsError('Preserve any prior camera-mount experiment')
    original = verify(baseline)
    completion = read(baseline / 'completion.json')
    if not completion['complete'] or completion['valid_cases'] != 12:
        raise ValueError('The complete original physical factorial is required')
    plan = copy.deepcopy(original)
    plan['cases'] = select_cases(original['cases'])
    plan['source_inputs_sha256'].update({
        str(Path(__file__).resolve()): sha(__file__),
        str(baseline / 'plan.json'): sha(baseline / 'plan.json'),
        str(baseline / 'completion.json'): sha(baseline / 'completion.json')})
    output.mkdir(parents=True)
    pairs = []
    for case in plan['cases']:
        original_path = baseline / case['scenario']
        recording_path = baseline / case['case_id'] / 'recording/recording.json'
        recording = read(recording_path)
        specification = normalize_specification(original_path)
        if recording['provenance']['source']['specification'] != specification:
            raise ValueError('Baseline recording and scenario differ')
        for source in (original_path, recording_path):
            plan['source_inputs_sha256'][str(source)] = sha(source)
        destination = output / case['scenario']
        destination.parent.mkdir(parents=True)
        save(destination, camera_variant(specification))
        case['scenario_sha256'] = sha(destination)
        pairs.append(dict(case_id=case['case_id'], baseline_recording=str(recording_path),
            baseline_recording_sha256=sha(recording_path),
            baseline_recording_identity=recording['recording_identity']))
    plan.update(schema_version='wide_area_viewpoint_development_diagnostic_v1',
        paired_baseline_cohort=str(baseline), pairs=pairs,
        factors=dict(camera_pitch_rad=dict(baseline=math.pi/2, candidate=1.2),
                     offset_above_common_safe_envelope_m=14., peak_speed_mps=1.5),
        camera='Only fixed body mount pitch changes, from nadir pi/2 to 1.2 rad. Native 960x720 and HFOV1.2 unchanged; no target look-at or gimbal tracking.',
        site_rule='All three previously declared low/slow sites. Mechanistic development choice after earlier diagnostics, never outcome-selected sites or a held-out sample.',
        geometry='Identical absolute XYZ trajectory, controller, clock, seed and full world in each matched pair; verify actual poses and timestamps after acquisition.',
        causal_scope='Camera pitch changes appearance, field-of-view coverage and projection together. All invisible target opportunities remain in the denominator. Actual body-pose differences must be quantified before attributing a difference to the mount.',
        comparison='All 18 motion captures per site, same three native model families at original confidence0.5 and original input scale. No endpoint settle frames.',
        success_gate='Raw focal-sector evidence permits paid testing only. No full-mission repair or routing ranking is implied.',
        pose_tolerance=dict(max_position_difference_m=.1, max_quaternion_angle_rad=.02,
            failure_interpretation='Report as a coupled physical-repeat/viewpoint comparison, not camera-only, if tolerance is exceeded.'))
    save(output / 'plan.json', plan)
    verify(output)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--baseline', type=Path, default=BASELINE)
    args = parser.parse_args()
    prepare(args.output, args.baseline)
