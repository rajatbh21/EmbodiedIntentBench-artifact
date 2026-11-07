"""Matched physical development clips for altitude and exposure sensitivity.

These short inspection passes are not replacements for the full wide-area
missions. Sites, complete worlds, factor levels and scoring intervals are fixed
before acquisition. No model or visibility result chooses the trajectory.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

from experiments.mission_runtime_profile import read, save, sha
from experiments.lake_missions.placement import collision_bounds, terrain_height
from experiments.wide_area_missions.cohort import clearance_surface, _new_bounds
from experiments.wide_area_missions.runtime import Trajectory, normalize_specification, capture, digest

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'results/wide_area_missions_v2_20261028'
OUTPUT = ROOT / 'results/wide_area_physical_diagnosis_v1_20261028'
FLIGHTS = ('shoreline_loop', 'wide_area_search', 'sector_revisit')


def pass_geometry(center, tangent, height, peak_speed):
    """36 m rest-to-rest Hermite pass; peak speed is 1.5 distance/duration."""
    if peak_speed not in (1.5, 4.0):
        raise ValueError('Only the declared speed levels are accepted')
    norm = math.hypot(*tangent)
    if norm <= 0:
        raise ValueError('A horizontal direction is required')
    direction = [value / norm for value in tangent]
    start = [center[i] - 18 * direction[i] for i in range(2)] + [height]
    end = [center[i] + 18 * direction[i] for i in range(2)] + [height]
    motion_s = 54 / peak_speed
    duration_s = math.ceil((motion_s + 6) / 2) * 2
    points = [dict(time_s=0., position_m=start, velocity_mps=[0., 0., 0.]),
              dict(time_s=motion_s, position_m=end, velocity_mps=[0., 0., 0.]),
              dict(time_s=duration_s, position_m=end, velocity_mps=[0., 0., 0.])]
    return points, motion_s, duration_s


def implementation_paths():
    names = ['experiments/wide_area_diagnosis/acquisition.py',
             'experiments/wide_area_missions/runtime.py',
             'experiments/wide_area_missions/worker.py',
             'experiments/wide_area_missions/cohort.py',
             'experiments/wide_area_missions/scenery.py',
             'experiments/lake_missions/placement.py',
             'experiments/lake_missions/assets.py',
             'experiments/v4_gazebo_pilot/assets.py',
             'experiments/lake_missions/world.py',
             'experiments/routing_mission/world_worker.py',
             'experiments/v4_gazebo_pilot/world.py',
             'experiments/v4_gazebo_pilot/transport.py']
    return {str(ROOT / name): sha(ROOT / name) for name in names}


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the previous development acquisition')
    cases = []
    sources = {str(PARENT / 'protocol.json'): sha(PARENT / 'protocol.json')}
    parent = read(PARENT / 'protocol.json')
    sources.update({str(PARENT / path): checksum for path, checksum in parent['frozen_files'].items()})
    sources.update(parent['external_mesh_manifest_sha256'])
    for path, checksum in sources.items():
        if sha(path) != checksum:
            raise ValueError('Parent acquisition assets changed before calibration lock')
    output.mkdir(parents=True)
    for flight in FLIGHTS:
        original_path = PARENT / flight / 'scenario.json'
        design_path = PARENT / flight / 'mission_design.json'
        sources[str(original_path)] = sha(original_path)
        sources[str(design_path)] = sha(design_path)
        original = normalize_specification(original_path)
        sector = read(design_path)['sectors'][0]
        original_trajectory = Trajectory(original['world']['trajectory'])
        position, velocity = original_trajectory.sample(sector['nominal_pass_time_s'])
        norm = math.hypot(*velocity[:2])
        direction = [v / norm for v in velocity[:2]]
        assets = Path(original['asset_directory'])
        sources[str(assets / 'manifest.json')] = sha(assets / 'manifest.json')
        bounds = collision_bounds(assets) + _new_bounds(original['scenery'])
        surface = clearance_surface(assets, bounds)
        sample_xy = [[position[i] + offset * direction[i] for i in range(2)]
                     for offset in [n / 2 for n in range(-36, 37)]]
        envelope_z = max(surface(*xy) for xy in sample_xy)
        target_positions = {o['object_id']: o['position_m'] for o in original['world']['objects']
                            if o['object_id'] in sector['evaluator_target_ids']}
        for clearance in (14., 28.):
            for speed in (1.5, 4.):
                case_id = f'{flight}/offset_{clearance:g}/speed_{speed:g}'
                folder = output / case_id
                folder.mkdir(parents=True)
                points, motion_s, duration_s = pass_geometry(position[:2], direction,
                                                            envelope_z + clearance, speed)
                specification = copy.deepcopy(original)
                specification.update(scenario_id=case_id, duration_s=duration_s,
                                     warmup_s=2., capture_interval_s=2.)
                specification['world'].update(initial_position_m=points[0]['position_m'],
                                              trajectory=points, max_speed_mps=speed)
                specification['controller'].update(command_speed_limit_mps=speed + 1,
                                                   minimum_clearance_m=10.)
                specification['scope'] = 'Posthoc development altitude-by-speed diagnostic, not a full mission'
                save(folder / 'scenario.json', specification)
                eligible = [i for i in range(round(duration_s / 2)) if i * 2 < motion_s]
                cases.append(dict(case_id=case_id, flight=flight, scenario=f'{case_id}/scenario.json',
                    scenario_sha256=sha(folder / 'scenario.json'),
                    focal_sector=sector['sector_id'], evaluator_target_ids=sector['evaluator_target_ids'],
                    full_world_objects=len(original['world']['objects']),
                    focal_target_positions_m=target_positions,
                    offset_above_safe_envelope_m=clearance, safe_envelope_z_m=envelope_z,
                    center_terrain_z_m=terrain_height(assets, *position[:2]),
                    planned_peak_speed_mps=speed, motion_s=motion_s, duration_s=duration_s,
                    evaluation_capture_ids=eligible,
                    excluded_settle_capture_ids=[i for i in range(round(duration_s / 2)) if i not in eligible]))
    plan = dict(schema_version='wide_area_physical_development_diagnostic_v1',
        development_only=True, parent_cohort=str(PARENT), cases=cases,
        factors=dict(offset_above_common_safe_envelope_m=[14, 28], peak_speed_mps=[1.5, 4]),
        site_rule='Authored sector_1 of each of the three existing flights; no outcome-based site selection',
        geometry='Identical 36 m XY centerline per site, rest-to-rest Hermite velocity profile; only absolute height and retiming vary',
        camera='Identical native 960x720, 1.2 rad horizontal FOV, body-mounted nadir, all 24 original world targets and scenery retained',
        scoring='All original world labels remain in component matching. Focal three target IDs define a diagnostic sector: two of three people, two IoU>=0.3 observations separated by>=2s, <=2 false components. No full-mission claim.',
        capture_rule='All 2 s captures stored. Only route times 0<=t<motion_s scored; endpoint settle captures excluded a priori for every arm.',
        causal_scope='Altitude changes apparent size, viewing geometry and field-of-view coverage together; it is not a pure image-size intervention. Speed changes exposure, capture count and actual body attitude during acceleration; original per-frame body quaternions are retained.',
        acquisition_failures='Preserve and report failures; never choose replacement sites based on model outcomes',
        outcome_selection=False, source_inputs_sha256=sources, implementation_sha256=implementation_paths())
    save(output / 'plan.json', plan)
    return plan


def verify(output):
    output = Path(output)
    plan = read(output / 'plan.json')
    if implementation_paths() != plan['implementation_sha256']:
        raise ValueError('Acquisition implementation changed after lock')
    for path, checksum in plan['source_inputs_sha256'].items():
        if sha(path) != checksum:
            raise ValueError('Changed acquisition source')
    for case in plan['cases']:
        if sha(output / case['scenario']) != case['scenario_sha256']:
            raise ValueError('Changed acquisition scenario')
    return plan


def execute(output=OUTPUT):
    output = Path(output).resolve()
    plan = verify(output)
    receipts = []
    for number, case in enumerate(plan['cases'], 1):
        folder = output / case['case_id'] / 'recording'
        failure = None
        if (folder / 'recording.json').exists():
            recording = read(folder / 'recording.json')
            expected = normalize_specification(output / case['scenario'])
            if recording['provenance']['source']['specification'] != expected:
                raise ValueError('Prior acquisition belongs to another scenario')
            if recording['recording_identity'] != digest(recording['provenance']['source']):
                raise ValueError('Changed recording identity')
            for frame in recording['frames']:
                if sha(folder / frame['file']) != frame['sha256']:
                    raise ValueError('Changed recorded frame')
        elif (folder / 'attempt_failure.json').exists():
            failure = read(folder / 'attempt_failure.json')
            recording = None
        elif (folder / 'acquisition_failure.json').exists():
            failure = read(folder / 'acquisition_failure.json')
            recording = None
        else:
            try:
                recording = capture(output / case['scenario'], folder)
            except Exception as error:
                failure = dict(type=type(error).__name__, message=str(error))
                folder.mkdir(parents=True, exist_ok=True)
                save(folder / 'attempt_failure.json', failure)
                recording = read(folder / 'recording.json') if (folder / 'recording.json').exists() else None
        receipts.append(dict(case_id=case['case_id'],
            recording_sha256=sha(folder / 'recording.json') if recording else None,
            physical_passed=bool(recording and recording['provenance']['physical_route_passed']),
            captures=len(recording['frames']) if recording else 0, failure=failure))
        save(output / 'progress.json', dict(completed=len(receipts), expected=len(plan['cases'])))
        print(f"Completed {number}/{len(plan['cases'])}: {case['case_id']}", flush=True)
    save(output / 'completion.json', dict(complete=True, meaning='All declared cases attempted; inspect physical_passed for every case',
        valid_cases=sum(r['physical_passed'] for r in receipts),
        plan_sha256=sha(output / 'plan.json'), cases=receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'execute', 'verify'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.mode == 'prepare':
        plan = prepare(args.output)
        print(f"Locked {len(plan['cases'])} physical development clips")
    elif args.mode == 'execute':
        execute(args.output)
    else:
        verify(args.output)
