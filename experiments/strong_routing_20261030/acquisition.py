"""Fresh fixed-path inspection clips chosen by geometry before model inference.

Six predeclared regions retain the complete existing scenery and target worlds.
The camera mount and flight controller remain fixed. Mission-author ROI geometry
defines the acquisition only; target coordinates and rendered labels are private
evaluator data and never become routing-policy inputs.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import numpy as np

from experiments.mission_runtime_profile import read, save, sha
from experiments.lake_missions.placement import collision_bounds, terrain_height
from experiments.wide_area_missions.cohort import clearance_surface, _new_bounds
from experiments.wide_area_missions.runtime import Trajectory, normalize_specification, capture, digest
from experiments.wide_area_diagnosis.acquisition import pass_geometry, implementation_paths as parent_implementation

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'results/wide_area_missions_v2_20261028'
OUTPUT = ROOT / 'results/strong_routing_acquisition_v2_20261030'
FLIGHTS = ('shoreline_loop', 'wide_area_search', 'sector_revisit')
SECTORS = ('sector_2', 'sector_6')
PITCH = 1.2
HEIGHT_OFFSETS = (14., 18., 22., 28., 36., 44.)
CAPTURE_IDS = list(range(18))


def project_point(point, position, pitch=PITCH, shape=(720, 960), hfov=1.2):
    """Pinhole projection with the existing fixed-yaw camera calibration."""
    origin = np.asarray(position, dtype=float) + [0., 0., -.1]
    delta = np.asarray(point, dtype=float) - origin
    c, s = math.cos(pitch), math.sin(pitch)
    ray = np.array([c * delta[0] - s * delta[2], delta[1], s * delta[0] + c * delta[2]])
    if ray[0] <= 0:
        return None
    h, w = shape
    focal = w / (2 * math.tan(hfov / 2))
    return ((w - 1) / 2 - focal * ray[1] / ray[0],
            (h - 1) / 2 - focal * ray[2] / ray[0])


def segment_intersects_box(start, end, low, high):
    """Closed finite-segment/AABB intersection, including parallel rays."""
    first, last = 0., 1.
    for a, b, lo, hi in zip(start, end, low, high):
        d = b - a
        if abs(d) < 1e-12:
            if not lo <= a <= hi:
                return False
        else:
            near, far = sorted(((lo - a) / d, (hi - a) / d))
            first, last = max(first, near), min(last, far)
            if first > last:
                return False
    return True


def visible_geometry(point, position, surface, bounds):
    """Conservative terrain/AABB line of sight; not rendered visibility."""
    start = np.asarray(position, dtype=float) + [0., 0., -.1]
    end = np.asarray(point, dtype=float)
    if any(segment_intersects_box(start, end, low, high) for _, low, high in bounds):
        return False
    steps = max(2, math.ceil(math.dist(start, end) / .5))
    for fraction in np.linspace(0., 1., steps + 1)[1:-1]:
        p = start + fraction * (end - start)
        if p[2] <= surface(p[0], p[1]) + .05:
            return False
    return True


def geometry_candidate(targets, altitude, terrain, bounds):
    """Center a fixed oblique Y-axis pass on the known mission ROI."""
    positions = np.asarray(list(targets.values()), dtype=float)
    center = positions.mean(axis=0)
    torso_z = float(np.median(positions[:, 2]) + .9)
    midpoint = [float(center[0] - (altitude - .1 - torso_z) / math.tan(PITCH)), float(center[1])]
    points, motion, duration = pass_geometry(midpoint, [0., 1.], altitude, 1.5)
    trajectory = Trajectory(points)
    samples = [trajectory.sample(t)[0] for t in np.linspace(0., 36., 145)]
    def envelope(x, y):
        z = terrain(x, y)
        for _, low, high in bounds:
            if low[0] - 5 <= x <= high[0] + 5 and low[1] - 5 <= y <= high[1] + 5:
                z = max(z, float(high[2]))
        return z
    minimum_clearance = float(min(p[2] - envelope(*p[:2]) for p in samples))
    observations = {}
    for identity, xyz in targets.items():
        torso = [xyz[0], xyz[1], xyz[2] + .9]
        rows = []
        for index in CAPTURE_IDS:
            position = trajectory.sample(index * 2.)[0]
            uv = project_point(torso, position)
            contained = bool(uv and 48 <= uv[0] < 912 and 36 <= uv[1] < 684)
            clear = contained and visible_geometry(torso, position, terrain, bounds)
            foot, head = project_point(xyz, position), project_point([xyz[0], xyz[1], xyz[2] + 1.8], position)
            rows.append(dict(capture_id=index, projected_torso_px=uv, in_margin=contained,
                             unobstructed=clear,
                             nominal_1_8m_projected_extent_px=math.dist(foot, head) if foot and head else None))
        observations[identity] = rows
    counts = {identity: sum(r['unobstructed'] for r in rows) for identity, rows in observations.items()}
    return dict(trajectory=points, motion_s=motion, duration_s=duration,
                midpoint_xy_m=midpoint, altitude_m=altitude,
                minimum_envelope_clearance_m=minimum_clearance,
                geometry_visible_capture_counts=counts, observations=observations,
                passed=minimum_clearance >= 14 - 1e-9 and all(n >= 4 for n in counts.values()))


def implementation_paths():
    # Preserve the established physical-probe verifier API. The new design
    # implementation is separately pinned in source_inputs_sha256 below.
    return parent_implementation()


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Preserve the previous acquisition design and attempts')
    parent = read(PARENT / 'protocol.json')
    sources = {str(PARENT / 'protocol.json'): sha(PARENT / 'protocol.json'),
               str(Path(__file__).resolve()): sha(__file__)}
    sources.update({str(PARENT / path): checksum for path, checksum in parent['frozen_files'].items()})
    sources.update(parent['external_mesh_manifest_sha256'])
    for path, checksum in sources.items():
        if sha(path) != checksum:
            raise ValueError('Parent acquisition assets changed')
    output.mkdir(parents=True)
    cases, rejected = [], []
    for flight in FLIGHTS:
        specification = normalize_specification(PARENT / flight / 'scenario.json')
        assets = Path(specification['asset_directory'])
        sources[str(assets / 'manifest.json')] = sha(assets / 'manifest.json')
        bounds = collision_bounds(assets) + _new_bounds(specification['scenery'])
        terrain = lambda x, y: terrain_height(assets, x, y)
        sectors = {s['sector_id']: s for s in read(PARENT / flight / 'mission_design.json')['sectors']}
        for sector_id in SECTORS:
            sector = sectors[sector_id]
            targets = {o['object_id']: o['position_m'] for o in specification['world']['objects']
                       if o['object_id'] in sector['evaluator_target_ids']}
            reference_z = max(p[2] for p in targets.values())
            attempts, selected = [], None
            for offset in HEIGHT_OFFSETS:
                candidate = geometry_candidate(targets, reference_z + offset, terrain, bounds)
                candidate['offset_above_highest_target_base_m'] = offset
                attempts.append(candidate)
                if candidate['passed']:
                    selected = candidate
                    break
            case_id = f'{flight}/{sector_id}'
            folder = output / case_id
            folder.mkdir(parents=True)
            save(folder / 'geometry_preflight.json', dict(case_id=case_id, attempts=attempts, accepted=bool(selected)))
            if selected is None:
                rejected.append(case_id)
                continue
            new = copy.deepcopy(specification)
            new.update(scenario_id=case_id, duration_s=42., warmup_s=2., capture_interval_s=2.)
            new['world'].update(initial_position_m=selected['trajectory'][0]['position_m'],
                                trajectory=selected['trajectory'], max_speed_mps=1.5, camera_pitch_rad=PITCH)
            new['controller'].update(command_speed_limit_mps=2.5, minimum_clearance_m=10.)
            new['scope'] = 'Fresh geometry-designed local inspection, known terrain; not a complete eight-sector mission'
            save(folder / 'scenario.json', new)
            cases.append(dict(case_id=case_id, flight=flight, focal_sector=sector_id,
                scenario=f'{case_id}/scenario.json', scenario_sha256=sha(folder / 'scenario.json'),
                evaluator_target_ids=sector['evaluator_target_ids'], focal_target_positions_m=targets,
                full_world_objects=len(new['world']['objects']), planned_peak_speed_mps=1.5,
                motion_s=36., duration_s=42., evaluation_capture_ids=CAPTURE_IDS,
                excluded_settle_capture_ids=[18, 19, 20],
                geometry_preflight=f'{case_id}/geometry_preflight.json',
                geometry_preflight_sha256=sha(folder / 'geometry_preflight.json'),
                selected_altitude_m=selected['altitude_m'],
                minimum_planned_envelope_clearance_m=selected['minimum_envelope_clearance_m']))
    plan = dict(schema_version='strong_routing_fresh_geometry_acquisition_v1',
        development_only=False, known_map=True, parent_cohort=str(PARENT), cases=cases,
        declared_case_count=6, geometry_rejected_cases=rejected,
        site_rule='Sector_2 and sector_6 in every existing wide layout, declared before any new RGB or model output.',
        geometry_rule='Fixed yaw-zero oblique camera, Y-axis 36m pass. Center on mission ROI torso plane, choose first ordered altitude offset passing conservative terrain/AABB line-of-sight and capture coverage.',
        height_offsets_above_highest_target_base_m=HEIGHT_OFFSETS,
        geometry_gate='At least four planned unoccluded torso captures for EACH of three people, within a 5% image margin; at least14m planned obstacle-envelope clearance.',
        camera='960x720 RGB, HFOV1.2, fixed pitch1.2, no look-at tracking. All24 world targets and scenery retained.',
        capture_rule='All21 captures stored; 18 motion captures at0,2,...,34s scored; three settle frames excluded before inference.',
        causal_scope='Fresh local inspection regions on a previously seen terrain and asset family. Geometry-led route/camera repair changes observation opportunity; no cross-map or full-wide-mission generalization claim.',
        outcome_selection=False, source_inputs_sha256=sources, implementation_sha256=implementation_paths())
    save(output / 'plan.json', plan)
    return plan


def verify(output=OUTPUT):
    output = Path(output)
    plan = read(output / 'plan.json')
    if plan['implementation_sha256'] != implementation_paths():
        raise ValueError('Acquisition implementation changed after lock')
    for path, checksum in plan['source_inputs_sha256'].items():
        if sha(path) != checksum:
            raise ValueError(f'Changed acquisition source: {path}')
    for case in plan['cases']:
        for key in ('scenario', 'geometry_preflight'):
            if sha(output / case[key]) != case[key + '_sha256']:
                raise ValueError('Changed predeclared geometry/scenario')
    return plan


def execute(output=OUTPUT):
    output = Path(output).resolve()
    plan = verify(output)
    receipts = []
    for number, case in enumerate(plan['cases'], 1):
        folder = output / case['case_id'] / 'recording'
        failure, recording = None, None
        if (folder / 'recording.json').exists():
            recording = read(folder / 'recording.json')
            expected = normalize_specification(output / case['scenario'])
            if recording['provenance']['source']['specification'] != expected:
                raise ValueError('Recording/scenario mismatch')
            if recording['recording_identity'] != digest(recording['provenance']['source']):
                raise ValueError('Recording identity mismatch')
            for frame in recording['frames']:
                if sha(folder / frame['file']) != frame['sha256']:
                    raise ValueError('Changed recorded frame')
        elif (folder / 'attempt_failure.json').exists():
            failure = read(folder / 'attempt_failure.json')
        else:
            try:
                recording = capture(output / case['scenario'], folder)
            except Exception as error:
                failure = dict(type=type(error).__name__, message=str(error))
                folder.mkdir(parents=True, exist_ok=True)
                save(folder / 'attempt_failure.json', failure)
                recording = read(folder / 'recording.json') if (folder / 'recording.json').exists() else None
        receipts.append(dict(case_id=case['case_id'], recording_sha256=sha(folder / 'recording.json') if recording else None,
                             physical_passed=bool(recording and recording['provenance']['physical_route_passed']),
                             captures=len(recording['frames']) if recording else 0, failure=failure))
        save(output / 'progress.json', dict(completed=len(receipts), expected=len(plan['cases'])))
        print(f'Completed {number}/{len(plan["cases"])}: {case["case_id"]}', flush=True)
    save(output / 'completion.json', dict(complete=True, meaning='All declared accepted geometry cases attempted; failures retained',
        valid_cases=sum(r['physical_passed'] for r in receipts), plan_sha256=sha(output / 'plan.json'), cases=receipts))


def audit(output=OUTPUT):
    output = Path(output).resolve()
    plan = verify(output)
    rows = []
    for case in plan['cases']:
        folder = output / case['case_id'] / 'recording'
        if not (folder / 'recording.json').exists():
            rows.append(dict(case_id=case['case_id'], recorded=False, geometry_feasible=False))
            continue
        recording = read(folder / 'recording.json')
        labels = {o['object_id']: o['label'] for o in recording['world']['objects']}
        measurements = {identity: [] for identity in case['evaluator_target_ids']}
        for index in case['evaluation_capture_ids']:
            frame = recording['frames'][index]
            path = folder / frame['file']
            if sha(path) != frame['sha256']:
                raise ValueError('Changed frame before coverage audit')
            with np.load(path, allow_pickle=False) as data:
                ids = data['object_ids']
                for identity in measurements:
                    ys, xs = np.where(ids == labels[identity])
                    measurements[identity].append(dict(capture_id=index, pixels=len(xs),
                        bbox_longest_px=int(max(xs.max()-xs.min()+1, ys.max()-ys.min()+1)) if len(xs) else 0))
        visible = {identity: sum(m['pixels'] >= 4 for m in values) for identity, values in measurements.items()}
        rows.append(dict(case_id=case['case_id'], recorded=True,
            physical_passed=recording['provenance']['physical_route_passed'],
            geometry_feasible=all(n >= 4 for n in visible.values()),
            focal_visible_capture_counts=visible, measurements=measurements,
            recording_sha256=sha(folder / 'recording.json')))
    document = dict(plan_sha256=sha(output / 'plan.json'), cases=rows,
        evaluator_only=True, model_outputs_used=False,
        passing_cases=sum(r['geometry_feasible'] for r in rows), declared_cases=plan['declared_case_count'],
        rule='Each focal target has at least4 original rendered masks of >=4px among18 motion captures. This is observability, not perception success. No case removed after this audit.')
    save(output / 'geometry_audit.json', document)
    return document


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'execute', 'audit'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = globals()[args.mode](args.output)
    if args.mode in ('prepare', 'audit'):
        print(f'{args.mode}: {len(result["cases"])} cases')
