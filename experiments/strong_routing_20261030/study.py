"""Prospective fixed-flight routing comparison after sensor-domain calibration.

The original paid runner is reused byte for byte with explicit dependencies.
New native results cannot alter the locked policies, resource grid or intent.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import multiprocessing
from pathlib import Path
import shutil
import time
from types import FunctionType

from experiments.lake_intent_contract import canonical
from experiments.lake_causal_router import MissionCosts
from experiments.mission_network_emulation import LinkSegment, LinkTrace
from experiments.mission_network_policies import PolicyCosts
from experiments.mission_runtime_profile import read, save, sha
from experiments.mission_task_deadlines.study import require
from experiments.wide_area_diagnosis import paid_clip as paid
from experiments.wide_area_diagnosis.physical_probe import verify_protocol
from experiments.wide_area_routing.inference import freeze_json
from experiments.strong_routing_20261030 import policy as calibrated

PARENT = Path('results/wide_area_viewpoint_paid_clip_v1_20261028')
POLICIES = (*paid.POLICIES, 'fastest_feasible', 'calibrated_quality', 'calibrated_evidence_rate')
LINKS = ('good', 'degraded', 'offline')
BATTERIES = ('ample', 'residual_250j')


def initial_battery(case, name):
    require(name in BATTERIES, 'Undeclared energy condition')
    # Reserve and full recorded flight are protected before allocating task energy.
    return .95 if name == 'ample' else .2 + (case['duration_s'] * 120. + 250.) / (120. * 3600.)


def link_trace(name, duration):
    require(name in LINKS, 'Undeclared network condition')
    end = round((duration + 30) * 1e9)
    if name == 'degraded':
        return LinkTrace((LinkSegment(0, end, 2_000_000, 10_000_000, 150_000_000),))
    return LinkTrace((LinkSegment(0, end, 50_000_000, 100_000_000, 30_000_000, name == 'offline'),))


class InspectionIntent(paid.LocalInspectionIntent):
    def __init__(self, case, battery):
        super().__init__(case)
        doc = self.document()
        doc['schema_version'] = 'prospective_local_inspection_intent_v1'
        doc['mission']['title'] = 'Fixed-path local inspection'
        doc['mission']['initial_battery_frac'] = battery
        self._json = canonical(doc)


def load_case(root, plan, item):
    state = paid.load_case(root, plan, dict(item, link='good'))
    expected = initial_battery(plan['cases'][item['flight']], item['energy_condition'])
    require(item['battery'] == expected, 'Initial charge differs from declared energy envelope')
    state['scenario'] = replace(state['scenario'], initial_battery_frac=expected)
    state['contract'] = InspectionIntent(plan['cases'][item['flight']], expected)
    state['costs'] = PolicyCosts(MissionCosts.from_scenario(state['scenario']), **plan['policy_costs'])
    state['trace'] = link_trace(item['link'], state['scenario'].duration_s)
    state['learned_profile'] = read(Path(root) / plan['calibration_path'])
    return state


def make_policy(name, actions, costs, profile, quality, risk, learned_profile=None):
    if name in paid.POLICIES:
        return paid.make_policy(name, actions, costs, profile, quality, risk)
    require(name in POLICIES, 'Undeclared protocol')
    return calibrated.make_policy(name, actions, costs, profile, quality, risk, learned_profile)


def source_hashes():
    return {**paid.source_hashes(), str(Path(__file__).resolve()): sha(__file__),
            str(Path(calibrated.__file__).resolve()): sha(calibrated.__file__)}


def bound_run_loop():
    namespace = dict(paid.run_case.__globals__)
    namespace.update(load_case=load_case, make_policy=make_policy, source_hashes=source_hashes)
    return FunctionType(paid.run_case.__code__, namespace, paid.run_case.__name__,
                        paid.run_case.__defaults__, paid.run_case.__closure__)


def run_case(root, plan, item, terrain, bank, manifest):
    return bound_run_loop()(root, plan, item, terrain, bank, manifest)


def make_grid(cases):
    return [dict(flight=key, policy=policy, perception_budget_s=budget, link=link,
                 energy_condition=energy, battery=initial_battery(case, energy),
                 run_id=f'{key}/{policy}/budget_{budget}/{link}/{energy}')
            for key, case in sorted(cases.items()) for policy in POLICIES for budget in (2, 8)
            for link in LINKS for energy in BATTERIES]


def prepare(root, probe, calibration):
    root, probe, calibration = map(lambda p: Path(p).resolve(), (root, probe, calibration))
    require(not root.exists(), 'Preserve previous comparisons')
    require(not (probe / 'completion.json').exists() and not (probe / 'cases').exists(),
            'Lock comparison before any new native inference')
    parent = paid.verify(PARENT)
    native = verify_protocol(probe)
    learned = read(calibration)
    calibrated.verify_calibration(learned)
    cases = {c['case_id']: c for c in native['cases']}
    require(len(cases) == 6 and all(c['motion_s'] == 36 and c['duration_s'] == 42 for c in cases.values()),
            'All six predeclared fresh inspection passes required')
    root.mkdir(parents=True)
    for path in parent['profiles'].values():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PARENT / path, target)
    calibration_path = 'inputs/oblique_calibration.json'
    shutil.copyfile(calibration, root / calibration_path)
    grid = make_grid(cases)
    plan = copy.deepcopy(parent)
    plan.update(schema_version='prospective_fixed_flight_routing_comparison_v1',
        development_only=False, prepared_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes=source_hashes(), probe=str(probe), native_protocol_sha256=sha(probe / 'protocol.json'),
        reference_parent=str(PARENT.resolve()), reference_parent_plan_sha256=sha(PARENT / 'plan.json'),
        cases=cases, grid=grid, expected_runs=len(grid),
        calibration_path=calibration_path, calibration_sha256=sha(root / calibration_path),
        factors=dict(policies=list(POLICIES), perception_budget_s=[2, 8], links=list(LINKS),
                     energy_conditions=list(BATTERIES), source_case_ids=list(cases)),
        network=dict(good=dict(upload_mbps=50, download_mbps=100, rtt_ms=30),
                     degraded=dict(upload_mbps=2, download_mbps=10, rtt_ms=150),
                     offline=dict(full_duration_outage=True),
                     semantics='Synthetic constant regimes, identical absolute clock for every protocol; no measured-network claim.'),
        battery_design=dict(capacity_wh=120, flight_power_w=120, reserve_fraction=.2,
            ample_initial_fraction=.95, residual_task_energy_j=250,
            formula='reserve + (42s * 120W + 250J) / (120Wh * 3600)',
            rationale='Analytical resource stress fixed before new outputs; 250J lies between 18 cheap calls and 9 heavy calls under frozen profiles.',
            caveat='Synthetic residual-energy envelope; not a calibrated BMS, complete battery cycle, or return-to-home mission.'),
        raw_phase_trigger='Every predeclared case retained. Existing binder requires any raw model success; that trigger does not select clips or policies.',
        outcome_awareness='New native outputs unavailable at plan lock. Prior three oblique sites are development data; every one of their 54 frames trains the global priors.',
        policy_changes='Three causal reference policies with common resource feasibility guard and sensor-domain global reliability/risk priors; all old controls retained unchanged.',
        trained_heads='Old policies retain old heads. New policies use only development aggregate scalars/model signatures; no runtime site, ground truth or future link.',
        hypotheses=['H1: feasible sensor geometry permits deployable paid local success.',
            'H2: reliable fast perception avoids local-heavy task-deadline failures.',
            'H3: causal feasibility fallback preserves local success when offload is unavailable.',
            'H4: full-flight reserve protection separates execution quality from resource validity.'],
        primary_outcome='Complete local-inspection contract, separately reported for every network, task budget and energy envelope.',
        statistical_unit='Six prospective recordings in three known layouts; paired replay cells are correlated conditions, not independent mission samples.',
        scope='Perception-only fixed-path local inspection; all original world targets retained for correctness. Not full eight-sector mission evaluation.',
        interpretation='Prospective new-recording validation within the same Lake assets; no unseen-map, real-flight, real-network or adaptive-superiority claim without supporting results.')
    freeze_json(root / 'plan.json', plan)
    return plan


def verify(root):
    root = Path(root)
    plan = read(root / 'plan.json')
    require(plan['source_hashes'] == source_hashes(), 'Frozen routing implementation changed')
    require(plan['reference_parent_plan_sha256'] == sha(PARENT / 'plan.json'), 'Reference study changed')
    require(plan['native_protocol_sha256'] == sha(Path(plan['probe']) / 'protocol.json'), 'Native protocol changed')
    verify_protocol(plan['probe'])
    for name, path in plan['profiles'].items():
        require(sha(root / path) == plan['profile_sha256'][name], 'Old control profile changed')
    require(sha(root / plan['calibration_path']) == plan['calibration_sha256'], 'Calibration changed')
    calibrated.verify_calibration(read(root / plan['calibration_path']))
    require(plan['grid'] == make_grid(plan['cases']) and plan['expected_runs'] == len(plan['grid']) == 648,
            'Incomplete or changed paired factorial')
    return plan


def _initialize(root, plan, binding):
    global _STATE
    terrain, _ = paid.load_public_terrain(Path(plan['assets']))
    _STATE = (root, plan, terrain, paid.load_bank(binding),
              read(Path(root) / 'predictions' / binding['case_id'] / 'manifest.json'))


def _run(item):
    root, plan, terrain, bank, manifest = _STATE
    row = run_case(root, plan, item, terrain, bank, manifest)
    save(Path(root) / row['path'] / 'row.json', row)
    return row


def execute(root, workers=3):
    root = Path(root).resolve()
    plan = verify(root)
    require(workers in (1, 2, 3), 'One to three workers required')
    bindings = paid.bind_predictions(root, plan)
    rows, started = [], time.perf_counter()
    for binding in bindings:
        pending = []
        for item in (i for i in plan['grid'] if i['flight'] == binding['case_id']):
            path = root / 'runs' / item['run_id'] / 'row.json'
            if path.exists():
                row = read(path)
                require(all(row[k] == v for k, v in item.items()), 'Completed cell changed')
                require(read(path.parent / 'protocol.json')['plan_sha256'] == sha(root / 'plan.json'), 'Wrong prior plan')
                for name, value in row['artifacts_sha256'].items():
                    require(sha(path.parent / name) == value, 'Completed artifact changed')
                rows.append(row)
            else:
                pending.append(item)
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=_initialize, initargs=(root, plan, binding)) as pool:
            for future in as_completed([pool.submit(_run, item) for item in pending]):
                row = future.result()
                rows.append(row)
                save(root / 'progress.json', dict(completed_runs=len(rows), expected_runs=len(plan['grid'])))
                print(f'{len(rows)}/{len(plan["grid"])} {row["run_id"]}: success={row["local_inspection_success"]}', flush=True)
    require(len(rows) == len({r['run_id'] for r in rows}) == plan['expected_runs'], 'Incomplete comparison')
    verify(root)
    rows.sort(key=lambda r: r['run_id'])
    freeze_json(root / 'run_index.json', rows)
    freeze_json(root / 'completion.json', dict(complete=True, runs=len(rows),
        plan_sha256=sha(root / 'plan.json'), run_index_sha256=sha(root / 'run_index.json'),
        native_bindings_sha256=sha(root / 'native_bindings.json'),
        prediction_manifest_sha256={c:sha(root/'predictions'/c/'manifest.json') for c in plan['cases']},
        local_inspection_successes=sum(r['local_inspection_success'] for r in rows),
        elapsed_host_s=time.perf_counter()-started, full_wide_mission_evaluated=False))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.output, args.probe, args.calibration)
    elif args.mode == 'verify':
        verify(args.output)
    else:
        execute(args.output, args.workers)
