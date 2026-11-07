"""Prospective second cohort with unchanged geometry rules and fresh sectors.

The first six clips may inform router calibration. This wrapper changes only
the predeclared sector indices and output location of the frozen acquisition
implementation. Every candidate height, route, camera and coverage rule remains
identical; no second-cohort model result is used to select or replace a case.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import types

from experiments.mission_runtime_profile import read, save, sha
from experiments.strong_routing_20261030 import acquisition as original

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / 'results/strong_routing_round2_acquisition_v1_20261030'
SECTORS = ('sector_3', 'sector_7')
FIRST_COHORT = original.OUTPUT


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    first = original.verify(FIRST_COHORT)
    if first['declared_case_count'] != 6 or len(first['cases']) != 6:
        raise ValueError('Complete original six-case acquisition design required')
    namespace = dict(original.prepare.__globals__)
    namespace.update(SECTORS=SECTORS)
    bound = types.FunctionType(original.prepare.__code__, namespace,
                               original.prepare.__name__, original.prepare.__defaults__,
                               original.prepare.__closure__)
    plan = bound(output)
    plan.update(schema_version='strong_routing_round2_geometry_acquisition_v1',
        site_rule='Sector_3 and sector_7 in every existing wide layout, declared before second-cohort RGB or native model outcomes.',
        round=2,
        first_cohort_plan=str(FIRST_COHORT / 'plan.json'),
        first_cohort_plan_sha256=sha(FIRST_COHORT / 'plan.json'),
        acquisition_changes=['Focal sector indices:2,6 to3,7', 'Output directory and provenance labels'],
        unchanged_geometry_implementation=str(Path(original.__file__).resolve()),
        unchanged_geometry_implementation_sha256=sha(original.__file__),
        algorithm_identity='The original prepare function code object executes in an isolated globals copy; only SECTORS changes. No global monkeypatch or geometry function replacement.',
        outcome_awareness='First-cohort model outcomes are known and may be used for router development. No second-cohort native result has been inspected; every declared case and geometry rejection will be retained.',
        causal_scope='Prospective fresh local inspection regions on the same known terrain and target assets. Not a new-map or full eight-sector mission evaluation.')
    plan['source_inputs_sha256'].update({str(Path(__file__).resolve()): sha(__file__),
        str(FIRST_COHORT / 'plan.json'): sha(FIRST_COHORT / 'plan.json')})
    save(output / 'plan.json', plan)
    original.verify(output)
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'verify', 'execute', 'audit'])
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    operation = prepare if args.mode == 'prepare' else getattr(original, args.mode)
    result = operation(args.output)
    if args.mode in ('prepare', 'audit'):
        print(f'{args.mode}: {len(result["cases"])} cases')
