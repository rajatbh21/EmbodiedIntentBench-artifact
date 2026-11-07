"""Paired resource interventions retain the paid replay and public boundary."""
from dataclasses import replace

import pytest

from experiments.mission_runtime_profile import save
from experiments.strong_routing_20261030 import study
from experiments.wide_area_diagnosis import paid_clip as paid
from tests.test_wide_area_paid_clip import local_fixture
from tests.test_wide_area_runner import outputs


def test_complete_factorial_has_unique_paired_cells():
    cases = {f'site_{i}': dict(duration_s=42) for i in range(6)}
    grid = study.make_grid(cases)
    assert grid == study.make_grid(dict(reversed(list(cases.items()))))
    assert len(grid) == len({r['run_id'] for r in grid}) == 648
    for policy in study.POLICIES:
        cells = [(r['flight'], r['perception_budget_s'], r['link'], r['energy_condition'])
                 for r in grid if r['policy'] == policy]
        assert len(set(cells)) == 72
    charge = study.initial_battery(dict(duration_s=42), 'residual_250j')
    assert (charge - .2) * 432000 - 42 * 120 == pytest.approx(250)


@pytest.mark.parametrize('name,up,down,rtt,outage', [
    ('good', 50_000_000, 100_000_000, 30_000_000, False),
    ('degraded', 2_000_000, 10_000_000, 150_000_000, False),
    ('offline', 0, 0, 30_000_000, True),
])
def test_whole_clip_network_regimes_do_not_target_detection_times(name, up, down, rtt, outage):
    for t in range(72):
        snapshot = study.link_trace(name, 42).snapshot(t * 10**9)
        assert (snapshot.upload_bps, snapshot.download_bps, snapshot.rtt_ns, snapshot.outage) == (up, down, rtt, outage)


def test_only_declared_runner_dependencies_change():
    original = dict(paid.run_case.__globals__)
    bound = study.bound_run_loop()
    assert bound.__code__ is paid.run_case.__code__
    assert {k for k, v in bound.__globals__.items() if original[k] is not v} == {'load_case', 'make_policy', 'source_hashes'}
    assert all(paid.run_case.__globals__[k] is v for k, v in original.items())


@pytest.mark.parametrize('energy', study.BATTERIES)
def test_initial_charge_is_consistent_in_intent_scenario_and_paid_accounting(tmp_path, monkeypatch, energy):
    settings, state = local_fixture(tmp_path, monkeypatch)
    state['scenario'] = replace(state['scenario'], battery_capacity_wh=120, flight_power_w=120)
    case = dict(case_id='synthetic_flight', duration_s=30, motion_s=26,
                focal_sector='sector_1', evaluator_target_ids=state['contract'].document()['evaluator_target_ids'])
    charge = study.initial_battery(case, energy)
    settings['item'].update(energy_condition=energy, battery=charge)
    settings['plan'].update(cases={'synthetic_flight': case}, policy_costs={}, calibration_path='calibration.json')
    save(settings['root'] / 'calibration.json', {'fixture': True})
    save(settings['root'] / 'plan.json', settings['plan'])
    new = study.load_case(settings['root'], settings['plan'], settings['item'])
    assert new['scenario'].initial_battery_frac == new['contract'].public_document()['mission']['initial_battery_frac'] == charge
    assert 'evaluator_target_ids' not in str(new['contract'].public_document())
    row = study.run_case(**settings)
    saved = outputs(settings, row)
    assert row['flight_energy_j'] == 3600
    assert row['final_battery_frac'] == pytest.approx(charge - row['total_energy_j'] / 432000)
    assert saved['intent']['mission']['initial_battery_frac'] == charge
    assert row['nonflight_energy_j'] > 0


def test_unknown_resource_conditions_rejected():
    with pytest.raises(ValueError):
        study.link_trace('target_selected_outage', 42)
    with pytest.raises(ValueError):
        study.initial_battery(dict(duration_s=42), 'fitted_after_results')
