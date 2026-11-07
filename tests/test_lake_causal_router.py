"""Synthetic split-leakage and causal resource/intent decision regressions."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.lake_causal_router import (
    CausalRouter,
    MissionCosts,
    check_split,
    fit_profile,
    forecast,
)
from experiments.v4_benchmark.resources import ResourceConfig
from experiments.v4_benchmark.scenario import load_scenario

ROOT = Path(__file__).parents[1]
LINEAGE = dict(recording="record", world="world", scenery="map", layout="layout", route="route")


def fixture(mode="intent_context"):
    scenario = load_scenario(ROOT / "experiments/v4_benchmark/example_scenario.json")
    specs = scenario.executors
    rows = [
        dict(
            capture_id=i,
            config_id=s.config_id,
            visible_target_ids=["secret"],
            matched_target_ids=["secret"] if i < (5 if n == 0 else 8) else [],
            matched_components=int(i < (5 if n == 0 else 8)),
            predicted_components=1,
        )
        for i in range(8)
        for n, s in enumerate(specs)
    ]
    profile = fit_profile(rows, {i: "low_contrast" for i in range(8)}, specs, LINEAGE)
    costs = MissionCosts(8, 0.5, 1000, 1, 0.01, 0.02, 10000000)
    observation = SimpleNamespace(
        intent_contract=dict(
            mission=dict(
                phases=[
                    dict(
                        phase_id="p",
                        start_s=0,
                        end_s=6,
                        deadline_s=6,
                        min_hits=2,
                        min_evidence_span_s=1,
                        required_recall=1,
                        weight=1,
                    )
                ]
            ),
            reporting=dict(mode="evidence_only"),
        ),
        state=SimpleNamespace(current_time_s=0, frame_id=0, battery_frac=1, current_config_id=None),
        tracking=dict(cost=dict(latency_s=0.1, energy_j=1, memory_mb=64)),
        contract=replace(scenario.contract, min_final_battery_frac=0.2, deadline_s=10),
        configs=scenario.catalog,
        resources=ResourceConfig(),
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        confirmed_reports_so_far=0,
    )
    return CausalRouter(profile, specs, costs, mode=mode), observation, profile, rows, specs


@pytest.mark.parametrize("key", LINEAGE)
def test_any_training_overlap_rejects_held_out_including_lighting_change(key):
    candidate = {k: v + "new" for k, v in LINEAGE.items()}
    candidate[key] = LINEAGE[key]
    with pytest.raises(ValueError, match=key):
        check_split([LINEAGE], candidate, scope="held_out")
    result = check_split([LINEAGE], candidate, scope="development")
    assert result["overlaps"] == [key]
    assert not result["independent_test"]


def test_independent_split_and_missing_lineage():
    assert check_split([LINEAGE], {k: v + "new" for k, v in LINEAGE.items()}, scope="held_out")[
        "independent_test"
    ]
    with pytest.raises(ValueError):
        check_split([], LINEAGE, scope="held_out")
    with pytest.raises(ValueError):
        check_split([LINEAGE], LINEAGE, scope="test")


def test_training_discards_target_ids_and_rejects_missing_pairs():
    router, obs, profile, rows, specs = fixture()
    assert "secret" not in repr(profile)
    assert "training_lineage" not in vars(router)
    with pytest.raises(ValueError, match="Complete"):
        fit_profile(rows[:-1], dict.fromkeys(range(8), "low_contrast"), specs, LINEAGE)
    selected = router.select(obs)
    rows[0]["matched_target_ids"].append("future_secret")
    assert router.select(obs) == selected


def test_phase_deadline_changes_choice_without_hidden_target_data():
    router, obs, profile, _, specs = fixture()
    assert router.select(obs) == "CFG_V4_STRONG"
    obs.intent_contract["mission"]["phases"][0].update(
        end_s=0.9, deadline_s=0.9, min_evidence_span_s=0.5
    )
    assert router.select(obs) == "CFG_V4_LIGHT"
    ablation = CausalRouter(profile, specs, router.costs, mode="no_intent")
    assert ablation.select(obs) == "CFG_V4_STRONG"


def test_context_and_no_context_ablations_differ_with_supported_calibration():
    router, obs, profile, _, specs = fixture()
    strong = profile["models"]["CFG_V4_STRONG"]
    strong["contexts"]["low_contrast"]["recall"] = 0
    contextual = CausalRouter(profile, specs, router.costs)
    global_only = CausalRouter(profile, specs, router.costs, mode="no_context")
    assert contextual.select(obs) == "CFG_V4_LIGHT"
    assert global_only.select(obs) == "CFG_V4_STRONG"
    strong["contexts"]["low_contrast"]["frames"] = 1
    sparse = CausalRouter(profile, specs, router.costs)
    assert sparse.select(obs) == "CFG_V4_STRONG"


def test_report_count_cannot_be_used_as_correct_unique_target_count():
    router, obs, *_ = fixture()
    selected = router.select(obs)
    obs.confirmed_reports_so_far = 100000
    assert router.select(obs) == selected
    assert not router.last_decision["report_count_used_for_stopping"]


def test_battery_pressure_and_tracking_memory_are_consumed():
    router, obs, *_ = fixture()
    assert router.select(obs) == "CFG_V4_STRONG"
    obs.state.battery_frac = 0.28
    assert router.select(obs) == "CFG_V4_LIGHT"
    obs.resources = ResourceConfig(memory_capacity_mb=63)
    with pytest.raises(ValueError, match="memory"):
        router.select(obs)


def test_report_deadline_changes_tail_choice_and_never_hides_infeasibility():
    router, obs, *_ = fixture()
    obs.state.frame_id, obs.state.current_time_s = 15, 7.5
    obs.intent_contract["reporting"] = dict(mode="report_required", deadline_s=8)
    assert router.select(obs) == "CFG_V4_LIGHT"
    obs.intent_contract["reporting"]["deadline_s"] = 7.51
    router.select(obs)
    assert router.last_decision["reason"] == "no_feasible_candidate_best_effort"


def test_exact_tick_rounding_switch_delay_and_final_overshoot():
    costs = MissionCosts(3, 1, 1000, 1, 0, 0, 10000000)
    assert forecast(costs, 0, 0.999999999, 0.01) == [(0, 1.01), (2, 3)]
    assert forecast(costs, 2, 1.001, 0) == [(2, 3.01)]


def test_configuration_signature_mismatch_is_rejected():
    router, _, profile, _, specs = fixture()
    with pytest.raises(ValueError, match="signature"):
        CausalRouter(profile, (replace(specs[0], parameters={}), specs[1]), router.costs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_public_costs_rejected(value):
    with pytest.raises(ValueError):
        MissionCosts(8, 0.5, 1000, value, 0, 0, 10000000)
