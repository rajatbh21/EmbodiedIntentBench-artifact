"""Risk units, capture-local ledgers, ablations, and genuine paid runner integration."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from experiments.lake_causal_router import MissionCosts
from experiments.mission_routing_v2.features import NAMES, encode
from experiments.mission_routing_v2.policy import plain
from experiments.mission_routing_v3.policy import Policy as EvidencePolicy
from experiments.mission_routing_v4.policy import MODES, Policy
from experiments.mission_routing_v4.risk import estimate, fit_count, phase_projection
from tests import test_mission_routing_v2 as prior
from tests.test_mission_routing_v3 import observation_fixture


def profile_for(profile, feature):
    return dict(
        schema_version="mission_false_component_risk_v1",
        feature_names=list(NAMES),
        training_lineage=[{"evaluator_target_ids": "never retained"}],
        models={
            c: dict(signature=m["signature"], risk=fit_count([feature] * 8, [0] * 8, ["a"] * 8))
            for c, m in profile["models"].items()
        },
    )


def fixture(mode="risk_aware"):
    profile, scenario, o = observation_fixture()
    o.intent_contract = plain(o.intent_contract)
    specs = tuple(
        replace(s, mission_latency_s=0.2, energy_j_per_call=1) for s in scenario.executors
    )
    for spec, quality in zip(specs, [0.9, 0.5], strict=True):
        model = profile["models"][spec.config_id]["quality"]
        model["coefficient"] = [quality] + [0] * (len(NAMES) - 1)
    risk = profile_for(profile, encode(o.rgb))
    for spec, rate in zip(specs, [0.5, 0.02], strict=True):
        risk["models"][spec.config_id]["risk"]["floor"] = rate
        risk["models"][spec.config_id]["risk"]["pseudocount_per_capture"] = 0.001
    costs = MissionCosts(120, 2, 100000, 1, 0, 0, 10_000_000)
    return Policy(profile, specs, costs, risk, mode), o, profile, specs, costs, risk


def advance(policy, o, index):
    o.state.frame_id, o.state.current_time_s = index, index * 2
    o.state.current_config_id = policy.last_decision["selected_config_id"]
    o.evidence_history.update(available_at_s=index * 2, last_completion_s=policy._previous[2])


def test_counts_are_not_probabilities_and_zero_events_have_positive_floor():
    f = encode(np.zeros((8, 8, 3), np.uint8))
    assert estimate(fit_count([f] * 20, [4] * 20, ["a"] * 20), f) > 1
    assert estimate(fit_count([f] * 20, [0] * 20, ["a"] * 20), f) > 0
    head = fit_count([f] * 20, [0] * 10 + [4] * 10, ["a"] * 10 + ["b"] * 10)
    assert head["floor"] == pytest.approx(41 / 11)


@pytest.mark.parametrize("bad", [-1, 0.5, float("nan"), float("inf")])
def test_invalid_count_labels_rejected(bad):
    f = encode(np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(ValueError):
        fit_count([f], [bad], ["a"])


def test_late_results_are_charged_to_capture_phase():
    p = dict(phase_id="one", start_s=0, end_s=30, deadline_s=30, max_false_components=2)
    r = phase_projection(p, 28, [(28, 35), (36, 43)], 0.4, 1.7)
    assert r["remaining_captures"] == 1
    assert r["projected_false_components"] == pytest.approx(2.1)
    assert not r["within_estimated_budget"]


def test_error_budget_changes_choice_without_ground_truth():
    policy, o, profile, specs, costs, _ = fixture()
    assert EvidencePolicy(profile, specs, costs).select(o) == specs[0].config_id
    assert policy.select(o) == specs[1].config_id
    relaxed, relaxed_o, *_ = fixture()
    relaxed_o.intent_contract["mission"]["phases"][0]["max_false_components"] = 100
    assert relaxed.select(relaxed_o) == specs[0].config_id
    assert not hasattr(policy, "training_lineage")
    assert "evaluator_target_ids" not in repr(policy.last_decision)
    assert "evaluator_target_ids" not in repr(policy.risk_models)


def test_ledger_is_idempotent_and_revisit_is_a_new_budget():
    policy, o, *_ = fixture()
    choice = policy.select(o)
    before = copy.deepcopy(policy.last_decision)
    assert policy.select(o) == choice and policy.last_decision == before
    first_phase = o.intent_contract["mission"]["phases"][0]["phase_id"]
    assert policy.ledger[first_phase] == pytest.approx(0.02)
    advance(policy, o, 1)
    policy.select(o)
    assert policy.ledger[first_phase] == pytest.approx(0.04)
    advance(policy, o, 45)
    policy.select(o)
    revisit = o.intent_contract["mission"]["phases"][3]["phase_id"]
    assert policy.ledger[first_phase] == pytest.approx(0.04)
    assert policy.ledger[revisit] == pytest.approx(0.02)


def test_different_repeat_input_and_uncompleted_action_rejected():
    policy, o, *_ = fixture()
    policy.select(o)
    o.intent_contract["mission"]["phases"][0]["max_false_components"] += 1
    with pytest.raises(ValueError, match="different public input"):
        policy.select(o)
    policy, o, *_ = fixture()
    policy.select(o)
    advance(policy, o, 1)
    o.evidence_history["last_completion_s"] = None
    with pytest.raises(ValueError, match="previous selected execution"):
        policy.select(o)


def test_accumulated_risk_changes_decision_vs_ledger_ablation():
    full, o, *_ = fixture()
    ablated, other, *_ = fixture("no_risk_ledger_v4")
    # At the last capture, both actions fit from zero, but not after accumulated risk.
    for policy, obs in ((full, o), (ablated, other)):
        phase = obs.intent_contract["mission"]["phases"][0]
        policy.ledger[phase["phase_id"]] = 1.8
        obs.state.frame_id, obs.state.current_time_s = 14, 28
        obs.evidence_history["available_at_s"] = 28
        obs.evidence_history["tracks"] = [dict(capture_times_s=[26], reported=False)]
    assert full.select(o) != ablated.select(other)


def test_no_admissible_risk_is_explicit_and_does_not_relax_contract():
    policy, o, *_ = fixture()
    o.intent_contract["mission"]["phases"][0]["max_false_components"] = 0
    policy.select(o)
    assert not policy.last_decision["risk_admissible"]
    assert policy.last_decision["risk_fallback"] == "minimum_estimated_risk_excess"
    assert o.intent_contract["mission"]["phases"][0]["max_false_components"] == 0


@pytest.mark.parametrize("mode", MODES)
def test_paid_tracking_runtime(tmp_path, monkeypatch, mode):
    def factory(profile, specs, costs, mode):
        risk = profile_for(profile, encode(np.zeros((8, 8, 3), np.uint8)))
        return Policy(profile, specs, costs, risk, mode)

    monkeypatch.setattr(prior, "Policy", factory)
    prior.test_feedback_controller_runs_through_paid_runtime(tmp_path, mode)
