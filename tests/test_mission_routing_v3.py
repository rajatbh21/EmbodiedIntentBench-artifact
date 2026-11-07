"""Resource DP, causal boundary and mission/history counterfactual regressions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.lake_causal_router import MissionCosts, forecast
from experiments.mission_routing_v3.budget import Action, Continuation
from experiments.mission_routing_v3.policy import support_probability


def brute(plan, index, previous, first=None):
    values = []
    for config in [first] if first else plan.actions:
        completion, next_index, energy, final = plan.transition(index, previous, config)
        if completion > plan.deadline_s or (final and plan.costs.duration_s > plan.deadline_s):
            value = float("inf")
        else:
            value = energy + (0 if final else brute(plan, next_index, config))
        values.append(value)
    return min(values)


@pytest.mark.parametrize("duration", [6, 8, 10])
@pytest.mark.parametrize("deadline", [8, 10, 12])
def test_budget_matches_exhaustive_switching_sequences(duration, deadline):
    costs = MissionCosts(duration, 2, 1000, 7, 0, 0, 10_000_000)
    actions = [Action("fast", 0.3, 1, 0.4, 2), Action("slow", 2.7, 4, 1.4, 3)]
    plan = Continuation(costs, actions, deadline)
    for previous in (None, "fast", "slow"):
        for first in ("fast", "slow"):
            assert plan.required(0, previous, first) == pytest.approx(
                brute(plan, 0, previous, first)
            )
            if plan.required(0, previous, first) < float("inf"):
                assert sum(row[3] for row in plan.witness(0, previous, first)) == pytest.approx(
                    plan.required(0, previous, first)
                )


def test_one_expensive_action_can_fit_when_constant_model_does_not():
    costs = MissionCosts(10, 2, 1000, 1, 0, 0, 10_000_000)
    plan = Continuation(costs, [Action("cheap", 0.1, 1, 0, 0), Action("good", 0.1, 20, 0, 0)], 10)
    assert plan.required(0, None, "good") == 34
    assert 10 + 5 * 20 > 40 > plan.required(0, None, "good")
    assert [row[1] for row in plan.witness(0, None, "good")] == ["good"] + ["cheap"] * 4


def test_last_call_overshoot_and_tick_rounding_are_charged():
    costs = MissionCosts(6, 2, 1000, 10, 0, 0, 10_000_000)
    plan = Continuation(costs, [Action("slow", 3.001, 2, 0, 0)], 8)
    completion, _, energy, final = plan.transition(2, "slow", "slow")
    assert completion == 7.01 and final
    assert energy == pytest.approx(32.1)
    assert plan.required(2, "slow", "slow") == energy


def test_repeat_support_changes_ranking_with_history_and_intent():
    fast_slots, slow_slots = [2, 4, 6, 8], [2, 6]
    fast = support_probability([], fast_slots, 0.5, 2, 2, 8)
    slow = support_probability([], slow_slots, 0.82, 2, 2, 8)
    assert fast > slow  # Fresh discovery favors more opportunities.
    fast += support_probability([0], fast_slots, 0.5, 2, 2, 8)
    slow += support_probability([0], slow_slots, 0.82, 2, 2, 8)
    assert slow > fast  # One pending confirmation changes the preferred model.
    fast = support_probability([], fast_slots, 0.5, 3, 4, 8)
    slow = support_probability([], slow_slots, 0.82, 3, 4, 8)
    fast += support_probability([0], fast_slots, 0.5, 3, 4, 8)
    slow += support_probability([0], slow_slots, 0.82, 3, 4, 8)
    assert fast > slow  # A stricter intent changes the choice again.


def test_gap_reset_span_and_empty_horizon():
    assert support_probability([0], [10], 1, 2, 2, 8) == 0
    assert support_probability([0], [1], 1, 2, 2, 8) == 0
    assert support_probability([0], [2], 1, 2, 2, 8) == 1
    assert support_probability([], [], 1, 2, 2, 8) == 0


# The same paid runner fixture used by frozen v2; it supplies genuine typed
# public resources/contracts and checks repeated public feedback deliveries.
from experiments.mission_routing_v3.policy import MODES, Policy  # noqa: E402
from tests import test_mission_routing_v2 as prior  # noqa: E402


@pytest.mark.parametrize("mode", MODES)
def test_v3_paid_runtime(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(prior, "Policy", Policy)
    prior.test_feedback_controller_runs_through_paid_runtime(tmp_path, mode)


def observation_fixture():
    from experiments.lake_intent_contract import IntentContract, immutable
    from experiments.mission_routing_v2.features import NAMES, encode, fit
    from experiments.mission_routing_v2.flights import contract_document
    from experiments.v4_benchmark.calibrated_router import model_signature
    from experiments.v4_benchmark.scenario import load_scenario
    from tests.test_v4_benchmark import ROOT

    scenario = load_scenario(ROOT / "experiments/v4_benchmark/example_scenario.json")
    # Retain local candidates only; the causal controller deliberately excludes remote.
    specs = tuple(s for s in scenario.executors if s.kind != "simulated_remote")
    rgb = np.zeros((8, 8, 3), np.uint8)
    feature = encode(rgb)
    profile = dict(
        feature_names=list(NAMES),
        models={
            s.config_id: dict(signature=model_signature(s), quality=fit([feature] * 4, [0.5] * 4))
            for s in scenario.executors
        },
    )
    contract = IntentContract(contract_document([str(i) for i in range(6)], "rapid_survey"))
    o = SimpleNamespace(
        state=SimpleNamespace(
            current_time_s=0, frame_id=0, current_config_id=None, battery_frac=0.9
        ),
        contract=replace(scenario.contract, deadline_s=130),
        configs=scenario.catalog,
        resources=scenario.resources,
        rgb=rgb,
        intent_contract=immutable(contract.public_document()),
        tracking={
            "cost": {"memory_mb": 0, "energy_j": 1, "latency_s": 0.1},
            "configuration": {"max_gap_s": 8, "min_hits": 2},
        },
        evidence_history={
            "available_at_s": 0,
            "last_completion_s": None,
            "counts_are_ground_truth": False,
            "tracks": [],
        },
    )
    assert specs
    return profile, scenario, o


def test_future_feedback_is_rejected_and_profile_lineage_is_not_stored():
    profile, scenario, o = observation_fixture()
    profile["training_lineage"] = [{"hidden": "not runtime data"}]
    policy = Policy(profile, scenario.executors, MissionCosts.from_scenario(scenario))
    assert not hasattr(policy, "training_lineage")
    o.evidence_history["last_completion_s"] = 1
    with pytest.raises(ValueError, match="completed"):
        policy.select(o)


def test_policy_budget_matches_every_recorded_candidate_cost():
    profile, scenario, o = observation_fixture()
    policy = Policy(profile, scenario.executors, MissionCosts.from_scenario(scenario))
    policy.select(o)
    for candidate in policy.last_decision["candidates"]:
        c = candidate["config_id"]
        assert candidate["mixed_energy_j"] <= candidate["constant_energy_j"] + 1e-8
        plan = policy._budget.witness(0, None, c)
        assert sum(row[3] for row in plan) == pytest.approx(candidate["mixed_energy_j"])
        a = policy._budget.actions[c]
        slots = forecast(policy.costs, 0, a.latency_s, a.load_latency_s)
        assert candidate["first_completion_s"] == slots[0][1]


def test_full_policy_choices_change_with_public_history_and_intent():
    import copy

    profile, scenario, o = observation_fixture()
    specs = tuple(
        replace(s, mission_latency_s=t, energy_j_per_call=1)
        for s, t in zip(scenario.executors, [0.2, 3.3], strict=True)
    )
    for spec, quality in zip(specs, [0.5, 0.82], strict=True):
        model = profile["models"][spec.config_id]["quality"]
        model["coefficient"] = [quality] + [0.0] * (len(model["coefficient"]) - 1)
    o.state.current_time_s, o.state.frame_id = 2, 1
    o.evidence_history.update(
        available_at_s=2, last_completion_s=1, tracks=[dict(capture_times_s=[0], reported=False)]
    )
    costs = MissionCosts(10, 2, 100000, 1, 0, 0, 10_000_000)
    full = Policy(profile, specs, costs)
    no_history = Policy(profile, specs, costs, "no_history_v3")
    assert full.select(o) == specs[1].config_id
    assert no_history.select(o) == specs[0].config_id
    # Public mission contract only; no evaluator target identities.
    from experiments.mission_routing_v2.policy import plain

    o.intent_contract = copy.deepcopy(plain(o.intent_contract))
    o.intent_contract["mission"]["phases"][0].update(min_hits=3, min_evidence_span_s=4)
    assert full.select(o) == specs[0].config_id
