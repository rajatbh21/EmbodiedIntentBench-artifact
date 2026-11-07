"""Temporal label causality, distinct objectives, ablations, and paid integration."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from experiments.mission_routing_v2.features import NAMES, encode
from experiments.mission_routing_v5.quality import fit
from experiments.mission_temporal_routing.fitting import fit as temporal_fit
from experiments.mission_temporal_routing.policy import STRATEGIES, Policy
from experiments.mission_temporal_routing.report import assert_grid
from experiments.mission_temporal_routing.temporal import (
    PATTERNS,
    REQUIREMENTS,
    head_key,
    possible,
    support,
    window_labels,
)
from experiments.resource_routing_study.policy import BatteryBlind
from tests import test_mission_routing_v2 as prior
from tests.test_mission_routing_v4 import advance, fixture, profile_for
from tests.test_mission_routing_v5 import recall_profile


def temporal_profile(old, rgb):
    feature = encode(rgb)
    return dict(
        schema_version="temporal_support_v1",
        feature_names=list(NAMES),
        models={
            c: dict(
                signature=m["signature"],
                heads={
                    head_key(*r, pat): fit([feature] * 4, [1] * 4, [2] * 4)
                    for r in REQUIREMENTS
                    for pat in PATTERNS
                    if possible(*r, pat)
                },
            )
            for c, m in old["models"].items()
        },
    )


def build(strategy="temporal_mission"):
    _, o, old, specs, costs, risk = fixture()
    o.confirmed_reports_so_far = 0
    temporal = temporal_profile(old, o.rgb)
    recall = recall_profile(old, "base_recall")
    return (
        Policy(recall, specs, costs, risk, temporal, strategy=strategy),
        o,
        recall,
        specs,
        costs,
        risk,
        temporal,
    )


def rows(matches, visible=None):
    return [
        dict(
            capture_id=i,
            capture_time_s=i * 2,
            visible_target_ids=["a", "b"] if visible is None else visible[i],
            matched_target_ids=m,
        )
        for i, m in enumerate(matches)
    ]


def test_temporal_dependence_distinguishes_identical_frame_hit_counts():
    persistent = rows([["a"], ["a"], ["a"], ["a"]])
    changing = rows([["a"], ["b"], ["a"], ["b"]])
    assert [len(r["matched_target_ids"]) for r in persistent] == [
        len(r["matched_target_ids"]) for r in changing
    ]
    assert window_labels(persistent, 3, 4, (0, 2, 4, 6)) == [(0, 1, 2)]
    assert window_labels(changing, 3, 4, (0, 2, 4, 6)) == [(0, 0, 2)]


def test_visibility_denominator_is_current_and_tail_is_not_padded():
    r = rows([["a"], ["b"], ["b"], ["b"]], [["a"], ["b"], ["b"], ["b"]])
    assert window_labels(r, 3, 4, (0, 2, 4, 6)) == [(0, 0, 1)]
    assert window_labels(r[:2], 3, 4, (0, 2, 4, 6)) == []
    assert window_labels(rows([[], []], [[], []]), 2, 2, (0, 2)) == [(0, 0, 0)]


def test_skipped_capture_changes_learned_repetition_label():
    r = rows([["a"], ["a"], ["b"], ["b"]])
    assert window_labels(r, 2, 2, (0, 2))[0] == (0, 1, 2)
    assert window_labels(r, 2, 2, (0, 4))[0] == (0, 0, 2)


def test_structurally_impossible_support_is_zero_without_loading_a_head():
    assert support({}, [], 3, 4, (0, 2)) == 0
    assert support({}, [], 3, 4, ()) == 0
    with pytest.raises(ValueError):
        support({}, [], 3, 4, (0, 3, 6))
    with pytest.raises(ValueError):
        window_labels(rows([["unknown"]], [["a"]]), 2, 2, (0,))


def test_newton_safeguard_preserves_binomial_objective_and_sparse_counts():
    dark = encode(np.zeros((8, 8, 3), np.uint8))
    light = encode(np.full((8, 8, 3), 255, np.uint8))
    x = [dark, light] * 50
    for hits in ([0, 0] * 50, [0, 2] * 50, [1, 1] * 50):
        a, b = fit(x, hits, [2] * 100), temporal_fit(x, hits, [2] * 100)
        assert np.allclose(a["coefficient"], b["coefficient"], atol=1e-6)
        assert a["target_hits"] == b["target_hits"]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_idempotence_causal_completion_and_no_runtime_gt(strategy):
    p, o, *_ = build(strategy)
    p.select(o)
    original = copy.deepcopy((p.last_decision, p.ledger, p.decisions))
    p.select(o)
    assert original == (p.last_decision, p.ledger, p.decisions)
    assert "evaluator_target_ids" not in repr(p.last_decision)
    assert not p.last_decision["learned_score_is_mission_probability"]
    advance(p, o, 1)
    o.evidence_history["last_completion_s"] += 0.1
    with pytest.raises(ValueError, match="must complete"):
        p.select(o)
    assert p.decisions == 1 and p.ledger == original[1]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_resource_guard_cannot_be_bypassed_by_high_temporal_score(strategy):
    _, o, recall, specs, costs, risk, temporal = build(strategy)
    specs = (replace(specs[0], energy_j_per_call=1e9), specs[1])
    p = Policy(recall, specs, costs, risk, temporal, strategy=strategy)
    assert p.select(o) == specs[1].config_id


def test_image_ablation_makes_candidate_scores_image_invariant():
    a, o, *_ = build("temporal_no_image")
    b, other, *_ = build("temporal_no_image")
    other.rgb = np.full_like(other.rgb, 255)
    assert a.select(o) == b.select(other)
    assert [c["temporal_log_score"] for c in a.last_decision["candidates"]] == [
        c["temporal_log_score"] for c in b.last_decision["candidates"]
    ]


def test_repetition_ablation_is_input_mask_not_evaluator_edit():
    a, o, *_ = build("temporal_no_repetition")
    b, other, *_ = build("temporal_no_repetition")
    for phase in other.intent_contract["mission"]["phases"]:
        phase.update(min_hits=3, min_evidence_span_s=4)
    original = copy.deepcopy(other.intent_contract)
    assert a.select(o) == b.select(other)
    assert a.last_decision == b.last_decision
    assert other.intent_contract == original


def test_learned_temporal_head_can_reverse_selection_with_intent():
    _, o, recall, specs, costs, risk, temporal = build("temporal_only")
    # Equal frame recall, but opposite learned temporal ordering by repetition demand.
    import math

    for i, spec in enumerate(specs):
        for key, head in temporal["models"][spec.config_id]["heads"].items():
            high = key.startswith("k2") == (i == 0)
            head["coefficient"] = [math.log(9) if high else -math.log(9)] + [0] * (len(NAMES) - 1)
    survey = Policy(recall, specs, costs, risk, temporal, strategy="temporal_only")
    inspection = Policy(recall, specs, costs, risk, temporal, strategy="temporal_only")
    for phase in o.intent_contract["mission"]["phases"]:
        phase.update(min_hits=2, min_evidence_span_s=2)
    assert survey.select(o) == specs[0].config_id
    for phase in o.intent_contract["mission"]["phases"]:
        phase.update(min_hits=3, min_evidence_span_s=4)
    assert inspection.select(o) == specs[1].config_id


@pytest.mark.parametrize("strategy", [*STRATEGIES, "temporal_no_battery"])
def test_paid_runtime(tmp_path, monkeypatch, strategy):
    def factory(old, specs, costs, mode):
        rgb = np.zeros((8, 8, 3), np.uint8)
        risk = profile_for(old, encode(rgb))
        p = Policy(
            recall_profile(old, "base_recall"),
            specs,
            costs,
            risk,
            temporal_profile(old, rgb),
            strategy="temporal_mission" if strategy == "temporal_no_battery" else strategy,
        )
        return BatteryBlind(p, costs.capacity_j) if strategy == "temporal_no_battery" else p

    monkeypatch.setattr(prior, "Policy", factory)
    prior.test_feedback_controller_runs_through_paid_runtime(tmp_path, "confirmation")


def test_comparison_denominator_rejects_duplicates_and_missing_cells():
    r = dict(flight="f", mission="m", policy="p")
    assert_grid([r], ["f"], ["m"], ["p"])
    for bad in ([], [r, r], [dict(r, policy="unknown")]):
        with pytest.raises(ValueError):
            assert_grid(bad, ["f"], ["m"], ["p"])
