"""Conditional-recall statistics, causal boundary and paid-runner regression tests."""

import copy

import numpy as np
import pytest

from experiments.mission_routing_v2.features import encode as base_encode
from experiments.mission_routing_v5.policy import Policy
from experiments.mission_routing_v5.quality import BASE_NAMES, NAMES, encode, fit, predict
from tests import test_mission_routing_v2 as prior
from tests.test_mission_routing_v4 import fixture, profile_for


def recall_profile(old, variant):
    feature = encode(np.zeros((8, 8, 3), np.uint8))
    return dict(
        schema_version="conditional_recall_v5",
        feature_names=list(NAMES),
        routing_variant=variant,
        training_lineage=[{"secret": "excluded"}],
        models={
            c: dict(
                signature=m["signature"],
                spatial=fit([feature] * 4, [1] * 4, [2] * 4),
                base=fit([feature[: len(BASE_NAMES)]] * 4, [1] * 4, [2] * 4),
            )
            for c, m in old["models"].items()
        },
    )


def test_constant_head_matches_smoothed_empirical_recall():
    x = encode(np.zeros((8, 8, 3), np.uint8))
    head = fit([x] * 3, [0, 1, 3], [0, 2, 4])
    assert predict(head, x)[0] == pytest.approx(5 / 8)
    assert head["empty_frames_excluded"] == 1
    assert head["target_trials"] == 6


def test_empty_frames_do_not_change_normalization_or_fit():
    x, other = encode(np.zeros((8, 8, 3), np.uint8)), encode(np.full((8, 8, 3), 255, np.uint8))
    base = fit([x] * 3, [1, 1, 1], [2, 2, 2])
    augmented = fit([x] * 3 + [other] * 100, [1] * 3 + [0] * 100, [2] * 3 + [0] * 100)
    assert base["coefficient"] == augmented["coefficient"]
    assert base["mean"] == augmented["mean"]
    assert base["scale"] == augmented["scale"]


@pytest.mark.parametrize(
    "hits,trials",
    [([-1], [1]), ([2], [1]), ([0.5], [1]), ([0], [0.5]), ([0], [0]), ([float("nan")], [1])],
)
def test_bad_counts_rejected(hits, trials):
    with pytest.raises(ValueError):
        fit([encode(np.zeros((8, 8, 3), np.uint8))], hits, trials)


def test_contextual_head_learns_direction_without_labels_at_prediction():
    dark, light = encode(np.zeros((8, 8, 3), np.uint8)), encode(np.full((8, 8, 3), 255, np.uint8))
    head = fit([dark, light] * 20, [0, 2] * 20, [2, 2] * 20)
    assert predict(head, dark)[0] < 0.2 < 0.8 < predict(head, light)[0]
    assert len(dark) == len(NAMES) == 36
    assert np.array_equal(dark[: len(BASE_NAMES)], base_encode(np.zeros((8, 8, 3), np.uint8)))
    with pytest.raises(ValueError):
        predict(head, [float("nan")] * len(NAMES))


@pytest.mark.parametrize("variant", ["spatial_recall", "base_recall", "global_recall"])
def test_paid_runtime(tmp_path, monkeypatch, variant):
    def build(old, specs, costs, mode):
        risk = profile_for(old, base_encode(np.zeros((8, 8, 3), np.uint8)))
        return Policy(recall_profile(old, mode), specs, costs, risk)

    monkeypatch.setattr(prior, "Policy", build)
    prior.test_feedback_controller_runs_through_paid_runtime(tmp_path, variant)


def test_ledger_repeat_is_idempotent_and_private_training_ids_are_discarded():
    _, o, old, specs, costs, risk = fixture()
    policy = Policy(recall_profile(old, "spatial_recall"), specs, costs, risk)
    choice = policy.select(o)
    before = copy.deepcopy(policy.last_decision)
    assert policy.select(o) == choice
    assert policy.last_decision == before
    assert "secret" not in repr(vars(policy))
    o.evidence_history["tracks"] = [dict(capture_times_s=[99], reported=False)]
    with pytest.raises(ValueError, match="completed prediction history"):
        policy.select(o)


def test_wrong_pool_and_signature_rejected():
    _, _, old, specs, costs, risk = fixture()
    profile = recall_profile(old, "spatial_recall")
    profile["models"][specs[0].config_id]["signature"] = "wrong"
    with pytest.raises(ValueError, match="signature"):
        Policy(profile, specs, costs, risk)
