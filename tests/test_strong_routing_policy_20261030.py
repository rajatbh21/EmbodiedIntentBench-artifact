"""Calibrated routing retains public boundaries and paid resource semantics."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from experiments.strong_routing_20261030.policy import (
    CalibratedPolicy, POLICIES, REGIME, SCHEMA, aggregate_counts, make_policy,
    verify_calibration,
)
from tests.test_mission_task_deadline_policy import FAST, HEAVY, REMOTE, ingredients, inputs


def calibration(args, fast_hits=39, heavy_hits=89):
    hits = {FAST: fast_hits, HEAVY: heavy_hits}
    models = {}
    for action in args["actions"]:
        name = action.source_config_id
        models[name] = dict(signature=action.source_signature, frames=54,
            visible_trials=98, hits=hits[name], false_components=0,
            recall=(hits[name]+1)/100, false_components_per_capture=1/55)
    return dict(schema_version=SCHEMA, sensor_regime=dict(REGIME),
                training_frame_count=54, models=models)


def policy(name, args=None, document=None):
    args = ingredients() if args is None else args
    return make_policy(name, args["actions"], args["costs"], args["profile"],
        args["quality_profile"], args["risk_profile"],
        learned_profile=document or calibration(args))


@pytest.mark.parametrize("name", POLICIES)
def test_public_outage_deadline_and_battery_are_hard_constraints(name):
    args, data = ingredients(), inputs()
    router = policy(name, args)
    assert router.select(**data)["action_id"] == REMOTE
    data["link"] = replace(data["link"], outage=True)
    out = router.select(**data)
    assert out["action_id"] == FAST
    assert not next(c for c in out["candidates"] if c["action_id"] == HEAVY)["task_deadline_feasible"]
    data["battery_frac"] = .21
    out = router.select(**data)
    assert out["action_id"] is None
    assert out["reason"] == "no_feasible_action"
    assert all(not c["battery_feasible"] for c in out["candidates"])


@pytest.mark.parametrize("name", POLICIES)
def test_no_ground_truth_future_link_or_site_feature_is_accepted(name):
    router, data = policy(name), inputs()
    data["intent"]["evaluator_target_ids"] = ["secret"]
    with pytest.raises(ValueError, match="redacted"):
        router.select(**data)
    del data["intent"]["evaluator_target_ids"]
    data["link"] = replace(data["link"], timestamp_ns=30_000_000)
    with pytest.raises(ValueError, match="current"):
        router.select(**data)
    data = inputs()
    with pytest.raises(TypeError):
        router.select(**data, site_id="unseen_site")


@pytest.mark.parametrize("name", POLICIES)
def test_stale_quality_risk_profiles_and_runtime_rgb_cannot_change_constant_priors(name):
    args, data = ingredients(), inputs()
    first = policy(name, args).select(**data)
    args["quality_profile"], args["risk_profile"] = object(), None
    data["rgb"] = np.full_like(data["rgb"], 255)
    second = policy(name, args).select(**data)
    assert first == second
    assert not second["stale_quality_risk_heads_used"]


def test_evidence_rate_distinguishes_slow_quality_from_repeated_observation_rate():
    args, data = ingredients(), inputs()
    args["profile"] = replace(args["profile"], perception_ns=8_000_000_000)
    data["link"] = replace(data["link"], outage=True)
    doc = calibration(args, fast_hits=79, heavy_hits=89)
    assert policy("calibrated_quality", args, doc).select(**data)["action_id"] == HEAVY
    evidence = policy("calibrated_evidence_rate", args, doc).select(**data)
    assert evidence["action_id"] == FAST
    rows = {row["action_id"]: row for row in evidence["candidates"]}
    assert rows[HEAVY]["evidence_window_s"] > rows[FAST]["evidence_window_s"]


def test_exact_ties_are_deterministic_under_action_order_reversal():
    args, data = ingredients(), inputs()
    first = args["actions"][0]
    args["actions"] = (replace(first, action_id="z_same_cost"), replace(first, action_id="a_same_cost"))
    for name in POLICIES:
        assert policy(name, args).select(**data)["action_id"] == "a_same_cost"
        args["actions"] = tuple(reversed(args["actions"]))
        assert policy(name, args).select(**data)["action_id"] == "a_same_cost"


def test_runtime_retains_no_training_lineage_and_performs_no_updates():
    args, data = ingredients(), inputs()
    doc = calibration(args)
    doc["training_frames"] = [{"case_id": "private_training_site", "target_id": "training_only"}]
    before = copy.deepcopy(doc)
    router = policy("calibrated_evidence_rate", args, doc)
    first = router.select(**data)
    assert first == router.select(**data)
    assert first["evaluation_learning_updates"] == 0
    assert doc == before
    assert not hasattr(router, "update")
    assert not hasattr(router, "calibration")
    assert all(set(row) == {"signature", "recall", "false_components_per_capture"}
               for row in router.models.values())


@pytest.mark.parametrize("defect", ("signature", "recall", "risk", "frames", "regime"))
def test_changed_training_identity_or_formula_is_rejected(defect):
    args = ingredients()
    doc = calibration(args)
    if defect == "signature":
        doc["models"][FAST]["signature"] = "other_weights"
    elif defect == "recall":
        doc["models"][FAST]["recall"] = .99
    elif defect == "risk":
        doc["models"][FAST]["false_components_per_capture"] = 0
    elif defect == "frames":
        doc["models"][FAST]["frames"] = 53
    else:
        doc["sensor_regime"]["camera_pitch_rad"] = 1.57
    with pytest.raises(ValueError):
        policy("calibrated_quality", args, doc)


def test_count_training_keeps_empty_frames_and_cannot_hide_model_failures():
    def frame(index, visible, matched, false):
        return dict(capture_id=index, scores={
            "perfect_gt": dict(visible_target_ids=visible),
            "model": dict(visible_target_ids=visible, matched_target_ids=matched,
                          false_positive_components=false)})
    rows = [frame(0, ["a", "b"], ["a"], 0), frame(1, [], [], 2), frame(2, ["c"], [], 0)]
    model = aggregate_counts([("site", rows)], {"model": "signature"})["model"]
    assert model["frames"] == 3
    assert model["recall"] == 2/5
    assert model["false_components_per_capture"] == 3/4
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_counts([("site", rows), ("site", rows)], {"model": "signature"})
    rows[0]["scores"]["model"]["visible_target_ids"] = ["a"]
    with pytest.raises(ValueError, match="Consistent"):
        aggregate_counts([("site", rows)], {"model": "signature"})
