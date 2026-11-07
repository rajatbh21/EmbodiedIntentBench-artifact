"""Phase-level expected-count gating is distinct from both quality and resources."""
from dataclasses import replace

import pytest

from experiments.strong_routing_20261030.mission_risk import NAME, make_policy
from tests.test_mission_task_deadline_policy import FAST, HEAVY, REMOTE, ingredients, inputs
from tests.test_strong_routing_calibration_v2 import fixture


def router(args=None, document=None):
    args=ingredients() if args is None else args
    return make_policy(NAME,args["actions"],args["costs"],args["profile"],None,object(),
                       learned_profile=document or fixture(args))


def test_high_recall_model_is_filtered_by_full_phase_expected_false_count():
    args,data=ingredients(),inputs();doc=fixture(args)
    # At 2-second cadence the 20-second public phase gives ten calls.
    doc["models"][HEAVY].update(false_components=48,false_components_per_capture=49/163)
    out=router(args,doc).select(**data)
    assert out["action_id"]==FAST
    remote=next(row for row in out["candidates"] if row["action_id"]==REMOTE)
    assert remote["feasible"] and not remote["mission_risk_feasible"]
    assert remote["full_phase_expected_calls"]==10
    assert remote["full_phase_expected_false_components"]==10*49/163
    assert out["mission_risk_guard"]["uses_actual_feedback"] is False
    assert out["mission_risk_guard"]["guarantees_false_component_cap"] is False


def test_phase_duration_not_current_time_changes_full_phase_risk_projection():
    args,data=ingredients(),inputs();doc=fixture(args)
    doc["models"][HEAVY].update(false_components=48,false_components_per_capture=49/163)
    policy=router(args,doc)
    first=policy.select(**data)
    data["release_ns"]=10_000_000_000
    data["link"]=replace(data["link"],timestamp_ns=10_020_000_000)
    later=policy.select(**data)
    a=next(row for row in first["candidates"] if row["action_id"]==REMOTE)
    b=next(row for row in later["candidates"] if row["action_id"]==REMOTE)
    assert a["full_phase_expected_false_components"]==b["full_phase_expected_false_components"]
    assert later["action_id"]==FAST


def test_resource_guards_still_precede_risk_and_cannot_admit_heavy_or_outage_action():
    args,data=ingredients(),inputs()
    data["link"]=replace(data["link"],outage=True)
    out=router(args).select(**data)
    assert out["action_id"]==FAST
    heavy=next(row for row in out["candidates"] if row["action_id"]==HEAVY)
    assert not heavy["task_deadline_feasible"] and heavy["full_phase_expected_calls"] is None
    data["battery_frac"]=.21
    out=router(args).select(**data)
    assert out["action_id"] is None and out["reason"]=="no_feasible_action"


def test_zero_intent_false_allowance_abstains_with_positive_smoothed_risks():
    args,data=ingredients(),inputs()
    data["intent"]["mission"]["phases"][0]["max_false_components"]=0
    out=router(args).select(**data)
    assert any(row["feasible"] for row in out["candidates"])
    assert out["action_id"] is None
    assert out["reason"]=="no_model_within_expected_phase_false_component_budget"


def test_slow_local_cadence_counts_skipped_capture_slots():
    args,data=ingredients(),inputs()
    args["profile"]=replace(args["profile"],perception_ns=8_000_000_000)
    data["link"]=replace(data["link"],outage=True)
    out=router(args).select(**data)
    heavy=next(row for row in out["candidates"] if row["action_id"]==HEAVY)
    assert heavy["forecast_capture_cadence_s"]==4
    assert heavy["full_phase_expected_calls"]==5


def test_private_labels_future_link_and_unrecognized_policy_are_rejected():
    args,data=ingredients(),inputs()
    data["intent"]["evaluator_target_ids"]=["secret"]
    with pytest.raises(ValueError,match="redacted"):router(args).select(**data)
    data=inputs();data["link"]=replace(data["link"],timestamp_ns=30_000_000)
    with pytest.raises(ValueError,match="current"):router(args).select(**data)
    with pytest.raises(ValueError,match="separately declared"):
        make_policy("unknown",args["actions"],args["costs"],args["profile"],None,None,fixture(args))
