"""Public constraint interventions and explicit infeasibility for reference routing."""

import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from experiments.lake_causal_router import MissionCosts
from experiments.mission_network_emulation import LinkSnapshot
from experiments.mission_network_policies import ActionDescriptor, PolicyCosts
from experiments.mission_routing_v2.features import NAMES
from experiments.mission_routing_v5.quality import NAMES as QUALITY_NAMES
from experiments.mission_task_deadlines.contract import DeadlineProfile
from experiments.mission_task_deadlines.policy import DeadlineAwarePolicy

FAST = "CFG_YOLO11S_960_ROT0"
HEAVY = "CFG_MASKRCNN_1280_ROT0_90"
REMOTE = "REMOTE__" + HEAVY


def ingredients():
    actions = (
        ActionDescriptor(FAST, FAST, "local", "fast", 0.2, 1, 0.05, 0.1, 50),
        ActionDescriptor(HEAVY, HEAVY, "local", "heavy", 2, 10, 0.05, 0.1, 100),
        ActionDescriptor(REMOTE, HEAVY, "remote", "heavy", 0.1, 0, 0, 0, 10),
    )
    recall, risk = {}, {}
    for source, signature, q in ((FAST, "fast", 0.4), (HEAVY, "heavy", 0.9)):
        head = dict(
            mean=[0.0] * len(NAMES), scale=[1.0] * len(NAMES), coefficient=[0.0] * len(NAMES)
        )
        quality = copy.deepcopy(head)
        quality["coefficient"][0] = math.log(q / (1 - q))
        quality.update(train_min=[-1.0] * len(NAMES), train_max=[1.0] * len(NAMES))
        recall[source] = dict(signature=signature, base=quality)
        risk[source] = dict(
            signature=signature, risk=dict(head=head, floor=0.1, pseudocount_per_capture=0.01)
        )
    return dict(
        actions=actions,
        costs=PolicyCosts(MissionCosts(20, 2, 1000, 1, 0.05, 0.1, 10_000_000)),
        profile=DeadlineProfile("test", 1_000_000_000, 150_000_000, 30_000_000),
        quality_profile=dict(
            schema_version="conditional_recall_v5", feature_names=list(QUALITY_NAMES), models=recall
        ),
        risk_profile=dict(
            schema_version="mission_false_component_risk_v1", feature_names=list(NAMES), models=risk
        ),
    )


def inputs():
    return dict(
        release_ns=0,
        current_action_id=None,
        battery_frac=0.8,
        cumulative_communication_mb=0.0,
        link=LinkSnapshot(20_000_000, 100_000_000, 100_000_000, 10_000_000, False),
        payload_bytes=1000,
        response_bytes=100,
        rgb=np.zeros((12, 12, 3), dtype=np.uint8),
        intent=dict(
            limits=dict(
                min_final_battery_frac=0.2,
                deadline_s=25,
                communication_budget_mb=100,
                privacy_level="remote_allowed",
            ),
            mission=dict(
                phases=[
                    dict(
                        phase_id="search",
                        start_s=0,
                        end_s=20,
                        deadline_s=20,
                        min_hits=2,
                        min_evidence_span_s=2,
                        max_false_components=2,
                    )
                ]
            ),
        ),
    )


def test_current_link_intervention_changes_legal_adaptive_choice():
    policy = DeadlineAwarePolicy("deadline_aware", **ingredients())
    data = inputs()
    assert policy.select(**data)["action_id"] == REMOTE
    data["link"] = replace(data["link"], outage=True)
    output = policy.select(**data)
    assert output["action_id"] == FAST
    assert not next(c for c in output["candidates"] if c["action_id"] == HEAVY)["feasible"]


def test_no_infeasible_fastest_fallback_when_task_budget_cannot_be_met():
    args, data = ingredients(), inputs()
    args["profile"] = replace(args["profile"], perception_ns=100_000_000)
    data["link"] = replace(data["link"], outage=True)
    out = DeadlineAwarePolicy("deadline_aware", **args).select(**data)
    assert out["action_id"] is None
    assert out["reason"] == "no_feasible_action"
    assert all(not c["task_deadline_feasible"] for c in out["candidates"])


def test_fixed_heavy_still_executes_an_infeasible_control():
    out = DeadlineAwarePolicy("fixed_local_heavy", **ingredients()).select(**inputs())
    assert out["action_id"] == HEAVY
    assert not next(c for c in out["candidates"] if c["action_id"] == HEAVY)["feasible"]


def test_battery_reserve_includes_future_flight_and_processing():
    data = inputs()
    data["battery_frac"] = 0.21
    out = DeadlineAwarePolicy("deadline_aware", **ingredients()).select(**data)
    assert out["action_id"] is None
    assert all(not c["battery_feasible"] for c in out["candidates"])
    assert min(c["reserve_required_j"] for c in out["candidates"]) > 20


@pytest.mark.parametrize("intervention", ["privacy", "communication", "memory"])
def test_independent_admission_constraints(intervention):
    args, data = ingredients(), inputs()
    if intervention == "privacy":
        data["intent"]["limits"]["privacy_level"] = "local_only"
    elif intervention == "communication":
        data["intent"]["limits"]["communication_budget_mb"] = 0
    else:
        args["memory_capacity_mb"] = 120  # Fast50+68 fits; heavy100+68 cannot.
        data["intent"]["limits"]["privacy_level"] = "local_only"
        args["profile"] = replace(args["profile"], perception_ns=3_000_000_000)
    assert DeadlineAwarePolicy("deadline_aware", **args).select(**data)["action_id"] == FAST


def test_intent_changes_score_and_is_recorded_without_changing_predictions():
    policy, data = DeadlineAwarePolicy("deadline_aware", **ingredients()), inputs()
    first = policy.select(**data)
    phase = data["intent"]["mission"]["phases"][0]
    phase.update(min_hits=3, min_evidence_span_s=4, max_false_components=0)
    strict = policy.select(**data)
    assert first["intent_requirements"] != strict["intent_requirements"]
    for a, b in zip(first["candidates"], strict["candidates"], strict=True):
        assert a["predicted_recall"] == b["predicted_recall"]
        assert a["score"] > b["score"]


def test_perception_budget_includes_preparation_routing_loading_and_rounding():
    args, data = ingredients(), inputs()
    args["profile"] = replace(args["profile"], perception_ns=320_000_000)
    data["link"] = replace(data["link"], outage=True)
    output = DeadlineAwarePolicy("deadline_aware", **args).select(**data)
    fast = next(c for c in output["candidates"] if c["action_id"] == FAST)
    assert fast["predicted_usable_finish_ns"] == 320_000_000
    assert fast["task_deadline_feasible"]
    args["profile"] = replace(args["profile"], perception_ns=319_999_999)
    assert DeadlineAwarePolicy("deadline_aware", **args).select(**data)["action_id"] is None


def test_whole_mission_deadline_does_not_substitute_for_phase_or_task():
    data = inputs()
    data["intent"]["limits"]["deadline_s"] = 1000
    data["intent"]["mission"]["phases"][0]["deadline_s"] = 0.1
    output = DeadlineAwarePolicy("deadline_aware", **ingredients()).select(**data)
    assert output["action_id"] is None
    assert all(not c["phase_deadline_feasible"] for c in output["candidates"])


def test_forbid_hidden_labels_and_future_link():
    policy, data = DeadlineAwarePolicy("deadline_aware", **ingredients()), inputs()
    data["intent"]["evaluator_target_ids"] = ["secret"]
    with pytest.raises(ValueError, match="redacted"):
        policy.select(**data)
    del data["intent"]["evaluator_target_ids"]
    data["link"] = replace(data["link"], timestamp_ns=30_000_000)
    with pytest.raises(ValueError, match="current"):
        policy.select(**data)


def test_profile_identity_cannot_be_changed():
    args = ingredients()
    args["quality_profile"]["models"][FAST]["signature"] = "different model"
    with pytest.raises(ValueError, match="identity"):
        DeadlineAwarePolicy("deadline_aware", **args)


def test_repeated_stateless_decisions_are_identical_and_public_data_unmodified():
    data, policy = inputs(), DeadlineAwarePolicy("deadline_aware", **ingredients())
    intent = copy.deepcopy(data["intent"])
    assert policy.select(**data) == policy.select(**data)
    assert data["intent"] == intent
