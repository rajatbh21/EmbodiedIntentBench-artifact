"""Version-two pooled priors keep the frozen selection rule and causal boundary."""
import copy
from dataclasses import replace

import pytest

from experiments.strong_routing_20261030 import calibration_v2 as pooled
from experiments.strong_routing_20261030 import policy as original
from tests.test_mission_task_deadline_policy import FAST, REMOTE, ingredients, inputs
from tests.test_strong_routing_policy_20261030 import calibration as prior_fixture


def fixture(args):
    doc=prior_fixture(args)
    doc.update(schema_version=pooled.SCHEMA,training_frame_count=162,training_case_count=9)
    for row in doc["models"].values():
        row["frames"]=162
        row["false_components_per_capture"]=(row["false_components"]+1)/163
    return doc


@pytest.mark.parametrize("name",pooled.POLICIES)
def test_pooled_policy_uses_identical_public_select_code_and_resource_guards(name):
    args,data=ingredients(),inputs()
    router=pooled.make_policy(name,args["actions"],args["costs"],args["profile"],object(),None,
                              learned_profile=fixture(args))
    assert router.select.__func__ is original.CalibratedPolicy.select
    assert router.select(**data)["action_id"]==REMOTE
    data["link"]=replace(data["link"],outage=True)
    assert router.select(**data)["action_id"]==FAST
    data["battery_frac"]=.21
    assert router.select(**data)["action_id"] is None


def test_original54_schema_is_not_misrepresented_as_pooled162():
    args=ingredients()
    with pytest.raises(ValueError,match="162"):
        pooled.verify_calibration(prior_fixture(args),verify_files=False)
    with pytest.raises(ValueError,match="oblique calibration"):
        original.verify_calibration(fixture(args),verify_files=False)


@pytest.mark.parametrize("defect",("frame_count","case_count","model_frames","formula","signature"))
def test_changed_pooled_denominators_formulas_or_identity_fail(defect):
    args=ingredients();doc=fixture(args)
    if defect=="frame_count":doc["training_frame_count"]=54
    elif defect=="case_count":doc["training_case_count"]=8
    elif defect=="model_frames":doc["models"][FAST]["frames"]=54
    elif defect=="formula":doc["models"][FAST]["false_components_per_capture"]=0
    else:doc["models"][FAST]["signature"]="other"
    with pytest.raises(ValueError):
        pooled.PoolCalibratedPolicy("calibrated_quality",args["actions"],args["costs"],args["profile"],doc)


def test_runtime_cannot_use_training_site_labels_future_state_or_evaluation_feedback():
    args,data=ingredients(),inputs();doc=fixture(args)
    doc["training_frames"]=[dict(case_id="private_training_site",target_id="training_only")]
    before=copy.deepcopy(doc)
    router=pooled.PoolCalibratedPolicy("calibrated_quality",args["actions"],args["costs"],args["profile"],doc)
    assert router.select(**data)==router.select(**data)
    assert doc==before
    assert not hasattr(router,"update") and not hasattr(router,"calibration")
    assert all(set(row)=={"signature","recall","false_components_per_capture"} for row in router.models.values())
    data["intent"]["evaluator_target_ids"]=["hidden"]
    with pytest.raises(ValueError,match="redacted"):
        router.select(**data)
