import copy
import pytest
from experiments.strong_routing_20261030.endpoint_sensitivity import rescore


def outcome(**changes):
    checks=dict(execution=True, sector_evidence=False, focal_report_recall=True,
        world_false_report_limit=True, world_duplicate_report_limit=True,
        battery_constraint_success=True, deadline_success=True, privacy_constraint_success=True,
        memory_constraint_success=True, communication_constraint_success=True)
    checks.update(changes)
    return dict(checks=checks, local_inspection_success=all(checks.values()),
                phases=[dict(evidence_recall=1.,required_recall=2/3,false_components=5)])


def test_only_component_predicate_changes_without_mutating_original():
    source=outcome();before=copy.deepcopy(source)
    result=rescore(source)
    assert result['raw_component_cap_only_failure'] and result['report_centered_sensitivity_success']
    assert source==before and not source['local_inspection_success']


@pytest.mark.parametrize('flag', ['execution','focal_report_recall','world_false_report_limit',
    'world_duplicate_report_limit','battery_constraint_success','deadline_success',
    'privacy_constraint_success','memory_constraint_success','communication_constraint_success'])
def test_other_failure_is_never_erased(flag):
    assert not rescore(outcome(**{flag:False}))['report_centered_sensitivity_success']


def test_inadequate_repeated_evidence_still_fails():
    source=outcome();source['phases'][0]['evidence_recall']=1/3
    assert not rescore(source)['report_centered_sensitivity_success']
