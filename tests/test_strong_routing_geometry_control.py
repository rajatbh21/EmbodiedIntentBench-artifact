from experiments.strong_routing_20261030.geometry_control import focal_semantics
from experiments.wide_area_diagnosis.paid_clip import LocalInspectionIntent


def test_nonfocal_correct_reports_do_not_satisfy_focal_recall():
    contract = LocalInspectionIntent(dict(case_id='case', focal_sector='sector',
        motion_s=36., duration_s=42., evaluator_target_ids=['a', 'b', 'c']))
    quality = dict(reports=[dict(target_id=target, completion_time_s=30.)
                           for target in ('a', 'elsewhere_1', 'elsewhere_2')],
                   false_reports=0, duplicate_correct_reports=0)
    result = focal_semantics(contract, [dict(success=True)], quality)
    assert not result['evidence_and_reporting_pass']
    assert result['focal_report_recall'] == 1 / 3
    assert result['correct_nonfocal_reports'] == 2
    quality['reports'].append(dict(target_id='b', completion_time_s=40.))
    assert focal_semantics(contract, [dict(success=True)], quality)['evidence_and_reporting_pass']


def test_late_correct_focal_report_is_not_timely_evidence():
    contract = LocalInspectionIntent(dict(case_id='case', focal_sector='sector',
        motion_s=36., duration_s=42., evaluator_target_ids=['a', 'b', 'c']))
    quality = dict(reports=[dict(target_id='a', completion_time_s=30.),
                           dict(target_id='b', completion_time_s=53.)],
                   false_reports=0, duplicate_correct_reports=0)
    result = focal_semantics(contract, [dict(success=True)], quality)
    assert result['timely_focal_reports'] == ['a']
    assert not result['evidence_and_reporting_pass']
