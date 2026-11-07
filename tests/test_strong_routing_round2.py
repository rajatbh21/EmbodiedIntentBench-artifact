"""Second round changes only declared calibration and fixed comparison set."""
from experiments.strong_routing_20261030 import round2, study


def test_same_complete_resource_grid_for_all_twelve_policies():
    cases={f'{layout}/sector_{sector}':dict(duration_s=42,motion_s=36)
           for layout in ('a','b','c') for sector in (3,7)}
    grid=round2.make_grid(cases)
    assert len(grid)==len({r['run_id'] for r in grid})==864
    assert grid==round2.make_grid(dict(reversed(list(cases.items()))))
    reference={(r['flight'],r['perception_budget_s'],r['link'],r['energy_condition'],r['battery'])
               for r in grid if r['policy']=='fixed_local_yolo'}
    for name in round2.POLICIES:
        assert {(r['flight'],r['perception_budget_s'],r['link'],r['energy_condition'],r['battery'])
                for r in grid if r['policy']==name}==reference
    assert set(study.POLICIES) < set(round2.POLICIES)


def test_previous_prior_and_current_prior_are_not_substituted(monkeypatch):
    seen=[]
    class Delegate:
        def select(self, **kwargs):
            return {'policy':'calibrated_evidence_rate','action_id':'unchanged'}
    def old(*args):
        seen.append(('old',args[-1]));return Delegate()
    def new(*args):
        seen.append(('new',args[-1]));return Delegate()
    monkeypatch.setattr(round2.policy,'make_policy',old)
    monkeypatch.setattr(round2.calibration_v2,'make_policy',new)
    previous={'version':'54frames'};current={'version':'162frames'}
    inputs=([],None,None,None,None)
    old_policy=round2.make_policy('previous_calibrated_evidence_rate',*inputs,
        learned_profile=dict(previous=previous,current=current))
    round2.make_policy('calibrated_quality',*inputs,learned_profile=dict(previous=previous,current=current))
    assert seen==[('old',previous),('new',current)]
    assert old_policy.select()==dict(policy='previous_calibrated_evidence_rate',action_id='unchanged')
