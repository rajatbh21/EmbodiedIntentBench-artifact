"""Matched current-intent and initial-intent routers; no test-label inputs."""
from experiments.strong_routing_20261030.calibration_v2 import PoolCalibratedPolicy
from experiments.dynamic_intent.protocol import deadline_profile

FIXED = {'fixed_local_yolo':'CFG_YOLO11S_960_ROT0',
         'fixed_local_mask':'CFG_MASKRCNN_1280',
         'fixed_local_cabinet':'CFG_CABINET_UAVID_1280',
         'fixed_remote_mask':'REMOTE__CFG_MASKRCNN_1280'}
POLICIES = (*FIXED, 'fastest_current', 'quality_current', 'quality_initial')

class UpdatingPolicy:
    def __init__(self, name, actions, costs, profile, calibration):
        if name not in POLICIES: raise ValueError('Undeclared dynamic-intent policy')
        self.name=name
        self.delegate=PoolCalibratedPolicy('fastest_feasible' if name=='fastest_current' else 'calibrated_quality', actions, costs, profile, calibration)
        self.initial=None

    def select(self, **inputs):
        update=dict(inputs['intent']['current_intent'])
        if self.initial is None:self.initial=update
        observed=self.initial if self.name=='quality_initial' else update
        self.delegate.forecaster.profile=deadline_profile(observed)
        # A blind policy receives the original update, even inside its delegate input.
        routed=dict(inputs,intent=dict(inputs['intent'],current_intent=dict(observed)))
        out=self.delegate.select(**routed)
        if self.name in FIXED:
            candidate=next(c for c in out['candidates'] if c['action_id']==FIXED[self.name])
            out['action_id']=candidate['action_id'] if candidate['legal'] else None
            out['reason']='Fixed action; resource forecast does not veto legal execution'
        out.update(policy=self.name,observed_intent=dict(observed))
        return out

def make_policy(name,actions,costs,profile,quality,risk,learned_profile=None):
    return UpdatingPolicy(name,actions,costs,profile,learned_profile)
