"""Matched intent-aware policies, with an optional expected-error budget."""
from experiments.dynamic_intent.policy import UpdatingPolicy, POLICIES as BASE
from experiments.dynamic_intent.protocol import deadline_profile
from experiments.strong_routing_20261030.mission_risk import MissionRiskBudgetPolicy

POLICIES=(*BASE,'risk_current','risk_initial')

class UpdatingRisk:
    def __init__(self,name,actions,costs,profile,calibration):
        self.name=name;self.initial=None
        self.delegate=MissionRiskBudgetPolicy(actions,costs,profile,calibration)
    def select(self,**inputs):
        current=dict(inputs['intent']['current_intent'])
        if self.initial is None:self.initial=current
        observed=self.initial if self.name=='risk_initial' else current
        self.delegate.delegate.forecaster.profile=deadline_profile(observed)
        routed=dict(inputs,intent=dict(inputs['intent'],current_intent=dict(observed)))
        out=self.delegate.select(**routed)
        out.update(policy=self.name,observed_intent=dict(observed))
        return out

def make_policy(name,actions,costs,profile,quality,risk,learned_profile=None):
    if name in BASE:return UpdatingPolicy(name,actions,costs,profile,learned_profile)
    if name not in ('risk_current','risk_initial'):raise ValueError('Undeclared policy')
    return UpdatingRisk(name,actions,costs,profile,learned_profile)
