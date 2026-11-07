"""Reuse the frozen paid execution loop with one continuous evidence window.

Isolated dependency binding changes the public contract and reporting lifetime.
It does not modify predictions, executor timing, load costs or battery accounting.
"""
from types import FunctionType
from experiments.dynamic_intent import runtime as previous
from experiments.dynamic_intent.scoring import dynamic_outcome
from experiments.strong_routing_20261030 import study as parent
from experiments.intent_mission_v2.protocol import MissionIntent, WINDOWS
from experiments.intent_mission_v2.policy import make_policy


def load_case(root,plan,item):
    state=parent.load_case(root,plan,item)
    state['contract']=MissionIntent(plan['cases'][item['flight']],item['battery'])
    return state

def outcome(contract,events,result):
    out=dynamic_outcome(contract,events,result)
    out.update(schema_version='cumulative_inspection_dynamic_urgency_outcome_v2',
        semantics='One full-pass inspection with persistent evidence, tracking and reports. Updated urgency changes only budgets of new perception requests. All request, quality, report and resource checks remain enforced. Not a periodic resurvey.')
    return out

def run_case(*args,**kwargs):
    from experiments.intent_mission_v2.study import source_hashes
    namespace=dict(previous.run_case.__globals__)
    namespace.update(WINDOWS=WINDOWS,load_case=load_case,make_policy=make_policy,
        dynamic_outcome=outcome,source_hashes=source_hashes)
    return FunctionType(previous.run_case.__code__,namespace,'run_case')(*args,**kwargs)
