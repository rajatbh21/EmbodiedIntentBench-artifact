"""Causal urgency updates and a common three-window reporting contract."""
from copy import deepcopy
import json
from experiments.lake_intent_contract import canonical
from experiments.wide_area_diagnosis.paid_clip import LocalInspectionIntent
from experiments.mission_task_deadlines.contract import DeadlineProfile

NS = 10**9
SCHEDULES = {
    'no_change': ((0, 8),),
    'urgency_increase': ((0, 8), (12, 2)),
    'urgency_recovery': ((0, 8), (12, 2), (24, 8)),
}
WINDOWS = ((0, 12), (12, 24), (24, 36))


def current_update(name, release_ns):
    if name not in SCHEDULES or type(release_ns) is not int or release_ns < 0:
        raise ValueError('Declared schedule and nonnegative integer clock required')
    eligible = [(i, t, budget) for i, (t, budget) in enumerate(SCHEDULES[name]) if t*NS <= release_ns]
    i, t, budget = eligible[-1]
    return dict(version=i, effective_from_ns=t*NS, perception_budget_ns=budget*NS)


def deadline_profile(update):
    return DeadlineProfile('intent_version_'+str(update['version']), update['perception_budget_ns'], 150_000_000, 30_000_000)


class WindowIntent(LocalInspectionIntent):
    def __init__(self, case, battery):
        super().__init__(case)
        if case['motion_s'] != 36 or case['duration_s'] != 42:
            raise ValueError('This pilot requires complete 36 s motion / 42 s recordings')
        d=self.document();phase=d['mission']['phases'][0]
        d['schema_version']='dynamic_intent_periodic_inspection_v1'
        d['mission']['initial_battery_frac']=battery
        d['mission']['title']='Periodic status reports during a fixed inspection flight'
        d['mission']['user_request']='For each of three 12-second observation windows, confirm and report at least two of the three people in the scheduled sector using fresh repeated evidence. Perception request urgency may be updated during flight.'
        # Delivery allowance is identical across all schedules and independent of urgency.
        d['mission']['phases']=[dict(deepcopy(phase),phase_id=f'window_{i}',start_s=a,end_s=b,
            deadline_s=b+8,instruction='Report current focal-sector status using observations from this window only.')
            for i,(a,b) in enumerate(WINDOWS)]
        d['reporting']['duplicate_scope']='Within each reporting window; renewed reports across windows are requested.'
        self._json=canonical(d)

    def at(self, update):
        d=self.public_document();d['current_intent']=deepcopy(update)
        # No schedule name, future change times/budgets or evaluator identities.
        return d
