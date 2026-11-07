"""Urgency changes do not erase already completed inspection work."""
from copy import deepcopy
from experiments.dynamic_intent.protocol import SCHEDULES, current_update, deadline_profile
from experiments.wide_area_diagnosis.paid_clip import LocalInspectionIntent
from experiments.lake_intent_contract import canonical

WINDOWS = ((0, 36),)

class MissionIntent(LocalInspectionIntent):
    def __init__(self, case, battery):
        super().__init__(case)
        if case['motion_s'] != 36 or case['duration_s'] != 42:
            raise ValueError('Complete 36 s observation / 42 s recording required')
        d=self.document()
        d['schema_version']='cumulative_inspection_dynamic_urgency_v2'
        d['mission']['initial_battery_frac']=battery
        d['mission']['title']='Complete one inspection under changing urgency'
        d['mission']['user_request']='During the fixed pass, confirm and report at least two of the three people in the scheduled sector using two observations at least two seconds apart. Updated urgency applies to new perception requests. Retain valid earlier evidence and reports; a new status survey is not requested.'
        d['mission']['phases'][0].update(phase_id='inspection',deadline_s=44)
        d['reporting'].update(deadline_s=44,duplicate_scope='Entire mission')
        self._json=canonical(d)

    def at(self, update):
        d=self.public_document();d['current_intent']=deepcopy(update)
        return d
