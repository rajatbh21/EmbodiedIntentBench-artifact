"""Expose immutable prediction geometry separately from the unchanged evidence history."""

from dataclasses import dataclass, fields
from typing import Any

from experiments.lake_intent_contract import immutable
from experiments.lake_tracking_runtime import TrackingIntentRouter
from experiments.mission_frustum_routing.geometry import snapshot as frustum_snapshot
from experiments.mission_routing_v2.feedback import FeedbackInput, snapshot
from experiments.v4_benchmark.routers import RouterInput


@dataclass(frozen=True)
class FrustumInput(FeedbackInput):
    frustum_history: Any = None


def decision_trace(policy, frustum):
    return dict(policy.last_decision, frustum_feedback=frustum)


class FeedbackAdapter(TrackingIntentRouter):
    def __init__(self, delegate, contract, tracking, tracker, pitch):
        super().__init__(delegate, contract, tracking)
        self.tracker, self.pitch, self.frustum = tracker, pitch, None

    @property
    def last_decision(self):
        return decision_trace(self.delegate, self.frustum)

    def select(self, observation):
        self.calls += 1
        now = observation.state.current_time_s
        history = snapshot(self.tracker, now)
        self.frustum = frustum_snapshot(
            self.tracker,
            now,
            dict(position_m=observation.position_m, quaternion_xyzw=observation.quaternion_xyzw),
            observation.rgb.shape[:2],
            self.pitch,
        )
        return self.delegate.select(
            FrustumInput(
                **{field.name: getattr(observation, field.name) for field in fields(RouterInput)},
                intent_contract=self.public,
                public_contract_sha256=self.public_sha256,
                tracking=self.tracking,
                evidence_history=immutable(history),
                frustum_history=immutable(self.frustum),
            )
        )
