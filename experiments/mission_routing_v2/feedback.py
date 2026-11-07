"""Immutable prediction-only history delivered after inference completion."""

from dataclasses import dataclass, fields
from typing import Any

from experiments.lake_intent_contract import immutable
from experiments.lake_tracking_runtime import TrackingIntentRouter, TrackingRouterInput
from experiments.v4_benchmark.routers import RouterInput


@dataclass(frozen=True)
class FeedbackInput(TrackingRouterInput):
    evidence_history: Any = None


def snapshot(tracker, now):
    state = tracker.tracker
    if state.last_completion > now + 1e-8:
        raise ValueError("Future inference cannot enter policy history")
    return dict(
        available_at_s=now,
        last_completion_s=state.last_completion if state.last_completion >= 0 else None,
        tracks=[
            dict(
                track_id=tid,
                last_capture_s=t["last_s"],
                capture_times_s=[e["capture_s"] for e in t["evidence"]],
                reported=t["reported"],
            )
            for tid, t in sorted(state.tracks.items())
            if now - t["last_s"] <= state.config.max_gap_s
        ],
        emitted_prediction_reports=state.report_count,
        counts_are_ground_truth=False,
    )


class FeedbackAdapter(TrackingIntentRouter):
    def __init__(self, delegate, contract, tracking, tracker):
        super().__init__(delegate, contract, tracking)
        self.tracker = tracker

    def select(self, observation):
        self.calls += 1
        history = snapshot(self.tracker, observation.state.current_time_s)
        return self.delegate.select(
            FeedbackInput(
                **{field.name: getattr(observation, field.name) for field in fields(RouterInput)},
                intent_contract=self.public,
                public_contract_sha256=self.public_sha256,
                tracking=self.tracking,
                evidence_history=immutable(history),
            )
        )
