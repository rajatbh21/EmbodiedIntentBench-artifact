"""Policy-input leakage audit.

Responsibility: observe (never alter) every policy input and every stored
prediction access during a run, and record violations of the benchmark's
information boundary:

1. Inputs are exactly the declared public fields.
2. The intent document carries no evaluator target identity, no target
   position, and only the update in force at release time (no future updates).
3. The link observation is the snapshot at decision time (no future trace).
4. The RGB array is the current capture's frame (no future frames, no GT array).
5. Stored predictions are read only for the action the policy selected, only
   for the current capture, at most once, and never for an abstention.

Static check: the routing calibration's training cases are disjoint from the
six evaluation recordings.
"""

from __future__ import annotations

import numpy as np

EXPECTED_INPUTS = {
    "release_ns",
    "current_action_id",
    "battery_frac",
    "cumulative_communication_mb",
    "link",
    "payload_bytes",
    "response_bytes",
    "intent",
    "rgb",
}
FORBIDDEN_KEYS = {
    "evaluator_target_ids",
    "focal_target_positions_m",
    "object_ids",
    "targets",
    "ground_truth",
    "visible_target_ids",
    "matched_target_ids",
    "intent_schedule",
}


def _walk(value, path=()):
    if isinstance(value, dict):
        for k, v in value.items():
            yield (*path, k), k, v
            yield from _walk(v, (*path, k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield (*path, i), None, v  # list elements are checked as values too
            yield from _walk(v, (*path, i))


class LeakageObserver:
    def __init__(self, focal_ids, frames, schedule_name):
        from experiments.dynamic_intent.protocol import current_update

        self.focal_ids = set(focal_ids)
        self.frames = frames
        self.schedule_name = schedule_name
        self.current_update = current_update
        self.violations = []
        self.decisions = self.mask_reads = 0
        self._pending = None  # (index, rgb_key, action_id) after selection

    def _flag(self, kind, detail):
        self.violations.append(dict(kind=kind, detail=detail))

    def policy_input(self, *, index, decision_ns, rgb_key, update, inputs):
        self.decisions += 1
        if self._pending is not None and self._pending[2] is not None and not self._pending[3]:
            pass  # a selected request may legitimately be unadmitted (no read)
        if set(inputs) != EXPECTED_INPUTS:
            self._flag("unexpected_input_fields", sorted(set(inputs) ^ EXPECTED_INPUTS))
        intent = inputs["intent"]
        for path, key, value in _walk(intent):
            if key in FORBIDDEN_KEYS:
                self._flag("forbidden_intent_key", list(map(str, path)))
            if isinstance(value, str) and any(t in value for t in self.focal_ids):
                self._flag("target_identity_in_intent", list(map(str, path)))
        expected = self.current_update(self.schedule_name, inputs["release_ns"])
        if intent.get("current_intent") != expected:
            self._flag("intent_update_not_current", dict(index=index))
        if any(
            str(k).startswith("future") or k in ("updates", "schedule") for _, k, _ in _walk(intent)
        ):
            self._flag("future_update_in_intent", dict(index=index))
        link = inputs["link"]
        if type(link).__name__ != "LinkSnapshot" or link.timestamp_ns != decision_ns:
            self._flag("link_not_decision_time_snapshot", dict(index=index))
        rgb = inputs["rgb"]
        frame = self.frames[index]
        if not isinstance(rgb, np.ndarray) or rgb is not frame.rgb or rgb.ndim != 3:
            self._flag("rgb_not_current_capture", dict(index=index))
        for other_index, other in self.frames.items():
            if other_index != index and np.shares_memory(rgb, other.rgb):
                self._flag("rgb_shares_other_capture", dict(index=index, other=other_index))
        if np.shares_memory(rgb, frame.object_ids):
            self._flag("rgb_shares_ground_truth", dict(index=index))
        self._pending = [index, rgb_key, None, False]

    def selected(self, *, index, action_id, rgb_key):
        self._pending[2] = action_id

    def mask_access(self, action_id, key):
        self.mask_reads += 1
        index, rgb_key, selected, read = self._pending
        if selected is None:
            self._flag("prediction_read_after_abstention", dict(index=index, action=action_id))
        elif action_id != selected:
            self._flag(
                "unselected_prediction_read", dict(index=index, action=action_id, selected=selected)
            )
        if key != rgb_key:
            self._flag("prediction_for_other_capture", dict(index=index))
        if read:
            self._flag("repeated_prediction_read", dict(index=index))
        self._pending[3] = True

    def summary(self):
        return dict(
            decisions=self.decisions,
            prediction_reads=self.mask_reads,
            violations=self.violations,
            passed=not self.violations,
        )


def calibration_disjointness(plan, calibration):
    evaluation = set(plan["cases"])
    training = {c for cohort in calibration["training_cohorts"] for c in cohort["case_ids"]}
    families = lambda ids: {i.split("/")[0] for i in ids}  # noqa: E731
    return dict(
        evaluation_recordings=sorted(evaluation),
        training_cases=sorted(training),
        overlapping_cases=sorted(evaluation & training),
        shared_route_families=sorted(families(evaluation) & families(training)),
        passed=not (evaluation & training),
        note="Case IDs are disjoint; training and evaluation share the Lake map and route "
        "families.",
    )
