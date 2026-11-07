"""Instrumented copy of the frozen 648-cell execution loop.

Responsibility: execute one declared grid cell exactly as
``experiments/dynamic_intent/runtime.py::run_case`` does when bound by
``tools/replay.py``, with five explicit, opt-in hooks used by the validation
experiments:

* ``protocol``: ``"causal"`` (frozen semantics; blocking computation makes
  intervening captures unavailable) or ``"no_skip"`` (ablation; see below).
* ``latency_multiplier``: scales every action's configured inference/service
  latency, and local compute energy under the declared 15 W compute model.
  Stored predictions, load costs, network transfer and downstream costs stay fixed.
* ``contract``: public contract object supplied to the policy and pipeline in
  place of the frozen ``MissionIntent`` (intent-variation reruns).
* ``policy``: a policy object in place of the declared factory (hindsight
  witness schedules).
* ``observer``: receives every policy input and every prediction-cache access
  (leakage audit). It can only observe; it cannot alter execution.

With all hooks at their defaults the loop must reproduce every scientific row
field of ``data/replay/expected_rows.json``; ``study.py parity`` checks all 648.
Nothing is written to disk here; callers decide what to persist.

``no_skip`` is an intentionally optimistic, physically unrealizable ablation:
every eligible capture is processed regardless of overlapping computation.
Perception latency still sets each request's completion time, deadline
admission and energy, and requests are still issued in capture order with
sequential model residency. The serial downstream tracker receives completed
predictions in capture order as a FIFO; its per-stage deadline is measured from
dequeue, so queueing behind a slower earlier request is not penalised; the
tracker's completion-order check is relaxed in the same FIFO sense (capture
order still enforced). Any downstream execution failure aborts a no-skip run.
Concurrent memory is not charged (per-call accounting is kept).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

PROTOCOLS = ("causal", "no_skip")
LOCAL_COMPUTE_POWER_W = 15.0


# --------------------------------------------------------------------------
# Pure-function memoisation (performance only; parity-checked)
# --------------------------------------------------------------------------
_LABEL_CACHE: dict = {}


def _memoized_label_components(mask):
    from missionbench.v2.executors import label_components as original

    key = (mask.shape, mask.dtype.str, hashlib.blake2b(mask.tobytes(), digest_size=20).digest())
    hit = _LABEL_CACHE.get(key)
    if hit is None:
        labels, count = original(mask)
        labels.setflags(write=False)  # any in-place caller mutation would raise
        hit = _LABEL_CACHE[key] = (labels, count)
    return hit


def install_memoization():
    """Replace the module-level references used on the hot path."""
    import experiments.v4_benchmark.reporting as reporting
    import missionbench.v2.evaluation as evaluation

    evaluation.label_components = _memoized_label_components
    reporting.label_components = _memoized_label_components


# --------------------------------------------------------------------------
# Workspace: one recording's verified inputs, loaded once per process
# --------------------------------------------------------------------------
class Workspace:
    """Read-only access to a prepared replay root (``tools/replay.py`` output)."""

    def __init__(self, root):
        from experiments.lake_tracking_runtime import load_public_terrain
        from experiments.mission_runtime_profile import read
        from tools import replay

        self.replay = replay
        self.root = Path(root).resolve()
        # A prepared replay root has plan.json; the committed bundle carries the same grid
        # in study.json.
        plan_path = self.root / "plan.json"
        self.plan = read(plan_path if plan_path.exists() else self.root / "study.json")
        self.bindings = {
            b["case_id"]: b for b in read(self.root / "native_bindings.json")["bindings"]
        }
        self.terrain, _ = load_public_terrain(replay.checked_path(self.root, self.plan["assets"]))
        self._banks = {}

    def recording(self, flight):
        if flight not in self._banks:
            from experiments.mission_runtime_profile import read
            from experiments.wide_area_diagnosis.paid_clip import load_bank

            binding = dict(self.bindings[flight])
            binding["recording_path"] = str(
                self.replay.checked_path(self.root, binding["recording_path"])
            )
            manifest = read(self.root / "predictions" / flight / "manifest.json")
            self._banks = {flight: (load_bank(binding), manifest)}  # keep one recording resident
        return self._banks[flight]

    def load_case(self, item):
        return self.replay.load_case(self.root, self.plan, item)


def scale_actions(actions, multiplier):
    if multiplier == 1:
        return tuple(actions)
    if not (isinstance(multiplier, (int, float)) and math.isfinite(multiplier) and multiplier > 0):
        raise ValueError("Latency multiplier must be finite and positive")
    out = []
    for a in actions:
        energy = a.energy_j * multiplier if a.placement == "local" else a.energy_j
        if a.placement == "local" and abs(a.energy_j - LOCAL_COMPUTE_POWER_W * a.latency_s) > 1e-6:
            raise ValueError("Local energy no longer follows the declared compute-power model")
        out.append(replace(a, latency_s=a.latency_s * multiplier, energy_j=energy))
    return tuple(out)


class _ObservedMasks:
    def __init__(self, inner, action_id, observer):
        self.inner, self.action_id, self.observer = inner, action_id, observer

    def __contains__(self, key):
        return key in self.inner

    def __getitem__(self, key):
        self.observer.mask_access(self.action_id, key)
        return self.inner[key]


class _QueuedPipeline:
    """FIFO admission for the no-skip ablation; everything else is the frozen pipeline."""

    @staticmethod
    def make(**kwargs):
        from experiments.mission_task_deadlines.pipeline import DeadlinePipeline

        class QueuedDeadlinePipeline(DeadlinePipeline):
            def _validate(self, selection, prediction, context):
                saved = self._last_busy_end_ns
                self._last_busy_end_ns = -1  # captures may arrive while downstream is busy
                try:
                    return super()._validate(selection, prediction, context)
                finally:
                    self._last_busy_end_ns = saved

        return QueuedDeadlinePipeline(**kwargs)


class _NoSkipTrackerOrder:
    """Scoped FIFO relaxation of the tracker's serial-order check (no_skip runs only).

    ``StaticReportTracker.update`` rejects a capture that precedes the previous
    update's completion. Under the no-skip ablation such captures are queued
    rather than unavailable, so the check on completion order is relaxed while
    capture order and every other tracker rule stay enforced.
    """

    def __enter__(self):
        from experiments.lake_tracking import StaticReportTracker

        self.cls, self.original = StaticReportTracker, StaticReportTracker.update
        original = self.original

        def update(tracker, measurements, capture_s, completion_s):
            tracker.last_completion = min(tracker.last_completion, capture_s)
            return original(tracker, measurements, capture_s, completion_s)

        StaticReportTracker.update = update
        return self

    def __exit__(self, *exc):
        self.cls.update = self.original
        return False


def run_case(ws, item, *, protocol="causal", **kwargs):
    """Execute one cell in memory. Returns dict(row, events, result, outcome, pipeline_failures)."""
    if protocol == "no_skip":
        with _NoSkipTrackerOrder():
            return _run_case(ws, item, protocol=protocol, **kwargs)
    return _run_case(ws, item, protocol=protocol, **kwargs)


def _run_case(
    ws,
    item,
    *,
    protocol="causal",
    latency_multiplier=1.0,
    contract=None,
    policy=None,
    outcome_fn=None,
    observer=None,
    policy_label=None,
):
    from experiments.dynamic_intent.protocol import current_update, deadline_profile
    from experiments.dynamic_intent.scoring import combined_reports
    from experiments.intent_mission_v2.policy import make_policy
    from experiments.intent_mission_v2.protocol import WINDOWS
    from experiments.intent_mission_v2.runtime import outcome as frozen_outcome
    from experiments.lake_hindsight_witness import VerifiedMasks
    from experiments.lake_tracking_evaluation import RunnerReportEvaluator
    from experiments.mission_pipeline_routing.runtime import CompletedPrediction, PublicPose
    from experiments.mission_pipeline_routing.study import public_context
    from experiments.mission_task_deadlines.contract import DeadlineProfile
    from experiments.mission_task_deadlines.perception import execute_perception
    from experiments.mission_task_deadlines.pipeline import DeadlinePipeline
    from experiments.mission_task_deadlines.study import DOWNSTREAM, require
    from experiments.v4_benchmark.runner import evaluation_frame
    from experiments.v4_gazebo_pilot.timing import seconds_to_ns
    from experiments.wide_area_routing.runner import (
        advance_opportunities,
        latency_summary,
        validate_bank,
    )
    from missionbench.v2.evaluation import MissionEvaluator

    if protocol not in PROTOCOLS:
        raise ValueError("Undeclared execution protocol")
    plan, root = ws.plan, ws.root
    require(item in plan["grid"], "Run is outside the declared study grid")
    bank, manifest = ws.recording(item["flight"])
    case = ws.load_case(item)
    scenario = case["scenario"]
    contract = case["contract"] if contract is None else contract
    actions = scale_actions(case["actions"], latency_multiplier)
    costs, trace = case["costs"], case["trace"]
    profiles, learned_profile = case["profiles"], case["learned_profile"]
    outcome_fn = frozen_outcome if outcome_fn is None else outcome_fn
    require(
        manifest["recording_identity"] == scenario.recording_identity, "Wrong prediction flight"
    )
    require(
        manifest["recording_sha256"] == bank["provenance"]["recording_sha256"],
        "Predictions and RGB bank bind different recording bytes",
    )
    predictions_dir = root / "predictions" / item["flight"]
    indices = validate_bank(scenario, bank)
    descriptors = {a.action_id: a for a in actions}
    for action in actions:
        model = manifest["models"][action.source_config_id]
        require(model["signature"] == action.source_signature, "Prediction model identity changed")
        require(
            all(bank["payloads"][i].rgb_sha256 in model["predictions"] for i in indices),
            "Incomplete prediction bank for declared opportunities",
        )
    budget = 8
    profile = DeadlineProfile(
        f"perception_{budget:g}s", seconds_to_ns(budget), 150_000_000, 30_000_000
    )
    if policy is None:
        policy = make_policy(
            item["policy"],
            actions,
            costs,
            profile,
            profiles["recall"],
            profiles["risk"],
            learned_profile=learned_profile,
        )
    pipeline_kwargs = dict(
        terrain=ws.terrain,
        step_ns=scenario.schedule.step_ns,
        battery_capacity_j=scenario.battery_capacity_wh * 3600,
        flight_power_w=scenario.flight_power_w,
    )

    def new_pipeline(p):
        if protocol == "no_skip":
            return _QueuedPipeline.make(profile=p, **pipeline_kwargs)
        return DeadlinePipeline(profile=p, **pipeline_kwargs)

    pipeline = new_pipeline(profile)
    scorer = MissionEvaluator(
        scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
    )
    reporters = [
        RunnerReportEvaluator(scenario.matching_iou_threshold, scenario.targets.values(), 2)
        for _ in WINDOWS
    ]
    active_window = None
    masks = {
        a.action_id: VerifiedMasks(predictions_dir, manifest["models"][a.source_config_id])
        for a in actions
    }
    if observer is not None:
        masks = {k: _ObservedMasks(v, k, observer) for k, v in masks.items()}
    frames, payloads, provenance = bank["frames"], bank["payloads"], bank["provenance"]
    require(
        provenance["recording_identity"] == scenario.recording_identity, "Wrong shared RGB bank"
    )
    schedule = scenario.schedule
    capacity = scenario.battery_capacity_wh * 3600
    nonflight = communication = peak_memory = 0.0
    current_action = None
    now_ns = position = 0
    work_end_ns = 0  # no_skip only: latest completion of any issued work
    events, skipped_ids = [], []
    pipeline_failures = 0
    termination = "mission_window_complete"

    def battery(at_ns, extra_j=0.0):
        energy = at_ns / 1e9 * scenario.flight_power_w + nonflight + extra_j
        return max(0.0, scenario.initial_battery_frac - energy / capacity)

    while position < len(indices):
        index = indices[position]
        capture_ns = schedule.at(index) - schedule.epoch_ns
        if protocol == "no_skip":
            now_ns = capture_ns  # the sensing timeline is never blocked
        if battery(now_ns) <= 0:
            termination = "battery_depleted"
            break
        require(capture_ns >= now_ns, "Capture occurred while preceding work was busy")
        if battery(capture_ns) <= 0:
            now_ns, termination = capture_ns, "battery_depleted"
            break
        if capture_ns > seconds_to_ns(scenario.contract.deadline_s):
            now_ns, termination = capture_ns, "deadline_exceeded"
            break
        frame, prepared = frames[index], payloads[index]
        window = next(i for i, (a, b) in enumerate(WINDOWS) if a <= capture_ns / 1e9 < b)
        if active_window != window:
            pipeline = new_pipeline(profile)
            active_window = window
        reporter = reporters[window]
        update = current_update(item["intent_schedule"], capture_ns)
        profile = deadline_profile(update)
        pipeline.profile = profile
        public_intent = contract.at(update)
        decision_ns = capture_ns + seconds_to_ns(costs.encode_s, allow_zero=True)
        observed_battery = battery(decision_ns, costs.encode_s * costs.codec_power_w)
        inputs = dict(
            release_ns=capture_ns,
            current_action_id=current_action,
            battery_frac=observed_battery,
            cumulative_communication_mb=communication,
            link=trace.snapshot(decision_ns),
            payload_bytes=len(prepared.payload),
            response_bytes=prepared.response_bytes,
            intent=public_intent,
            rgb=frame.rgb,
        )
        if observer is not None:
            observer.policy_input(
                index=index,
                decision_ns=decision_ns,
                rgb_key=prepared.rgb_sha256,
                update=update,
                inputs=inputs,
            )
        selection = policy.select(**inputs)
        action_id = selection["action_id"]
        action = descriptors[action_id] if action_id is not None else None
        if observer is not None:
            observer.selected(index=index, action_id=action_id, rgb_key=prepared.rgb_sha256)
        perception = execute_perception(
            action=action,
            rgb=frame.rgb,
            prepared=prepared,
            predictions=masks[action_id] if action_id is not None else None,
            release_ns=capture_ns,
            profile=profile,
            mission_deadline_ns=seconds_to_ns(scenario.contract.deadline_s),
            current_action_id=current_action,
            trace=trace,
            costs=costs,
            router_latency_ns=seconds_to_ns(scenario.router_latency_s, allow_zero=True),
            router_energy_j=scenario.router_energy_j,
            step_ns=schedule.step_ns,
        )
        nonflight += perception.nonflight_energy_j
        communication += perception.communication_mb
        current_action = perception.resident_action_id
        if protocol == "no_skip":
            done_ns = max(perception.completed_ns, pipeline.final_time_ns)  # FIFO dequeue
        else:
            done_ns = now_ns = perception.completed_ns
        output = None
        reports = []
        downstream_skip_reason = None
        if perception.mask is not None and battery(done_ns) > 0:
            resource = scenario.resources.for_config(action_id)
            context = public_context(
                contract, capture_ns, done_ns, battery(done_ns), resource.memory_mb
            )
            output = pipeline.execute(
                DOWNSTREAM,
                CompletedPrediction(
                    action.source_config_id,
                    capture_ns,
                    done_ns,
                    perception.mask,
                    PublicPose(
                        tuple(frame.body_pose["position_m"]),
                        tuple(frame.body_pose["quaternion_xyzw"]),
                        scenario.world.get("camera_pitch_rad", math.pi / 2),
                    ),
                ),
                context,
            )
            failed_stages = sum(s.get("status") == "execution_failed" for s in output["stages"])
            pipeline_failures += failed_stages
            if failed_stages and protocol == "no_skip":
                raise RuntimeError(
                    "Downstream stage failed under the no-skip ablation; the ablation must "
                    "not silently discard queued evidence"
                )
            nonflight += output["charged_energy_j"]
            done_ns = output["completed_ns"]
            reports = output["emitted_reports"]
            peak_memory = max(
                peak_memory,
                action.memory_mb,
                *(s["costs"]["resident_memory_mb"] for s in output["stages"] if s["dispatched"]),
            )
        elif perception.mask is not None:
            downstream_skip_reason = "battery_depleted_before_association_release"
            peak_memory = max(peak_memory, action.memory_mb + 68)
        elif action is not None:
            downstream_skip_reason = "perception_output_not_admitted"
            peak_memory = max(peak_memory, action.memory_mb + 68)
        if protocol == "no_skip":
            work_end_ns = max(work_end_ns, done_ns, perception.completed_ns)
            next_position, skipped = position + 1, ()
        else:
            now_ns = done_ns
            completion, next_position, skipped = advance_opportunities(
                schedule, indices, position, now_ns
            )
            require(completion == now_ns, "Inconsistent usable completion ticks")
        truth = evaluation_frame(frame, index, capture_ns / 1e9, scenario)
        prediction = (
            perception.mask
            if perception.mask is not None
            else np.zeros(frame.rgb.shape[:2], dtype=np.bool_)
        )
        score = scorer.update(truth, prediction)
        reporter.update(truth, prediction, reports)
        for missed in skipped:
            missed_truth = evaluation_frame(
                frames[missed], missed, (schedule.at(missed) - schedule.epoch_ns) / 1e9, scenario
            )
            scorer.note_skipped_visibility(missed_truth.visible_target_ids)
            reporter.note_visibility(missed_truth)
        events.append(
            dict(
                window_id=window,
                applied_intent=update,
                observation_id=index,
                capture_time_s=capture_ns / 1e9,
                capture_timestamp_ns=frame.timestamp_ns,
                completion_time_s=done_ns / 1e9,
                perception_completed_ns=perception.completed_ns,
                selected_action_id=action_id,
                source_config_id=action.source_config_id if action else None,
                source_rgb_key=prepared.rgb_sha256,
                selected_mask_sha256=hashlib.sha256(perception.mask.tobytes()).hexdigest()
                if perception.mask is not None
                else None,
                capture_body_pose=frame.body_pose,
                public_link=trace.snapshot(decision_ns).to_dict(),
                battery_at_decision=observed_battery,
                router_decision=selection,
                perception=perception.event,
                pipeline=output,
                downstream_skip_reason=downstream_skip_reason,
                execution=dict(
                    success=perception.mask is not None,
                    config_id=action_id,
                    predicted_pixels=int(prediction.sum()),
                ),
                score=score.to_dict(),
                emitted_reports=reports,
                cumulative_nonflight_energy_j=nonflight,
                cumulative_energy_j=(done_ns if protocol == "no_skip" else now_ns)
                / 1e9
                * scenario.flight_power_w
                + nonflight,
                cumulative_communication_mb=communication,
                battery_after=battery(done_ns if protocol == "no_skip" else now_ns),
                skipped_observation_ids=list(skipped),
            )
        )
        skipped_ids.extend(skipped)
        position = next_position
    if protocol == "no_skip":
        now_ns = max(now_ns, work_end_ns)
    if termination == "mission_window_complete":
        require(
            len(events) + len(skipped_ids) == len(indices),
            "Missing declared opportunity accounting",
        )
        now_ns = max(now_ns, seconds_to_ns(scenario.duration_s))
    if battery(now_ns) <= 0:
        termination = "battery_depleted"
    quality, report_quality = (
        scorer.quality_details(now_ns / 1e9),
        combined_reports(reporters, now_ns / 1e9),
    )
    flags = dict(
        memory_constraint_success=peak_memory <= 4096,
        deadline_success=now_ns / 1e9 <= scenario.contract.deadline_s,
        battery_constraint_success=battery(now_ns) >= scenario.contract.min_final_battery_frac,
        communication_constraint_success=communication <= scenario.contract.communication_budget_mb,
        privacy_constraint_success=True,
    )
    result = dict(
        schema_version="local_inspection_task_deadline_execution_v1",
        contract=asdict(scenario.contract),
        status="completed",
        error=None,
        termination_reason=termination,
        mission_completed=termination == "mission_window_complete",
        final_time_s=now_ns / 1e9,
        final_battery_frac=battery(now_ns),
        cumulative_energy_j=now_ns / 1e9 * scenario.flight_power_w + nonflight,
        cumulative_nonflight_energy_j=nonflight,
        flight_energy_j=now_ns / 1e9 * scenario.flight_power_w,
        cumulative_communication_mb=communication,
        peak_configured_memory_mb=peak_memory,
        processed_observations=len(events),
        eligible_opportunities=len(indices),
        inactive_capture_count=(schedule.end_ns - schedule.epoch_ns + schedule.interval_ns - 1)
        // schedule.interval_ns
        - len(indices),
        unprocessed_opportunity_ids=list(indices[position:]),
        perception_latency=latency_summary(events),
        selected_action_counts=dict(
            Counter(e["selected_action_id"] for e in events if e["selected_action_id"] is not None)
        ),
        skipped_observation_ids=skipped_ids,
        router_errors=0,
        invalid_actions=0,
        privacy_violations=0,
        inference_failures=sum(not e["execution"]["success"] for e in events),
        constraints=flags,
        quality=quality,
        report_quality=report_quality,
        recording_identity=scenario.recording_identity,
    )
    outcome = outcome_fn(contract, events, result)
    row = dict(
        **item,
        dynamic_mission_success=outcome["dynamic_mission_success"],
        window_successes=outcome["window_successes"],
        window_metrics=outcome["windows"],
        path="runs/" + item["run_id"],
        local_inspection_success=outcome["local_inspection_success"],
        full_wide_mission_evaluated=False,
        raw_false_components=outcome["phases"][0]["false_components"],
        evidence_recall=outcome["phases"][0]["evidence_recall"],
        world_false_reports=outcome["reporting"]["world_false_reports"],
        world_duplicate_reports=outcome["reporting"]["world_duplicates"],
        correct_nonfocal_reports=outcome["reporting"]["correct_nonfocal_reports"],
        focal_correct_reports=len(outcome["reporting"]["timely_focal_reports"]),
        successful_phases=sum(p["success"] for p in outcome["phases"]),
        report_recall=outcome["reporting"]["recall_by_deadline"],
        final_battery_frac=result["final_battery_frac"],
        communication_mb=communication,
        decision_count=len(events),
        skipped_opportunity_count=len(skipped_ids),
        eligible_opportunity_count=len(indices),
        selected_action_counts=result["selected_action_counts"],
        perception_latency=result["perception_latency"],
        final_time_s=result["final_time_s"],
        flight_energy_j=result["flight_energy_j"],
        nonflight_energy_j=nonflight,
        total_energy_j=result["cumulative_energy_j"],
        mission_completed=result["mission_completed"],
        termination_reason=termination,
        task_metrics=outcome["task_metrics"],
        abstentions=outcome["abstentions"],
        failed_constraints=outcome["failed_constraints"],
    )
    if policy_label is not None:
        row["policy"] = policy_label
    return dict(
        row=row, events=events, result=result, outcome=outcome, pipeline_failures=pipeline_failures
    )
