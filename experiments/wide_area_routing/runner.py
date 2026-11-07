"""Paid perception routing on declared opportunities from a fixed wide flight.

Only the invocation schedule differs from the dense legacy replay: original
capture IDs and clocks remain intact, inactive transit captures are not misses,
and all flight time is charged. The existing policy factory, paid perception,
deadline pipeline, public context and recoverable intent scorer are reused.
"""

from __future__ import annotations

import hashlib
import math
from bisect import bisect_left
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.mission_pipeline_routing.runtime import CompletedPrediction, PublicPose
from experiments.mission_pipeline_routing.study import public_context
from experiments.mission_runtime_profile import read, save, sha
from experiments.mission_task_deadlines.contract import DeadlineProfile
from experiments.mission_task_deadlines.pipeline import DeadlinePipeline
from experiments.mission_task_deadlines.study import DOWNSTREAM, evaluate_recoverable, require
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


def advance_opportunities(schedule, indices, position, completed_ns):
    """Return completion, next list position and only busy-missed eligible IDs.

    ``indices`` contains validated, strictly increasing original capture IDs.
    ``completed_ns`` is relative to the recording epoch. A result ready exactly
    at the next eligible capture does not skip it. Inactive captures never count
    as lost observations, including gaps spanned by a nonpreemptive invocation.
    """
    index = indices[position]
    capture_ns = schedule.at(index) - schedule.epoch_ns
    completion, next_index, _ = schedule.complete(index, completed_ns - capture_ns)
    next_position = bisect_left(indices, next_index, lo=position + 1)
    return completion - schedule.epoch_ns, next_position, tuple(indices[position + 1:next_position])


def validate_bank(scenario, bank):
    """Check sparse scheduling/identity before opening a writable run directory."""
    indices = tuple(bank["opportunity_indices"])
    require(bool(indices), "No declared perception opportunities")
    require(all(type(i) is int and i >= 0 for i in indices), "Invalid opportunity index")
    require(tuple(sorted(set(indices))) == indices, "Opportunities must be unique and sorted")
    require(
        set(bank["frames"]) == set(bank["payloads"]) == set(indices),
        "RGB/payload banks must exactly match declared opportunities",
    )
    require(
        bank["provenance"]["recording_identity"] == scenario.recording_identity,
        "Wrong shared RGB bank",
    )
    if "opportunity_indices" in bank["provenance"]:
        require(list(indices) == bank["provenance"]["opportunity_indices"], "Opportunity receipt changed")
    schedule = scenario.schedule
    require(
        bank["provenance"]["capture_schedule"]
        == {name: getattr(schedule, name) for name in ("epoch_ns", "interval_ns", "end_ns")},
        "Capture schedule receipt changed",
    )
    require(
        schedule.end_ns - schedule.epoch_ns == seconds_to_ns(scenario.duration_s),
        "Capture schedule differs from the full flight duration",
    )
    for index in indices:
        require(schedule.at(index) < schedule.end_ns, "Opportunity outside the flight")
        require(bank["frames"][index].timestamp_ns == schedule.at(index), "Noncanonical RGB capture")
        require(
            bank["frames"][index].body_pose["timestamp_ns"] == schedule.at(index),
            "Pose timestamp differs from its RGB capture",
        )
    return indices


def source_hashes():
    """Bind orchestration and the unchanged policy/runtime/scoring implementations."""
    repo = Path(__file__).resolve().parents[2]
    paths = (
        "experiments/wide_area_routing/runner.py",
        "experiments/wide_area_routing/adapter.py",
        "experiments/perception_router_suite/policy.py",
        "experiments/perception_router_suite/learning.py",
        "experiments/mission_task_deadlines/policy.py",
        "experiments/mission_task_deadlines/perception.py",
        "experiments/mission_task_deadlines/contract.py",
        "experiments/mission_task_deadlines/pipeline.py",
        "experiments/mission_task_deadlines/study.py",
        "experiments/mission_pipeline_routing/runtime.py",
        "experiments/mission_pipeline_routing/study.py",
        "experiments/mission_network_emulation.py",
        "experiments/mission_network_policies.py",
        "experiments/lake_hindsight_witness.py",
        "experiments/lake_intent_contract.py",
        "experiments/lake_tracking_evaluation.py",
        "experiments/v4_benchmark/runner.py",
        "experiments/v4_gazebo_pilot/timing.py",
        "missionbench/v2/evaluation.py",
    )
    return {str(repo / path): sha(repo / path) for path in paths}


def latency_summary(events):
    """Configured capture-to-perception completion, including failed attempts."""
    durations = [
        (e["perception_completed_ns"] - round(e["capture_time_s"] * 1e9)) / 1e9
        for e in events if e["selected_action_id"] is not None
    ]
    return dict(
        count=len(durations),
        mean_s=float(np.mean(durations)) if durations else None,
        p50_s=float(np.percentile(durations, 50)) if durations else None,
        p95_s=float(np.percentile(durations, 95)) if durations else None,
        max_s=max(durations) if durations else None,
        scope="Configured capture-to-perception finish, including overhead and failed attempts; excludes abstentions",
    )


def run_case(root, plan, item, terrain, bank, manifest):
    from experiments.mission_task_deadlines.perception import execute_perception
    from experiments.perception_router_suite.policy import make_policy
    from experiments.wide_area_routing.adapter import load_case

    root = Path(root)
    require(item in plan["grid"], "Run is outside the declared study grid")
    case = load_case(root, plan, item)
    scenario, contract = case["scenario"], case["contract"]
    actions, costs, trace = case["actions"], case["costs"], case["trace"]
    profiles, learned_profile = case["profiles"], case["learned_profile"]
    require(manifest["recording_identity"] == scenario.recording_identity, "Wrong prediction flight")
    require(
        manifest["recording_sha256"] == bank["provenance"]["recording_sha256"],
        "Predictions and RGB bank bind different recording bytes",
    )
    predictions_dir = root / "predictions" / item["flight"]
    require(read(predictions_dir / "manifest.json") == manifest, "Prediction manifest changed")
    require(read(root / "plan.json") == plan, "Study plan changed")
    indices = validate_bank(scenario, bank)
    descriptors = {a.action_id: a for a in actions}
    for action in actions:
        model = manifest["models"][action.source_config_id]
        require(model["signature"] == action.source_signature, "Prediction model identity changed")
        require(
            all(bank["payloads"][i].rgb_sha256 in model["predictions"] for i in indices),
            "Incomplete prediction bank for declared opportunities",
        )
    budget = item["perception_budget_s"]
    require(type(budget) in (int, float) and budget in (2, 8), "Undeclared perception budget")
    profile = DeadlineProfile(f"perception_{budget:g}s", seconds_to_ns(budget), 150_000_000, 30_000_000)
    policy = make_policy(
        item["policy"],
        actions,
        costs,
        profile,
        profiles["recall"],
        profiles["risk"],
        learned_profile=learned_profile,
    )
    pipeline = DeadlinePipeline(
        profile=profile,
        terrain=terrain,
        step_ns=scenario.schedule.step_ns,
        battery_capacity_j=scenario.battery_capacity_wh * 3600,
        flight_power_w=scenario.flight_power_w,
    )
    scorer = MissionEvaluator(
        scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
    )
    reporter = RunnerReportEvaluator(scenario.matching_iou_threshold, scenario.targets.values(), 2)
    masks = {
        a.action_id: VerifiedMasks(
            predictions_dir,
            manifest["models"][a.source_config_id],
        )
        for a in actions
    }
    frames, payloads, provenance = bank["frames"], bank["payloads"], bank["provenance"]
    require(
        provenance["recording_identity"] == scenario.recording_identity, "Wrong shared RGB bank"
    )
    destination = root / "runs" / item["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    save(
        destination / "protocol.json",
        dict(
            schema_version="wide_area_paid_routing_v1",
            case=item,
            implementation_sha256=source_hashes(),
            opportunity_indices=list(indices),
            observation_clock="Original recording capture indices; only predeclared opportunities are eligible",
            flight_energy_scope="Entire elapsed flight including initial transit, inactive gaps and tail",
            plan_sha256=sha(root / "plan.json"),
            contract_sha256=contract.sha256,
            public_contract_sha256=contract.public_sha256,
            recording=provenance,
            prediction_manifest_sha256=sha(
                predictions_dir / "manifest.json"
            ),
            deadline_profile=profile.to_dict(),
            downstream_selection=asdict(DOWNSTREAM),
        ),
    )
    save(destination / "intent.json", contract.document())
    schedule = scenario.schedule
    capacity = scenario.battery_capacity_wh * 3600
    nonflight = communication = peak_memory = 0.0
    current_action = None
    now_ns = position = 0
    events, skipped_ids = [], []
    termination = "mission_window_complete"

    def battery(at_ns, extra_j=0.0):
        energy = at_ns / 1e9 * scenario.flight_power_w + nonflight + extra_j
        return max(0.0, scenario.initial_battery_frac - energy / capacity)

    while position < len(indices):
        index = indices[position]
        if battery(now_ns) <= 0:
            termination = "battery_depleted"
            break
        capture_ns = schedule.at(index) - schedule.epoch_ns
        require(capture_ns >= now_ns, "Capture occurred while preceding work was busy")
        if battery(capture_ns) <= 0:
            now_ns, termination = capture_ns, "battery_depleted"
            break
        if capture_ns > seconds_to_ns(scenario.contract.deadline_s):
            now_ns, termination = capture_ns, "deadline_exceeded"
            break
        frame, prepared = frames[index], payloads[index]
        public_intent = contract.public_document()
        decision_ns = capture_ns + seconds_to_ns(costs.encode_s, allow_zero=True)
        observed_battery = battery(decision_ns, costs.encode_s * costs.codec_power_w)
        selection = policy.select(
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
        action_id = selection["action_id"]
        action = descriptors[action_id] if action_id is not None else None
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
        now_ns = perception.completed_ns
        output = None
        reports = []
        downstream_skip_reason = None
        if perception.mask is not None and battery(now_ns) > 0:
            resource = scenario.resources.for_config(action_id)
            context = public_context(
                contract, capture_ns, now_ns, battery(now_ns), resource.memory_mb
            )
            output = pipeline.execute(
                DOWNSTREAM,
                CompletedPrediction(
                    action.source_config_id,
                    capture_ns,
                    now_ns,
                    perception.mask,
                    PublicPose(
                        tuple(frame.body_pose["position_m"]),
                        tuple(frame.body_pose["quaternion_xyzw"]),
                        scenario.world.get("camera_pitch_rad", math.pi / 2),
                    ),
                ),
                context,
            )
            nonflight += output["charged_energy_j"]
            now_ns = output["completed_ns"]
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
        completion, next_position, skipped = advance_opportunities(schedule, indices, position, now_ns)
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
                observation_id=index,
                capture_time_s=capture_ns / 1e9,
                capture_timestamp_ns=frame.timestamp_ns,
                completion_time_s=now_ns / 1e9,
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
                cumulative_energy_j=now_ns / 1e9 * scenario.flight_power_w + nonflight,
                cumulative_communication_mb=communication,
                battery_after=battery(now_ns),
                skipped_observation_ids=list(skipped),
            )
        )
        skipped_ids.extend(skipped)
        position = next_position
    if termination == "mission_window_complete":
        require(len(events) + len(skipped_ids) == len(indices), "Missing declared opportunity accounting")
        now_ns = max(now_ns, seconds_to_ns(scenario.duration_s))
    if battery(now_ns) <= 0:
        termination = "battery_depleted"
    quality, report_quality = scorer.quality_details(now_ns / 1e9), reporter.summary(now_ns / 1e9)
    flags = dict(
        memory_constraint_success=peak_memory <= 4096,
        deadline_success=now_ns / 1e9 <= scenario.contract.deadline_s,
        battery_constraint_success=battery(now_ns) >= scenario.contract.min_final_battery_frac,
        communication_constraint_success=communication <= scenario.contract.communication_budget_mb,
        privacy_constraint_success=True,
    )
    result = dict(
        schema_version="wide_area_task_deadline_execution_v1",
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
        inactive_capture_count=(schedule.end_ns - schedule.epoch_ns + schedule.interval_ns - 1) // schedule.interval_ns - len(indices),
        unprocessed_opportunity_ids=list(indices[position:]),
        perception_latency=latency_summary(events),
        selected_action_counts=dict(Counter(e["selected_action_id"] for e in events if e["selected_action_id"] is not None)),
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
    outcome = evaluate_recoverable(contract, events, result)
    for name, value in (("events", events), ("results", result), ("outcome", outcome)):
        save(destination / (name + ".json"), value)
    return dict(
        **item,
        path=str(destination.relative_to(root)),
        mission_success=outcome["mission_success"],
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
        artifacts_sha256={
            name + ".json": sha(destination / (name + ".json"))
            for name in ("protocol", "intent", "events", "results", "outcome")
        },
    )
