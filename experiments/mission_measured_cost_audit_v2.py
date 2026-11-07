"""V2 audit of the metadata-repaired development cost replays.

Chronology checks are versioned from the frozen transfer integrity checker, with
an explicit TrackingCost argument. The original checker and study are unchanged.
Raw masks, report identity, public-only policy decisions and terminal accounting
are rebuilt independently of the runner; every declared cell must be present.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from math import isfinite

from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label, tolerance=1e-8):
    require(
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and isfinite(actual)
        and abs(actual - expected) <= tolerance,
        label + " mismatch",
    )


def reconstruct_states(scenario, events, tracking):
    """Build the entire public state from the paid action sequence, never log fields."""
    from missionbench.schemas.runtime_state import EvidenceSummary, RuntimeState

    states = {}
    specs = {s.config_id: s for s in scenario.executors}
    schedule = scenario.schedule
    capacity = scenario.battery_capacity_wh * 3600
    initial = scenario.initial_battery_frac * capacity
    energy, previous_time, previous_config, index = 0.0, 0.0, None, 0
    for processed, event in enumerate(events):
        capture_s = (schedule.at(index) - schedule.epoch_ns) / 1e9
        energy += (capture_s - previous_time) * scenario.flight_power_w
        state = RuntimeState(
            current_time_s=capture_s,
            frame_id=index,
            battery_frac=max(0.0, (initial - energy) / capacity),
            power_mode="v4_configured",
            network=scenario.network.observation_at(capture_s),
            current_config_id=previous_config,
            remaining_deadline_s=scenario.contract.deadline_s - capture_s,
            cumulative_energy_j=energy,
            cumulative_communication_mb=0.0,
            path_progress=min(1.0, capture_s / scenario.duration_s),
            evidence_summary=EvidenceSummary(0, processed, 0.0),
        )
        require(state.to_dict() == event["runtime_at_capture"], "Complete public state differs")
        states[index] = state
        config = event["execution"]["config_id"]
        spec, resource = specs[config], scenario.resources.for_config(config)
        switched = config != previous_config
        completion, next_index, skipped = schedule.complete(
            index,
            seconds_to_ns(
                spec.mission_latency_s
                + scenario.router_latency_s
                + tracking.latency_s
                + (resource.load_latency_s if switched else 0),
                allow_zero=True,
            ),
        )
        completion_s = (completion - schedule.epoch_ns) / 1e9
        # Match floating-point addition order, while independently deriving every charge.
        last_advance = capture_s
        for missed in skipped:
            next_advance = (schedule.at(missed) - schedule.epoch_ns) / 1e9
            energy += (next_advance - last_advance) * scenario.flight_power_w
            last_advance = next_advance
        energy += (completion_s - last_advance) * scenario.flight_power_w
        energy += spec.energy_j_per_call + 0.0 + scenario.router_energy_j
        energy += (resource.load_energy_j if switched else 0) + tracking.energy_j
        previous_time, previous_config, index = completion_s, config, next_index
    return states


def check_execution(scenario, events, result, row, canonical, tracking=DEFAULT_TRACKING_COST):
    """Reconstruct paid chronology from fixed source costs; consume no GT labels."""
    require(bool(events), "Empty execution cannot verify this positive-battery cohort")
    schedule = scenario.schedule
    require(
        result["tracking_cost"] == asdict(tracking), "Tracking costs differ from the assigned arm"
    )
    require(result["decision_stride"] == 1, "Changed study decision cadence")
    close(result["decision_period_s"], scenario.observation_interval_s, "Decision period")
    require(result["cadence_skipped_observation_ids"] == [], "Unexpected cadence omissions")
    specs = {s.config_id: s for s in scenario.executors}
    require(
        all(
            not scenario.catalog.get(c).strategy.is_remote and s.communication_mb_per_call == 0
            for c, s in specs.items()
        ),
        "Integrity checker requires the declared local-only pool",
    )
    require(
        scenario.resources.memory_capacity_mb is None
        or all(
            scenario.resources.for_config(c).memory_mb + tracking.memory_mb
            <= scenario.resources.memory_capacity_mb
            for c in specs
        ),
        "Integrity checker requires the study's nonbinding memory pool",
    )
    capacity = scenario.battery_capacity_wh * 3600
    initial = scenario.initial_battery_frac
    index, previous, completion_s = 0, None, 0.0
    compute_j = router_j = tracking_j = load_j = load_s = 0.0
    memory = 0.0
    skipped, selected = [], []

    def battery(time_s):
        energy = time_s * scenario.flight_power_w + compute_j + router_j + tracking_j + load_j
        return max(0.0, initial - energy / capacity)

    for event in events:
        capture_ns = schedule.at(index)
        capture_s = (capture_ns - schedule.epoch_ns) / 1e9
        require(capture_ns < schedule.end_ns, "Execution contains post-window capture")
        require(
            battery(completion_s) > 0
            and battery(capture_s) > 0
            and completion_s <= scenario.contract.deadline_s
            and capture_s <= scenario.contract.deadline_s,
            "Execution continued after a terminal resource condition",
        )
        state = event["runtime_at_capture"]
        require(
            event["observation_id"] == state["frame_id"] == index
            and event["capture_timestamp_ns"] == capture_ns,
            "Parallel capture index/clock mismatch",
        )
        close(event["capture_time_s"], capture_s, "Capture time")
        close(state["current_time_s"], capture_s, "Runtime capture time")
        require(state["current_config_id"] == previous, "Previous configuration mismatch")
        close(state["battery_frac"], battery(capture_s), "Capture battery")
        close(
            state["cumulative_energy_j"],
            capture_s * scenario.flight_power_w + compute_j + router_j + tracking_j + load_j,
            "Capture energy",
            1e-6,
        )
        execution = event["execution"]
        config = execution["config_id"]
        require(config in specs, "Unknown executed configuration")
        require(
            event["action"]["config_id"] == event["action"]["requested_config_id"] == config
            and event["action"]["outcome"] == "valid"
            and event["router_error"] is None
            and execution["success"],
            "Action, selection or execution failure",
        )
        spec, resource = specs[config], scenario.resources.for_config(config)
        switch_s = resource.load_latency_s if config != previous else 0
        switch_j = resource.load_energy_j if config != previous else 0
        completion_ns, next_index, omitted = schedule.complete(
            index,
            seconds_to_ns(
                spec.mission_latency_s + scenario.router_latency_s + tracking.latency_s + switch_s,
                allow_zero=True,
            ),
        )
        completion_s = (completion_ns - schedule.epoch_ns) / 1e9
        close(event["completion_time_s"], completion_s, "Completion time")
        close(event["effective_latency_s"], completion_s - capture_s, "Effective latency")
        close(execution["mission_latency_s"], spec.mission_latency_s, "Executor latency")
        close(execution["energy_j"], spec.energy_j_per_call, "Executor energy")
        close(execution["communication_mb"], 0, "Local communication")
        close(execution["communication_energy_j"], 0, "Local radio energy")
        close(event["configured_router_latency_s"], scenario.router_latency_s, "Router latency")
        close(event["configured_tracking_latency_s"], tracking.latency_s, "Tracking latency")
        close(event["tracking_energy_j"], tracking.energy_j, "Tracking energy")
        close(event["model_load_latency_s"], switch_s, "Model load time")
        close(event["model_load_energy_j"], switch_j, "Model load energy")
        close(event["resident_memory_mb"], resource.memory_mb + tracking.memory_mb, "Memory")
        require(event["skipped_observation_ids"] == list(omitted), "Busy capture omissions differ")
        compute_j += spec.energy_j_per_call
        router_j += scenario.router_energy_j
        tracking_j += tracking.energy_j
        load_j += switch_j
        load_s += switch_s
        memory = max(memory, resource.memory_mb + tracking.memory_mb)
        close(event["battery_after"], battery(completion_s), "Completion battery")
        close(
            event["cumulative_energy_j"],
            completion_s * scenario.flight_power_w + compute_j + router_j + tracking_j + load_j,
            "Completion energy",
            1e-6,
        )
        close(event["cumulative_communication_mb"], 0, "Cumulative local communication")
        selected.append(config)
        skipped.extend(omitted)
        index, previous = next_index, config

    # Reconstruct the runner's next loop test and its final flight-energy tail.
    final_s = completion_s
    if schedule.at(index) >= schedule.end_ns:
        if battery(final_s) <= 0:
            termination = "battery_depleted"
        else:
            final_s = max(final_s, scenario.duration_s)
            termination = "battery_depleted" if battery(final_s) <= 0 else "mission_window_complete"
    elif battery(final_s) <= 0:
        termination = "battery_depleted"
    else:
        next_capture_s = (schedule.at(index) - schedule.epoch_ns) / 1e9
        if final_s > scenario.contract.deadline_s or next_capture_s > scenario.contract.deadline_s:
            if final_s < scenario.contract.deadline_s:
                deadline_ns = schedule.epoch_ns + seconds_to_ns(scenario.contract.deadline_s)
                rounded_ns = (
                    (deadline_ns + schedule.step_ns - 1) // schedule.step_ns * schedule.step_ns
                )
                final_s = (rounded_ns - schedule.epoch_ns) / 1e9
            termination = "deadline_exceeded"
        elif battery(next_capture_s) <= 0:
            final_s, termination = next_capture_s, "battery_depleted"
        else:
            raise ValueError("Unexplained truncated execution before final capture")
    require(result["termination_reason"] == termination, "Terminal reason mismatch")
    require(
        result["mission_completed"] == (termination == "mission_window_complete")
        and result["status"] == "completed"
        and result["error"] is None,
        "Terminal status mismatch",
    )
    close(result["final_time_s"], final_s, "Final time")
    close(result["final_battery_frac"], battery(final_s), "Final battery")
    for key, expected in (
        ("flight_energy_j", final_s * scenario.flight_power_w),
        ("inference_energy_j", compute_j),
        ("router_energy_j", router_j),
        ("tracking_energy_j", tracking_j),
        ("model_load_energy_j", load_j),
        ("model_load_latency_s", load_s),
        ("peak_configured_memory_mb", memory),
        ("cumulative_communication_mb", 0),
        ("communication_energy_j", 0),
        (
            "cumulative_energy_j",
            final_s * scenario.flight_power_w + compute_j + router_j + tracking_j + load_j,
        ),
    ):
        close(result[key], expected, key, 1e-6)
    require(result["processed_observations"] == len(events), "Processed count mismatch")
    require(result["config_selection_history"] == selected, "Selection history mismatch")
    require(
        result["config_switches"] == sum(a != b for a, b in pairwise(selected)),
        "Switch count mismatch",
    )
    require(
        result["busy_skipped_observation_ids"] == result["skipped_observation_ids"] == skipped,
        "Aggregate skipped observations mismatch",
    )
    flags = dict(
        memory_constraint_success=True,
        deadline_success=final_s <= scenario.contract.deadline_s,
        battery_constraint_success=battery(final_s) >= scenario.contract.min_final_battery_frac,
        communication_constraint_success=scenario.contract.communication_budget_mb >= 0,
        privacy_constraint_success=True,
    )
    require(result["privacy_violations"] == 0, "Local pool cannot introduce privacy violations")
    for key, expected in flags.items():
        require(result["constraints"][key] is expected, "Resource flag differs: " + key)
    reconstruct_states(scenario, events, tracking)
    checks = dict(
        full_success=canonical["mission_success"],
        phases_succeeded=sum(p["success"] for p in canonical["phases"]),
        report_recall=canonical["reporting"]["recall_by_deadline"],
        false_reports=canonical["reporting"]["false_reports"],
        duplicate_reports=canonical["reporting"]["duplicate_reports"],
        final_battery=result["final_battery_frac"],
        calls=len(events),
        selections=dict(Counter(selected)),
        failed_constraints=canonical["failed_constraints"],
    )
    for key, expected in checks.items():
        require(row[key] == expected, "Summary " + key + " mismatch")
    return dict(
        events=len(events),
        next_capture_index=index,
        terminal_reason=termination,
        complete_capture_window=schedule.at(index) >= schedule.end_ns,
        final_time_s=final_s,
        final_battery_frac=battery(final_s),
    )


def complete_grid(plan, rows):
    """Reject missing, repeated, substituted or out-of-scope comparisons."""
    expected = {
        (arm, cell, mission, policy)
        for arm in plan["arms"]
        for cell in plan["cases"]
        for mission in plan["missions"]
        for policy in plan["policies"]
    }
    actual = [(r["arm"], r["cell"], r["mission"], r["policy"]) for r in rows]
    require(len(actual) == len(expected) and set(actual) == expected, "Incomplete replay grid")
    paths = [r["path"] for r in rows]
    require(len(set(paths)) == len(paths), "Multiple comparisons share one output path")
    return len(expected)


def compare_outcome(canonical, outcome):
    for key in ("mission_success", "constraints", "failed_constraints", "reporting", "phases"):
        require(canonical[key] == outcome[key], "Canonical outcome mismatch: " + key)


def run(root):
    """Audit complete adaptive executions without calling the mission runner."""
    from collections import defaultdict
    from pathlib import Path

    import numpy as np

    from missionbench.v2.evaluation import MissionEvaluator
    from missionbench.v2.executors import label_components
    from experiments.lake_hindsight_witness import VerifiedMasks
    from experiments.lake_intent_contract import IntentContract, evaluate, immutable
    from experiments.lake_intent_runner import bind_scenario
    from experiments.lake_tracking import SurfaceProjector, TrackingConfig, TrackingPipeline
    from experiments.lake_tracking_evaluation import RunnerReportEvaluator
    from experiments.lake_tracking_runtime import load_public_terrain
    from experiments.mission_measured_cost_study_v2 import (
        load_profiles,
        scenario_for,
        tracking_cost,
        verify,
    )
    from experiments.mission_routing_v2.feedback import FeedbackInput, snapshot
    from experiments.mission_routing_v2.models import CONTROL, ENSEMBLE, ROTATED
    from experiments.mission_routing_v2.train import checked_manifest
    from experiments.mission_runtime_profile import read, save, sha
    from experiments.mission_transfer_comparison_v4.study import make_policy
    from experiments.v4_benchmark.backends import ReplayBackend
    from experiments.v4_benchmark.calibrated_router import model_signature
    from experiments.v4_benchmark.probe_cache import rgb_key
    from experiments.v4_benchmark.runner import evaluation_frame, immutable_rgb
    from experiments.v4_benchmark.scenario import digest

    root = Path(root)
    require(not (root / "validation.json").exists(), "Refuse to overwrite an existing audit")
    plan = verify(root)
    index = read(root / "run_index.json")
    require(
        index["complete_execution_grid"] and index["plan_sha256"] == sha(root / "plan.json"),
        "Execution grid or plan binding differs",
    )
    rows = index["cases"]
    executions = complete_grid(plan, rows)
    require(executions == index["expected_replays"], "Replay count differs from declared index")
    profiles = load_profiles(root, plan)
    source = Path(plan["cohort"])
    terrain, metadata = load_public_terrain(Path(plan["assets"]))
    counts = dict(executions=0, events=0, public_decisions_reproduced=0, reports=0)
    poses, terminals, hashes = {}, {}, {}
    for flight in sorted({case["source_flight"] for case in plan["cases"].values()}):
        folder = source / flight / "predictions"
        manifest = checked_manifest(folder, source / "protocol.json")
        masks = {c: VerifiedMasks(folder, m) for c, m in manifest["models"].items()}
        by_capture = defaultdict(list)
        scorers = []
        for row in rows:
            case = plan["cases"][row["cell"]]
            if case["source_flight"] != flight:
                continue
            require(row["flight"] == flight, "Source recording label differs")
            destination = root / row["path"]
            require(destination.resolve().is_relative_to(root.resolve()), "Escaped output path")
            for filename, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
                ("protocol.json", "protocol_sha256"),
            ):
                require(sha(destination / filename) == row[key], "Changed result artifact")
                hashes[str(destination.relative_to(root) / filename)] = row[key]
            scenario = scenario_for(root, plan, row["cell"], row["arm"])
            contract = IntentContract(
                read(root / "missions" / row["arm"] / row["cell"] / row["mission"] / "intent.json")
            )
            scenario = bind_scenario(scenario, contract)
            require(set(masks) == {s.config_id for s in scenario.executors}, "Model pool differs")
            for spec in scenario.executors:
                require(
                    manifest["models"][spec.config_id]["signature"] == model_signature(spec),
                    "Cost intervention changed prediction semantics",
                )
            events = read(destination / "execution/events.json")
            result = read(destination / "execution/results.json")
            outcome = read(destination / "outcome.json")
            protocol = read(destination / "protocol.json")
            require(
                outcome["protocol_sha256"] == row["protocol_sha256"]
                and outcome["events_sha256"] == row["events_sha256"]
                and outcome["execution_sha256"] == row["results_sha256"],
                "Outcome input binding differs",
            )
            require(
                protocol["router"] == row["policy"]
                and protocol["gt_aware"] is False
                and protocol["deployable_policy"] is True
                and outcome["gt_aware"] is False
                and outcome["deployable_policy"] is True
                and outcome["public_contract_deliveries"] == len(events)
                and protocol["contract_sha256"] == contract.sha256
                and protocol["public_contract_sha256"] == contract.public_sha256
                and protocol["scenario_sha256"] == digest(scenario.source)
                and result["scenario_sha256"] == digest(scenario.source)
                and protocol["recording_identity"] == scenario.recording_identity
                and result["backend"]["recording_identity"] == scenario.recording_identity,
                "Policy, contract, scenario or recording binding differs",
            )
            cost = tracking_cost(plan, row["arm"])
            canonical = evaluate(contract, events, result)
            terminal = check_execution(scenario, events, result, row, canonical, cost)
            states = reconstruct_states(scenario, events, cost)
            compare_outcome(canonical, outcome)
            require(
                not any(
                    result[k] for k in ("router_errors", "inference_failures", "invalid_actions")
                ),
                "Execution errors cannot be accepted as an audited comparison",
            )
            policy = make_policy(row["policy"], profiles, scenario)
            projector = SurfaceProjector(terrain)
            tracker = TrackingPipeline(TrackingConfig(remember_reports=True), projector)
            tracker.provenance.update(
                map=metadata,
                anchor_height_m=projector.anchor_height_m,
                hfov_rad=projector.hfov,
                static_target_assumption=True,
            )
            tracking = dict(configuration=tracker.provenance, cost=asdict(cost))
            require(
                protocol["public_tracking"] == tracking
                and protocol["tracking"] == tracker.provenance
                and result["tracking"] == tracker.provenance,
                "Prediction tracking or public cost differs",
            )
            frame_scorer = MissionEvaluator(
                scenario.matching_iou_threshold,
                len(scenario.targets),
                scenario.observation_interval_s,
            )
            report_scorer = RunnerReportEvaluator(
                scenario.matching_iou_threshold, scenario.targets.values(), 2
            )
            scorers.append((report_scorer, result))
            public_counts = {"predicted_detections": 0}
            for event in events:
                by_capture[event["observation_id"]].append(
                    (
                        policy,
                        tracker,
                        event,
                        scenario,
                        contract,
                        tracking,
                        frame_scorer,
                        report_scorer,
                        states[event["observation_id"]],
                        public_counts,
                    )
                )
            counts["executions"] += 1
            counts["events"] += len(events)
            terminals[row["path"]] = terminal
        backend = ReplayBackend(scenario, source / flight / "recording")
        recording_poses = []
        try:
            for capture, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                recording_poses.append((timestamp, frame.body_pose))
                key = rgb_key(frame.rgb)
                predictions = {c: mask[key] for c, mask in masks.items()}
                require(
                    np.array_equal(
                        predictions[ENSEMBLE], predictions[CONTROL] | predictions[ROTATED]
                    ),
                    "Frozen ensemble prediction differs from its component union",
                )
                for (
                    policy,
                    tracker,
                    event,
                    scenario,
                    contract,
                    tracking,
                    scorer,
                    reports,
                    state,
                    public_counts,
                ) in by_capture[capture]:
                    require(frame.body_pose == event["capture_body_pose"], "Physical pose differs")
                    history = snapshot(tracker, event["capture_time_s"])
                    observation = FeedbackInput(
                        contract=scenario.contract,
                        intent_contract=immutable(contract.public_document()),
                        public_contract_sha256=contract.public_sha256,
                        state=state,
                        configs=scenario.catalog,
                        resources=scenario.resources,
                        tracking=immutable(tracking),
                        evidence_history=immutable(history),
                        confirmed_reports_so_far=tracker.report_count,
                        rgb=immutable_rgb(frame.rgb),
                        position_m=tuple(frame.body_pose["position_m"]),
                        quaternion_xyzw=tuple(frame.body_pose["quaternion_xyzw"]),
                        detections_so_far=public_counts["predicted_detections"],
                        requirements=scenario.requirements,
                    )
                    choice = policy.select(observation)
                    require(
                        choice == event["execution"]["config_id"], "Public-only selection differs"
                    )
                    if getattr(policy, "last_decision", None) is not None:
                        require(
                            policy.last_decision == event["router_decision"],
                            "Decision trace differs",
                        )
                    emitted = tracker.update(
                        predictions[choice],
                        event["capture_time_s"],
                        event["completion_time_s"],
                        pose=frame.body_pose,
                        camera_pitch_rad=scenario.world["camera_pitch_rad"],
                    )
                    require(
                        emitted == event["emitted_reports"], "Prediction report history differs"
                    )
                    labels, n_components = label_components(predictions[choice])
                    areas = np.bincount(labels.ravel(), minlength=n_components + 1)
                    public_counts["predicted_detections"] += int(
                        np.count_nonzero(areas[1:] >= scorer.min_component_px)
                    )
                    truth = evaluation_frame(frame, capture, event["capture_time_s"], scenario)
                    score = scorer.update(truth, predictions[choice])
                    require(
                        score.to_dict() == event["score"],
                        "Mask score differs",
                    )
                    reports.update(truth, predictions[choice], emitted)
                    counts["public_decisions_reproduced"] += 1
                    counts["reports"] += len(emitted)
        finally:
            backend.close()
        for scorer, result in scorers:
            require(
                scorer.summary(result["final_time_s"]) == result["report_quality"],
                "Report identity or repeated support score differs",
            )
        poses[flight] = digest(recording_poses)
        print(
            f"Reconstructed {flight}: {counts['public_decisions_reproduced']} decisions", flush=True
        )
    require(counts["executions"] == executions, "Incomplete independent audit")
    require(counts["events"] == counts["public_decisions_reproduced"], "Unreproduced decisions")
    verify(root)
    validation = dict(
        complete=True,
        schema_version="mission_measured_cost_audit_v2",
        plan_sha256=sha(root / "plan.json"),
        run_index_sha256=sha(root / "run_index.json"),
        validator_sha256=sha(Path(__file__)),
        source_output_sha256=hashes,
        terminal_reconstructions=terminals,
        recording_pose_sha256=poses,
        public_decisions_reconstructed_without_ground_truth=True,
        raw_selected_mask_scores_reconstructed=True,
        report_identity_support_reconstructed=True,
        per_event_cost_clock_memory_battery_verified=True,
        canonical_outcomes_verified=True,
        fresh_evaluation=False,
        **counts,
    )
    save(root / "validation.json", validation)
    return validation


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checked = run(args.output)
    print(
        {k: checked[k] for k in ("complete", "executions", "events", "public_decisions_reproduced")}
    )
