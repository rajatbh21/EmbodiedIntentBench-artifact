"""Independent raw-mask, clock, cost, public-policy and endpoint reconstruction.

The selected algorithm implementations and canonical scorers are shared, but the
mission runner is never called. No logged battery, candidate, report, score or
resource-success field is accepted without reconstruction from locked inputs.
"""

import hashlib
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_pipeline_routing.catalog import default_catalog
from experiments.mission_pipeline_routing.policy import select
from experiments.mission_pipeline_routing.runtime import CompletedPrediction, Pipeline, PublicPose
from experiments.mission_pipeline_routing.study import (
    MODEL,
    plain,
    public_context,
    require,
    scenario_for,
    verify,
)
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_runtime_profile import read, save, sha
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


def compare(actual, expected, label):
    for key, value in expected.items():
        require(key in actual and actual[key] == value, f"{label}.{key} differs")


def chosen_costs(plan, selection, previous, memory):
    catalog = {a["algorithm_id"]: a for a in plan["catalog"]}
    names = (selection.association_id, selection.verification_id)
    reset = tuple(
        task
        for task, old, new in zip(
            ("association", "report_verification"), previous, names, strict=True
        )
        if old != new
    )
    loads = [catalog[new]["costs"] for old, new in zip(previous, names, strict=True) if old != new]
    service = [catalog[name]["costs"] for name in names]
    load_ns = sum(c["load_latency_ns"] for c in loads)
    load_j = sum(c["load_energy_j"] for c in loads)
    service_ns = sum(c["latency_ns"] for c in service)
    service_j = sum(c["energy_j"] for c in service)
    return dict(
        routing_latency_ns=plan["downstream_routing_latency_ns"],
        routing_energy_j=plan["downstream_routing_energy_j"],
        load_latency_ns=load_ns,
        load_energy_j=load_j,
        service_latency_ns=service_ns,
        service_energy_j=service_j,
        charged_energy_j=plan["downstream_routing_energy_j"] + load_j + service_j,
        resident_memory_mb=memory + sum(c["memory_mb"] for c in service),
        reset_slots=list(reset),
    )


def run(
    root,
    *,
    verify_plan=None,
    catalog_document=None,
    pipeline_factory=None,
    policy_selector=None,
):
    """Reconstruct executions with separately pinned public-only extension hooks."""
    root = Path(root)
    require(not (root / "validation.json").exists(), "Preserve existing audit")
    verify_inputs = verify_plan or verify
    plan = verify_inputs(root)
    plan_sha, index_sha = sha(root / "plan.json"), sha(root / "run_index.json")
    index = read(root / "run_index.json")
    keys = ("flight", "mission", "policy")
    expected = {tuple(r[k] for k in keys) for r in plan["grid"]}
    rows = index["cases"]
    require(
        index["complete_grid"]
        and index["plan_sha256"] == plan_sha
        and len(rows) == len(expected) == plan["expected_runs"]
        and {tuple(r[k] for k in keys) for r in rows} == expected
        and len({r["path"] for r in rows}) == len(rows),
        "Missing, duplicate or substituted grid cell",
    )
    require(
        plan["catalog"]
        == (default_catalog().document() if catalog_document is None else catalog_document),
        "Changed executable catalog",
    )
    terrain, _ = load_public_terrain(Path(plan["assets"]))
    hashes = {}
    counts = dict(runs=0, decisions=0, reports=0, independent_cost_checks=0)
    for row in rows:
        flight, mission, policy = (row[k] for k in keys)
        destination = (root / row["path"]).resolve()
        require(destination.is_relative_to(root.resolve()), "Escaped run path")
        for name, expected_sha in row["artifacts_sha256"].items():
            require(
                name in ("events.json", "results.json", "outcome.json"),
                "Unexpected indexed artifact",
            )
            require(sha(destination / name) == expected_sha, "Changed indexed artifact")
            hashes[str((destination / name).relative_to(root.resolve()))] = expected_sha
        require(
            set(row["artifacts_sha256"]) == {"events.json", "results.json", "outcome.json"},
            "Incomplete artifact binding",
        )
        events, result, outcome = (
            read(destination / n) for n in ("events.json", "results.json", "outcome.json")
        )
        contract = IntentContract(read(root / "missions" / flight / mission / "intent.json"))
        scenario = bind_scenario(scenario_for(root, plan, flight, mission), contract)
        schedule = scenario.schedule
        resource = scenario.resources.for_config(MODEL)
        capacity = scenario.battery_capacity_wh * 3600
        source = Path(plan["source"])
        manifest = checked_manifest(source / flight / "predictions", source / "protocol.json")
        require(
            manifest["recording_identity"] == scenario.recording_identity
            and manifest["models"][MODEL]["signature"] == plan["upstream_signature"],
            "Changed selected prediction bank",
        )
        masks = VerifiedMasks(source / flight / "predictions", manifest["models"][MODEL])
        backend = ReplayBackend(scenario, source / flight / "recording")
        pipeline = (pipeline_factory or Pipeline)(terrain=terrain)
        scorer = MissionEvaluator(
            scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
        )
        reporter = RunnerReportEvaluator(
            scenario.matching_iou_threshold, scenario.targets.values(), 2
        )
        next_capture = 0
        previous = (None, None)
        upstream_j = load_j = pipeline_j = peak_memory = 0.0
        now = 0.0
        expected_events, all_skipped = [], []
        try:
            for number, event in enumerate(events):
                capture = schedule.at(next_capture)
                relative = capture - schedule.epoch_ns
                require(
                    capture < schedule.end_ns and relative / 1e9 <= scenario.contract.deadline_s,
                    "Ineligible capture processed",
                )
                require(
                    scenario.initial_battery_frac
                    - (now * scenario.flight_power_w + upstream_j + load_j + pipeline_j) / capacity
                    > 0,
                    "New work after exhausted battery",
                )
                backend.advance_to(capture)
                frame = backend.capture()
                mask = masks[rgb_key(frame.rgb)]
                cold = number == 0
                cold_ns = seconds_to_ns(resource.load_latency_s, allow_zero=True) if cold else 0
                cold_j = resource.load_energy_j if cold else 0.0
                ready = relative + cold_ns + plan["upstream_latency_ns"]
                upstream_j += plan["upstream_energy_j"]
                load_j += cold_j
                pre_energy = (
                    ready / 1e9 * scenario.flight_power_w + upstream_j + load_j + pipeline_j
                )
                context = public_context(
                    contract,
                    relative,
                    ready,
                    max(0.0, scenario.initial_battery_frac - pre_energy / capacity),
                    resource.memory_mb,
                )
                history = plain(pipeline.snapshot(ready))
                selection = (policy_selector or select)(policy, context, history)
                costs = chosen_costs(plan, selection, previous, resource.memory_mb)
                stage_ready = (
                    ready
                    + costs["routing_latency_ns"]
                    + costs["load_latency_ns"]
                    + costs["service_latency_ns"]
                )
                completion, following, skipped = schedule.complete(
                    next_capture, stage_ready - relative
                )
                output_time = (completion - schedule.epoch_ns) / 1e9
                pending = pipeline.dispatch(
                    selection,
                    CompletedPrediction(
                        MODEL,
                        relative,
                        ready,
                        mask,
                        PublicPose(
                            tuple(frame.body_pose["position_m"]),
                            tuple(frame.body_pose["quaternion_xyzw"]),
                            scenario.world.get("camera_pitch_rad", math.pi / 2),
                        ),
                    ),
                    context,
                )
                compare(
                    plain(pending.to_dict()),
                    dict(
                        costs,
                        job_id=number + 1,
                        start_ns=ready,
                        ready_ns=stage_ready,
                        selection=asdict(selection),
                    ),
                    "independent stage costs",
                )
                for missed in skipped:
                    backend.advance_to(schedule.at(missed))
                    truth = evaluation_frame(
                        backend.capture(),
                        missed,
                        (schedule.at(missed) - schedule.epoch_ns) / 1e9,
                        scenario,
                    )
                    scorer.note_skipped_visibility(truth.visible_target_ids)
                    reporter.note_visibility(truth)
                backend.advance_to(completion)
                output = pipeline.complete(pending.job_id, completion - schedule.epoch_ns)
                pipeline_j += costs["charged_energy_j"]
                require(
                    pipeline.cumulative_energy_j == pipeline_j,
                    "Runtime payment differs from independent algebra",
                )
                energy = output_time * scenario.flight_power_w + upstream_j + load_j + pipeline_j
                truth = evaluation_frame(frame, next_capture, relative / 1e9, scenario)
                score = scorer.update(truth, mask)
                reporter.update(truth, mask, output["emitted_reports"])
                expected_event = dict(
                    observation_id=next_capture,
                    capture_timestamp_ns=capture,
                    capture_time_s=relative / 1e9,
                    completion_time_s=output_time,
                    capture_body_pose=frame.body_pose,
                    source_rgb_key=rgb_key(frame.rgb),
                    selected_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                    upstream_available_ns=ready,
                    upstream_load_latency_ns=cold_ns,
                    upstream_load_energy_j=cold_j,
                    upstream_latency_ns=plan["upstream_latency_ns"],
                    upstream_energy_j=plan["upstream_energy_j"],
                    public_context=asdict(context),
                    public_history=history,
                    selection=asdict(selection),
                    pipeline=plain(output),
                    execution=dict(config_id=MODEL, success=True, predicted_pixels=int(mask.sum())),
                    score=score.to_dict(),
                    emitted_reports=output["emitted_reports"],
                    battery_after=max(0.0, scenario.initial_battery_frac - energy / capacity),
                    cumulative_energy_j=energy,
                    skipped_observation_ids=list(skipped),
                )
                require(
                    event == expected_event,
                    "Raw selected output/public state/pipeline event reconstruction differs",
                )
                expected_events.append(expected_event)
                all_skipped.extend(skipped)
                peak_memory = max(peak_memory, costs["resident_memory_mb"])
                next_capture, now, previous = (
                    following,
                    output_time,
                    (selection.association_id, selection.verification_id),
                )
                counts["decisions"] += 1
                counts["reports"] += len(output["emitted_reports"])
                counts["independent_cost_checks"] += 1
        finally:
            backend.close()
        energy = now * scenario.flight_power_w + upstream_j + load_j + pipeline_j
        reason = "mission_window_complete"
        if schedule.at(next_capture) < schedule.end_ns:
            if scenario.initial_battery_frac - energy / capacity <= 0:
                reason = "battery_depleted"
            elif (
                now > scenario.contract.deadline_s
                or (schedule.at(next_capture) - schedule.epoch_ns) / 1e9
                > scenario.contract.deadline_s
            ):
                reason = "deadline_exceeded"
            else:
                raise ValueError("Unaccounted eligible capture omission")
        else:
            now = max(now, scenario.duration_s)
            energy = now * scenario.flight_power_w + upstream_j + load_j + pipeline_j
            if scenario.initial_battery_frac - energy / capacity <= 0:
                reason = "battery_depleted"
        battery = max(0.0, scenario.initial_battery_frac - energy / capacity)
        quality, reports = scorer.quality_details(now), reporter.summary(now)
        extra, first = scenario.requirements.evaluate(quality, expected_events, reports)
        flags = dict(
            extra,
            memory_constraint_success=peak_memory <= 4096,
            quality_success=scenario.contract.quality_satisfied(
                quality[scenario.contract.quality_metric]
            ),
            deadline_success=now <= scenario.contract.deadline_s,
            battery_constraint_success=battery >= scenario.contract.min_final_battery_frac,
            communication_constraint_success=scenario.contract.communication_budget_mb >= 0,
            privacy_constraint_success=True,
        )
        rebuilt = dict(
            schema_version="task_pipeline_execution_v1",
            contract=asdict(scenario.contract),
            status="completed",
            error=None,
            termination_reason=reason,
            mission_completed=reason == "mission_window_complete",
            final_time_s=now,
            final_battery_frac=battery,
            cumulative_energy_j=energy,
            cumulative_communication_mb=0.0,
            inference_energy_j=upstream_j,
            model_load_energy_j=load_j,
            pipeline_energy_j=pipeline_j,
            flight_energy_j=now * scenario.flight_power_w,
            peak_configured_memory_mb=peak_memory,
            processed_observations=len(events),
            skipped_observation_ids=all_skipped,
            router_errors=0,
            inference_failures=0,
            invalid_actions=0,
            privacy_violations=0,
            constraints=flags,
            quality=quality,
            report_quality=reports,
            first_correct_report_s=first,
            mission_success=reason == "mission_window_complete" and all(flags.values()),
            contract_sha256=contract.sha256,
            recording_identity=scenario.recording_identity,
            policy=policy,
            selected_executor_calls=pipeline.executed_calls,
            association_resets=sum(
                "association" in e["pipeline"]["costs"]["reset_slots"] for e in expected_events
            ),
            verification_resets=sum(
                "report_verification" in e["pipeline"]["costs"]["reset_slots"]
                for e in expected_events
            ),
        )
        require(result == rebuilt, "Independent terminal/resource/report reconstruction differs")
        canonical = evaluate(contract, expected_events, rebuilt)
        require(outcome == canonical, "Canonical endpoint differs")
        compare(
            row,
            dict(
                calls=len(events),
                full_success=canonical["mission_success"],
                phases_succeeded=sum(p["success"] for p in canonical["phases"]),
                report_recall=canonical["reporting"]["recall_by_deadline"],
                failed_constraints=canonical["failed_constraints"],
                report_count=reports["total_reports"],
                false_reports=reports["false_reports"],
                duplicate_reports=reports["duplicate_correct_reports"],
                final_battery=battery,
                pipeline_energy_j=pipeline_j,
                association_resets=rebuilt["association_resets"],
                verification_resets=rebuilt["verification_resets"],
                selections=dict(
                    Counter(
                        e["selection"]["association_id"] + "/" + e["selection"]["verification_id"]
                        for e in expected_events
                    )
                ),
            ),
            "index",
        )
        counts["runs"] += 1
        print(f"Audited {counts['runs']}/{len(rows)}", flush=True)
    verify_inputs(root)
    require(
        sha(root / "plan.json") == plan_sha and sha(root / "run_index.json") == index_sha,
        "Changed audit inputs",
    )
    for path, expected_sha in hashes.items():
        require(sha(root / path) == expected_sha, "Output changed during audit")
    validation = dict(
        complete=True,
        schema_version="task_pipeline_audit_v1",
        plan_sha256=plan_sha,
        run_index_sha256=index_sha,
        auditor_sha256=sha(Path(__file__)),
        output_sha256=hashes,
        all_public_decisions_reselected=True,
        raw_selected_masks_and_reports_reconstructed=True,
        clock_resources_and_terminal_flags_reconstructed=True,
        original_canonical_endpoint_verified=True,
        no_mission_runner_called=True,
        shared_selected_algorithms_and_scorers=True,
        **counts,
    )
    save(root / "validation.json", validation)
    return validation


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output)
    print({k: result[k] for k in ("complete", "runs", "decisions", "reports")})
