"""Independent receipt reconstruction for the wide-area scheduled pilot.

This module never calls the study runner or inference. It reconstructs the
sparse clock, selected-mask reads, public decisions, paid task receipts,
downstream state, semantic matching and final energy accounting from frozen
inputs. Existing policy, transport, executor, tracking and scoring primitives
are reused and checked against saved receipts; this is not an independent
implementation of those underlying algorithms or a hardware validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import math
import multiprocessing
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_pipeline_routing.runtime import CompletedPrediction, PublicPose
from experiments.mission_pipeline_routing.study import public_context
from experiments.mission_runtime_profile import read, save, sha
from experiments.mission_task_deadline_audit import check_deadline, check_transport
from experiments.mission_task_deadlines.contract import DeadlineProfile
from experiments.mission_task_deadlines.perception import execute_perception
from experiments.mission_task_deadlines.pipeline import DeadlinePipeline
from experiments.mission_task_deadlines.study import DOWNSTREAM, evaluate_recoverable
from experiments.perception_router_suite.policy import make_policy
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.wide_area_routing.adapter import load_bank, load_case


def require(value, message):
    if not value:
        raise ValueError(message)


def near(actual, expected, message):
    require(math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), message)


def ns(value):
    return round(value * 1_000_000_000)


def next_opportunity(indices, position, finish_ns, interval_ns):
    """Independent sparse chronology, without the execution scheduler helper."""
    next_position = position + 1
    missed = []
    while next_position < len(indices) and indices[next_position] * interval_ns < finish_ns:
        missed.append(indices[next_position])
        next_position += 1
    return next_position, missed


class CountedMasks:
    """Observe exactly when the paid executor requests an original selected mask."""

    def __init__(self, directory, model):
        self.verified = VerifiedMasks(directory, model)
        self.reads = []

    def __getitem__(self, key):
        self.reads.append(key)
        return self.verified[key]


def audit_run(root, plan, item, bank, manifest, terrain):
    root = Path(root)
    directory = root / "runs" / item["run_id"]
    row, protocol, events, result, outcome = (
        read(directory / (name + ".json"))
        for name in ("row", "protocol", "events", "results", "outcome")
    )
    require(all(row.get(key) == value for key, value in item.items()), "Result belongs to another grid cell")
    require(protocol["case"] == item, "Execution protocol names another grid cell")
    require(protocol["plan_sha256"] == sha(root / "plan.json"), "Execution uses another plan")
    for path, expected in protocol["implementation_sha256"].items():
        require(sha(path) == expected, "Changed execution implementation: " + path)
    for name, expected in row["artifacts_sha256"].items():
        require(sha(directory / name) == expected, "Changed execution artifact: " + name)
    case = load_case(root, plan, item)
    scenario, contract = case["scenario"], case["contract"]
    require(result["status"] == "completed" and result["error"] is None,
            "Unexpected execution error or state")
    require(result["contract"] == asdict(scenario.contract), "Runtime resource contract differs")
    require(result["router_errors"] == result["invalid_actions"] == result["privacy_violations"] == 0,
            "Unexpected routing or privacy error")
    require(read(directory / "intent.json") == contract.document(), "Changed execution intent")
    require(protocol["contract_sha256"] == contract.sha256, "Changed private contract identity")
    require(protocol["public_contract_sha256"] == contract.public_sha256, "Changed public contract identity")
    require(protocol["recording"] == bank["provenance"], "Changed bank provenance")
    require(manifest["recording_identity"] == scenario.recording_identity, "Wrong prediction flight")
    require(manifest["recording_sha256"] == bank["provenance"]["recording_sha256"], "Wrong prediction recording bytes")
    predictions = root / "predictions" / item["flight"]
    require(protocol["prediction_manifest_sha256"] == sha(predictions / "manifest.json"), "Changed prediction manifest")
    actions = {action.action_id: action for action in case["actions"]}
    for action in actions.values():
        require(manifest["models"][action.source_config_id]["signature"] == action.source_signature,
                "Selected neural semantics differ")
    budget = item["perception_budget_s"]
    profile = DeadlineProfile(f"perception_{budget:g}s", ns(budget), 150_000_000, 30_000_000)
    require(protocol["deadline_profile"] == profile.to_dict(), "Changed invocation budgets")
    require(protocol["downstream_selection"] == asdict(DOWNSTREAM), "Changed fixed downstream stages")
    policy = make_policy(item["policy"], case["actions"], case["costs"], profile,
                         case["profiles"]["recall"], case["profiles"]["risk"], case["learned_profile"])
    pipeline = DeadlinePipeline(profile=profile, terrain=terrain,
                                step_ns=scenario.schedule.step_ns,
                                battery_capacity_j=scenario.battery_capacity_wh * 3600,
                                flight_power_w=scenario.flight_power_w)
    scorer = MissionEvaluator(scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s)
    reporter = RunnerReportEvaluator(scenario.matching_iou_threshold, scenario.targets.values(), 2)
    masks = {action_id: CountedMasks(predictions, manifest["models"][action.source_config_id])
             for action_id, action in actions.items()}
    indices = bank["opportunity_indices"]
    require(protocol["opportunity_indices"] == indices, "Changed opportunity receipt")
    clock = scenario.schedule
    capacity = scenario.battery_capacity_wh * 3600
    costs, trace = case["costs"], case["trace"]
    position = finish = 0
    nonflight = communication = peak_memory = 0.0
    resident = None
    skipped_all = []
    read_count = 0

    def battery(at_ns, extra=0.0):
        return max(0.0, item["battery"] - (at_ns / 1e9 * scenario.flight_power_w + nonflight + extra) / capacity)

    for event in events:
        require(position < len(indices), "Extra event outside declared opportunities")
        index = indices[position]
        capture = index * clock.interval_ns
        require(event["observation_id"] == index, "Repeated, omitted, or busy observation")
        require(event["capture_time_s"] == capture / 1e9, "Capture time was renumbered")
        require(capture >= finish, "Policy observed RGB before previous work finished")
        require(battery(capture) > 0, "Task dispatched after battery depletion")
        frame, payload = bank["frames"][index], bank["payloads"][index]
        require(event["capture_timestamp_ns"] == frame.timestamp_ns == clock.epoch_ns + capture,
                "Physical RGB timestamp differs")
        require(event["capture_body_pose"] == frame.body_pose, "Capture pose was substituted")
        require(event["source_rgb_key"] == payload.rgb_sha256, "Selected RGB identity differs")
        decision_ns = capture + ns(costs.encode_s)
        observed_battery = battery(decision_ns, costs.encode_s * costs.codec_power_w)
        near(event["battery_at_decision"], observed_battery, "Uncharged transit or preparation before decision")
        require(event["public_link"] == trace.snapshot(decision_ns).to_dict(), "Policy observed wrong current link")
        expected_decision = policy.select(
            release_ns=capture, current_action_id=resident, battery_frac=observed_battery,
            cumulative_communication_mb=communication, link=trace.snapshot(decision_ns),
            payload_bytes=len(payload.payload), response_bytes=payload.response_bytes,
            intent=contract.public_document(), rgb=frame.rgb)
        require(event["router_decision"] == expected_decision, "Public-only frozen policy decision differs")
        action_id = expected_decision["action_id"]
        require(event["selected_action_id"] == action_id, "Selected action differs from public decision")
        action = actions[action_id] if action_id is not None else None
        cache = masks[action_id] if action is not None else None
        previous_reads = len(cache.reads) if cache is not None else 0
        task = execute_perception(
            action=action, rgb=frame.rgb, prepared=payload, predictions=cache,
            release_ns=capture, profile=profile, mission_deadline_ns=ns(scenario.contract.deadline_s),
            current_action_id=resident, trace=trace, costs=costs,
            router_latency_ns=ns(scenario.router_latency_s), router_energy_j=scenario.router_energy_j,
            step_ns=clock.step_ns)
        require(event["perception"] == task.event, "Paid perception receipt differs from frozen inputs")
        require(event["perception_completed_ns"] == task.completed_ns, "Perception availability differs")
        check_deadline(event["perception"], capture, profile.perception_ns, clock.step_ns)
        if event["perception"]["transport"] is not None:
            audited_bytes, audited_radio = check_transport(event["perception"]["transport"], plan["policy_costs"])
            near(audited_bytes, task.communication_mb, "Transport byte ledger differs")
            near(audited_radio, task.event["costs"]["radio_energy_j"], "Transport radio ledger differs")
        new_reads = len(cache.reads) - previous_reads if cache is not None else 0
        require(new_reads == int(task.mask is not None), "Canceled or abstained task read a prediction")
        read_count += new_reads
        expected_hash = hashlib.sha256(task.mask.tobytes()).hexdigest() if task.mask is not None else None
        require(event["selected_mask_sha256"] == expected_hash, "Selected prediction bytes differ")
        require(event["source_config_id"] == (action.source_config_id if action else None), "Changed base model identity")
        nonflight += task.nonflight_energy_j
        communication += task.communication_mb
        resident = task.resident_action_id
        finish = task.completed_ns
        output, reports, reason = None, [], None
        if task.mask is not None and battery(finish) > 0:
            resource = scenario.resources.for_config(action_id)
            output = pipeline.execute(
                DOWNSTREAM,
                CompletedPrediction(action.source_config_id, capture, finish, task.mask,
                    PublicPose(tuple(frame.body_pose["position_m"]), tuple(frame.body_pose["quaternion_xyzw"]),
                               scenario.world.get("camera_pitch_rad", math.pi / 2))),
                public_context(contract, capture, finish, battery(finish), resource.memory_mb))
            nonflight += output["charged_energy_j"]
            finish = output["completed_ns"]
            reports = output["emitted_reports"]
            memory = [stage["costs"]["resident_memory_mb"] for stage in output["stages"] if stage["dispatched"]]
            peak_memory = max([peak_memory, action.memory_mb, *memory])
        elif action is not None:
            reason = "battery_depleted_before_association_release" if task.mask is not None else "perception_output_not_admitted"
            peak_memory = max(peak_memory, action.memory_mb + 68)
        require(event["pipeline"] == output and event["emitted_reports"] == reports,
                "Tracking, verification, or emitted reports differ")
        require(event["downstream_skip_reason"] == reason, "Downstream output gate differs")
        require(event["completion_time_s"] == finish / 1e9, "Usable completion differs")
        truth = evaluation_frame(frame, index, capture / 1e9, scenario)
        predicted = task.mask if task.mask is not None else np.zeros(frame.rgb.shape[:2], dtype=bool)
        require(event["execution"] == dict(success=task.mask is not None, config_id=action_id,
                                           predicted_pixels=int(predicted.sum())), "Execution mask receipt differs")
        require(event["score"] == scorer.update(truth, predicted).to_dict(), "Canonical semantic matching differs")
        reporter.update(truth, predicted, reports)
        position, missed = next_opportunity(indices, position, finish, clock.interval_ns)
        require(event["skipped_observation_ids"] == missed, "Sparse eligible skips differ")
        for missed_id in missed:
            missed_truth = evaluation_frame(bank["frames"][missed_id], missed_id,
                                             missed_id * clock.interval_ns / 1e9, scenario)
            scorer.note_skipped_visibility(missed_truth.visible_target_ids)
            reporter.note_visibility(missed_truth)
        skipped_all.extend(missed)
        near(event["cumulative_nonflight_energy_j"], nonflight, "Nonflight energy ledger differs")
        near(event["cumulative_communication_mb"], communication, "Communication ledger differs")
        near(event["cumulative_energy_j"], finish / 1e9 * scenario.flight_power_w + nonflight,
             "Flight or task energy omitted")
        near(event["battery_after"], battery(finish), "Post-task battery differs")

    termination = "mission_window_complete"
    if position < len(indices):
        next_capture = indices[position] * clock.interval_ns
        if battery(finish) <= 0:
            termination = "battery_depleted"
        elif battery(next_capture) <= 0:
            finish, termination = next_capture, "battery_depleted"
        elif next_capture > ns(scenario.contract.deadline_s):
            finish, termination = next_capture, "deadline_exceeded"
        else:
            raise ValueError("Run stopped while another eligible task was available")
    else:
        require(len(events) + len(skipped_all) == len(indices), "Opportunity accounting is incomplete")
        finish = max(finish, ns(scenario.duration_s))
    if battery(finish) <= 0:
        termination = "battery_depleted"
    require(result["termination_reason"] == termination, "Wrong terminal state")
    require(result["mission_completed"] is (termination == "mission_window_complete"), "Wrong completion flag")
    require(result["final_time_s"] == finish / 1e9, "Flight tail or overshoot time omitted")
    near(result["flight_energy_j"], finish / 1e9 * scenario.flight_power_w, "Flight energy differs")
    near(result["cumulative_nonflight_energy_j"], nonflight, "Final task energy differs")
    near(result["cumulative_energy_j"], finish / 1e9 * scenario.flight_power_w + nonflight, "Final total energy differs")
    near(result["final_battery_frac"], battery(finish), "Final reserve differs")
    near(result["cumulative_communication_mb"], communication, "Final bytes differ")
    near(result["peak_configured_memory_mb"], peak_memory, "Peak memory differs")
    require(result["skipped_observation_ids"] == skipped_all, "Final skipped opportunities differ")
    require(result["unprocessed_opportunity_ids"] == indices[position:], "Unprocessed opportunities differ")
    require(result["processed_observations"] == len(events), "Incorrect decision count")
    require(result["inference_failures"] == sum(not event["execution"]["success"] for event in events),
            "Incorrect admitted-output failure count")
    require(result["eligible_opportunities"] == len(indices), "Incorrect eligible opportunity count")
    require(result["inactive_capture_count"] == (clock.end_ns - clock.epoch_ns) // clock.interval_ns - len(indices),
            "Transit captures were relabeled as tasks")
    require(result["selected_action_counts"] == dict(Counter(event["selected_action_id"] for event in events
                                                             if event["selected_action_id"] is not None)),
            "Selected action counts differ")
    require(result["quality"] == scorer.quality_details(finish / 1e9), "Final perception metrics differ")
    require(result["report_quality"] == reporter.summary(finish / 1e9), "Final report metrics differ")
    expected_flags = dict(memory_constraint_success=peak_memory <= 4096,
        deadline_success=finish / 1e9 <= scenario.contract.deadline_s,
        battery_constraint_success=battery(finish) >= scenario.contract.min_final_battery_frac,
        communication_constraint_success=communication <= scenario.contract.communication_budget_mb,
        privacy_constraint_success=True)
    require(result["constraints"] == expected_flags, "Resource pass/fail differs")
    require(result["recording_identity"] == scenario.recording_identity, "Final result uses another physical flight")
    require(outcome == evaluate_recoverable(contract, events, result), "Canonical mission endpoint differs")
    expected_row = dict(mission_success=outcome["mission_success"],
        successful_phases=sum(phase["success"] for phase in outcome["phases"]),
        report_recall=outcome["reporting"]["recall_by_deadline"],
        final_battery_frac=result["final_battery_frac"], communication_mb=communication,
        decision_count=len(events), skipped_opportunity_count=len(skipped_all),
        eligible_opportunity_count=len(indices), selected_action_counts=result["selected_action_counts"],
        final_time_s=finish / 1e9, flight_energy_j=finish / 1e9 * scenario.flight_power_w,
        nonflight_energy_j=nonflight, total_energy_j=finish / 1e9 * scenario.flight_power_w + nonflight,
        mission_completed=result["mission_completed"], termination_reason=termination,
        task_metrics=outcome["task_metrics"], abstentions=outcome["abstentions"],
        failed_constraints=outcome["failed_constraints"])
    for key, expected in expected_row.items():
        require(row[key] == expected, "Summary metric differs: " + key)
    durations = [(event["perception_completed_ns"] - ns(event["capture_time_s"])) / 1e9
                 for event in events if event["selected_action_id"] is not None]
    latency = dict(count=len(durations), mean_s=float(np.mean(durations)) if durations else None,
        p50_s=float(np.percentile(durations, 50)) if durations else None,
        p95_s=float(np.percentile(durations, 95)) if durations else None,
        max_s=max(durations) if durations else None)
    for key, expected in latency.items():
        require(result["perception_latency"][key] == expected and row["perception_latency"][key] == expected,
                "Latency summary differs: " + key)
    return dict(run_id=item["run_id"], passed=True, events=len(events), cache_reads=read_count,
                canceled_or_abstained_without_cache_read=len(events) - read_count,
                busy_skipped_opportunities=len(skipped_all), full_flight_time_s=finish / 1e9,
                mission_success=outcome["mission_success"])


def _audit_flight(arguments):
    """One isolated process owns a flight's read-only bank and local state."""
    root, plan, flight, indexed, complete, diagnostic_summary = arguments
    terrain, _ = load_public_terrain(Path(plan["assets"]))
    bank = load_bank(root, plan, flight)
    manifest_path = root / "predictions" / flight / "manifest.json"
    manifest = read(manifest_path)
    require(complete["prediction_manifest_sha256"][flight] == sha(manifest_path), "Changed neural cache manifest")
    diagnostic_receipt = None
    if diagnostic_summary is not None:
        from experiments.wide_area_routing.diagnostics import cross_model_bound, verify_saved

        path = root / "model_diagnostics" / (flight + ".json")
        diagnostic = read(path)
        require(diagnostic_summary["case_artifact_sha256"][flight] == sha(path), "Changed model diagnostic receipt")
        require(diagnostic_summary["cases"][flight] == diagnostic, "Diagnostic summary and per-flight receipt differ")
        verify_saved(diagnostic)
        item = next(item for item in plan["grid"] if item["flight"] == flight)
        phases = load_case(root, plan, item)["contract"].document()["mission"]["phases"]
        recomputed = cross_model_bound(phases, {name: value["frame_scores"]
                                              for name, value in diagnostic["models"].items()})
        require(recomputed == diagnostic["optimistic_cross_model_evidence_bound"],
                "Optimistic evidence bound differs from its native-model score receipts")
        diagnostic_receipt = dict(flight=flight, passed=True, source_sha256=sha(path),
            failed_evidence_sectors=len(recomputed["failed_phase_ids"]),
            check="Input/source/mask hashes and bound recomputation from canonical per-model score receipts; "
                  "all-model native scoring is not rerun here.")
    audited = []
    for item in (item for item in plan["grid"] if item["flight"] == flight):
        require(indexed[item["run_id"]] == read(root / "runs" / item["run_id"] / "row.json"),
                "Index row differs from saved cell")
        audited.append(audit_run(root, plan, item, bank, manifest, terrain))
        if len(audited) % 8 == 0:
            print(f"Audited {flight}: {len(audited)} paired replays", flush=True)
    return audited, diagnostic_receipt


def validate_workers(workers):
    require(type(workers) is int and 1 <= workers <= 3, "Use one to three isolated audit workers")


def audit(root, output=None, *, workers=1):
    from experiments.wide_area_routing.study import verify

    validate_workers(workers)
    root = Path(root).resolve()
    plan = verify(root)
    complete = read(root / "completion.json")
    require(complete["complete"] and complete["plan_sha256"] == sha(root / "plan.json"),
            "Complete frozen execution required")
    require(complete["run_index_sha256"] == sha(root / "run_index.json"), "Changed run index")
    rows = read(root / "run_index.json")
    require(len(rows) == len(plan["grid"]) == plan["expected_runs"], "Incomplete paired grid")
    indexed = {row["run_id"]: row for row in rows}
    require(len(indexed) == len(rows) and set(indexed) == {item["run_id"] for item in plan["grid"]},
            "Missing, duplicate, or extra run")
    audited = []
    diagnostic_receipts = []
    diagnostic_summary_path = root / "model_diagnostics.json"
    diagnostic_summary = read(diagnostic_summary_path) if diagnostic_summary_path.exists() else None
    if diagnostic_summary is not None:
        require(diagnostic_summary["complete"] and diagnostic_summary["plan_sha256"] == sha(root / "plan.json"),
                "Diagnostic summary belongs to another or incomplete study")
    before = {name: sha(root / name) for name in ("plan.json", "run_index.json", "completion.json")}
    arguments = [(root, plan, flight, indexed, complete, diagnostic_summary) for flight in plan["cases"]]
    if workers == 1:
        outputs = [_audit_flight(argument) for argument in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            outputs = [future.result() for future in as_completed(
                [pool.submit(_audit_flight, argument) for argument in arguments])]
    for cases, diagnostic_receipt in outputs:
        audited.extend(cases)
        if diagnostic_receipt is not None:
            diagnostic_receipts.append(diagnostic_receipt)
    order = {item["run_id"]: index for index, item in enumerate(plan["grid"])}
    audited.sort(key=lambda row: order[row["run_id"]])
    diagnostic_receipts.sort(key=lambda row: row["flight"])
    require(len(audited) == len(order) and {row["run_id"] for row in audited} == set(order),
            "Incomplete or duplicate audit shard")
    require(all(sha(root / name) == value for name, value in before.items()), "Study changed during audit")
    report = dict(schema_version="wide_area_independent_receipt_audit_v1",
        inspected_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        audit_workers=workers, process_isolation="One flight per process; all artifacts are read-only until the final combined audit receipt",
        plan_sha256=sha(root / "plan.json"), audit_source_sha256=sha(__file__),
        run_index_sha256=sha(root / "run_index.json"), verified_runs=len(audited),
        verified_events=sum(row["events"] for row in audited),
        verified_selected_cache_reads=sum(row["cache_reads"] for row in audited),
        verified_canceled_or_abstained_without_cache_reads=sum(row["canceled_or_abstained_without_cache_read"] for row in audited),
        uses_runner=False, reruns_neural_inference=False, repairs_original_artifacts=False,
        independent_checks="Original physical identity and capture indices; sparse scheduling and full-flight energy arithmetic; "
                           "exact current link, frozen public-only policy decision, selected-mask bytes and model identity; "
                           "paid deadline and cancellation receipts; no dropped-output cache reads; canonical scores and mission endpoint.",
        reused_primitives="Existing policies, network transport, execute_perception, DeadlinePipeline, and canonical evaluators "
                          "are rerun from immutable inputs. Their internal algorithms are not independently reimplemented.",
        model_diagnostic_receipts=diagnostic_receipts,
        model_diagnostic_summary_sha256=sha(diagnostic_summary_path) if diagnostic_summary is not None else None,
        cases=audited)
    save(Path(output) if output else root / "independent_audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Frozen study directory")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = audit(args.output, args.report, workers=args.workers)
    print(f"Verified {result['verified_runs']} runs / {result['verified_events']} events", flush=True)
