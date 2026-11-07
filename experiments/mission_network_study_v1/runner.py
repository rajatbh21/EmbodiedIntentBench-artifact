"""Versioned paid network replay; original tracking runtime remains immutable.

Preparation precedes policy observation; only delivered predictions reach tracking.
Virtual link timestamps use mission origin; physical timestamps retain their epoch.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from missionbench.schemas.runtime_state import EvidenceSummary, RuntimeState
from missionbench.simulator.action_validator import ActionOutcome, ActionValidator
from missionbench.v2.evaluation import MissionEvaluator
from missionbench.v2.executors import ImageExecutionResult, ImageExecutor
from experiments.lake_tracking_runtime import ZERO_TRACKING_COST, TrackingCost
from experiments.mission_network_policies import AttemptOutcome
from experiments.mission_network_study_v1.executors import prepare_payload
from experiments.v4_benchmark.backends import Frame, WorldBackend, save_json
from experiments.v4_benchmark.reporting import ReportEvaluator, ReportTracker
from experiments.v4_benchmark.routers import Router, RouterInput, immutable_rgb
from experiments.v4_benchmark.scenario import Scenario, digest, exported_source
from experiments.v4_gazebo_pilot.timing import seconds_to_ns

FROZEN_RUNNER_SHA256 = "75513809d0ea11a93a15d6ec0e3683822913b663a63aca3c73e6373bb5b39fc9"


@dataclass(frozen=True)
class EvaluationFrame:
    """The fields consumed by the shared evaluator, without a fabricated 2D footprint."""

    observation_id: int
    scheduled_capture_time_s: float
    semantic_gt: np.ndarray
    instance_gt: np.ndarray
    visible_target_ids: tuple[str, ...]
    provenance: dict[str, Any]


def evaluation_frame(
    frame: Frame, index: int, time_s: float, scenario: Scenario
) -> EvaluationFrame:
    ids = set(int(v) for v in np.unique(frame.object_ids)) - {0}
    known = {obj["label"] for obj in scenario.world["objects"]}
    if not ids <= known:
        raise ValueError(f"unmapped object labels: {ids - known}")
    target_ids = scenario.targets
    return EvaluationFrame(
        index,
        time_s,
        np.isin(frame.object_ids, list(target_ids)).astype(np.uint8),
        frame.object_ids,
        tuple(sorted(target_ids[i] for i in ids if i in target_ids)),
        {"instance_object_ids": {str(k): v for k, v in target_ids.items()}},
    )


def validate_result(result: ImageExecutionResult, config_id: str, shape: tuple[int, ...]) -> None:
    if result.config_id != config_id:
        raise ValueError("executor returned a different config_id")
    if result.prediction_mask.shape != shape or result.prediction_mask.dtype != np.bool_:
        raise ValueError("executor must return a bool mask at input resolution")
    for key in ("mission_latency_s", "energy_j", "communication_energy_j", "communication_mb"):
        value = getattr(result, key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"executor returned invalid {key}")


def run_mission(
    scenario: Scenario,
    backend: WorldBackend,
    router: Router,
    executors: dict[str, ImageExecutor],
    *,
    router_name: str,
    output: Path | None = None,
    decision_stride: int = 1,
    tracker=None,
    report_scoring=None,
    tracking_cost: TrackingCost = ZERO_TRACKING_COST,
    trace=None,
    costs=None,
) -> dict[str, Any]:
    """Return a complete result, including explicit error status if a component fails.

    Callers own/close the backend. Executors and routers are fresh per mission.
    Resource budgets are evaluation constraints; comm overspend is not silently blocked.
    Battery exhaustion and deadline stop new work. In-flight results keep full costs.
    """
    if (tracker is None) != (report_scoring is None):
        raise ValueError("Tracking and report evaluation must be supplied together")
    if type(decision_stride) is not int or decision_stride < 1:
        raise ValueError("decision_stride must be a positive integer")
    if set(executors) != set(scenario.catalog.ids()):
        raise ValueError("executor pool must exactly match the scenario catalog")
    if trace is None or costs is None:
        raise ValueError("Explicit private link trace and declared network costs required")
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    schedule = scenario.schedule
    epoch = schedule.epoch_ns
    if backend.time_ns != epoch:
        raise ValueError("backend must start at the mission epoch")
    evaluator = MissionEvaluator(
        scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
    )
    report_tracker = ReportTracker(scenario.reporting) if tracker is None else tracker
    report_evaluator = (
        ReportEvaluator(scenario.matching_iou_threshold, len(scenario.targets))
        if report_scoring is None
        else report_scoring
    )
    validator = ActionValidator(
        catalog=scenario.catalog,
        allowed_config_ids=scenario.catalog.ids(),
        privacy_level=scenario.contract.privacy_level,
        fallback_config_id=scenario.fallback_config_id,
    )
    capacity_j = scenario.battery_capacity_wh * 3600
    initial_energy_j = scenario.initial_battery_frac * capacity_j
    energy_j = communication_mb = radio_j = 0.0
    compute_j = routing_j = load_j = tracking_j = preparation_j = 0.0
    expected_timeouts = 0
    last_attempt = None
    load_time_s = peak_memory_mb = 0.0
    memory_ok = True
    mission_time_s = 0.0
    current_config = None
    events, skipped_ids = [], []
    cadence_skipped_ids, busy_skipped_ids = [], []
    invalid_actions = privacy_violations = inference_failures = predicted_detections = 0
    router_errors = 0
    termination = "mission_window_complete"
    error = None
    index = 0

    def battery() -> float:
        return max(0.0, (initial_energy_j - energy_j) / capacity_j)

    def advance(absolute_ns: int) -> None:
        nonlocal mission_time_s, energy_j
        backend.advance_to(absolute_ns)
        new_time_s = (absolute_ns - epoch) / 1e9
        energy_j += (new_time_s - mission_time_s) * scenario.flight_power_w
        mission_time_s = new_time_s

    try:
        while schedule.at(index) < schedule.end_ns:
            capture_ns = schedule.at(index)
            capture_s = (capture_ns - epoch) / 1e9
            if battery() <= 0:
                termination = "battery_depleted"
                break
            if (
                mission_time_s > scenario.contract.deadline_s
                or capture_s > scenario.contract.deadline_s
            ):
                if mission_time_s < scenario.contract.deadline_s:
                    stop_ns = epoch + seconds_to_ns(scenario.contract.deadline_s)
                    stop_ns = (
                        (stop_ns + schedule.step_ns - 1) // schedule.step_ns
                    ) * schedule.step_ns
                    advance(stop_ns)
                termination = "deadline_exceeded"
                break
            advance(capture_ns)
            if battery() <= 0:
                termination = "battery_depleted"
                break
            frame = backend.capture()
            if frame.timestamp_ns != capture_ns:
                raise ValueError("backend returned stale or future observation")
            report_evaluator.note_visibility(evaluation_frame(frame, index, capture_s, scenario))
            if index % decision_stride:
                # Fixed sampling schedule, independent of RGB, labels or outcomes.
                # Sensor/flight continue; only evaluator diagnostics inspect skipped GT.
                evaluator.note_skipped_visibility(
                    evaluation_frame(frame, index, capture_s, scenario).visible_target_ids
                )
                cadence_skipped_ids.append(index)
                skipped_ids.append(index)
                index += 1
                continue
            # Deterministic causal payload preparation is common to every action.
            prepared = prepare_payload(frame.rgb)
            prep_s = costs.encode_s
            prep_j = prep_s * costs.codec_power_w
            decision_ns = capture_ns + seconds_to_ns(prep_s, allow_zero=True)
            advance(decision_ns)
            energy_j += prep_j
            preparation_j += prep_j
            decision_s = (decision_ns - epoch) / 1e9
            public_link = trace.snapshot(decision_ns - epoch)
            router.set_public_context(public_link, prepared, last_attempt)
            # No ground-truth-derived evidence enters this frozen observation.
            state = RuntimeState(
                current_time_s=decision_s,
                frame_id=index,
                battery_frac=battery(),
                power_mode="v4_configured",
                network=scenario.network.observation_at(decision_s),
                current_config_id=current_config,
                remaining_deadline_s=scenario.contract.deadline_s - decision_s,
                cumulative_energy_j=energy_j,
                cumulative_communication_mb=communication_mb,
                path_progress=min(1.0, decision_s / scenario.duration_s),
                evidence_summary=EvidenceSummary(
                    predicted_unique_targets=0,  # Unavailable: no predicted track identity.
                    processed_frames=len(events),
                    mean_prediction_confidence=0.0,
                ),
            )
            public = RouterInput(
                scenario.contract,
                state,
                scenario.catalog,
                immutable_rgb(frame.rgb),
                tuple(frame.body_pose["position_m"]),
                tuple(frame.body_pose["quaternion_xyzw"]),
                predicted_detections,
                scenario.requirements,
                scenario.resources,
                report_tracker.report_count,
            )
            started = time.perf_counter()
            router_error = None
            try:
                requested = router.select(public)
            except Exception as exc:
                requested = None
                router_errors += 1
                router_error = f"{type(exc).__name__}: {exc}"
            measured_router_s = time.perf_counter() - started
            action = validator.validate(requested, current_config_id=current_config)
            invalid_actions += int(not action.is_valid)
            privacy_violations += int(action.outcome is ActionOutcome.PRIVACY_VIOLATION)
            model_resources = scenario.resources.for_config(action.config_id)
            if (
                scenario.resources.memory_capacity_mb is not None
                and model_resources.memory_mb + tracking_cost.memory_mb
                > scenario.resources.memory_capacity_mb
            ):
                memory_ok = False
                termination = "memory_capacity_exceeded"
                energy_j += scenario.router_energy_j
                routing_j += scenario.router_energy_j
                stop_ns = decision_ns + seconds_to_ns(scenario.router_latency_s, allow_zero=True)
                advance(((stop_ns + schedule.step_ns - 1) // schedule.step_ns) * schedule.step_ns)
                break
            loading = current_config != action.config_id
            transition_s = model_resources.load_latency_s if loading else 0.0
            transition_j = model_resources.load_energy_j if loading else 0.0
            peak_memory_mb = max(
                peak_memory_mb, model_resources.memory_mb + tracking_cost.memory_mb
            )
            executor = executors[action.config_id]
            dispatch_ns = decision_ns + seconds_to_ns(
                scenario.router_latency_s + transition_s, allow_zero=True
            )
            result = executor.run_selected(
                frame.rgb.copy(),
                prepared,
                dispatch_ns - epoch,
                seconds_to_ns(scenario.contract.deadline_s),
            )
            validate_result(result, action.config_id, frame.rgb.shape[:2])
            if (
                not scenario.catalog.get(action.config_id).strategy.is_remote
                and result.communication_mb
            ):
                raise ValueError("local executor reported remote communication")
            inference_failures += int(not result.success)
            attempt_status = result.diagnostics["attempt_status"]
            delivered = result.success
            expected_timeouts += int(not delivered and attempt_status in ("timeout", "deadline"))
            paid_tracking_s = tracking_cost.latency_s if delivered else 0.0
            paid_tracking_j = tracking_cost.energy_j if delivered else 0.0
            latency_ns = seconds_to_ns(
                prep_s
                + scenario.router_latency_s
                + transition_s
                + result.mission_latency_s
                + paid_tracking_s,
                allow_zero=True,
            )
            completion_ns, next_index, skipped = schedule.complete(index, latency_ns)
            # Hidden GT for missed frames is used only for missed-visibility diagnostics.
            for missed in skipped:
                advance(schedule.at(missed))
                missed_frame = backend.capture()
                expected_ns = schedule.at(missed)
                if missed_frame.timestamp_ns != expected_ns:
                    raise ValueError("stale skipped-frame diagnostic")
                report_evaluator.note_visibility(
                    evaluation_frame(missed_frame, missed, (expected_ns - epoch) / 1e9, scenario)
                )
                evaluator.note_skipped_visibility(
                    evaluation_frame(
                        missed_frame,
                        missed,
                        (expected_ns - epoch) / 1e9,
                        scenario,
                    ).visible_target_ids
                )
            advance(completion_ns)
            energy_j += result.energy_j + result.communication_energy_j + scenario.router_energy_j
            energy_j += transition_j + paid_tracking_j
            tracking_j += paid_tracking_j
            load_j += transition_j
            load_time_s += transition_s
            radio_j += result.communication_energy_j
            compute_j += result.energy_j
            routing_j += scenario.router_energy_j
            communication_mb += result.communication_mb
            prediction = (
                result.prediction_mask if result.success else np.zeros_like(result.prediction_mask)
            )
            score = evaluator.update(
                evaluation_frame(frame, index, capture_s, scenario), prediction
            )
            reports = []
            if delivered:
                reports = report_tracker.update(
                    prediction,
                    capture_s,
                    mission_time_s,
                    pose=frame.body_pose,
                    camera_pitch_rad=scenario.world.get("camera_pitch_rad", math.pi / 2),
                )
                report_evaluator.update(
                    evaluation_frame(frame, index, capture_s, scenario), prediction, reports
                )
            last_attempt = AttemptOutcome(
                action.config_id, capture_s, mission_time_s, delivered, attempt_status
            )
            predicted_detections += score.predicted_components
            skipped_ids.extend(skipped)
            busy_skipped_ids.extend(skipped)
            current_config = action.config_id
            summary = result.to_summary()
            event = {
                "observation_id": index,
                "capture_time_s": capture_s,
                "completion_time_s": mission_time_s,
                "capture_timestamp_ns": capture_ns,
                "configured_router_latency_s": scenario.router_latency_s,
                "configured_tracking_latency_s": paid_tracking_s,
                "tracking_energy_j": paid_tracking_j,
                "model_load_latency_s": transition_s,
                "model_load_energy_j": transition_j,
                "resident_memory_mb": model_resources.memory_mb + tracking_cost.memory_mb,
                "effective_latency_s": (completion_ns - capture_ns) / 1e9,
                "action": action.to_dict(),
                "router_error": router_error,
                "router_decision": (
                    getattr(router, "last_decision", None) if router_error is None else None
                ),
                "runtime_at_decision": state.to_dict(),
                "decision_time_s": decision_s,
                "decision_timestamp_ns": decision_ns,
                "dispatch_timestamp_ns": dispatch_ns,
                "dispatch_network_ns": dispatch_ns - epoch,
                "preparation_latency_s": prep_s,
                "preparation_energy_j": prep_j,
                "prepared_payload": prepared.summary(),
                "public_link": public_link.to_dict(),
                "attempt_status": attempt_status,
                "delivered_prediction": delivered,
                "source_config_id": result.diagnostics["source_config_id"],
                "placement": result.diagnostics["placement"],
                "transport": result.diagnostics["transport"],
                "capture_body_pose": frame.body_pose,
                "battery_after": battery(),
                "cumulative_energy_j": energy_j,
                "cumulative_communication_mb": communication_mb,
                "execution": summary,
                "score": score.to_dict(),
                "emitted_reports": reports,
                "skipped_observation_ids": list(skipped),
                "measured_router_wall_clock_s": measured_router_s,
            }
            events.append(event)
            if output is not None:
                np.savez_compressed(
                    output / f"observation_{index:05d}.npz",
                    rgb=frame.rgb,
                    object_ids=frame.object_ids,
                    prediction_mask=prediction,
                )
                save_json(output / "events.json", events)
            index = next_index
        if termination == "mission_window_complete":
            # Charge the tail after the last inference, including when no frame lands there.
            if battery() <= 0:
                termination = "battery_depleted"
            elif mission_time_s < scenario.duration_s:
                advance(schedule.end_ns)
                if battery() <= 0:
                    termination = "battery_depleted"
    except Exception as exc:
        termination = "component_error"
        error = f"{type(exc).__name__}: {exc}"

    quality = evaluator.quality_details(mission_time_s)
    report_quality = report_evaluator.summary(mission_time_s)
    extra_constraints, first_report_s = scenario.requirements.evaluate(
        quality, events, report_quality
    )
    constraints = {
        **extra_constraints,
        "memory_constraint_success": memory_ok,
        "quality_success": scenario.contract.quality_satisfied(
            quality[scenario.contract.quality_metric]
        ),
        "deadline_success": mission_time_s <= scenario.contract.deadline_s,
        "battery_constraint_success": battery() >= scenario.contract.min_final_battery_frac,
        "communication_constraint_success": communication_mb
        <= scenario.contract.communication_budget_mb,
        "privacy_constraint_success": privacy_violations == 0,
    }
    configs = [event["action"]["config_id"] for event in events]
    metrics = {
        "result_schema_version": "network_tracking_runtime_v1",
        "preparation_energy_j": preparation_j,
        "expected_transport_timeouts": expected_timeouts,
        "tracking_cost": asdict(tracking_cost),
        "tracking_energy_j": tracking_j,
        "tracking": getattr(report_tracker, "provenance", {"association": "legacy"}),
        "scenario_id": scenario.scenario_id,
        "router": router_name,
        "contract": asdict(scenario.contract),
        "mission_requirements": scenario.requirements.to_dict(),
        "first_correct_report_s": first_report_s,
        "status": "error" if error else "completed",
        "error": error,
        "termination_reason": termination,
        "mission_completed": termination == "mission_window_complete",
        "mission_success": termination == "mission_window_complete" and all(constraints.values()),
        "constraints": constraints,
        "quality": quality,
        "report_quality": report_quality,
        "reporting": scenario.reporting.to_dict(),
        "final_time_s": mission_time_s,
        "final_battery_frac": battery(),
        "cumulative_energy_j": energy_j,
        "cumulative_communication_mb": communication_mb,
        "communication_energy_j": radio_j,
        "processed_observations": len(events),
        "decision_stride": decision_stride,
        "decision_period_s": decision_stride * scenario.observation_interval_s,
        "cadence_skipped_observation_ids": cadence_skipped_ids,
        "busy_skipped_observation_ids": busy_skipped_ids,
        "flight_energy_j": mission_time_s * scenario.flight_power_w,
        "inference_energy_j": compute_j,
        "router_energy_j": routing_j,
        "model_load_energy_j": load_j,
        "model_load_latency_s": load_time_s,
        "peak_configured_memory_mb": peak_memory_mb,
        "resources": scenario.resources.to_dict(),
        "skipped_observation_ids": skipped_ids,
        "invalid_actions": invalid_actions,
        "privacy_violations": privacy_violations,
        "inference_failures": inference_failures,
        "router_errors": router_errors,
        "config_selection_history": configs,
        "config_switches": sum(a != b for a, b in pairwise(configs)),
        "scenario_sha256": digest(scenario.source),
        "evaluation_protocol_sha256": digest(
            {
                "version": "network_tracking_runtime_v1",
                "network_costs": asdict(costs),
                "tracking_cost": asdict(tracking_cost),
                "tracking": getattr(report_tracker, "provenance", {"association": "legacy"}),
                "reporting": scenario.reporting.to_dict(),
                "resources": scenario.resources.to_dict(),
                "decision_stride": decision_stride,
                "matching_iou_threshold": scenario.matching_iou_threshold,
            }
        ),
        "contract_sha256": digest(asdict(scenario.contract)),
        "mission_contract_sha256": digest(
            {"contract": asdict(scenario.contract), "requirements": scenario.requirements.to_dict()}
        ),
        "backend": backend.provenance,
        "provenance": {
            "evaluation_purpose": "integration_diagnostic",
            "perception": "RGB-dependent executor; see scenario kinds and event provenance",
            "latency": "Common prep + router + load + placement + delivered tracking; tick rounded",
            "tracking_costs": "Host warm p95; assumed 1 J/64 MB; no electrical measurement",
            "energy": (
                "Assumed flight, preparation, compute, radio, router, load and tracking energy"
            ),
            "residency": (
                "optional single resident config; unload before load, including cold start; "
                "host executors may be preloaded"
            ),
            "network": "Current post-prep snapshot; private trace integrates in-flight transfer",
            "network_clock": "Mission-origin ns = physical ns minus recording epoch",
            "timeout": "8s transport/service from dispatch; delivered unpack outside timeout",
            "failure": "Paid recoverable cancellation; no output/tracking/report/fallback",
            "targets": (
                "rendered static meshes; not real aerial human perception"
                if any("mesh" in obj for obj in scenario.world["objects"])
                else "procedural boxes; not aerial human perception"
            ),
            "telemetry": "ideal simulator odometry; no target GT provided to router",
            "path_progress": "elapsed fraction of fixed mission window; not measured path length",
            "unavailable_evidence": (
                "unique predicted tracks and confidence unavailable; legacy fields zero"
            ),
            "warmup": "excluded; initial battery is specified at mission start",
            "host_time": "diagnostic event fields only; never drives mission state",
        },
    }
    if output is not None:
        save_json(output / "results.json", metrics)
        save_json(output / "events.json", events)
        save_json(output / "contract.json", {"schema_version": "1.0", **asdict(scenario.contract)})
        save_json(output / "scenario.json", exported_source(scenario))
    return metrics
