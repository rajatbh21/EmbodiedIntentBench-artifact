"""Paid selected perception with a deadline measured from capture availability.

Preparation, routing, loading, service, transport, decoding and tick rounding all
consume the same invocation budget. Local execution finishes and discards late
outputs. Virtual transport can stop at a deadline, retaining actual bytes and
radio cost. The independently configured transport timeout is not a task budget.
Only an admitted selected result may read a cached prediction. No free fallback
or retry is performed. Flight energy is paid by the replay clock, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from experiments.mission_network_emulation import account_radio_energy, emulate_job
from experiments.mission_network_policies import ActionDescriptor, PolicyCosts
from experiments.mission_network_study_v1.executors import PreparedPayload
from experiments.mission_task_deadlines.contract import (
    DeadlineProfile,
    TaskDeadline,
    TaskKind,
    assess_completion,
)
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


@dataclass(frozen=True)
class PerceptionResult:
    mask: np.ndarray | None
    completed_ns: int
    nonflight_energy_j: float
    communication_mb: float
    resident_action_id: str | None
    event: dict


def execute_perception(
    *,
    action: ActionDescriptor | None,
    rgb,
    prepared: PreparedPayload,
    predictions,
    release_ns: int,
    profile: DeadlineProfile,
    mission_deadline_ns: int,
    current_action_id: str | None,
    trace,
    costs: PolicyCosts,
    router_latency_ns: int,
    router_energy_j: float,
    step_ns: int = 10_000_000,
) -> PerceptionResult:
    if not isinstance(profile, DeadlineProfile) or not isinstance(costs, PolicyCosts):
        raise ValueError("Typed deadline profile and policy costs required")
    request = TaskDeadline(TaskKind.PERCEPTION, release_ns, profile.perception_ns)
    for name, value in (
        ("mission deadline", mission_deadline_ns),
        ("routing latency", router_latency_ns),
        ("physics step", step_ns),
    ):
        if type(value) is not int or value < (1 if name == "physics step" else 0):
            raise ValueError(f"Invalid {name}")
    if (
        type(router_energy_j) not in (int, float)
        or not math.isfinite(router_energy_j)
        or router_energy_j < 0
    ):
        raise ValueError("Finite nonnegative routing energy required")
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) <= 0
    ):
        raise ValueError("RGB uint8 input required")
    if (
        not isinstance(prepared, PreparedPayload)
        or prepared.rgb_sha256 != rgb_key(rgb)
        or not isinstance(prepared.payload, bytes)
        or not prepared.payload
        or type(prepared.response_bytes) is not int
        or prepared.response_bytes != (rgb.shape[0] * rgb.shape[1] + 7) // 8
    ):
        raise ValueError("Prepared payload must belong to the current RGB")
    if action is not None and not isinstance(action, ActionDescriptor):
        raise ValueError("Typed selected action or explicit abstention required")
    if current_action_id is not None and (
        not isinstance(current_action_id, str) or not current_action_id
    ):
        raise ValueError("Resident action identity must be a nonempty string")

    prep_ns = seconds_to_ns(costs.encode_s, allow_zero=True)
    switched = action is not None and action.action_id != current_action_id
    load_ns = seconds_to_ns(action.load_latency_s, allow_zero=True) if switched else 0
    load_j = action.load_energy_j if switched else 0.0
    decision_ns = release_ns + prep_ns
    dispatch_ns = decision_ns + router_latency_ns + load_ns
    charges = dict(
        preparation_latency_ns=prep_ns,
        preparation_energy_j=costs.encode_s * costs.codec_power_w,
        routing_latency_ns=router_latency_ns,
        routing_energy_j=router_energy_j,
        load_latency_ns=load_ns,
        load_energy_j=load_j,
        compute_energy_j=0.0,
        decode_latency_ns=0,
        decode_energy_j=0.0,
        radio_energy_j=0.0,
    )
    raw_finish_ns, transport, radio = dispatch_ns, None, None
    communication_mb = 0.0
    output_arrived, deadline_canceled = False, False
    status = "abstained"
    stop_reason, bound = None, None
    pre_dispatch_budget_exhausted = False
    if action is not None:
        if action.placement == "local":
            # Local service is non-preemptive even when loading spent the budget.
            raw_finish_ns += seconds_to_ns(action.latency_s, allow_zero=True)
            charges["compute_energy_j"] = action.energy_j
            output_arrived, status = True, "completed"
        else:
            bound = min(request.deadline_ns, mission_deadline_ns)
            stop_reason = (
                "task_deadline_miss"
                if request.deadline_ns <= mission_deadline_ns
                else "mission_deadline_exceeded"
            )
            # Every prepared request has a positive upload/response. Starting
            # exactly at its inclusive bound leaves no time to transfer either.
            if dispatch_ns >= bound:
                status = stop_reason
                deadline_canceled = status == "task_deadline_miss"
                pre_dispatch_budget_exhausted = True
            else:
                job = emulate_job(
                    trace,
                    start_ns=dispatch_ns,
                    upload_bits=len(prepared.payload) * 8,
                    download_bits=prepared.response_bytes * 8,
                    service_ns=seconds_to_ns(action.latency_s, allow_zero=True),
                    timeout_ns=seconds_to_ns(costs.timeout_s),
                    deadline_ns=bound,
                )
                transport = job.to_dict()
                radio = account_radio_energy(
                    job,
                    upload_power_w=costs.tx_power_w,
                    download_power_w=costs.rx_power_w,
                    idle_power_w=costs.wait_power_w,
                )
                charges["radio_energy_j"] = radio["configured_radio_j"]
                communication_mb = float(job.uploaded_bits + job.downloaded_bits) / 8e6
                output_arrived, raw_finish_ns = job.prediction_eligible, job.finish_ns
                status = {
                    "deadline": stop_reason,
                    "timeout": "transport_timeout",
                }.get(job.status, job.status)
                deadline_canceled = status == "task_deadline_miss"
                if output_arrived:
                    charges["decode_latency_ns"] = seconds_to_ns(costs.decode_s, allow_zero=True)
                    charges["decode_energy_j"] = costs.decode_s * costs.codec_power_w
                    raw_finish_ns += charges["decode_latency_ns"]
    assessment = assess_completion(request, raw_finish_ns, step_ns)
    deadline_missed = deadline_canceled or not assessment.on_time
    if action is not None and not assessment.on_time:
        status = "task_deadline_miss"
    elif action is not None and assessment.usable_finish_ns > mission_deadline_ns:
        status = "mission_deadline_exceeded"
    admitted = action is not None and output_arrived and status == "completed"
    mask = None
    if admitted:
        selected = predictions[prepared.rgb_sha256]
        if selected.dtype != np.bool_ or selected.shape != rgb.shape[:2]:
            raise ValueError("Invalid selected cached prediction")
        if action.placement == "remote":
            packed = np.packbits(selected.reshape(-1), bitorder="big")
            restored = np.unpackbits(packed, bitorder="big")[: selected.size].reshape(
                selected.shape
            )
            if len(packed) != prepared.response_bytes or not np.array_equal(restored, selected):
                raise ValueError("Lossless selected response parity failed")
        # Immutable backing bytes prevent state consumers from editing the cache.
        mask = np.frombuffer(selected.tobytes(), dtype=np.bool_).reshape(selected.shape)
    energy_j = sum(v for k, v in charges.items() if k.endswith("energy_j"))
    charges["charged_energy_j"] = energy_j
    charges["availability_rounding_ns"] = assessment.usable_finish_ns - raw_finish_ns
    event = dict(
        **assessment.to_dict(),
    )
    event.update(
        status=status,
        deadline_status=assessment.status,
        deadline_missed=deadline_missed,
        task_deadline_met=admitted and assessment.on_time,
        admitted=admitted,
        output_arrived=output_arrived,
        action_id=action.action_id if action else None,
        source_config_id=action.source_config_id if action else None,
        placement=action.placement if action else None,
        source_signature=action.source_signature if action else None,
        decision_ns=decision_ns,
        dispatch_ns=dispatch_ns,
        reset_slot=switched,
        costs=charges,
        communication_mb=communication_mb,
        transport=transport,
        radio=radio,
        deadline_stop_reason=stop_reason
        if (
            pre_dispatch_budget_exhausted
            or (transport is not None and transport["status"] == "deadline")
        )
        else None,
        transport_bound_ns=bound,
        transport_deadline_constraint=stop_reason,
        network_dispatched=transport is not None,
        pre_dispatch_budget_exhausted=pre_dispatch_budget_exhausted,
        prediction_cache_read=admitted,
        provenance="cached real inference; virtual transport; configured onboard energy",
    )
    return PerceptionResult(
        mask,
        assessment.usable_finish_ns,
        energy_j,
        communication_mb,
        action.action_id if action is not None else current_action_id,
        event,
    )
