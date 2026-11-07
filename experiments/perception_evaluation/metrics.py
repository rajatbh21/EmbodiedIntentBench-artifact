"""Supplemental perception-routing diagnostics; the frozen intent remains primary.

This module reads completed replay receipts. It neither changes actions nor
rescoring rules. All energy is configured onboard energy; cloud-server energy
and measured electrical energy are unavailable. Readiness is reconstructed
retrospectively and does not authorize early flight termination.
"""

from __future__ import annotations

import math
from dataclasses import asdict

from experiments.lake_intent_contract import IntentContract, canonical
from experiments.lake_missions.objectives import enough_evidence
from experiments.mission_task_deadlines.contract import (
    DeadlineProfile,
    TaskDeadline,
    TaskKind,
    assess_completion,
)
from experiments.mission_task_deadlines.study import DOWNSTREAM, evaluate_recoverable

ENERGY_COMPONENTS = ("preparation", "routing", "load", "compute", "decode", "radio")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, name):
    require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        "Invalid nonnegative finite " + name,
    )
    return value


def close(left, right, name):
    number(left, name)
    number(right, name)
    require(math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-7), "Inconsistent " + name)


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def quantile(values, probability):
    """Linear interpolation at (n - 1) * probability, with no empty-set zero."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _report_readiness(intent, result):
    requirement = intent["reporting"]
    if requirement["mode"] == "evidence_only":
        return dict(
            report_quota_required=False,
            report_quota_reached=None,
            report_quota_time_s=None,
            report_quota_censored=None,
            report_quota_observation_end_s=None,
        )
    horizon = min(result["final_time_s"], requirement["deadline_s"])
    seen, reached_at = set(), None
    for report in sorted(result["report_quality"]["reports"], key=lambda r: r["completion_time_s"]):
        if report["completion_time_s"] > horizon:
            break
        if report["target_id"] is not None:
            seen.add(report["target_id"])
        if len(seen) / len(intent["evaluator_target_ids"]) >= requirement["min_report_recall"]:
            reached_at = report["completion_time_s"]
            break
    return dict(
        report_quota_required=True,
        report_quota_reached=reached_at is not None,
        report_quota_time_s=reached_at,
        report_quota_censored=reached_at is None,
        report_quota_observation_end_s=horizon,
    )


def _phase_readiness(intent, events, outcome):
    """First recall-quota availability under the unchanged phase clock.

    Later false outputs can invalidate a phase, so this is a recall diagnostic.
    Its success-conditioned counterpart below additionally requires the complete
    original endpoint, including every final false-output and resource gate.
    """
    rows = []
    for phase, scored in zip(intent["mission"]["phases"], outcome["phases"], strict=True):
        evidence = {target: set() for target in phase["evaluator_target_ids"]}
        ready_at = None
        for event in sorted(events, key=lambda e: e["completion_time_s"]):
            if not (
                phase["start_s"] <= event["capture_time_s"] < phase["end_s"]
                and event["completion_time_s"] <= phase["deadline_s"]
                and event["execution"]["success"]
            ):
                continue
            for target in set(event["score"]["matched_target_ids"]) & evidence.keys():
                evidence[target].add(event["capture_time_s"])
            supported = sum(enough_evidence(times, phase) for times in evidence.values())
            if supported / len(evidence) >= phase["required_recall"]:
                ready_at = event["completion_time_s"]
                break
        rows.append(
            dict(
                phase_id=phase["phase_id"],
                recall_quota_ready_time_s=ready_at,
                recall_quota_reached=ready_at is not None,
                final_phase_success=scored["success"],
                evidence_deadline_s=phase["deadline_s"],
            )
        )
    times = [row["recall_quota_ready_time_s"] for row in rows]
    return rows, max(times) if all(time is not None for time in times) else None


def evaluate_run(intent: dict, protocol: dict, events: list, result: dict, outcome: dict) -> dict:
    """Validate one frozen receipt bundle, then derive supplemental diagnostics.

    Deadline rates count selected perception attempts. Abstentions are decisions
    with real preparation/routing charges, but are not perception attempts.
    Skipped-capture rates use only explicitly accounted capture IDs; no unseen
    tail or nominal schedule is invented. Latency includes failed selected work.
    """
    contract = IntentContract(intent)
    require(protocol["contract_sha256"] == contract.sha256, "Changed intent binding")
    require(
        protocol["public_contract_sha256"] == contract.public_sha256,
        "Changed public intent binding",
    )
    require(
        protocol["case"]["mission"] == intent["mission"]["mission_id"],
        "Changed mission identity",
    )
    close(
        protocol["case"]["initial_battery_frac"],
        intent["mission"]["initial_battery_frac"],
        "initial battery binding",
    )
    profile = DeadlineProfile.from_dict(protocol["deadline_profile"])
    require(protocol["downstream_selection"] == asdict(DOWNSTREAM), "Changed fixed downstream")
    expected = evaluate_recoverable(contract, events, result)
    require(canonical(expected) == canonical(outcome), "Changed canonical intent outcome")
    number(result["final_time_s"], "charged duration")
    number(result["final_battery_frac"], "final battery")
    require(result["final_battery_frac"] <= 1, "Invalid final battery fraction")

    energy = dict.fromkeys(ENERGY_COMPONENTS, 0.0)
    perception_energy = downstream_energy = communication = 0.0
    observations, skipped, latencies = [], [], []
    admitted = misses = attempts = abstentions = 0
    for event in events:
        observation_id = event["observation_id"]
        require(type(observation_id) is int and observation_id >= 0, "Invalid capture ID")
        observations.append(observation_id)
        for skipped_id in event["skipped_observation_ids"]:
            require(type(skipped_id) is int and skipped_id >= 0, "Invalid skipped capture ID")
            skipped.append(skipped_id)
        perception = event["perception"]
        require(perception["task"] == "perception", "Changed perception task kind")
        require(perception["budget_ns"] == profile.perception_ns, "Changed per-call budget")
        assessment = assess_completion(
            TaskDeadline(TaskKind.PERCEPTION, perception["release_ns"], perception["budget_ns"]),
            perception["raw_finish_ns"],
            perception["step_ns"],
        )
        for field, value in assessment.to_dict().items():
            if field != "status":
                require(perception[field] == value, "Changed perception clock: " + field)
        require(
            perception["deadline_status"] == assessment.status,
            "Changed perception deadline assessment",
        )
        require(
            perception["action_id"] == event["selected_action_id"],
            "Changed selected action receipt",
        )
        require(
            perception["task_deadline_met"] is (perception["admitted"] and assessment.on_time),
            "Changed admitted deadline status",
        )
        require(
            perception["deadline_missed"]
            is (
                not assessment.on_time or perception["deadline_stop_reason"] == "task_deadline_miss"
            ),
            "Changed deadline miss classification",
        )
        close(event["capture_time_s"], perception["release_ns"] / 1e9, "capture release")
        require(
            event["perception_completed_ns"] == perception["usable_finish_ns"],
            "Changed perception availability",
        )
        number(event["completion_time_s"], "event completion")
        require(
            perception["usable_finish_ns"] / 1e9
            <= event["completion_time_s"]
            <= result["final_time_s"],
            "Invalid event completion horizon",
        )
        if event["selected_action_id"] is None:
            abstentions += 1
            require(not perception["admitted"], "Abstention cannot admit perception output")
        else:
            attempts += 1
            admitted += int(perception["admitted"])
            misses += int(perception["deadline_missed"])
            latencies.append((perception["usable_finish_ns"] - perception["release_ns"]) / 1e9)
        costs = perception["costs"]
        component_sum = 0.0
        for component in ENERGY_COMPONENTS:
            value = number(costs[component + "_energy_j"], component + " energy")
            energy[component] += value
            component_sum += value
        close(component_sum, costs["charged_energy_j"], "perception receipt energy")
        perception_energy += component_sum
        communication += number(perception["communication_mb"], "communication")
        pipeline = event["pipeline"]
        if pipeline is not None:
            require(pipeline["selection"] == asdict(DOWNSTREAM), "Changed downstream selection")
            require(
                pipeline["deadline_profile"] == profile.name, "Changed downstream deadline profile"
            )
            stage_energy = 0.0
            for stage in pipeline["stages"]:
                kind = TaskKind(stage["task"])
                require(kind != TaskKind.PERCEPTION, "Unexpected perception in fixed downstream")
                require(stage["budget_ns"] == profile.budget_for(kind), "Changed downstream budget")
                selected_id = asdict(DOWNSTREAM)[kind.value + "_id"]
                require(stage["algorithm_id"] == selected_id, "Changed downstream algorithm")
                stage_energy += number(
                    stage["costs"]["charged_energy_j"], "downstream stage energy"
                )
            close(stage_energy, pipeline["charged_energy_j"], "downstream receipt energy")
            downstream_energy += stage_energy
        close(
            event["cumulative_nonflight_energy_j"],
            perception_energy + downstream_energy,
            "cumulative receipt energy",
        )
        close(event["cumulative_communication_mb"], communication, "cumulative communication")

    require(len(observations) == len(set(observations)), "Repeated processed capture ID")
    require(len(skipped) == len(set(skipped)), "Repeated skipped capture ID")
    require(not set(observations) & set(skipped), "Processed and skipped capture overlap")
    require(skipped == result["skipped_observation_ids"], "Changed skipped capture ledger")
    require(result["processed_observations"] == len(events), "Changed processed capture count")
    close(
        perception_energy + downstream_energy,
        result["cumulative_nonflight_energy_j"],
        "total nonflight energy",
    )
    close(
        result["flight_energy_j"] + result["cumulative_nonflight_energy_j"],
        result["cumulative_energy_j"],
        "total onboard energy",
    )
    close(communication, result["cumulative_communication_mb"], "total communication")
    require(attempts == outcome["task_metrics"]["perception"]["attempts"], "Changed attempt count")
    require(
        misses == outcome["task_metrics"]["perception"]["deadline_misses"], "Changed miss count"
    )

    phases = outcome["phases"]
    weights = sum(phase["weight"] for phase in phases)
    false_components = sum(event["score"]["false_positive_components"] for event in events)
    matched = sum(event["score"]["matched_components"] for event in events)
    predictions = sum(event["score"]["predicted_components"] for event in events)
    require(matched + false_components == predictions, "Inconsistent detection counts")
    readiness = _report_readiness(intent, result)
    phase_readiness, phase_time = _phase_readiness(intent, events, outcome)
    successful_ready = None
    if outcome["mission_success"]:
        require(phase_time is not None, "Successful mission lacks phase readiness")
        successful_ready = phase_time
        if readiness["report_quota_required"]:
            require(readiness["report_quota_reached"], "Successful mission lacks report quota")
            successful_ready = max(successful_ready, readiness["report_quota_time_s"])
    reporting = outcome["reporting"]
    values = dict(
        schema_version="perception_supplemental_metrics_v1",
        mission_success=outcome["mission_success"],
        phase_completion_rate=sum(phase["success"] for phase in phases) / len(phases),
        weighted_phase_completion_rate=sum(phase["weight"] * phase["success"] for phase in phases)
        / weights,
        weighted_evidence_recall=sum(phase["weight"] * phase["evidence_recall"] for phase in phases)
        / weights,
        normalized_intent_recall_attainment=sum(
            phase["weight"] * min(1.0, phase["evidence_recall"] / phase["required_recall"])
            for phase in phases
        )
        / weights,
        report_recall=reporting.get("recall_by_deadline"),
        false_reports=reporting.get("false_reports"),
        duplicate_reports=reporting.get("duplicate_reports"),
        raw_false_components=false_components,
        false_components_per_mission_minute=ratio(false_components * 60, intent["duration_s"]),
        detection_precision=ratio(matched, predictions),
        perception_attempts=attempts,
        perception_admitted=admitted,
        perception_abstentions=abstentions,
        perception_deadline_misses=misses,
        perception_deadline_miss_rate=ratio(misses, attempts),
        perception_admission_rate=ratio(admitted, attempts),
        perception_abstention_rate=ratio(abstentions, len(events)),
        perception_latency_p50_s=quantile(latencies, 0.5),
        perception_latency_p95_s=quantile(latencies, 0.95),
        skipped_capture_count=len(skipped),
        accounted_capture_count=len(observations) + len(skipped),
        skipped_capture_fraction=ratio(len(skipped), len(observations) + len(skipped)),
        energy_total_j=result["cumulative_energy_j"],
        energy_flight_j=result["flight_energy_j"],
        energy_nonflight_j=result["cumulative_nonflight_energy_j"],
        energy_perception_j=perception_energy,
        energy_downstream_j=downstream_energy,
        battery_reserve_margin=result["final_battery_frac"]
        - intent["limits"]["min_final_battery_frac"],
        communication_mb=communication,
        nominal_flight_duration_s=intent["duration_s"],
        charged_duration_s=result["final_time_s"],
        required_phase_evidence_ready_time_s=phase_time,
        successful_mission_evidence_ready_time_s=successful_ready,
        successful_mission_evidence_ready_event=successful_ready is not None,
        successful_mission_evidence_ready_censored=successful_ready is None,
        successful_mission_evidence_ready_observation_end_s=result["final_time_s"],
        phase_readiness=phase_readiness,
        failed_constraints=outcome["failed_constraints"],
        **readiness,
    )
    values.update({"perception_" + name + "_energy_j": value for name, value in energy.items()})
    values["semantics"] = dict(
        primary="Unchanged recoverable intent endpoint, including all final "
        "resource/report/phase gates.",
        partial_progress="Diagnostic only; recall attainment does not forgive "
        "false outputs or resource failures.",
        latency="Capture release to usable perception completion; selected attempts "
        "include misses and failures.",
        capture_denominator="Union of explicit processed/skipped IDs; no unobserved tail inferred.",
        false_component_rate="Raw false components per nominal flight minute; diagnostic only.",
        energy="Configured onboard energy including paid failed work; "
        "cloud-server energy unavailable.",
        phase_clock="Original event completion after fixed downstream processing, "
        "not perception-only finish.",
        report_quota="Diagnostic first timely unique-correct report quota; "
        "does not imply mission success.",
        readiness="Retrospective evidence-ready time, conditional on eventual full "
        "mission success; no early stop.",
        censoring="A null time is no qualifying event observed by its horizon, "
        "not a zero-time success. "
        "No independent censoring assumption or survival estimator is claimed.",
    )
    return values
