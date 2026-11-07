"""Independent completeness, terminal state and summary checks for local replays.

This study fixes stride one, local execution and nonbinding configured memory.
Early battery/deadline termination is retained when independently justified;
an unexplained prefix of an execution is never accepted as a complete replay.
"""

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


def check_execution(scenario, events, result, row, canonical):
    """Reconstruct paid chronology from fixed source costs; consume no GT labels."""
    require(bool(events), "Empty execution cannot verify this positive-battery cohort")
    schedule = scenario.schedule
    tracking = DEFAULT_TRACKING_COST
    require(result["tracking_cost"] == asdict(tracking), "Changed fixed tracking costs")
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
