"""Posthoc inference-power sensitivity of fixed observed schedules, never policy reruns."""

import argparse
import math
from pathlib import Path

POWERS = (15, 16, 18, 20, 25)
REFERENCE_POWER = 15
FIXED_COMPONENTS = (
    "flight_energy_j",
    "router_energy_j",
    "tracking_energy_j",
    "model_load_energy_j",
    "communication_energy_j",
)


def finite(value, label, *, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError("Invalid " + label)
    return value


def close(actual, expected, label, tolerance=1e-6):
    finite(actual, label)
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(label + " differs from configured schedule")


def schedule_energy(scenario, intent, events, runtime, *, full_success):
    """Hold actions, timing and non-inference costs fixed; vary only active power."""
    if type(full_success) is not bool:
        raise ValueError("Explicit reference success required")
    mission = scenario["mission"]
    capacity = finite(mission["battery_capacity_wh"], "battery capacity") * 3600
    if capacity <= 0:
        raise ValueError("Positive battery capacity required")
    initial = finite(mission["initial_battery_frac"], "initial battery")
    reserve = finite(intent["limits"]["min_final_battery_frac"], "reserve")
    if initial > 1 or reserve > 1:
        raise ValueError("Battery fractions must be at most one")
    if (
        intent["mission"]["initial_battery_frac"] != initial
        or intent["mission"]["min_final_battery_frac"] != reserve
    ):
        raise ValueError("Scenario and intent battery settings differ")
    specs = {s["config_id"]: s for s in scenario["executor_configs"]}
    seconds = []
    for event in events:
        execution = event["execution"]
        spec = specs[execution["config_id"]]
        duration = finite(spec["mission_latency_s"], "declared inference duration")
        close(execution["mission_latency_s"], duration, "Selected declared latency", 1e-9)
        close(execution["energy_j"], REFERENCE_POWER * duration, "Selected inference energy")
        seconds.append(duration)
    inference_s = math.fsum(seconds)
    reference_inference = REFERENCE_POWER * inference_s
    close(runtime["inference_energy_j"], reference_inference, "Aggregate inference energy")
    fixed = {key: finite(runtime[key], key) for key in FIXED_COMPONENTS}
    final_time = finite(runtime["final_time_s"], "final time")
    close(
        fixed["flight_energy_j"],
        final_time * finite(mission["flight_power_w"], "flight power"),
        "Flight energy",
    )
    close(
        fixed["router_energy_j"],
        len(events) * finite(mission["router_energy_j"], "router cost"),
        "Router energy",
    )
    for total, event_key in (
        ("tracking_energy_j", "tracking_energy_j"),
        ("model_load_energy_j", "model_load_energy_j"),
    ):
        close(fixed[total], math.fsum(finite(e[event_key], event_key) for e in events), total)
    close(
        fixed["communication_energy_j"],
        math.fsum(finite(e["execution"]["communication_energy_j"], "radio energy") for e in events),
        "Radio energy",
    )
    fixed_j = math.fsum(fixed.values())
    reference_j = fixed_j + reference_inference
    close(runtime["cumulative_energy_j"], reference_j, "Total energy")
    close(
        runtime["final_battery_frac"],
        max(0, initial - reference_j / capacity),
        "Reference final battery",
        1e-8,
    )
    available = (initial - reserve) * capacity
    margin = available - reference_j
    complete = runtime["mission_completed"] is True
    if full_success and (not complete or not events or margin < -1e-8):
        raise ValueError("Reference success lacks a complete reserve-feasible schedule")
    threshold = (available - fixed_j) / inference_s if inference_s > 0 else None
    sensitivity = {}
    for watts in POWERS:
        energy = fixed_j + watts * inference_s
        difference = available - energy
        sensitivity[str(watts)] = dict(
            notional_energy_j=energy,
            reserve_margin_j=difference,
            unclamped_end_battery_frac=initial - energy / capacity,
            reserve_budget_satisfied=difference >= -1e-8,
            reference_success_retaining_reserve=full_success and difference >= -1e-8,
        )
    return dict(
        initial_battery_frac=initial,
        battery_capacity_wh=capacity / 3600,
        battery_capacity_j=capacity,
        required_reserve_frac=reserve,
        usable_energy_above_reserve_j=available,
        full_success_at_reference=full_success,
        complete_mission_at_reference=complete,
        reference_termination_reason=runtime["termination_reason"],
        processed_calls=len(events),
        declared_inference_seconds=inference_s,
        reference_active_inference_power_w=REFERENCE_POWER,
        reference_inference_energy_j=reference_inference,
        non_inference_energy_j=fixed_j,
        non_inference_components_j=fixed,
        reference_total_energy_j=reference_j,
        reference_reserve_margin_j=margin,
        successful_schedule_reserve_breakpoint_w=threshold if full_success else None,
        inference_power_affects_energy=inference_s > 0,
        powers_w=sensitivity,
        schedule_extent="complete_reference_mission" if complete else "observed_failed_prefix_only",
    )


def summarize(rows, policies):
    summaries = []
    for policy in policies:
        selected = [r for r in rows if r["policy"] == policy]
        if not selected:
            raise ValueError("Missing policy in sensitivity denominator")
        successes = [r for r in selected if r["full_success_at_reference"]]
        limits = [
            r["successful_schedule_reserve_breakpoint_w"]
            for r in successes
            if r["successful_schedule_reserve_breakpoint_w"] is not None
        ]
        summaries.append(
            dict(
                policy=policy,
                conditions=len(selected),
                reference_successes=len(successes),
                incomplete_reference_runs=sum(
                    not r["complete_mission_at_reference"] for r in selected
                ),
                min_successful_schedule_reserve_breakpoint_w=min(limits, default=None),
                reference_successes_retaining_reserve={
                    str(w): sum(
                        r["powers_w"][str(w)]["reference_success_retaining_reserve"]
                        for r in selected
                    )
                    for w in POWERS
                },
            )
        )
    return summaries


def run(root):
    from experiments.lake_feasibility_inference import read
    from experiments.lake_intent_contract import IntentContract, evaluate
    from experiments.lake_missions.assets import save, sha
    from experiments.mission_transfer_comparison_v4.audit import complete_grid
    from experiments.mission_transfer_comparison_v4.protocol import verify

    outputs = (root / "energy_robustness.json", root / "ENERGY_ROBUSTNESS.md")
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            "Preserve completed energy sensitivity; choose a new analysis version"
        )
    if not (root / "completion.json").is_file():
        raise ValueError("Complete primary study required before energy sensitivity")
    completion = read(root / "completion.json")
    if completion["complete"] is not True:
        raise ValueError("Complete primary study required before energy sensitivity")
    p = verify(root)
    if p["active_cpu_power_sensitivity_w"] != [REFERENCE_POWER]:
        raise ValueError("Expected the unchanged 15 W reference study")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed completed study artifact: " + name)
    analysis = read(root / "analysis.json")
    if (
        analysis["complete"] is not True
        or analysis["protocol_sha256"] != sha(root / "protocol.json")
        or analysis["routing_admission_sha256"] != p["routing_admission_sha256"]
    ):
        raise ValueError("Analysis does not bind the completed within-recording comparison")
    complete_grid(analysis["policy_cases"], p)
    rows, inputs = [], {}
    for row in analysis["policy_cases"]:
        destination = root / row["path"]
        for name, key in (
            ("execution/events.json", "events_sha256"),
            ("execution/results.json", "results_sha256"),
            ("outcome.json", "outcome_sha256"),
        ):
            path = destination / name
            if sha(path) != row[key]:
                raise ValueError("Changed audited execution: " + str(path))
            inputs[str(path.relative_to(root))] = row[key]
        folder = root / row["flight"] / row["mission"]
        scenario, intent = read(folder / "scenario.json"), read(folder / "intent.json")
        events, runtime = (
            read(destination / "execution/events.json"),
            read(destination / "execution/results.json"),
        )
        outcome = read(destination / "outcome.json")
        canonical = evaluate(IntentContract(intent), events, runtime)
        if (
            canonical["mission_success"] != row["full_success"]
            or canonical["mission_success"] != outcome["mission_success"]
        ):
            raise ValueError("Reference success differs from canonical mission endpoint")
        measured = schedule_energy(
            scenario, intent, events, runtime, full_success=row["full_success"]
        )
        close(row["energy_j"], measured["reference_total_energy_j"], "Published total energy")
        if abs(row["reserve_margin_j"] - measured["reference_reserve_margin_j"]) > 1e-6:
            raise ValueError("Published reserve margin differs from actual capacity and intent")
        if row["battery"] != measured["initial_battery_frac"]:
            raise ValueError("Published initial battery differs")
        for name in ("scenario.json", "intent.json"):
            inputs[str((folder / name).relative_to(root))] = sha(folder / name)
        rows.append(
            dict(
                flight=row["flight"],
                mission=row["mission"],
                policy=row["policy"],
                layout=row["layout"],
                lighting=row["lighting"],
                **measured,
            )
        )
    summaries = summarize(rows, p["policies"])
    batteries = sorted({r["initial_battery_frac"] for r in rows})
    strata = {
        str(b): summarize([r for r in rows if r["initial_battery_frac"] == b], p["policies"])
        for b in batteries
    }
    result = dict(
        schema_version="within_recording_fixed_schedule_energy_v1",
        complete=True,
        posthoc=True,
        cases=rows,
        policy_summary=summaries,
        battery_strata=strata,
        conditions=len(p["cases"]) * len(p["missions"]),
        executions=len(rows),
        powers_w=list(POWERS),
        reference_power_w=REFERENCE_POWER,
        protocol_sha256=sha(root / "protocol.json"),
        completion_sha256=sha(root / "completion.json"),
        analysis_sha256=sha(root / "analysis.json"),
        routing_admission_sha256=p["routing_admission_sha256"],
        inputs_sha256=inputs,
        source_sha256=sha(Path(__file__)),
        scope="Reserve-budget sensitivity of fixed observed schedules, not adaptive policy reruns",
        adaptive_policy_rerun=False,
        neural_inference_repeated=False,
        physics_repeated=False,
        mission_outcomes_under_changed_power_claimed=False,
        primary_results_changed=False,
        policy_selection_on_sensitivity=False,
        measured_onboard_energy=False,
        lighting_causal_effect_identified=False,
        assumptions=[
            "Configurations, capture subsets, declared latencies and final time stay fixed",
            "Flight, router, tracking, model-loading and communication energy stay fixed",
            "Only configured active inference power changes; host timing is never substituted",
            "A reserve-budget failure does not predict the actions an adaptive router would take",
            "A failed prefix cannot establish feasibility of an unfinished mission",
        ],
    )
    lines = [
        "# Fixed-schedule inference-power sensitivity",
        "",
        f"This posthoc diagnostic retains all {len(rows)} runs across {result['conditions']} "
        "conditions. It changes no policy, mission threshold or primary outcome. Each schedule's "
        "capacity, initial charge and required reserve come from its own scenario and intent.",
        "",
        "For each fixed schedule, E(P) = E_fixed + P * T_inference. T_inference sums declared "
        "latencies of selected configurations, including an ensemble as one configuration. "
        "E_fixed retains flight, router, tracking, model-loading and communication costs. "
        "The successful schedule's exact reserve breakpoint is "
        "P* = ((initial - reserve) * capacity - E_fixed) / T_inference.",
        "",
        "The table counts reference-successful schedules that still satisfy the energy reserve at "
        "each assumed inference power. These are not newly simulated mission-success rates. "
        "At larger power an adaptive policy could select different models or terminate earlier; "
        "this calculation keeps its original actions and timing. Failed runs remain in the JSON, "
        "and a truncated failed prefix is never counted as a successful full-mission schedule.",
        "",
        "| Policy | Reference success / conditions | Retain at 15 W | 16 W | 18 W | 20 W | 25 W | "
        "Minimum successful breakpoint (W) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        limit = row["min_successful_schedule_reserve_breakpoint_w"]
        counts = row["reference_successes_retaining_reserve"]
        shown_limit = "—" if limit is None else f"{limit:.6f}"
        lines.append(
            f"| {row['policy']} | {row['reference_successes']}/{row['conditions']} | "
            + " | ".join(str(counts[str(w)]) for w in POWERS)
            + f" | {shown_limit} |"
        )
    lines += [
        "",
        "Per-condition exact breakpoints and every declared battery stratum are included "
        "in `energy_robustness.json`. No policy is selected from this sensitivity analysis. "
        "This remains a configured CPU-power assumption, not measured drone energy. The "
        "scope amendment and both rejected-acquisition histories remain unchanged; lighting "
        "is an observational stratum, not an isolated causal intervention.",
        "",
    ]
    outputs[1].write_text("\n".join(lines))
    save(outputs[0], result)
    print(
        dict(complete=True, executions=len(rows), policies=len(summaries), powers_w=list(POWERS)),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
