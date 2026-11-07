"""Validated paired mission outcomes and evaluator-only risk calibration diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v4.flights import verify
from experiments.mission_routing_v4.policy import MODES
from experiments.mission_routing_v4.study import POLICIES


def witness_audit(root, protocol):
    """Optional evaluator witnesses never enter primary policy execution counts."""
    folder = root / "witnesses"
    if not folder.exists():
        return None
    plan = read(folder / "protocol.json")
    summary = read(folder / "summary.json")
    if not summary["complete"] or summary["protocol_sha256"] != sha(folder / "protocol.json"):
        raise ValueError("Incomplete witness search")
    if plan["protocol_sha256"] != sha(root / "protocol.json") or not plan["gt_aware"]:
        raise ValueError("Wrong witness protocol")
    if plan["source_sha256"] != sha(Path(__file__).with_name("mission_routing_v4_witness.py")):
        raise ValueError("Changed witness implementation")
    if plan["search_sha256"] != sha(Path(__file__).with_name("lake_hindsight_search.py")):
        raise ValueError("Changed witness search")
    rows = summary["cases"]
    keys = [(r["flight"], r["mission"]) for r in rows]
    expected = {(f, m) for f in protocol["cases"] for m in protocol["missions"]}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete witness grid")
    for row in rows:
        destination = folder / row["flight"] / row["mission"]
        if sha(destination / "search.json") != row["search_sha256"]:
            raise ValueError("Changed witness search result")
        if "outcome_sha256" in row:
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                if sha(destination / "replay" / name) != row[key]:
                    raise ValueError("Changed witness replay")
            outcome = read(destination / "replay/outcome.json")
            if not outcome["gt_aware"] or outcome["deployable_policy"]:
                raise ValueError("Evaluator witness was mislabeled as a policy")
            intent = IntentContract(read(root / row["flight"] / row["mission"] / "intent.json"))
            events = read(destination / "replay/execution/events.json")
            result = read(destination / "replay/execution/results.json")
            canonical = evaluate(intent, events, result)
            if any(
                canonical[k] != outcome[k]
                for k in ("mission_success", "failed_constraints", "reporting", "phases")
            ):
                raise ValueError("Witness canonical outcome differs")
            if outcome["mission_success"] != row["full_contract_witness"]:
                raise ValueError("False witness summary")
        elif row["full_contract_witness"]:
            raise ValueError("Full witness has no paid replay")
    return dict(
        gt_aware=True,
        deployable_policy=False,
        excluded_from_policy_grid=True,
        witnessed_cases=sum(r["full_contract_witness"] for r in rows),
        cases=len(rows),
        summary_sha256=sha(folder / "summary.json"),
    )


def risk_audit(events, phases):
    """Reconstruct the expected ledger without treating estimated counts as truth."""
    ledger, rows = {}, []
    changed, fallback = 0, 0
    for event in events:
        d = event["router_decision"]
        if d["risk_ledger_is_observed_error"] or d["uses_ground_truth"]:
            raise ValueError("Risk ledger has invalid provenance")
        if d["risk_ledger_before"] != ledger:
            raise ValueError("Risk ledger discontinuity")
        selected = next(
            c for c in d["candidates"] if c["config_id"] == event["execution"]["config_id"]
        )
        if selected["config_id"] != d["selected_config_id"]:
            raise ValueError("Risk choice differs from execution")
        for phase in phases:
            if phase["start_s"] <= event["capture_time_s"] < phase["end_s"]:
                pid = phase["phase_id"]
                ledger[pid] = ledger.get(pid, 0) + selected["estimated_fp_per_capture"]
        if ledger != d["risk_ledger_after"]:
            raise ValueError("Risk ledger did not charge exactly the selected capture")
        changed += d["selected_config_id"] != d["baseline_selected_config_id"]
        fallback += d["risk_fallback"] is not None
    for p in phases:
        eligible = [e for e in events if p["start_s"] <= e["capture_time_s"] < p["end_s"]]
        observed = sum(e["score"]["false_positive_components"] for e in eligible)
        rows.append(
            dict(
                phase_id=p["phase_id"],
                captures=len(eligible),
                expected_false_components=ledger.get(p["phase_id"], 0),
                observed_false_components=observed,
                max_false_components=p["max_false_components"],
                actual_limit_satisfied=observed <= p["max_false_components"],
            )
        )
    return dict(
        gt_aware=True,
        used_for_routing=False,
        same_state_choice_changes=changed,
        decisions=len(events),
        fallback_decisions=fallback,
        phases=rows,
    )


def run(root):
    p = verify(root)
    validation = read(root / "validation.json")
    if not validation["complete"] or not all(
        validation[k]
        for k in (
            "physical_poses_verified",
            "per_event_clock_energy_memory_verified",
            "prediction_history_rebuilt_without_ground_truth",
            "canonical_outcomes_verified",
        )
    ):
        raise ValueError("Independent validation required")
    rows = []
    for flight in p["cases"]:
        for power in p["active_cpu_power_sensitivity_w"]:
            path = root / f"summary_{flight}_{power}.json"
            summary = read(path)
            if (
                sha(path) != validation["source_summaries_sha256"][path.name]
                or summary["protocol_sha256"] != sha(root / "protocol.json")
                or not summary["complete"]
            ):
                raise ValueError("Changed or incomplete execution summary")
            rows.extend(summary["cases"])
    expected = {
        (f, m, c, w)
        for f in p["cases"]
        for m in p["missions"]
        for c in POLICIES
        for w in p["active_cpu_power_sensitivity_w"]
    }
    actual = [(r["flight"], r["mission"], r["policy"], r["power_w"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected or validation["cases"] != len(rows):
        raise ValueError("Incomplete comparison grid")
    groups, diagnostics = defaultdict(dict), []
    schedules = {}
    for row in rows:
        folder = root / row["path"]
        for name, key in (
            ("outcome.json", "outcome_sha256"),
            ("execution/events.json", "events_sha256"),
            ("execution/results.json", "results_sha256"),
        ):
            if sha(folder / name) != row[key]:
                raise ValueError("Changed execution artifact")
        events = read(folder / "execution/events.json")
        schedules[row["path"]] = [
            (e["observation_id"], e["execution"]["config_id"]) for e in events
        ]
        outcome = read(folder / "outcome.json")
        row["total_false_components"] = sum(p["false_components"] for p in outcome["phases"])
        row["phases_exceeding_fp_limit"] = sum(
            p["false_components"] > p["max_false_components"] for p in outcome["phases"]
        )
        groups[row["flight"], row["mission"], row["power_w"]][row["policy"]] = row
        if row["policy"] in MODES:
            public = read(root / row["flight"] / row["mission"] / "intent.json")["mission"][
                "phases"
            ]
            diagnostics.append(
                dict(
                    flight=row["flight"],
                    mission=row["mission"],
                    policy=row["policy"],
                    power_w=row["power_w"],
                    **risk_audit(events, public),
                )
            )
    ablations = []
    for key, group in groups.items():
        full = group["risk_aware"]
        for name in ("confirmation", "no_risk_ledger_v4", "no_risk_history_v4", "global_risk_v4"):
            other = group[name]
            ablations.append(
                dict(
                    flight=key[0],
                    mission=key[1],
                    power_w=key[2],
                    comparator=name,
                    schedule_changed=schedules[full["path"]] != schedules[other["path"]],
                    success_delta=int(full["full_success"]) - int(other["full_success"]),
                    phases_delta=full["phases_succeeded"] - other["phases_succeeded"],
                    report_recall_delta=full["report_recall"] - other["report_recall"],
                    false_components_delta=full["total_false_components"]
                    - other["total_false_components"],
                    final_battery_delta=full["final_battery"] - other["final_battery"],
                )
            )
    witnesses = witness_audit(root, p)
    feasibility = []
    dense = (
        read(root / "feasibility_diagnostics.json")
        if (root / "feasibility_diagnostics.json").exists()
        else None
    )
    if dense and dense["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Wrong feasibility cohort")
    for (flight, mission, power), group in groups.items():
        impossible = (
            dense["flights"][flight]["missions"][mission]["latency_certified_infeasible"]
            if dense
            else None
        )
        executed_success = any(r["full_success"] for r in group.values())
        if impossible and executed_success:
            raise ValueError("Infeasibility bound contradicts an executed full success")
        feasibility.append(
            dict(
                flight=flight,
                mission=mission,
                power_w=power,
                latency_certified_infeasible=impossible,
                predefined_policy_full_success_exists=executed_success,
            )
        )
    counts = {c: sum(r["full_success"] for r in rows if r["policy"] == c) for c in POLICIES}
    failure_counts = {
        c: dict(Counter(k for r in rows if r["policy"] == c for k in r["failed_constraints"]))
        for c in POLICIES
    }
    summary = dict(
        schema_version="mission_routing_v4_results_v1",
        complete=True,
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        reporter_sha256=sha(Path(__file__)),
        evaluator_witnesses=witnesses,
        feasibility_by_case=feasibility,
        executions=len(rows),
        cases_per_policy=len(groups),
        physical_flights=len(p["cases"]),
        independent_layouts=1,
        successes_by_policy=counts,
        failed_constraint_counts_by_policy=failure_counts,
        cases=rows,
        paired_comparisons=ablations,
        evaluator_only_risk_diagnostics=diagnostics,
        interpretation="Descriptive paired known-map cases, not independent mission samples",
    )
    save(root / "analysis_v4.json", summary)
    lines = [
        "# Prospective false-component risk routing",
        "",
        f"{len(rows)} independently validated paid replays on two new Gazebo flights. "
        "One shared new layout, two paired lighting conditions, three unchanged intents. "
        "All 14 policies use exactly the same physical flight within each condition. "
        "These correlated cases are not a population success estimate.",
        "",
        "Risk was fitted only on the original 356 development captures (3,204 frame/config "
        "pairs). No v2/v3 evaluation labels were used for fitting. Earlier inspected results "
        "motivated this new method; the fresh cohort was frozen before rendering or inference.",
        "",
        "| Policy | Full contracts satisfied |",
        "|---|---:|",
        *(f"| {c} | {counts[c]}/{len(groups)} |" for c in POLICIES),
        "",
        "## Paired full controller vs frozen v3",
        "",
        "| Flight / intent | Full success v4 / v3 | Phases v4 / v3 | "
        "False components v4 / v3 | Report recall v4 / v3 |",
        "|---|---:|---:|---:|---:|",
    ]
    for (flight, mission, _), group in groups.items():
        a, b = group["risk_aware"], group["confirmation"]
        lines.append(
            f"| {flight} / {mission} | {a['full_success']} / {b['full_success']} | "
            f"{a['phases_succeeded']} / {b['phases_succeeded']} | "
            f"{a['total_false_components']} / {b['total_false_components']} | "
            f"{a['report_recall']:.3f} / {b['report_recall']:.3f} |"
        )
    lines += [
        "",
        "## Expected risk vs evaluator-observed errors",
        "",
        "Expected false components are a causal model ledger, never observed truth. "
        "Observed counts below are evaluator-only diagnostics after execution. Both include "
        "late outputs in the capture phase. False confirmed reports are a separate unit.",
        "",
        "| Flight / intent | Expected / observed components | "
        "Choices changed vs same-state v3 | Fallback decisions |",
        "|---|---:|---:|---:|",
    ]
    for d in diagnostics:
        if d["policy"] == "risk_aware":
            expected = sum(p["expected_false_components"] for p in d["phases"])
            observed = sum(p["observed_false_components"] for p in d["phases"])
            lines.append(
                f"| {d['flight']} / {d['mission']} | {expected:.3f} / {observed} | "
                f"{d['same_state_choice_changes']}/{d['decisions']} | {d['fallback_decisions']} |"
            )
    if dense is not None:
        lines += [
            "",
            "## Fixed-flight feasibility strata",
            "",
            "| Flight / intent | Infeasible under optimistic serial timing? | "
            "Any predefined full success? |",
            "|---|---:|---:|",
        ]
        for f in feasibility:
            lines.append(
                f"| {f['flight']} / {f['mission']} | {f['latency_certified_infeasible']} | "
                f"{f['predefined_policy_full_success_exists']} |"
            )
        lines += [
            "",
            "Certified-infeasible cases remain in the complete frozen grid but cannot "
            "show which router completes the mission. The bound ignores energy, false "
            "positives, switching and reporting; failure even under this relaxation "
            "certifies inability to meet the phase evidence requirement for this pool, "
            "flight, capture interval and costs. A passing bound alone is inconclusive.",
        ]
    if witnesses is not None:
        lines += [
            "",
            "## Evaluator-only feasibility",
            "",
            f"Full paid-contract witnesses: {witnesses['witnessed_cases']}/{witnesses['cases']}. "
            "These GT-aware schedules are excluded from the 84 policy executions. "
            "A found witness establishes feasibility under the configured assumptions; "
            "a failed beam search is inconclusive, not proof of impossibility.",
        ]
    lines += [
        "",
        "## Limits",
        "",
        "The ridge count estimate and add-one floor are heuristics, not calibrated chance "
        "constraints or confidence bounds. Future image quality and risk are approximated "
        "using current RGB. Risk admission projects a constant candidate's phase cadence; "
        "resource admission separately checks a cheapest mixed continuation. Joint mission "
        "feasibility is not proven. No report-correctness predictor was added.",
        "",
        "The textured map is known and the corridor is nearby earlier cohorts. Missions "
        "contain authored static people and ideal pose/DEM. They are not real UAV mission "
        "logs. CPU energy assumes 15 W; flight, memory, switching, router and tracking costs "
        "remain configured. Strong fixed baselines may outperform adaptation.",
        "",
    ]
    (root / "REPORT_V4.md").write_text("\n".join(lines))
    print(dict(executions=len(rows), successes_by_policy=counts), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
