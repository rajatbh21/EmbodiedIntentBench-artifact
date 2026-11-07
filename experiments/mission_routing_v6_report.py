"""Paired fresh-layout results; all predeclared policies and failures remain visible."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v4_report import risk_audit
from experiments.mission_routing_v5_report import paired
from experiments.mission_routing_v6.flights import verify
from experiments.mission_routing_v6.study import POLICIES


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
    if plan["source_sha256"] != sha(Path(__file__).with_name("mission_routing_v6_witness.py")):
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


def validate_grid(rows, protocol):
    expected = {
        (f, m, n) for f in protocol["cases"] for m in protocol["missions"] for n in POLICIES
    }
    keys = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete or duplicated predeclared policy grid")


def classify(*, latency_impossible, policy_success, paid_witness):
    if latency_impossible and (policy_success or paid_witness):
        raise ValueError("Feasibility certificate contradicts successful paid execution")
    if latency_impossible:
        return "latency_certified_infeasible"
    if policy_success or paid_witness:
        return "full_contract_witnessed_feasible"
    return "unresolved"


def run(root):
    p = verify(root)
    validation = read(root / "validation.json")
    if (
        not validation["complete"]
        or validation["cases"] != 90
        or validation["protocol_sha256"] != sha(root / "protocol.json")
        or validation["validator_sha256"]
        != sha(Path(__file__).with_name("mission_routing_v6") / "validate.py")
    ):
        raise ValueError("All 90 paid runs must be independently validated")
    sensor = read(root / "sensor_gate.json")
    diagnostic = read(root / "feasibility_diagnostics.json")
    if (
        not sensor["passed"]
        or diagnostic["protocol_sha256"] != sha(root / "protocol.json")
        or diagnostic["diagnostic_implementation_sha256"]
        != sha(Path(__file__).with_name("mission_routing_v2_diagnostics.py"))
    ):
        raise ValueError("Missing or changed sensor/latency diagnostics")
    witnesses = witness_audit(root, p)
    if witnesses is None:
        raise ValueError("Complete evaluator-only witness inventory required")
    witness_rows = {
        (r["flight"], r["mission"]): r for r in read(root / "witnesses/summary.json")["cases"]
    }
    rows, hashes, schedules = [], {}, {}
    for flight in p["cases"]:
        filename = f"summary_{flight}_15.json"
        if sha(root / filename) != validation["source_summaries_sha256"][filename]:
            raise ValueError("Changed independently validated summary")
        summary = read(root / filename)
        if not summary["complete"] or summary["protocol_sha256"] != sha(root / "protocol.json"):
            raise ValueError("Wrong study summary")
        for row in summary["cases"]:
            destination = root / row["path"]
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                if sha(destination / name) != row[key]:
                    raise ValueError("Changed paid replay artifact")
            events = read(destination / "execution/events.json")
            schedules[flight, row["mission"], row["policy"]] = [
                (e["observation_id"], e["execution"]["config_id"], e["completion_time_s"])
                for e in events
            ]
            result = read(destination / "execution/results.json")
            contract = IntentContract(read(root / flight / row["mission"] / "intent.json"))
            outcome = evaluate(contract, events, result)
            expected = dict(
                full_success=outcome["mission_success"],
                phases_succeeded=sum(v["success"] for v in outcome["phases"]),
                report_recall=outcome["reporting"]["recall_by_deadline"],
                false_reports=outcome["reporting"]["false_reports"],
                duplicate_reports=outcome["reporting"]["duplicate_reports"],
                final_battery=result["final_battery_frac"],
                calls=len(events),
                failed_constraints=outcome["failed_constraints"],
                selections=dict(Counter(e["execution"]["config_id"] for e in events)),
            )
            if any(row[k] != v for k, v in expected.items()):
                raise ValueError("Incorrect outcome summary")
            if any(result[k] for k in ("router_errors", "inference_failures", "invalid_actions")):
                raise ValueError("Execution failure cannot be scored as a policy outcome")
            risk = None
            if (
                events
                and (events[0].get("router_decision") or {}).get("risk_ledger_is_observed_error")
                is False
            ):
                risk = risk_audit(events, contract.document()["mission"]["phases"])
            rows.append(
                dict(
                    row,
                    total_false_components=sum(
                        e["score"]["false_positive_components"] for e in events
                    ),
                    correct_targets_reported=outcome["reporting"]["recall_by_deadline"] * 6,
                    phases=outcome["phases"],
                    risk_audit=risk,
                )
            )
        hashes[filename] = sha(root / filename)
    validate_grid(rows, p)
    feasibility = []
    for flight in p["cases"]:
        d = diagnostic["flights"][flight]
        if d["outcomes_sha256"] != sha(root / flight / "diagnostic_outcomes.json"):
            raise ValueError("Changed dense evaluator outcomes")
        for mission in p["missions"]:
            group = [r for r in rows if r["flight"] == flight and r["mission"] == mission]
            feasibility.append(
                dict(
                    flight=flight,
                    mission=mission,
                    conclusion=classify(
                        latency_impossible=d["missions"][mission]["latency_certified_infeasible"],
                        policy_success=any(r["full_success"] for r in group),
                        paid_witness=witness_rows[flight, mission]["full_contract_witness"],
                    ),
                    sensor_bound="feasibility_unresolved",
                )
            )
    primary = [r for r in rows if r["policy"] == p["primary_policy"]]
    resource_contrast = [
        dict(
            flight=flight,
            policy=name,
            same_capture_model_schedule=(
                schedules[flight, "rapid_survey", name] == schedules[flight, "reserve_patrol", name]
            ),
        )
        for flight in p["cases"]
        for name in POLICIES
    ]
    result = dict(
        schema_version="development_selected_fresh_layout_report_v6",
        complete=True,
        scope=p["scope"],
        fresh_layout_evaluation=True,
        unseen_map=False,
        independent_layouts=1,
        paired_lighting_flights=2,
        primary_policy=p["primary_policy"],
        evaluation_labels_used_for_fit=False,
        primary_selected_before_capture=True,
        new_paid_replays=len(rows),
        actual_neural_calls=validation["actual_neural_calls"],
        reproduced_public_decisions=validation["public_decisions_reproduced"],
        cases=rows,
        rapid_reserve_schedule_contrast=resource_contrast,
        feasibility_by_case=feasibility,
        paired_primary_vs={
            name: paired(primary, [r for r in rows if r["policy"] == name])
            for name in POLICIES
            if name != p["primary_policy"]
        },
        successes_by_policy={
            n: sum(r["full_success"] for r in rows if r["policy"] == n) for n in POLICIES
        },
        failure_counts={
            n: dict(Counter(k for r in rows if r["policy"] == n for k in r["failed_constraints"]))
            for n in POLICIES
        },
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        sensor_gate_sha256=sha(root / "sensor_gate.json"),
        diagnostics_sha256=sha(root / "feasibility_diagnostics.json"),
        witness_summary_sha256=sha(root / "witnesses/summary.json"),
        summaries_sha256=hashes,
        reporter_sha256=sha(Path(__file__)),
        limitations=p["caveats"],
    )
    save(root / "analysis_v6.json", result)
    lines = [
        "# Development-selected routing: fresh physical evaluation",
        "",
        "All 15 policies, including the primary (`base_recall`), were frozen before new "
        "physical captures. The same known lake, one new layout, two paired lighting flights, "
        "three intents and the unchanged nine-configuration pool produced 90 paid policy "
        "replays. These are authored simulated missions, not real drone mission logs.",
        "",
        "Both 60-capture physical flights passed the pre-inference sensor screen. "
        f"Actual neural forward calls: {validation['actual_neural_calls']}. "
        "Independently reconstructed public decisions: "
        f"{validation['public_decisions_reproduced']}. "
        "Raw GT matching and identity-consistent report scoring were independently rebuilt.",
        "",
        "## Full mission contract success",
        "",
        "Each policy has six correlated intent/lighting conditions. No statistical "
        "population claim or post-hoc choice of the primary policy is made.",
        "",
        "| Policy | Full success / 6 |",
        "|---|---:|",
    ]
    lines += [f"| {n} | {result['successes_by_policy'][n]} |" for n in POLICIES]
    lines += [
        "",
        "## Every predeclared paid outcome",
        "",
        "Correct reports are unique target reports by deadline out of six. False "
        "components and false confirmed reports are different error measures.",
        "",
        "| Flight / intent | Policy | Full | Phases / 4 | Correct reports / 6 | "
        "False components | False reports | Battery |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['flight']} / {r['mission']} | {r['policy']} | {int(r['full_success'])} | "
            f"{r['phases_succeeded']} | {r['correct_targets_reported']:.0f} | "
            f"{r['total_false_components']} | {r['false_reports']} | {r['final_battery']:.6f} |"
        )
    lines += [
        "",
        "## Feasibility is separate from policy success",
        "",
        "Passing the sensor gate alone leaves feasibility unresolved. Optimistic latency "
        "bounds can certify impossibility; only a successful canonical paid replay establishes "
        "full feasibility. GT-aware witness schedules are never policy results.",
        "",
        "| Flight | Intent | Classification |",
        "|---|---|---|",
    ]
    lines += [f"| {r['flight']} | {r['mission']} | {r['conclusion']} |" for r in feasibility]
    lines += [
        "",
        "## Resource contrast",
        "",
        f"Rapid survey and reserve patrol have identical capture/model schedules in "
        f"{sum(r['same_capture_model_schedule'] for r in resource_contrast)}"
        f"/{len(resource_contrast)} "
        "paired policy/lighting comparisons. They differ in initial battery, but this "
        "cohort does not establish a routing response to that resource contrast. "
        "Do not claim evidence for every exposed runtime input from these results.",
        "",
        "## Limits",
        "",
        *["- " + c for c in p["caveats"]],
        "",
        "Do not infer that more development data guarantees better routing. Compare "
        "expanded and original base recall on these exact same paid conditions. "
        "Any controller redesign informed by this cohort makes future reuse exploratory.",
        "",
    ]
    (root / "REPORT_V6.md").write_text("\n".join(lines))
    print(dict(successes=result["successes_by_policy"], feasibility=feasibility), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
