"""Audit denominators, feasibility strata and fixed-schedule energy slack without new runs."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_missions.assets import save, sha
from experiments.mission_strategy_routing.protocol import verify
from experiments.mission_strategy_routing.report import checked_events

STRESS_FRACTIONS = (0.0, 0.01, 0.05)
FEASIBILITY = ("full_contract_witnessed_feasible", "unresolved", "sensor_infeasible")


def index_complete(rows, conditions, policies):
    """Reject missing, duplicate and extra conditions instead of silently changing denominators."""
    indexed = {}
    expected = {(f, m, p) for f, m in conditions for p in policies}
    for row in rows:
        key = row["flight"], row["mission"], row["policy"]
        if key in indexed:
            raise ValueError("Duplicate policy condition")
        if type(row["full_success"]) is not bool:
            raise ValueError("Boolean canonical success required")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("Incomplete or extra comparison grid")
    return indexed


def energy_slack(initial, reserve, capacity_wh, consumed_j, final):
    values = (initial, reserve, capacity_wh, consumed_j, final)
    if not all(math.isfinite(x) for x in values) or capacity_wh <= 0 or consumed_j < 0:
        raise ValueError("Finite, physically valid energy accounting required")
    if not 0 <= reserve <= initial <= 1:
        raise ValueError("Invalid initial or reserve fraction")
    capacity = capacity_wh * 3600
    expected_final = initial - consumed_j / capacity
    if not math.isclose(expected_final, final, abs_tol=1e-8, rel_tol=0):
        raise ValueError("Stored final battery does not match energy accounting")
    margin = (initial - reserve) * capacity - consumed_j
    return dict(
        reserve_margin_j=margin,
        max_relative_total_energy_overrun=margin / consumed_j if consumed_j else None,
        fixed_schedule_stress=[
            dict(relative_overrun=f, reserve_preserved=margin + 1e-6 >= f * consumed_j)
            for f in STRESS_FRACTIONS
        ],
    )


def summarize(protocol, rows, feasibility):
    conditions = {(f, m) for f in protocol["cases"] for m in protocol["missions"]}
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    expected_policies = list(protocol["policies"]) + list(protocol["fixed_policies"])
    if set(policies) != set(expected_policies) or len(expected_policies) != len(
        set(expected_policies)
    ):
        raise ValueError("Changed declared policy pool")
    indexed = index_complete(rows, conditions, policies)
    status = {}
    for item in feasibility:
        key = item["case"], item["mission"]
        if key in status or item["classification"] not in FEASIBILITY:
            raise ValueError("Duplicate or unknown feasibility status")
        if item["classification"] == FEASIBILITY[0] and not item["witness_success"]:
            raise ValueError("Feasible witness classification without a successful witness")
        if item["classification"] != FEASIBILITY[0] and item["witness_success"]:
            raise ValueError("Successful witness contradicts feasibility classification")
        status[key] = item["classification"]
    if set(status) != conditions:
        raise ValueError("Incomplete feasibility coverage")
    # A successful canonical policy replay is itself a feasibility witness.
    promoted = []
    for key in conditions:
        success = any(indexed[*key, p]["full_success"] for p in policies)
        if success and status[key] == "sensor_infeasible":
            raise ValueError("Successful execution contradicts certified sensor infeasibility")
        if success and status[key] == "unresolved":
            status[key] = "observed_policy_feasible"
            promoted.append(dict(flight=key[0], mission=key[1]))
    summaries = []
    for policy in expected_policies:
        selected = [indexed[f, m, policy] for f, m in sorted(conditions)]
        strata = {}
        for label in (*FEASIBILITY, "observed_policy_feasible"):
            subset = [r for r in selected if status[r["flight"], r["mission"]] == label]
            strata[label] = dict(
                cases=len(subset), successes=sum(r["full_success"] for r in subset)
            )
        summaries.append(
            dict(
                policy=policy,
                cases=len(selected),
                successes=sum(r["full_success"] for r in selected),
                feasibility_strata=strata,
                missions={
                    m: dict(
                        cases=sum(r["mission"] == m for r in selected),
                        successes=sum(r["full_success"] for r in selected if r["mission"] == m),
                    )
                    for m in protocol["missions"]
                },
            )
        )
    lineages = [c["lineage"] for c in protocol["cases"].values()]
    units = {
        name: len({x[name] for x in lineages})
        for name in ("recording", "layout", "route", "scenery", "world")
    }
    pairs = []
    for opponent in [p for p in expected_policies if p != "mission_survival"]:
        groups = {}
        for f, m in sorted(conditions):
            a, b = indexed[f, m, "mission_survival"], indexed[f, m, opponent]
            recording = protocol["cases"][f]["lineage"]["recording"]
            group = groups.setdefault(recording, dict(conditions=0, wins=0, losses=0, ties=0))
            delta = int(a["full_success"]) - int(b["full_success"])
            group["conditions"] += 1
            group["wins" if delta > 0 else "losses" if delta < 0 else "ties"] += 1
        pairs.append(
            dict(
                opponent=opponent,
                by_recording=groups,
                condition_wins=sum(g["wins"] for g in groups.values()),
                condition_losses=sum(g["losses"] for g in groups.values()),
                recordings_with_any_win=sum(g["wins"] > 0 for g in groups.values()),
                recordings_with_any_loss=sum(g["losses"] > 0 for g in groups.values()),
            )
        )
    return dict(
        policy_summary=summaries,
        comparison_conditions=len(conditions),
        lineage_units=units,
        paired_comparisons=pairs,
        observed_feasibility_promotions=promoted,
        feasibility_counts=dict(Counter(status.values())),
        fresh_test=protocol["fresh_test"],
        independent_generalization_supported=False,
        uncertainty_estimates=None,
        uncertainty_reason="Repeated conditions on one inspected layout are not independent scenes",
    )


def checked_bundle(root, manifest):
    doc = read(root / manifest)
    if not doc["complete"]:
        raise ValueError("Incomplete source artifact bundle")
    for name, expected in doc["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed source artifact: " + name)
    return doc


def run(study, output):
    study, output = study.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Preserve existing audit output")
    p = verify(study)
    for root in (
        study,
        *(
            Path(p[k]).resolve()
            for k in ("parent", "cohort", "training", "risk_training", "assets")
        ),
    ):
        if output.is_relative_to(root):
            raise ValueError("Audit output must be outside all existing source trees")
    checked_bundle(study, "release_audit.json")
    checked_bundle(study, "completion.json")
    a, v = read(study / "analysis.json"), read(study / "validation.json")
    if (
        not v["complete"]
        or v["protocol_sha256"] != sha(study / "protocol.json")
        or a["validation_sha256"] != sha(study / "validation.json")
    ):
        raise ValueError("Independent source validation is required")
    parent = Path(p["parent"])
    parent_p = read(parent / "protocol.json")
    fixed = [n for n in parent_p["policies"] if n.startswith("static:")]
    # The fixed pool is declared by the immutable parent, never inferred from surviving rows.
    audit_protocol = dict(p, fixed_policies=fixed)
    all_rows = a["policy_cases"] + a["reused_fixed_controls"]
    result = summarize(audit_protocol, all_rows, a["feasibility"])
    energy = []
    artifacts = {}
    for row in all_rows:
        root = parent if row["policy"].startswith("static:") else study
        events = checked_events(root, row)
        destination = root / row["path"]
        execution, outcome = (
            read(destination / "execution/results.json"),
            read(destination / "outcome.json"),
        )
        intent_path = root / row["flight"] / row["mission"] / "intent.json"
        scenario_path = root / row["flight"] / row["mission"] / "scenario.json"
        contract = IntentContract(read(intent_path))
        canonical = evaluate(contract, events, execution)
        for key in ("mission_success", "failed_constraints", "reporting", "phases"):
            if canonical[key] != outcome[key]:
                raise ValueError("Canonical stored-event outcome does not reproduce")
        if row["full_success"] != canonical["mission_success"]:
            raise ValueError("Summary success differs from canonical outcome")
        if row["final_battery"] != execution["final_battery_frac"]:
            raise ValueError("Summary battery differs from execution")
        mission = read(scenario_path)["mission"]
        slack = energy_slack(
            mission["initial_battery_frac"],
            contract.document()["limits"]["min_final_battery_frac"],
            mission["battery_capacity_wh"],
            execution["cumulative_energy_j"],
            row["final_battery"],
        )
        energy.append(
            dict(
                flight=row["flight"],
                mission=row["mission"],
                policy=row["policy"],
                full_success=row["full_success"],
                consumed_j=execution["cumulative_energy_j"],
                **slack,
            )
        )
        for file in (
            intent_path,
            scenario_path,
            destination / "outcome.json",
            destination / "execution/events.json",
            destination / "execution/results.json",
        ):
            artifacts[str(file)] = sha(file)
    stress = []
    for policy in (*p["policies"], *fixed):
        subset = [r for r in energy if r["policy"] == policy]
        stress.append(
            dict(
                policy=policy,
                original_successes=sum(r["full_success"] for r in subset),
                diagnostics=[
                    dict(
                        relative_total_energy_overrun=f,
                        original_successes_preserving_reserve=sum(
                            r["full_success"] and r["fixed_schedule_stress"][i]["reserve_preserved"]
                            for r in subset
                        ),
                    )
                    for i, f in enumerate(STRESS_FRACTIONS)
                ],
            )
        )
    result.update(
        schema_version="mission_evidence_audit_v1",
        complete=True,
        audited_at_utc=datetime.now(UTC).isoformat(),
        source=str(study),
        source_protocol_sha256=sha(study / "protocol.json"),
        historical_public_decisions_verified_by_source=v["public_decisions_reproduced"],
        outcomes_recomputed_from_stored_events=len(all_rows),
        raw_perception_reexecuted=False,
        source_artifacts_sha256=artifacts,
        policy_energy_stress=stress,
        energy_cases=energy,
        stress_scope="Fixed-schedule total-energy scaling only; not rerouted or rescored missions",
        new_policy_replays=0,
        new_neural_calls=0,
        new_physical_flights=0,
        audit_source_sha256=sha(Path(__file__)),
    )
    lines = [
        "# Mission evidence audit",
        "",
        f"Source: `{study.name}`. No new policy runs or neural inference.",
        "",
        f"{len(all_rows)} stored outcomes reconstructed; "
        f"{result['comparison_conditions']} conditions, "
        f"{result['lineage_units']['recording']} recordings, "
        f"{result['lineage_units']['layout']} layout, "
        f"{result['lineage_units']['scenery']} scenery asset.",
        "",
        "Historical split labels do not make this inspected study a fresh test. "
        "No independent-scene confidence interval is reported.",
        "",
        "| Policy | Full success | Witness-feasible | Unresolved | Strict |",
        "|---|---:|---:|---:|---:|",
    ]
    # Denominators are generated, never assumed from this particular cohort.

    def fraction(x):
        return f"{x['successes']}/{x['cases']}" if x["cases"] else "N/A (0 cases)"

    for s in result["policy_summary"]:
        lines.append(
            f"| {s['policy']} | {fraction(s)} | "
            f"{fraction(s['feasibility_strata'][FEASIBILITY[0]])} | "
            f"{fraction(s['feasibility_strata']['unresolved'])} | "
            f"{fraction(s['missions']['verified_inspection'])} |"
        )
    lines += [
        "",
        "Feasible and unresolved strata supplement the full denominator; "
        "no failed case is removed.",
        "",
        "Energy diagnostics scale total charged energy by 0%, 1% and 5% with the schedule fixed. "
        "They do not change official success, establish hardware robustness "
        "or simulate altered latency.",
        "",
        "| Policy | Original successes | Preserve reserve at +1% energy | At +5% energy |",
        "|---|---:|---:|---:|",
    ]
    for s in stress:
        lines.append(
            f"| {s['policy']} | {s['original_successes']} | "
            f"{s['diagnostics'][1]['original_successes_preserving_reserve']} | "
            f"{s['diagnostics'][2]['original_successes_preserving_reserve']} |"
        )
    lines += [
        "",
        "See audit.json for recording-grouped paired comparisons, "
        "per-case margins, lineage and hashes.",
        "",
    ]
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    (output / "REPORT.md").write_text("\n".join(lines))
    save(
        output / "completion.json",
        dict(
            complete=True,
            source_sha256=sha(Path(__file__)),
            artifacts_sha256={n: sha(output / n) for n in ("audit.json", "REPORT.md")},
        ),
    )
    print(
        dict(
            outcomes=len(all_rows),
            units=result["lineage_units"],
            feasibility=result["feasibility_counts"],
        )
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.study, args.output)
