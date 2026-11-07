"""Post-hoc endpoint discrimination and matched interventions on the complete transfer grid.

No new policy is selected or fitted. Frames and factorial cells are not treated
as independent map samples. A failed hindsight search never proves infeasibility.
"""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_interpretation import classify_cell, interactions, matrix
from experiments.mission_transfer_posthoc_integrity import check_diagnostics, implementation_sha256


def intervention_pairs(rows, dimension):
    """Compare exactly matched conditions; reject duplicate or unbalanced pairs."""
    keys = ("layout", "lighting", "battery", "mission", "policy")
    if dimension not in keys[:-1]:
        raise ValueError("Unknown intervention")
    values = sorted({r[dimension] for r in rows})
    if len(values) != 2:
        raise ValueError("A matched intervention requires exactly two levels")
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys if k != dimension)
        group = groups.setdefault(key, {})
        if row[dimension] in group:
            raise ValueError("Duplicate matched intervention cell")
        group[row[dimension]] = row
    result = []
    for key, group in sorted(groups.items()):
        if set(group) != set(values):
            raise ValueError("Missing matched intervention mate")
        left, right = (group[v] for v in values)
        same = left["schedule"] == right["schedule"]
        flip = left["full_success"] != right["full_success"]
        result.append(
            dict(
                matched_on=dict(zip((k for k in keys if k != dimension), key, strict=True)),
                levels=values,
                same_schedule=same,
                endpoint_changed=flip,
                same_schedule_endpoint_changed=same and flip,
                full_success=[left["full_success"], right["full_success"]],
                phase_difference=right["phases_succeeded"] - left["phases_succeeded"],
                report_recall_difference=right["report_recall"] - left["report_recall"],
            )
        )
    return result


def summarize_interventions(pairs, policies):
    return [
        dict(
            policy=policy,
            pairs=len(sub),
            changed_schedules=sum(not r["same_schedule"] for r in sub),
            changed_endpoints=sum(r["endpoint_changed"] for r in sub),
            unchanged_schedule_changed_endpoint=sum(
                r["same_schedule_endpoint_changed"] for r in sub
            ),
        )
        for policy in policies
        for sub in [[r for r in pairs if r["matched_on"]["policy"] == policy]]
    ]


def group_discrimination(rows, groups):
    """Keep the complete condition grid when describing each explicit reference set."""
    expected_cells = {(r["flight"], r["mission"]) for r in rows}
    available = {r["policy"] for r in rows}
    results = {}
    for name, policies in groups.items():
        if not policies or len(set(policies)) != len(policies) or not set(policies) <= available:
            raise ValueError("Invalid discrimination group")
        lookup, cells = matrix([r for r in rows if r["policy"] in policies], policies)
        if set(cells) != expected_cells:
            raise ValueError("Discrimination groups must retain every condition")
        results[name] = dict(
            policies=list(policies),
            conditions=len(cells),
            **{
                label: sum(len({lookup[f, m, p][field] for p in policies}) > 1 for f, m in cells)
                for label, field in (
                    ("success_varies", "full_success"),
                    ("phase_count_varies", "phases_succeeded"),
                    ("report_recall_varies", "report_recall"),
                )
            },
        )
    return results


def run(root):
    from experiments.mission_transfer_comparison_v4.protocol import verify

    p = verify(root)
    if (root / "interpretation.json").exists():
        raise FileExistsError("Preserve completed interpretation")
    completion = read(root / "completion.json")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed completed comparison")
    analysis = read(root / "analysis.json")
    cohort = Path(p["cohort"])
    witnesses = read(root / "witness_validation.json")
    diagnostics = read(cohort / "feasibility_diagnostics.json")
    if not witnesses["complete"] or witnesses["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Complete independent witness audit required")
    for flight, expected in witnesses["independent_audits_sha256"].items():
        audit_path = root / "witnesses" / flight / "validation.json"
        if sha(audit_path) != expected:
            raise ValueError("Changed witness audit")
        audit = read(audit_path)
        for name, digest in audit["source_summaries_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("Changed witness summary")
    check_diagnostics(diagnostics, p, root, cohort)
    for flight, data in diagnostics["flights"].items():
        if (
            sha(cohort / flight / "diagnostic_outcomes.json") != data["outcomes_sha256"]
            or sha(cohort / flight / "predictions/manifest.json") != data["source_manifest_sha256"]
        ):
            raise ValueError("Changed optimistic diagnostic evidence")
    rows, policies = analysis["policy_cases"], p["policies"]
    lookup, cells = matrix(rows, policies)
    if set(cells) != {(f, m) for f in p["cases"] for m in p["missions"]}:
        raise ValueError("Incomplete predeclared denominator")
    witness = {(r["flight"], r["mission"]): r for r in witnesses["cases"]}
    if len(witness) != len(witnesses["cases"]) or set(witness) != set(cells):
        raise ValueError("Unmatched witness grid")
    feasibility = []
    for flight, mission in cells:
        successes = [c for c in policies if lookup[flight, mission, c]["full_success"]]
        source = p["cases"][flight]["source_flight"]
        bound = diagnostics["flights"][source]["missions"][mission]["latency_certified_infeasible"]
        witnessed = witness[flight, mission]["full_contract_witness"]
        feasibility.append(
            dict(
                flight=flight,
                mission=mission,
                successful_policies=successes,
                full_contract_witness=witnessed,
                serial_bound_failed=bound,
                classification=classify_cell(successes, witnessed, bound),
            )
        )
    pairwise = {
        d: interactions(rows, policies, d) for d in ("mission", "battery", "lighting", "layout")
    }
    interventions = {d: intervention_pairs(rows, d) for d in ("mission", "battery")}
    intervention_summary = {
        d: summarize_interventions(values, policies) for d, values in interventions.items()
    }
    failures = []
    for policy in policies:
        sub = [r for r in rows if r["policy"] == policy]
        categories = sorted({c for r in sub for c in r["failed_constraints"]})
        failures.append(
            dict(
                policy=policy,
                failures=sum(not r["full_success"] for r in sub),
                overlapping_failure_counts={
                    c: sum(c in r["failed_constraints"] for r in sub) for c in categories
                },
            )
        )
    groups = group_discrimination(
        rows,
        {
            "all_predeclared_policies": policies,
            "all_nonstatic_policies_including_ablations": [
                c for c in policies if not c.startswith("static:")
            ],
            "routing_objectives_without_coverage_ablations": [
                "coverage_tail",
                "phase_survival",
                "mission_survival",
                "temporal_mission",
                "round_robin",
                "greedy_quality",
                "approxdet_eq5_adapter",
            ],
            "fixed_configurations": [c for c in policies if c.startswith("static:")],
        },
    )
    result = dict(
        complete=True,
        posthoc=True,
        hypothesis_test=False,
        policy_selection_on_evaluation=False,
        independent_maps=1,
        independent_layout_groups=p["independent_evaluation_layouts"],
        conditions=len(cells),
        executions=len(rows),
        source_sha256=sha(Path(__file__)),
        helper_sha256=implementation_sha256(),
        routing_admission_sha256=p["routing_admission_sha256"],
        protocol_sha256=sha(root / "protocol.json"),
        analysis_sha256=sha(root / "analysis.json"),
        witness_validation_sha256=sha(root / "witness_validation.json"),
        diagnostics_sha256=sha(cohort / "feasibility_diagnostics.json"),
        feasibility=feasibility,
        pairwise_strata=pairwise,
        matched_interventions=interventions,
        matched_intervention_summary=intervention_summary,
        failure_categories=failures,
        policy_group_discrimination=groups,
    )
    save(root / "interpretation.json", result)
    total_pairs = len(policies) * (len(policies) - 1) // 2
    lines = [
        "# Mission-level discrimination on the complete transfer grid",
        "",
        f"This post-hoc description retains all {len(rows)} runs and {len(cells)} conditions. "
        "No outcome selects a policy or changes the contracts. The map underwent a disclosed "
        "pre-inference geometry revision and post-capture admission-scope amendment. "
        "Lighting strata compare separate physical recordings with measured pose differences, "
        "not an illumination-only causal intervention. There is one map and two spatial groups; "
        "factorial cells and frames are repeated measures, not independent map samples.",
        "",
        "## Policy failure versus task infeasibility",
        "",
        "Successful evaluator-only witnesses pay the same serial inference, switching, "
        "tracking and flight costs. Their raw prediction/report outcomes are independently "
        "reconstructed. They use ground truth and are excluded from the policy comparison. "
        "An unsuccessful search is inconclusive. An optimistic bound that fails certifies "
        "infeasibility only for this fixed model pool, trajectory and authored contract.",
        "",
        "| Condition | Intent | Successful policies | Paid witness | Classification |",
        "|---|---|---:|---|---|",
    ]
    lines += [
        f"| {r['flight']} | {r['mission']} | {len(r['successful_policies'])} | "
        f"{r['full_contract_witness']} | {r['classification']} |"
        for r in feasibility
    ]
    lines += [
        "",
        "## Discrimination within explicit reference sets",
        "",
        "All conditions remain in each denominator. These post-hoc descriptive groups "
        "separate variation among routing objectives from variation across fixed "
        "configurations; group membership is explicit in JSON and does not select "
        "methods by their outcomes. The primary policy and predeclared contrasts remain "
        "unchanged. Differences among all policies alone are not evidence that adaptive "
        "routing improves on a strong fixed configuration.",
        "",
        "| Reference set | Conditions | Success differs | Phase count differs | "
        "Report recall differs |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {name} | {r['conditions']} | {r['success_varies']} | "
            f"{r['phase_count_varies']} | {r['report_recall_varies']} |"
            for name, r in groups.items()
        ],
        "",
        "## Context-dependent relative performance",
        "",
        "A strict sign reversal requires A to complete more contracts than B in one "
        "stratum and fewer in another; a tie does not count. All policy pairs are retained. "
        "This is descriptive and carries no population significance claim.",
        "",
        f"| Stratification | Strict sign reversals /{total_pairs} |",
        "|---|---:|",
    ]
    lines += [
        f"| {d} | {sum(r['strict_sign_reversal'] for r in values)} |"
        for d, values in pairwise.items()
    ]
    lines += [
        "",
        "## Exactly matched interventions",
        "",
        "Each pair holds layout, policy and the other declared factors fixed. "
        "Intent and battery mates use identical "
        "recorded RGB. A changed schedule is behavioral sensitivity, not evidence that the "
        "change helps. An identical selected-frame/configuration schedule with a changed "
        "endpoint shows why model selection alone does not define mission success.",
        "",
        "| Factor | Policy | Pairs | Changed schedule | Changed success | "
        "Same schedule, changed success |",
        "|---|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {d} | {r['policy']} | {r['pairs']} | {r['changed_schedules']} | "
        f"{r['changed_endpoints']} | {r['unchanged_schedule_changed_endpoint']} |"
        for d, values in intervention_summary.items()
        for r in values
    ]
    lines += [
        "",
        "Every condition, all pairwise strata and overlapping failure categories are "
        "available in `interpretation.json`. Counts are not independent statistical trials. "
        "A successful fixed configuration may outperform an adaptive router; such a result "
        "is retained rather than excluded from the benchmark.",
        "",
    ]
    (root / "INTERPRETATION.md").write_text("\n".join(lines))
    print(dict(conditions=len(cells), executions=len(rows), posthoc=True))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
