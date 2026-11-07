"""Post-hoc benchmark discrimination and feasibility, without changing frozen outcomes.

This describes the full grid. It does not select a winner, estimate population
uncertainty from repeated frames, or treat an evaluator witness as a policy.
"""

import argparse
from itertools import combinations
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.protocol import verify


def matrix(rows, policies):
    lookup = {}
    for r in rows:
        key = (r["flight"], r["mission"], r["policy"])
        if key in lookup or r["policy"] not in policies:
            raise ValueError("Duplicate or unexpected policy cell")
        lookup[key] = r
    cells = sorted({(r["flight"], r["mission"]) for r in rows})
    if not cells or set(lookup) != {(f, m, p) for f, m in cells for p in policies}:
        raise ValueError("Complete balanced policy grid required")
    return lookup, cells


def contrast(rows, left, right):
    lookup, cells = matrix(rows, (left, right))
    differences = [
        int(lookup[f, m, left]["full_success"]) - int(lookup[f, m, right]["full_success"])
        for f, m in cells
    ]
    return dict(
        wins=sum(d > 0 for d in differences),
        losses=sum(d < 0 for d in differences),
        ties=sum(d == 0 for d in differences),
        difference=sum(differences),
        paired_conditions=len(cells),
    )


def interactions(rows, policies, dimension):
    values = sorted({r[dimension] for r in rows})
    result = []
    for a, b in combinations(policies, 2):
        strata = []
        for value in values:
            sub = [r for r in rows if r[dimension] == value and r["policy"] in (a, b)]
            strata.append(dict(value=value, **contrast(sub, a, b)))
        differences = [s["difference"] for s in strata]
        result.append(
            dict(
                left=a,
                right=b,
                strata=strata,
                strict_sign_reversal=min(differences) < 0 < max(differences),
            )
        )
    return result


def classify_cell(policy_successes, witnessed, bound_failed):
    if (policy_successes or witnessed) and bound_failed:
        raise ValueError("Feasibility witness contradicts optimistic bound")
    if policy_successes:
        return "achieved_by_predeclared_policy"
    if witnessed:
        return "feasible_but_all_predeclared_policies_failed"
    if bound_failed:
        return "certified_pool_perception_infeasible"
    return "unresolved"


def run(root):
    p = verify(root)
    completion = read(root / "completion.json")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed completed comparison")
    analysis = read(root / "analysis.json")
    witnesses = read(root / "witness_validation.json")
    diagnostics = read(Path(p["cohort"]) / "feasibility_diagnostics.json")
    if not witnesses["complete"] or witnesses["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Complete independent witness audit required")
    for flight, expected in witnesses["independent_audits_sha256"].items():
        if sha(root / "witnesses" / flight / "validation.json") != expected:
            raise ValueError("Changed witness audit")
    if diagnostics["protocol_sha256"] != sha(Path(p["cohort"]) / "protocol.json"):
        raise ValueError("Wrong optimistic diagnostic cohort")
    rows, policies = analysis["policy_cases"], p["policies"]
    lookup, cells = matrix(rows, policies)
    if len(rows) != 504 or len(cells) != 24:
        raise ValueError("Wrong predeclared denominator")
    witness = {(r["flight"], r["mission"]): r for r in witnesses["cases"]}
    if len(witness) != len(witnesses["cases"]) or set(witness) != set(cells):
        raise ValueError("Unmatched witness grid")
    feasibility = []
    for f, m in cells:
        successes = [c for c in policies if lookup[f, m, c]["full_success"]]
        source = p["cases"][f]["source_flight"]
        bound = diagnostics["flights"][source]["missions"][m]["latency_certified_infeasible"]
        witnessed = witness[f, m]["full_contract_witness"]
        feasibility.append(
            dict(
                flight=f,
                mission=m,
                successful_policies=successes,
                full_contract_witness=witnessed,
                serial_bound_failed=bound,
                classification=classify_cell(successes, witnessed, bound),
            )
        )
    effects = {
        d: interactions(rows, policies, d) for d in ("mission", "battery", "lighting", "layout")
    }
    primary = p["primary_policy"]
    layout_effects = []
    for omitted in sorted({r["layout"] for r in rows}):
        for other in policies:
            if other == primary:
                continue
            sub = [r for r in rows if r["layout"] != omitted and r["policy"] in (primary, other)]
            layout_effects.append(
                dict(omitted_layout=omitted, comparator=other, **contrast(sub, primary, other))
            )
    errors = []
    for policy in policies:
        sub = [r for r in rows if r["policy"] == policy]
        categories = sorted({c for r in sub for c in r["failed_constraints"]})
        errors.append(
            dict(
                policy=policy,
                cases=len(sub),
                failures=sum(not r["full_success"] for r in sub),
                overlapping_failure_counts={
                    c: sum(c in r["failed_constraints"] for r in sub) for c in categories
                },
            )
        )
    result = dict(
        complete=True,
        posthoc=True,
        hypothesis_test=False,
        independent_layout_groups=3,
        maps=1,
        conditions=24,
        executions=504,
        source_sha256=sha(Path(__file__)),
        analysis_sha256=sha(root / "analysis.json"),
        witness_validation_sha256=sha(root / "witness_validation.json"),
        diagnostics_sha256=sha(Path(p["cohort"]) / "feasibility_diagnostics.json"),
        feasibility=feasibility,
        pairwise_strata=effects,
        leave_one_layout_out_descriptive_contrasts=layout_effects,
        failure_categories=errors,
        policy_selection_on_evaluation=False,
    )
    save(root / "interpretation.json", result)
    lines = [
        "# Benchmark discrimination and feasibility",
        "",
        "This is a post-hoc description of the complete frozen grid, not a new hypothesis "
        "test or policy-selection procedure. There are three authored spatial layout groups "
        "on one known map. The 24 intent/resource/lighting conditions are repeated measures.",
        "",
        "## Feasibility of every condition",
        "",
        "All successful witnesses pay the same inference, loading, tracking, integer-clock "
        "and flight costs as evaluated policies. Their raw mask scores and report identities "
        "are independently reconstructed. They use evaluator-only ground truth and are "
        "excluded from the 21-policy comparison. An unsuccessful beam search is inconclusive.",
        "",
        "| Condition | Intent | Successful policies /21 | Paid witness | Classification |",
        "|---|---|---:|---|---|",
    ]
    lines += [
        f"| {r['flight']} | {r['mission']} | {len(r['successful_policies'])} | "
        f"{r['full_contract_witness']} | {r['classification']} |"
        for r in feasibility
    ]
    lines += [
        "",
        "## Dependence of relative performance on mission and environment",
        "",
        "A strict sign reversal means that A completes more contracts than B in at least "
        "one stratum and fewer in another. Ties do not count. The following table counts "
        "all 210 unordered policy pairs; it is descriptive and does not claim that the "
        "underlying effects generalize beyond these layouts.",
        "",
        "| Stratification | Pairs with strict sign reversal /210 |",
        "|---|---:|",
    ]
    lines += [
        f"| {d} | {sum(r['strict_sign_reversal'] for r in values)} |"
        for d, values in effects.items()
    ]
    lines += [
        "",
        "## Primary method after omitting one layout",
        "",
        "Each row reports the smallest and largest paired success difference after "
        "omitting each of the three layouts in turn (16 conditions remain). These "
        "are stability descriptions, not confidence intervals or held-out model selection.",
        "",
        "| Comparator | Minimum difference | Maximum difference |",
        "|---|---:|---:|",
    ]
    for other in policies:
        if other != primary:
            ds = [r["difference"] for r in layout_effects if r["comparator"] == other]
            lines.append(f"| {other} | {min(ds):+d} | {max(ds):+d} |")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "Differences in full success demonstrate discrimination for this operating range. "
        "Witnessed all-policy failures identify controller gaps without weakening the mission "
        "contract. Certified infeasibility identifies insufficient evidence under the fixed "
        "sensor/model pool, and unresolved cases remain unresolved. None of these outcomes "
        "establishes adaptive superiority, calibrated success probabilities, independent-map "
        "generalization, or measured onboard resource validity.",
        "",
    ]
    (root / "INTERPRETATION.md").write_text("\n".join(lines))
    print({d: sum(r["strict_sign_reversal"] for r in v) for d, v in effects.items()}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
