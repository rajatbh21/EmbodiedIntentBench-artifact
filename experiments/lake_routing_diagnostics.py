"""Audit policy discrimination and model complementarity without changing the benchmark.

This is evaluator-only analysis of existing development outputs. Framewise free
union scores ignore costs and cannot be treated as realizable routing performance.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import validate_table
from experiments.lake_missions.assets import save, sha


def complementarity(rows: list[dict], times: list[float], configs: list[str]) -> dict:
    validate_table(rows, times, configs)
    by_frame = defaultdict(dict)
    totals, visible = dict.fromkeys(configs, 0), 0
    for row in rows:
        matched, seen = set(row["matched_target_ids"]), set(row["visible_target_ids"])
        if not matched <= seen:
            raise ValueError("Matched targets must be visible")
        by_frame[row["capture_id"]][row["config_id"]] = matched, seen
        totals[row["config_id"]] += len(matched)
    free_union, free_best_single, differing = 0, 0, 0
    for frame in by_frame.values():
        visibility = [seen for _, seen in frame.values()]
        if any(seen != visibility[0] for seen in visibility):
            raise ValueError("Same-frame visibility differs between configurations")
        visible += len(visibility[0])
        matches = [matched for matched, _ in frame.values()]
        free_union += len(set().union(*matches))
        free_best_single += max(map(len, matches))
        differing += int(any(matched != matches[0] for matched in matches))
    best = max(totals.values())
    return dict(
        schema_version="routing_complementarity_v1",
        capture_count=len(times),
        visible_target_observations=visible,
        matched_target_observations_by_config=totals,
        best_static_configs=sorted(c for c in configs if totals[c] == best),
        best_static_matched_observations=best,
        free_per_frame_best_single_matched_observations=free_best_single,
        free_per_frame_union_matched_observations=free_union,
        best_single_over_best_static_gain=free_best_single - best,
        union_over_best_single_gain=free_union - free_best_single,
        captures_with_prediction_disagreement=differing,
        gt_aware=True,
        deployable_policy=False,
        interpretation="Dense target-observation counts, not unique mission targets or success",
        ignored_constraints=[
            "latency",
            "energy",
            "loading",
            "tracking",
            "reporting",
            "phase deadlines",
        ],
    )


def discrimination(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = row["environment"], row["mission"], row["policy"]
        if key in seen:
            raise ValueError("Duplicate policy case")
        seen.add(key)
        groups[key[:2]].append(row)
    result = []
    for (environment, mission), cases in sorted(groups.items()):
        if len(cases) < 2:
            raise ValueError("At least two policies required per case")
        metrics = {}
        for key in ("full_success", "phases_passed", "weighted_evidence_recall", "report_recall"):
            values = [float(row[key]) for row in cases]
            metrics[key] = dict(
                minimum=min(values), maximum=max(values), spread=max(values) - min(values)
            )
        metrics["full_success"]["all_fail"] = all(not r["full_success"] for r in cases)
        metrics["full_success"]["all_pass"] = all(r["full_success"] for r in cases)
        result.append(
            dict(environment=environment, mission=mission, policy_count=len(cases), metrics=metrics)
        )
    return result


def run(causal: Path, prior: Path, output: Path) -> dict:
    analysis, validation = read(causal / "analysis.json"), read(causal / "validation.json")
    if (
        not validation["complete"]
        or analysis["validation_sha256"] != sha(causal / "validation.json")
        or analysis["summary_sha256"] != sha(causal / "summary.json")
    ):
        raise ValueError("Changed or unvalidated causal analysis")
    summary = read(causal / "summary.json")
    # Bind the diagnostic metrics back to canonical per-run outcomes.
    for row in analysis["cases"]:
        case = next(
            r
            for r in summary["cases"]
            if all(r[k] == row[k] for k in ("environment", "mission", "policy"))
        )
        destination = Path(case["output"])
        if sha(destination / "outcome.json") != case["outcome_sha256"]:
            raise ValueError("Changed canonical outcome")
        outcome = read(destination / "outcome.json")
        phases = outcome["phases"]
        expected = dict(
            full_success=outcome["mission_success"],
            phases_passed=sum(p["success"] for p in phases),
            report_recall=outcome["reporting"]["recall_by_deadline"],
            weighted_evidence_recall=sum(p["weight"] * p["evidence_recall"] for p in phases)
            / sum(p["weight"] for p in phases),
        )
        if any(row[k] != value for k, value in expected.items()):
            raise ValueError("Analysis metric differs from canonical outcome")
    dense = {}
    for environment in sorted({r["environment"] for r in analysis["cases"]}):
        rows = read(prior / environment / "outcomes.json")
        times = sorted({r["capture_time_s"] for r in rows})
        configs = sorted({r["config_id"] for r in rows})
        dense[environment] = complementarity(rows, times, configs)
    result = dict(
        schema_version="routing_discrimination_audit_v1",
        scope="development",
        independent_test=False,
        policy_discrimination=discrimination(analysis["cases"]),
        model_complementarity=dense,
        source_sha256={
            str(p.resolve()): sha(p)
            for p in [
                causal / "analysis.json",
                causal / "validation.json",
                causal / "summary.json",
                *[prior / env / "outcomes.json" for env in dense],
            ]
        },
        implementation_sha256=sha(Path(__file__)),
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "audit.json", result)
    lines = [
        "# Routing discrimination audit",
        "",
        "This is a post-hoc development diagnostic of existing outputs. It changes no score, "
        "scenario, requirement, policy or cost. No new simulation or inference was run.",
        "",
        "## Policy-dependent mission outcomes",
        "",
        "| Environment | Mission | Full success range | Phases passed range | "
        "Weighted evidence recall range | Report recall range |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for case in result["policy_discrimination"]:
        cells = [
            f"{case['metrics'][key]['minimum']:.3f}-{case['metrics'][key]['maximum']:.3f}"
            for key in (
                "full_success",
                "phases_passed",
                "weighted_evidence_recall",
                "report_recall",
            )
        ]
        lines.append(f"| {case['environment']} | {case['mission']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "The primary all-constraints success metric has a floor effect on these policies. "
        "Secondary outcomes already distinguish policies, but they do not replace full success. "
        "The same flight under lighting variants is not an independent statistical sample.",
        "",
        "## Dense perception complementarity (GT-aware, free diagnostic)",
        "",
        "| Environment | Visible target-observations | Best fixed model | "
        "Best single model per frame | All-model union per frame |",
        "|---|---:|---:|---:|---:|",
    ]
    for env, d in dense.items():
        lines.append(
            f"| {env} | {d['visible_target_observations']} | "
            f"{d['best_static_matched_observations']} | "
            f"{d['free_per_frame_best_single_matched_observations']} | "
            f"{d['free_per_frame_union_matched_observations']} |"
        )
    lines += [
        "",
        "Counts measure target-observations on every canonical capture, not unique people. "
        "The per-frame best model uses hidden labels and ignores all costs. Even when its "
        "gain over the best fixed model is small, useful mission routing can still arise from "
        "cost, cadence, phase priorities or energy. Conversely, large free-union gains do not "
        "prove that a legal one-model-per-decision policy can realize them.",
        "",
        "## Required comparison design",
        "",
        "Add a content/cost predictor baseline and a learned contextual policy adapted from "
        "published approaches; label adaptations explicitly. Keep full mission success primary "
        "and phase/report/cost diagnostics secondary. Use declared budget sweeps and independent "
        "scene-level splits; never select thresholds or test cases to favor one policy. "
        "Match resource limits and charge every feature, probe, switch and inference call. "
        "Include intent, environment and history ablations and timing controls. A feasible "
        "offline witness establishes opportunity, not an optimal oracle or deployable baseline.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("causal", "prior", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    run(args.causal, args.prior, args.output)
