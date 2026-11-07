"""Hash-bound supplementary analysis of frozen perception-only routing replays.

No simulations, neural calls, policy updates or primary endpoint revisions occur.
Every result is reconstructed from the original selected-output receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from experiments.perception_evaluation.contract import SCHEMA, public_specification
from experiments.perception_evaluation.metrics import evaluate_run
from experiments.perception_router_suite.study import verify

FACTORS = ("flight", "mission", "initial_battery_frac", "link", "deadline_profile")
FILES = ("protocol.json", "intent.json", "events.json", "results.json", "outcome.json")
MEAN_FIELDS = (
    "phase_completion_rate",
    "weighted_evidence_recall",
    "normalized_intent_recall_attainment",
    "report_recall",
    "false_reports",
    "duplicate_reports",
    "false_components_per_mission_minute",
    "perception_latency_p50_s",
    "perception_latency_p95_s",
    "skipped_capture_fraction",
    "energy_total_j",
    "energy_flight_j",
    "energy_nonflight_j",
    "energy_perception_j",
    "energy_downstream_j",
    "perception_radio_energy_j",
    "battery_reserve_margin",
    "communication_mb",
    "nominal_flight_duration_s",
    "charged_duration_s",
)
SUM_FIELDS = (
    "perception_attempts",
    "perception_admitted",
    "perception_abstentions",
    "perception_deadline_misses",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact path escapes study")
    return path


def condition(row):
    return tuple(row[k] for k in FACTORS)


def validate_grid(plan, rows):
    """Demand the full declared Cartesian product, with exactly one paired run."""
    factors = plan["factors"]
    expected = set(
        itertools.product(
            factors["flights"],
            factors["missions"],
            factors["batteries"],
            factors["links"],
            [p["name"] for p in factors["deadline_profiles"]],
            factors["policies"],
        )
    )
    actual = [(*condition(r), r["policy"]) for r in rows]
    require(len(set(actual)) == len(actual), "Duplicate paired policy condition")
    require(len({r["run_id"] for r in rows}) == len(rows), "Duplicate run identity")
    require(set(actual) == expected, "Missing or unexpected paired policy conditions")
    require(len(rows) == plan["expected_runs"], "Unexpected run count")


def mean(values):
    present = [v for v in values if v is not None]
    return math.fsum(present) / len(present) if present else None


def aggregate(rows, keys=("mission", "policy")):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    output = []
    for key, group in sorted(groups.items()):
        item = dict(zip(keys, key, strict=True))
        item.update(runs=len(group), mission_successes=sum(r["mission_success"] for r in group))
        item["mission_success_rate"] = item["mission_successes"] / len(group)
        item["reserve_violations"] = sum(r["battery_reserve_margin"] < 0 for r in group)
        item.update({"mean_" + k: mean(r[k] for r in group) for k in MEAN_FIELDS})
        item.update({k: sum(r[k] for r in group) for k in SUM_FIELDS})
        attempts = item["perception_attempts"]
        decisions = attempts + item["perception_abstentions"]
        item["micro_deadline_miss_rate"] = (
            item["perception_deadline_misses"] / attempts if attempts else None
        )
        item["micro_admission_rate"] = item["perception_admitted"] / attempts if attempts else None
        item["micro_abstention_rate"] = (
            item["perception_abstentions"] / decisions if decisions else None
        )
        # A mean of run p95 values is NOT the pooled p95 over every invocation.
        # Names above intentionally preserve this distinction and equal run weights.
        for prefix, time_key, event_key in (
            ("report_quota", "report_quota_time_s", "report_quota_reached"),
            (
                "successful_evidence",
                "successful_mission_evidence_ready_time_s",
                "successful_mission_evidence_ready_event",
            ),
        ):
            eligible = (
                [r for r in group if r["report_quota_required"]]
                if prefix == "report_quota"
                else group
            )
            attained = [r[time_key] for r in eligible if r[event_key]]
            item[prefix + "_applicable_runs"] = len(eligible)
            item[prefix + "_not_applicable_runs"] = len(group) - len(eligible)
            item[prefix + "_attained_runs"] = len(attained)
            item[prefix + "_unattained_runs"] = len(eligible) - len(attained)
            item[prefix + "_mean_time_s_conditional_on_attainment"] = mean(attained)
        output.append(item)
    return output


def paired_comparisons(rows):
    """Descriptive within-condition differences; no pseudo-independent CI."""
    indexed = {(r["policy"], condition(r)): r for r in rows}
    require(len(indexed) == len(rows), "Duplicate paired condition")
    policies = sorted({r["policy"] for r in rows})
    output = []
    for mission in sorted({r["mission"] for r in rows}):
        conditions = sorted({condition(r) for r in rows if r["mission"] == mission})
        for a, b in itertools.combinations(policies, 2):
            require(
                all((p, c) in indexed for c in conditions for p in (a, b)), "Unpaired comparison"
            )
            pairs = [(indexed[a, c], indexed[b, c]) for c in conditions]
            wins = sum(x["mission_success"] and not y["mission_success"] for x, y in pairs)
            losses = sum(y["mission_success"] and not x["mission_success"] for x, y in pairs)
            output.append(
                dict(
                    mission=mission,
                    policy_a=a,
                    policy_b=b,
                    matched_conditions=len(pairs),
                    a_only_success=wins,
                    b_only_success=losses,
                    tied_success_status=len(pairs) - wins - losses,
                    success_rate_difference_a_minus_b=(wins - losses) / len(pairs),
                    mean_phase_completion_difference_a_minus_b=mean(
                        x["phase_completion_rate"] - y["phase_completion_rate"] for x, y in pairs
                    ),
                    mean_nonflight_energy_difference_a_minus_b_j=mean(
                        x["energy_nonflight_j"] - y["energy_nonflight_j"] for x, y in pairs
                    ),
                )
            )
    return output


def render(summary):
    lines = [
        "# Perception-only routing: mission, latency and energy evaluation",
        "",
        f"Supplementary analysis of {summary['runs']} existing deterministic replays. "
        "No new flights, "
        "neural calls, router training, or replay execution were performed. Every original "
        "primary outcome was reconstructed without changing intent thresholds.",
        "",
        "The treatment is perception configuration/placement. Trajectory, camera inputs, "
        "association and report verification are controlled. Intent defines required "
        "phase evidence, revisit confirmation, reports and resource limits.",
        "",
        "## Mission performance and incremental onboard cost",
        "",
        "Policies are compared on the same matched conditions within each intent. "
        "All costs are configured "
        "onboard estimates; remote server energy is unknown. Nonflight energy includes "
        "perception, communication and fixed downstream processing. Failures are included.",
        "",
    ]
    for mission in sorted({r["mission"] for r in summary["by_intent_policy"]}):
        group = [r for r in summary["by_intent_policy"] if r["mission"] == mission]
        if not group:
            continue
        lines += ["### " + mission, ""]
        if mission == "verified_inspection":
            lines += [
                "Known sensor-infeasible control on these recordings; keep separate "
                "from attainable-task ranking.",
                "",
            ]
        lines += [
            "| Router | Success | Phase completion | Timely report recall | Nonflight J "
            "| Misses / attempts | Mean run p95 (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in group:
            latency = r["mean_perception_latency_p95_s"]
            latency_text = f"{latency:.3f}" if latency is not None else "n/a"
            lines.append(
                f"| {r['policy']} | {r['mission_successes']}/{r['runs']} "
                f"| {r['mean_phase_completion_rate']:.1%} | {r['mean_report_recall']:.1%} "
                f"| {r['mean_energy_nonflight_j']:.1f} "
                f"| {r['perception_deadline_misses']}/{r['perception_attempts']} "
                f"| {latency_text} |"
            )
        lines += [""]
    lines += [
        "## Interpretation",
        "",
        "- Primary success remains the conjunction of the original intent's phase, report, "
        "execution and resource gates. One paid missed invocation can be recovered later.",
        "- Phase completion and intent-normalized recall are diagnostics, not replacements "
        "for success. Raw false components, false reports and duplicate reports are distinct.",
        "- Latency starts at capture availability and includes preparation, routing, loading, "
        "inference or virtual transport, decode and availability rounding. It includes failed "
        "selected attempts. The table averages per-run p95 values; it is not pooled p95.",
        "- Report quota time means first attainment of the required unique timely report "
        "count. It can precede later failure. Missing attainment stays null; conditional means "
        "are accompanied by attained/unattained counts in report.json. They do not rank overall "
        "mission speed. Successful evidence readiness is also conditional on final success.",
        "- Nominal flight duration is 120 s. Charged elapsed time can include final service "
        "overrun. This protocol has no early-return flight savings and no path-efficiency score.",
        "- The same raw false-component cap is accumulated over calls. A false-component "
        "rate is now reported to diagnose exposure, but does not replace that public gate.",
        "- Two inspected recordings and deterministic resource interventions support paired "
        "development comparisons, not population significance. No independent-trial CI is "
        "computed from the 576 correlated replays.",
        "",
        "## Artifacts",
        "",
        "- `report.json`: per-run values, policy summaries, network/deadline/battery strata "
        "and paired success/energy differences.",
        "- `public_contracts.json`: redacted intent plus per-invocation budgets for each "
        "matched condition; a documentation envelope for inputs already supplied separately.",
        "- `analysis_manifest.json`: source, original receipt and output digests; all "
        "primary outcomes verified unchanged.",
        "",
        "## Evaluation precedents",
        "",
        "[ALFRED](https://arxiv.org/pdf/1912.01734) motivates full and partial task "
        "completion. [Towards Streaming Perception](https://arxiv.org/pdf/2005.10420) "
        "motivates accounting for availability and missed observations; these metrics are "
        "not streaming AP. [ApproxDet](https://arxiv.org/pdf/2010.10754) evaluates "
        "accuracy, p95 latency and latency violations. [MLPerf Tiny]"
        "(https://arxiv.org/abs/2106.07597) evaluates accuracy, latency and energy jointly; "
        "our energy remains modeled rather than electrically measured.",
        "",
    ]
    return "\n".join(lines)


def build(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    require(not output.exists(), "Preserve existing analysis; select a new output directory")
    require(not output.is_relative_to(source), "Keep supplemental outputs outside frozen source")
    plan = verify(source)
    index = read(source / "run_index.json")
    require(index["complete_grid"] is True, "Study is incomplete")
    require(index["plan_sha256"] == sha(source / "plan.json"), "Index plan mismatch")
    validate_grid(plan, index["cases"])
    inputs = {
        str(source / n): sha(source / n) for n in ("plan.json", "run_index.json", "completion.json")
    }
    rows, public_contracts, paired_protocols = [], [], {}
    for item in index["cases"]:
        run = inside(source, item["path"])
        require(set(item["artifacts_sha256"]) == set(FILES), "Incomplete receipt inventory")
        data = {}
        for name in FILES:
            path = inside(source, run / name)
            require(
                sha(path) == item["artifacts_sha256"][name], "Changed run artifact: " + str(path)
            )
            inputs[str(path)] = item["artifacts_sha256"][name]
            data[name] = read(path)
        protocol, intent = data["protocol.json"], data["intent.json"]
        require(protocol["plan_sha256"] == index["plan_sha256"], "Run plan mismatch")
        require(
            all(protocol["case"][k] == item[k] for k in (*FACTORS, "run_id", "policy")),
            "Run case differs from index",
        )
        profiles = {p["name"]: p for p in plan["factors"]["deadline_profiles"]}
        require(
            protocol["deadline_profile"] == profiles[item["deadline_profile"]],
            "Run deadline budget differs from plan",
        )
        envelope = public_specification(intent, protocol)
        signature = (envelope, protocol["prediction_manifest_sha256"])
        key = condition(item)
        if key in paired_protocols:
            require(
                signature == paired_protocols[key], "Uncontrolled input differs across policies"
            )
        else:
            paired_protocols[key] = signature
            public_contracts.append({**{k: item[k] for k in FACTORS}, "contract": envelope})
        metrics = evaluate_run(
            intent, protocol, data["events.json"], data["results.json"], data["outcome.json"]
        )
        require(metrics["mission_success"] == item["mission_success"], "Primary outcome changed")
        rows.append({**{k: item[k] for k in (*FACTORS, "run_id", "policy")}, **metrics})
    summary = dict(
        schema_version=SCHEMA,
        analysis_kind="posthoc_diagnostics_unchanged_primary_endpoint",
        source=str(source),
        runs=len(rows),
        matched_conditions=len(paired_protocols),
        new_replays=0,
        new_neural_calls=0,
        changed_primary_outcomes=0,
        inference_scope="Descriptive paired development comparisons; no population inference.",
        by_intent_policy=aggregate(rows),
        by_recording=aggregate(rows, ("mission", "flight", "policy")),
        by_network=aggregate(rows, ("mission", "link", "policy")),
        by_deadline=aggregate(rows, ("mission", "deadline_profile", "policy")),
        by_initial_battery=aggregate(rows, ("mission", "initial_battery_frac", "policy")),
        paired_comparisons=paired_comparisons(rows),
        cells=rows,
        failed_constraint_counts=dict(
            Counter(reason for row in rows for reason in row["failed_constraints"])
        ),
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "report.json", summary)
    save(output / "public_contracts.json", public_contracts)
    (output / "REPORT.md").write_text(render(summary))
    sources = list(Path(__file__).parent.glob("*.py"))
    repository = Path(__file__).resolve().parents[2]
    sources += list((repository / "tests").glob("test_perception_evaluation*.py"))
    sources += [repository / "docs/perception_evaluation_protocol.md"]
    save(
        output / "analysis_manifest.json",
        dict(
            schema_version=SCHEMA,
            complete=True,
            analysis_kind=summary["analysis_kind"],
            runs=len(rows),
            primary_outcomes_reconstructed=len(rows),
            primary_outcomes_changed=0,
            input_sha256=inputs,
            implementation_sha256={str(p.resolve()): sha(p) for p in sources},
            output_sha256={
                n: sha(output / n) for n in ("report.json", "public_contracts.json", "REPORT.md")
            },
        ),
    )
    return {"runs": len(rows), "changed_primary_outcomes": 0, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
