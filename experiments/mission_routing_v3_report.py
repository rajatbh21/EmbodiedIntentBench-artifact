"""Summarize independently validated v3 paired cases without population claims."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v3.flights import verify
from experiments.mission_routing_v3.study import POLICIES


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
    if plan["source_sha256"] != sha(Path(__file__).with_name("mission_routing_v3_witness.py")):
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
                raise ValueError("Summary changed since validation")
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
    for row in rows:
        folder = root / row["path"]
        for filename, key in (
            ("outcome.json", "outcome_sha256"),
            ("execution/events.json", "events_sha256"),
            ("execution/results.json", "results_sha256"),
        ):
            if sha(folder / filename) != row[key]:
                raise ValueError("Changed outcome after validation")
    by_case = defaultdict(dict)
    for row in rows:
        by_case[row["flight"], row["mission"], row["power_w"]][row["policy"]] = row
    counts = {c: sum(r["full_success"] for r in rows if r["policy"] == c) for c in POLICIES}
    ablations = []
    for key, group in by_case.items():
        full = group["confirmation"]

        def schedule(row):
            return [
                (e["observation_id"], e["execution"]["config_id"])
                for e in read(root / row["path"] / "execution/events.json")
            ]

        for name in ("no_history_v3", "no_intent_v3", "no_context_v3", "constant_budget_v3"):
            other = group[name]
            ablations.append(
                dict(
                    flight=key[0],
                    mission=key[1],
                    power_w=key[2],
                    ablation=name,
                    schedule_changed=schedule(full) != schedule(other),
                    success_delta=int(full["full_success"]) - int(other["full_success"]),
                    phases_delta=full["phases_succeeded"] - other["phases_succeeded"],
                    recall_delta=full["report_recall"] - other["report_recall"],
                    battery_delta=full["final_battery"] - other["final_battery"],
                )
            )
    witnesses = witness_audit(root, p)
    summary = dict(
        schema_version="mission_routing_v3_summary_v1",
        complete=True,
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        physical_flights=len(p["cases"]),
        layouts=1,
        executions=len(rows),
        successes_by_policy=counts,
        cases_per_policy=len(by_case),
        ablations=ablations,
        cases=rows,
        evaluator_witnesses=witnesses,
        reporter_sha256=sha(Path(__file__)),
        interpretation="Paired descriptive cases on one known-map layout; not independent trials",
    )
    save(root / "analysis_v3.json", summary)
    lines = [
        "# Prospective confirmation-routing results",
        "",
        f"{len(rows)} verified paired executions; {len(p['cases'])} physical Gazebo recordings, "
        "one shared new layout on the same known lake map. Compute power is assumed 15 W.",
        "",
        "Every policy uses the same recorded flight within each lighting condition. "
        "All outcomes use the canonical mission evaluator. Counts below are descriptive cases, "
        "not independent flights or a population success estimate.",
        "",
        "| Policy | Successful intent/lighting cases |",
        "|---|---:|",
    ]
    lines.extend(f"| {c} | {counts[c]}/{len(by_case)} |" for c in POLICIES)
    lines.extend(
        [
            "",
            "## Ablations",
            "",
            "Deltas are full controller minus ablation.",
            "",
            "| Ablation | Changed schedules | Success delta | Phase delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in ("no_history_v3", "no_intent_v3", "no_context_v3", "constant_budget_v3"):
        group = [r for r in ablations if r["ablation"] == name]
        lines.append(
            f"| {name} | {sum(r['schedule_changed'] for r in group)}/{len(group)} | "
            f"{sum(r['success_delta'] for r in group):+d} | "
            f"{sum(r['phases_delta'] for r in group):+d} |"
        )
    lines.extend(
        [
            "",
            "## Scope",
            "",
            *["- " + s for s in p["caveats"]],
            "",
            "The old v2 cohort was inspected before this revision. It remains historical "
            "evidence, not an independent test of v3. No superiority claim follows from "
            "this implementation or from changed choices alone.",
            "",
        ]
    )
    if witnesses is not None:
        lines.extend(
            [
                "",
                "## Evaluator-only feasibility",
                "",
                f"Full-contract witnesses: {witnesses['witnessed_cases']}/{witnesses['cases']}. "
                "These schedules use hidden evidence and are excluded from every policy count. "
                "A successful replay proves feasibility, not deployable routing performance.",
                "",
            ]
        )
    (root / "REPORT_V3.md").write_text("\n".join(lines))
    print(
        {
            "executions": len(rows),
            "successes": counts,
            "ablations": [
                {
                    "name": name,
                    "changed": sum(
                        r["schedule_changed"] for r in ablations if r["ablation"] == name
                    ),
                }
                for name in ("no_history_v3", "no_intent_v3", "no_context_v3", "constant_budget_v3")
            ],
        },
        flush=True,
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
