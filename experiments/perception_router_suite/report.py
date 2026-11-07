"""Audit the adaptive selectors and describe a completed paired replay grid.

This is an arithmetic/decision audit, not an independent neural accuracy or
physical simulator implementation. Reports make no population or significance
claims from the two reused deterministic development recordings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path

ADAPTIVE = ("resource_threshold", "accuracy_latency", "supervised_utility", "linucb_frozen")
ORIGINAL = ("fixed_local_fast", "fixed_local_heavy", "fixed_remote_union", "deadline_aware")
LABELS = {
    "fixed_local_fast": "Fixed local fast",
    "fixed_local_heavy": "Fixed local heavy",
    "fixed_remote_union": "Fixed remote union",
    "deadline_aware": "Intent heuristic",
    "resource_threshold": "Resource thresholds",
    "accuracy_latency": "Accuracy / latency",
    "supervised_utility": "Supervised utility",
    "linucb_frozen": "Frozen LinUCB",
}
FACTORS = ("flight", "deadline_profile", "link", "initial_battery_frac", "mission")
REPO = Path(__file__).resolve().parents[2]
PREVIOUS = REPO / "results/mission_factor_sweeps_v1_20261028"


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def close(actual, expected, message):
    require(
        type(actual) in (int, float)
        and math.isfinite(actual)
        and math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-10),
        message,
    )


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing or escaped artifact")
    return path


def condition(row):
    return tuple(row[k] for k in FACTORS)


def has_hidden_labels(value):
    forbidden = {
        "evaluator_target_ids",
        "ground_truth",
        "actual_reward",
        "test_reward",
        "oracle_reward",
        "evaluator_labels",
        "target_ids",
    }
    if isinstance(value, dict):
        return bool(forbidden & set(value)) or any(has_hidden_labels(v) for v in value.values())
    return isinstance(value, list) and any(has_hidden_labels(v) for v in value)


def reconstructed_context(rgb_features, row, event, decision):
    """Independent reconstruction of the documented, public 24-feature context."""
    latency = (row["predicted_usable_finish_ns"] - decision["release_ns"]) / 1e9
    budget = decision["task_deadlines"]["perception_ns"] / 1e9
    reserve = decision["reserve_available_j"]
    intent = decision["intent_requirements"]
    return [
        *rgb_features,
        math.log1p(latency),
        math.log1p(budget),
        min(4.0, latency / budget),
        math.log1p(row["estimate"]["energy_j"]),
        math.log1p(row["estimate"]["communication_mb"]),
        float(row["placement"] == "remote"),
        event["battery_at_decision"],
        math.log1p(max(0.0, reserve) / 1000),
        min(4.0, row["reserve_required_j"] / max(1.0, reserve)),
        float(intent["min_hits"]),
        math.log1p(intent["min_evidence_span_s"]),
        math.log1p(intent["max_false_components"]),
    ]


def linear_score(head, context):
    require(
        len(context) == len(head["mean"]) == len(head["scale"]) == len(head["coefficient"]),
        "Changed learned context dimensions",
    )
    require(all(math.isfinite(v) for v in context), "Nonfinite learned context")
    require(all(math.isfinite(v) and v > 0 for v in head["scale"]), "Invalid learned scale")
    value = math.fsum(
        (v - m) / s * c
        for v, m, s, c in zip(
            context, head["mean"], head["scale"], head["coefficient"], strict=True
        )
    )
    return min(1.0, max(0.0, value))


def audit_decision(event, intent, profile, *, rgb_features=None, actions=None):
    """Recompute ranking from forecasts, not by calling the selection policy."""
    decision = event["router_decision"]
    name = decision["policy"]
    require(name in ADAPTIVE, "Expected an adaptive policy")
    require(not has_hidden_labels(decision), "Hidden labels in decision record")
    require(decision["evaluation_learning_updates"] == 0, "Evaluation-time learning update")
    require(decision["common_resource_guard"] == "deadline_aware_v1", "Changed resource guard")
    require(decision["action_id"] == event["selected_action_id"], "Decision/execution mismatch")
    expected_digest = (
        document_sha(profile) if name in ("supervised_utility", "linucb_frozen") else None
    )
    require(decision["learned_profile_sha256"] == expected_digest, "Changed learned profile digest")
    rows = decision["candidates"]
    require(len({r["action_id"] for r in rows}) == len(rows), "Duplicate candidate")
    if actions is not None:
        require({r["action_id"] for r in rows} == set(actions), "Changed action pool")
    limits = intent["limits"]
    admitted = []
    for row in rows:
        if actions is not None:
            action = actions[row["action_id"]]
            require(row["placement"] == action["placement"], "Changed candidate placement")
            legal = (
                action["placement"] == "local" or limits["privacy_level"] == "remote_allowed"
            ) and action["memory_mb"] + 68 <= 4096
            require(row["legal"] == legal, "Changed legal admission")
        flags = (
            "task_deadline_feasible",
            "phase_deadline_feasible",
            "battery_feasible",
            "communication_feasible",
        )
        finish = row["predicted_usable_finish_ns"]
        if finish is None:
            require(not any(row[k] for k in flags), "Undeliverable candidate marked feasible")
        else:
            expected = (
                finish <= decision["release_ns"] + decision["task_deadlines"]["perception_ns"],
                finish + 150_000_000
                <= round(
                    min(decision["intent_requirements"]["deadline_s"], limits["deadline_s"]) * 1e9
                ),
                row["reserve_required_j"] <= decision["reserve_available_j"] + 1e-9,
                row["projected_communication_mb"] <= limits["communication_budget_mb"] + 1e-9,
            )
            require(tuple(row[k] for k in flags) == expected, "Inconsistent forecast feasibility")
        feasible = row["legal"] and all(row[k] for k in flags)
        require(row["feasible"] == feasible, "Inconsistent aggregate feasibility")
        if feasible:
            admitted.append(row)
        else:
            require(row["routing_score"] is None, "Excluded action assigned a ranking score")
    if not admitted:
        require(decision["action_id"] is None, "Action selected without an eligible candidate")
        require(decision["reason"] == "no_feasible_action", "Wrong abstention reason")
        return dict(decisions=1, abstentions=1, learned_scores=0)
    require(decision["action_id"] is not None, "Abstained despite eligible candidate")

    def tie(row):
        return (row["predicted_usable_finish_ns"], row["estimate"]["energy_j"], row["action_id"])

    scores = 0
    if name == "resource_threshold":
        require(
            decision["thresholds"]
            == dict(battery_headroom=0.20, upload_bps=10_000_000, rtt_ns=100_000_000),
            "Changed thresholds",
        )
        require(
            all(r["routing_score"] is None for r in admitted), "Threshold policy logged a score"
        )
        if event["battery_at_decision"] - limits["min_final_battery_frac"] <= 0.20:
            selected = min(admitted, key=lambda r: (r["estimate"]["energy_j"], *tie(r)))
        else:
            link = event["public_link"]
            good = (
                not link["outage"]
                and link["upload_bps"] >= 10_000_000
                and link["rtt_ns"] <= 100_000_000
            )
            preferred = ("REMOTE__" if good else "") + "CFG_MASKRCNN_1280_ROT0_90"
            selected = next((r for r in admitted if r["action_id"] == preferred), None)
            selected = selected or min(admitted, key=tie)
    else:
        for row in admitted:
            if name == "accuracy_latency":
                expected_score = row["predicted_recall"]
            else:
                require(rgb_features is not None, "Public RGB features required for learned audit")
                expected_context = reconstructed_context(rgb_features, row, event, decision)
                require(
                    len(expected_context) == len(row["learned_context"]),
                    "Changed context dimensions",
                )
                for logged, expected_value in zip(
                    row["learned_context"], expected_context, strict=True
                ):
                    close(logged, expected_value, "Learned context differs from public observation")
                section = "supervised" if name == "supervised_utility" else "bandit"
                expected_score = linear_score(profile[section][row["action_id"]], expected_context)
                scores += 1
            close(row["routing_score"], expected_score, "Changed routing objective score")
        selected = min(admitted, key=lambda r: (-r["routing_score"], *tie(r)))
    require(
        decision["action_id"] == selected["action_id"], "Selection violates objective or tie break"
    )
    return dict(decisions=1, abstentions=0, learned_scores=scores)


def audit_parity(root, rows, previous=PREVIOUS):
    """Match factors rather than run IDs; require byte-identical old-policy outputs."""
    previous = Path(previous).resolve()
    old_index = read(previous / "run_index.json")
    completion = read(previous / "completion.json")
    require(
        completion["complete"] is True and old_index["complete_grid"] is True,
        "Incomplete historical reference",
    )
    old = {(r["policy"], condition(r)): r for r in old_index["cases"]}
    require(len(old) == len(old_index["cases"]), "Duplicate historical conditions")
    bindings = {
        str(previous / "completion.json"): sha(previous / "completion.json"),
        str(previous / "run_index.json"): sha(previous / "run_index.json"),
    }
    count = 0
    for row in rows:
        if row["policy"] not in ORIGINAL:
            continue
        match = old.get((row["policy"], condition(row)))
        require(match is not None, "Historical paired reference missing")
        for name in ("events.json", "results.json", "outcome.json"):
            new_path = checked_path(root, Path(row["path"]) / name)
            old_path = checked_path(previous, Path(match["path"]) / name)
            expected = match["artifacts_sha256"][name]
            require(sha(old_path) == expected, "Changed historical output")
            require(
                sha(new_path) == expected, "Original policy output differs from historical replay"
            )
            bindings[str(old_path)] = expected
        count += 1
    return dict(
        runs=count,
        artifacts_sha256=bindings,
        comparison="Byte-identical events, results and outcomes for matched original policies.",
    )


def extract_cell(item, events, result, outcome):
    task = outcome["task_metrics"]["perception"]
    attempts = task["attempts"]
    actions = Counter(e["selected_action_id"] or "ABSTAIN" for e in events)
    # Resident state is retained across abstention. Initial load is not a switch.
    previous = None
    switches = 0
    for event in events:
        action = event["selected_action_id"]
        if action is not None:
            switches += previous is not None and action != previous
            previous = action
    return dict(
        **{k: item[k] for k in ("run_id", "policy", *FACTORS)},
        known_infeasible_intent=item["mission"] == "verified_inspection",
        mission_success=int(outcome["mission_success"]),
        phase_fraction=sum(p["success"] for p in outcome["phases"]) / len(outcome["phases"]),
        report_recall=outcome["reporting"]["recall_by_deadline"],
        perception_attempts=attempts,
        perception_deadline_misses=task["deadline_misses"],
        perception_deadline_miss_rate=task["deadline_misses"] / attempts if attempts else None,
        final_battery_frac=result["final_battery_frac"],
        battery_reserve_met=int(result["constraints"]["battery_constraint_success"]),
        energy_j=result["cumulative_energy_j"],
        communication_mb=result["cumulative_communication_mb"],
        abstentions=outcome["abstentions"],
        switches=switches,
        actions=dict(actions),
    )


MEAN_METRICS = (
    "phase_fraction",
    "report_recall",
    "perception_deadline_miss_rate",
    "final_battery_frac",
    "energy_j",
    "communication_mb",
    "abstentions",
    "switches",
)


def mean(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


def summarize(cells):
    result = []
    for mission, policy in itertools.product(
        sorted({c["mission"] for c in cells}), dict.fromkeys(c["policy"] for c in cells)
    ):
        group = [c for c in cells if c["mission"] == mission and c["policy"] == policy]
        require(group, "Missing policy/intent cells")
        result.append(
            dict(
                mission=mission,
                policy=policy,
                runs=len(group),
                mission_successes=sum(c["mission_success"] for c in group),
                battery_reserve_met=sum(c["battery_reserve_met"] for c in group),
                perception_attempts=sum(c["perception_attempts"] for c in group),
                perception_deadline_misses=sum(c["perception_deadline_misses"] for c in group),
                **{k: mean([c[k] for c in group]) for k in MEAN_METRICS},
            )
        )
    return result


def paired_differences(cells):
    """Within-condition differences; no random-replicate or statistical claims."""
    policies = list(dict.fromkeys(c["policy"] for c in cells))
    lookup = {(c["policy"], condition(c)): c for c in cells}
    require(len(lookup) == len(cells), "Duplicate paired cell")
    output = []
    for mission in sorted({c["mission"] for c in cells}):
        for reference, policy in itertools.combinations(policies, 2):
            a = [c for c in cells if c["mission"] == mission and c["policy"] == reference]
            b = [lookup.get((policy, condition(c))) for c in a]
            require(all(v is not None for v in b), "Incomplete paired policy conditions")
            diffs = [y["mission_success"] - x["mission_success"] for x, y in zip(a, b, strict=True)]
            output.append(
                dict(
                    mission=mission,
                    reference=reference,
                    policy=policy,
                    pairs=len(a),
                    mission_wins=sum(d > 0 for d in diffs),
                    mission_losses=sum(d < 0 for d in diffs),
                    mission_ties=sum(d == 0 for d in diffs),
                    mission_success_delta=mean(diffs),
                    **{
                        k + "_delta": mean(
                            [
                                y[k] - x[k]
                                for x, y in zip(a, b, strict=True)
                                if x[k] is not None and y[k] is not None
                            ]
                        )
                        for k in MEAN_METRICS
                    },
                )
            )
    return output


def csv_write(path, rows):
    require(rows, "Cannot write an empty metric table")
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary, paired, audit):
    def pct(value):
        return "NA" if value is None else f"{value * 100:.1f}%"

    lines = [
        "# Adaptive perception router comparison",
        "",
        f"**{audit['runs']} completed, audited replays** compare eight policies on two fixed "
        "Gazebo recordings, crossed with 2/4/8-second perception-call deadlines, three virtual "
        "networks, 34%/65% initial battery, and two intents. All policies share 18 local/remote "
        "actions, cached actual neural predictions, fixed terrain association and "
        "Repeat2 verification.",
        "",
        "| Policy | Mechanism |",
        "|---|---|",
        "| Fixed local fast / heavy / remote union | Original fixed controls. |",
        "| Intent heuristic | Original recall, evidence-window and false-component score. |",
        "| Resource thresholds | Low reserve: minimum service energy; otherwise prefer heavy "
        "local/remote according to current link, with feasible fastest fallback. |",
        "| Accuracy / latency | Maximum predicted recall among actions admitted by "
        "the common resource guard. |",
        "| Supervised utility | Ridge utility predictor trained with all development "
        "action rewards. |",
        "| Frozen LinUCB | Disjoint contextual bandit trained with one selected-action reward "
        "per development decision; greedy frozen means at evaluation. |",
        "",
        "All adaptive policies use the same forecast admission for invocation/phase deadlines, "
        "memory, privacy, communication and battery reserve. An empty feasible set produces "
        "paid abstention. "
        "These are mechanism-level baselines, not complete reproductions of ApproxDet or Chanakya. "
        "A confidence cascade is not implemented: it would require charging every "
        "sequential inference.",
        "",
        "## Results by intent",
        "",
        "Strict inspection is a known sensor-infeasible control, separated from rapid survey. "
        "Rates are averages of per-run rates; deadline counts are also provided in CSV. "
        "The denominator is 36 conditions per policy per intent, not 36 independent random trials.",
        "",
    ]
    for mission, title in (
        ("rapid_survey", "Rapid survey"),
        ("verified_inspection", "Strict inspection: infeasible control"),
    ):
        lines += [
            f"### {title}",
            "",
            "| Policy | Missions | Phases | Report recall | Deadline misses | Final battery | "
            "Reserve met | Traffic MB | Switches |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary:
            if row["mission"] != mission:
                continue
            lines.append(
                f"| {LABELS[row['policy']]} | {row['mission_successes']}/{row['runs']} | "
                f"{pct(row['phase_fraction'])} | {pct(row['report_recall'])} | "
                f"{pct(row['perception_deadline_miss_rate'])} | {pct(row['final_battery_frac'])} | "
                f"{row['battery_reserve_met']}/{row['runs']} | {row['communication_mb']:.2f} | "
                f"{row['switches']:.2f} |"
            )
        lines.append("")
    lines += [
        "## Paired differences from the original intent heuristic",
        "",
        "Positive deltas favor the new policy for success and phase completion. "
        "Every pair has the same recording and constraint settings. All policy pairs are in CSV.",
        "",
        "| Intent | New policy | Wins / losses / ties | Success delta (pp) | Phase delta (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in paired:
        if row["reference"] == "deadline_aware" and row["policy"] in ADAPTIVE:
            lines.append(
                f"| {row['mission']} | {LABELS[row['policy']]} | "
                f"{row['mission_wins']} / {row['mission_losses']} / {row['mission_ties']} | "
                f"{row['mission_success_delta'] * 100:+.1f} | "
                f"{row['phase_fraction_delta'] * 100:+.1f} |"
            )
    lines += [
        "",
        "## Verification and limits",
        "",
        f"The independent receipt audit passes all {audit['runs']} runs. The supplemental audit "
        "reconstructs adaptive eligibility and selection from logged forecasts, actual public RGB "
        "features and frozen numerical heads. Evaluation learning updates are zero. "
        f"All {audit['historical_parity']['runs']} original-policy replays have byte-identical "
        "events, results and outcomes to their matched earlier executions. Source, training and "
        "run artifacts are hash-bound. These checks do not independently reimplement forecast "
        "models, vision inference or the full mission scorer.",
        "",
        "Training and evaluation use development evidence, so results do not establish held-out "
        "generalization. Network, power and controller overhead remain configured; neural masks "
        "are cached. No live cloud inference, fresh flight or hardware energy measurement "
        "is claimed. The shared routing overhead does not establish measured runtime parity "
        "between controllers. Observed differences show behavior on this fixed grid; "
        "no p-values or significance claims are made.",
        "",
        "`cells.csv` contains per-run metrics; `policy_summary.csv` contains intent-separated "
        "means/counts; `paired_differences.csv` contains all matched policy comparisons; "
        "`action_distribution.csv` counts selections including abstentions. Switches exclude the "
        "initial load and preserve the resident model across abstentions. `router_audit.json` "
        "and `independent_audit.json` bind the supporting evidence.",
        "",
    ]
    return "\n".join(lines)


def build(root):
    from experiments.mission_factor_sweeps.execution import load_shared_inputs
    from experiments.mission_routing_v2.features import encode
    from experiments.mission_task_deadline_audit import audit as receipt_audit
    from experiments.perception_router_suite.study import verify

    root = Path(root).resolve()
    names = (
        "independent_audit.json",
        "router_audit.json",
        "analysis.json",
        "cells.csv",
        "policy_summary.csv",
        "paired_differences.csv",
        "action_distribution.csv",
        "REPORT.md",
    )
    require(not any((root / name).exists() for name in names), "Preserve previous report outputs")
    verified = verify(root)
    plan = verified[0] if isinstance(verified, tuple) else verified
    index, completion = read(root / "run_index.json"), read(root / "completion.json")
    require(
        completion["complete"] is True and index["complete_grid"] is True,
        "Completed study required",
    )
    require(plan["expected_runs"] == len(index["cases"]) == 576, "Full 576-run grid required")
    require(index["plan_sha256"] == sha(root / "plan.json"), "Changed index plan binding")
    independent = receipt_audit(root)
    _, banks, _ = load_shared_inputs(plan)
    rgb = {
        (flight, index): encode(frame.rgb).tolist()
        for flight, bank in banks.items()
        for index, frame in bank[0].items()
    }
    profile = read(plan["learned_profile_path"])
    parent = read(Path(plan["parent_root"]) / "plan.json")
    actions = {a["action_id"]: a for a in parent["actions_by_scale"]["1.0"]}
    require(
        profile["action_signatures"] == {k: v["source_signature"] for k, v in actions.items()},
        "Learned profile action identity mismatch",
    )
    cells, bindings, counts = [], {}, Counter()
    for row in index["cases"]:
        directory = root / row["path"]
        for relative, expected in row["artifacts_sha256"].items():
            path = checked_path(root, Path(row["path"]) / relative)
            require(sha(path) == expected, "Changed run artifact")
            bindings[str(path.relative_to(root))] = expected
        events, result, outcome, intent = (
            read(directory / (name + ".json"))
            for name in ("events", "results", "outcome", "intent")
        )
        if row["policy"] in ADAPTIVE:
            for event in events:
                require(event["router_decision"]["policy"] == row["policy"], "Wrong policy record")
                counts.update(
                    audit_decision(
                        event,
                        intent,
                        profile,
                        rgb_features=rgb[(row["flight"], event["observation_id"])],
                        actions=actions,
                    )
                )
        cells.append(extract_cell(row, events, result, outcome))
    require(len({(c["policy"], condition(c)) for c in cells}) == 576, "Duplicate condition")
    parity = audit_parity(root, index["cases"], plan.get("old_reference_root", PREVIOUS))
    require(parity["runs"] == 288, "Incomplete historical parity comparison")
    require(
        bindings == independent["audited_artifacts_sha256"], "Audit artifact inventory mismatch"
    )
    audit = dict(
        schema_version="perception_router_selection_audit_v1",
        passed=True,
        runs=len(cells),
        counts=dict(counts),
        plan_sha256=sha(root / "plan.json"),
        completion_sha256=sha(root / "completion.json"),
        learned_profile_sha256=sha(plan["learned_profile_path"]),
        learned_profile_document_sha256=document_sha(profile),
        evaluation_learning_updates=0,
        logged_hidden_labels=False,
        public_rgb_context_reconstructed=True,
        historical_parity=parity,
        audit_source_path=str(Path(__file__).resolve()),
        audit_source_sha256=sha(__file__),
        audited_artifacts_sha256=bindings,
    )
    summary, paired = summarize(cells), paired_differences(cells)
    action_counts = Counter()
    for cell in cells:
        for action, count in cell["actions"].items():
            action_counts[(cell["mission"], cell["policy"], action)] += count
    action_rows = [
        dict(mission=m, policy=p, action_id=a, selections=n)
        for (m, p, a), n in sorted(action_counts.items())
    ]
    for name, value in (("independent_audit.json", independent), ("router_audit.json", audit)):
        with (root / name).open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    csv_write(root / "cells.csv", [{k: v for k, v in c.items() if k != "actions"} for c in cells])
    csv_write(root / "policy_summary.csv", summary)
    csv_write(root / "paired_differences.csv", paired)
    csv_write(root / "action_distribution.csv", action_rows)
    with (root / "REPORT.md").open("x") as stream:
        stream.write(render_report(summary, paired, audit))
    analysis = dict(
        schema_version="perception_router_analysis_v1",
        complete=True,
        runs=len(cells),
        cells=cells,
        summary=summary,
        paired_differences=paired,
        source_sha256={
            name: sha(root / name)
            for name in (
                "plan.json",
                "completion.json",
                "run_index.json",
                "independent_audit.json",
                "router_audit.json",
                "cells.csv",
                "policy_summary.csv",
                "paired_differences.csv",
                "action_distribution.csv",
                "REPORT.md",
            )
        },
    )
    with (root / "analysis.json").open("x") as stream:
        json.dump(analysis, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = build(args.root)
    print(json.dumps(dict(runs=result["runs"], summary=result["summary"])))


if __name__ == "__main__":
    main()
