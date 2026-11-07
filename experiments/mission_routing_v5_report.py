"""Exploratory recall revision report; requires independently reconstructed paid runs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v4_report import risk_audit
from experiments.mission_routing_v5.protocol import verify
from experiments.mission_routing_v5.study import POLICIES


def paired(rows, reference):
    keys = [(r["flight"], r["mission"]) for r in rows]
    refs = {(r["flight"], r["mission"]): r for r in reference}
    if len(set(keys)) != len(keys) or len(refs) != len(reference) or set(keys) != set(refs):
        raise ValueError("Paired comparison requires unique identical conditions")
    return [
        dict(
            flight=r["flight"],
            mission=r["mission"],
            phase_delta=r["phases_succeeded"]
            - refs[(r["flight"], r["mission"])]["phases_succeeded"],
            report_recall_delta=r["report_recall"]
            - refs[(r["flight"], r["mission"])]["report_recall"],
            false_component_delta=r["total_false_components"]
            - refs[(r["flight"], r["mission"])]["total_false_components"],
            battery_delta=r["final_battery"] - refs[(r["flight"], r["mission"])]["final_battery"],
            full_success_delta=int(r["full_success"])
            - int(refs[(r["flight"], r["mission"])]["full_success"]),
        )
        for r in rows
    ]


def summarize_folds(folds):
    keys = [(f["heldout"], f["config_id"]) for f in folds]
    flights, configs = {k[0] for k in keys}, {k[1] for k in keys}
    if (
        len(set(keys)) != len(keys)
        or set(keys) != {(f, c) for f in flights for c in configs}
        or len(flights) < 2
    ):
        raise ValueError("Complete unique flight/config validation grid required")
    return {
        mode: dict(
            macro_binomial_log_loss=float(
                np.mean([f["scores"][mode]["binomial_log_loss"] for f in folds])
            ),
            macro_weighted_frame_recall_mse=float(
                np.mean([f["scores"][mode]["weighted_frame_recall_mse"] for f in folds])
            ),
        )
        for mode in ("spatial", "base", "global_recall", "legacy_quality")
    }


def run(root):
    p = verify(root)
    source, training = Path(p["source"]), Path(p["recall_training"])
    validation = read(root / "validation.json")
    if not validation["complete"] or validation["cases"] != 18:
        raise ValueError("All 18 exploratory runs must be independently validated")
    if validation["validator_sha256"] != sha(
        Path(__file__).with_name("mission_routing_v5") / "validate.py"
    ):
        raise ValueError("Changed validator")
    audit = read(source / "completion_audit.json")
    for name in ("analysis_v4.json", "validation.json", "protocol.json"):
        if audit["artifact_sha256"][name] != sha(source / name):
            raise ValueError("Changed original cohort result")
    old = read(source / "analysis_v4.json")
    rows, hashes = [], {}
    for flight in p["cases"]:
        filename = f"summary_{flight}_15.json"
        if sha(root / filename) != validation["source_summaries_sha256"][filename]:
            raise ValueError("Changed validated summary")
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
            events, result = (
                read(destination / "execution/events.json"),
                read(destination / "execution/results.json"),
            )
            contract = IntentContract(read(source / flight / row["mission"] / "intent.json"))
            outcome = evaluate(contract, events, result)
            if (
                row["full_success"] != outcome["mission_success"]
                or row["phases_succeeded"] != sum(v["success"] for v in outcome["phases"])
                or row["report_recall"] != outcome["reporting"]["recall_by_deadline"]
                or row["final_battery"] != result["final_battery_frac"]
                or row["calls"] != len(events)
            ):
                raise ValueError("Incorrect outcome summary")
            risk_audit(events, contract.document()["mission"]["phases"])
            row = dict(
                row,
                total_false_components=sum(e["score"]["false_positive_components"] for e in events),
                mean_selected_out_of_range_fraction=float(
                    np.mean(
                        [
                            next(
                                c["out_of_range_feature_fraction"]
                                for c in e["router_decision"]["candidates"]
                                if c["config_id"] == e["execution"]["config_id"]
                            )
                            for e in events
                        ]
                    )
                ),
            )
            rows.append(row)
        hashes[filename] = sha(root / filename)
    expected = {(f, m, name) for f in p["cases"] for m in p["missions"] for name in POLICIES}
    keys = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    if len(set(keys)) != 18 or set(keys) != expected:
        raise ValueError("Incomplete exploratory grid")
    folds = read(training / "blocked_validation.json")["folds"]
    scores = summarize_folds(folds)
    baseline = [r for r in old["cases"] if r["policy"] == "risk_aware"]
    comparison = {
        name: paired([r for r in rows if r["policy"] == name], baseline) for name in POLICIES
    }
    result = dict(
        schema_version="conditional_recall_exploratory_report_v1",
        complete=True,
        scope=p["scope"],
        fresh_test=False,
        fitted_on_evaluation_labels=False,
        design_informed_by_inspected_outcomes=True,
        new_paid_replays=len(rows),
        new_physical_recordings=0,
        new_neural_calls=0,
        reproduced_public_decisions=validation["public_decisions_reproduced"],
        development_scores=scores,
        cases=rows,
        paired_vs_v4_risk=comparison,
        successes_by_policy={
            n: sum(r["full_success"] for r in rows if r["policy"] == n) for n in POLICIES
        },
        failure_counts={
            n: dict(Counter(k for r in rows if r["policy"] == n for k in r["failed_constraints"]))
            for n in POLICIES
        },
        inherited_feasibility=old["feasibility_by_case"],
        readiness=dict(
            promote_as_validated_improvement=False,
            fresh_independent_evaluation=False,
            spatial_beats_global_blocked_recall_loss=scores["spatial"]["macro_binomial_log_loss"]
            < scores["global_recall"]["macro_binomial_log_loss"],
            assessment_is_posthoc=True,
        ),
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        source_analysis_sha256=sha(source / "analysis_v4.json"),
        development_diagnostics_sha256=sha(training / "blocked_validation.json"),
        summaries_sha256=hashes,
        reporter_sha256=sha(Path(__file__)),
    )
    save(root / "analysis_v5.json", result)
    lines = [
        "# Conditional recall v5: exploratory paid replay",
        "",
        "This is reuse of the inspected v4 cohort, not a fresh test. Three new controllers "
        "share the frozen v4 risk estimator, ledger and limits, the same route, RGB, "
        "cached neural masks, intent contracts, tracker, and paid cost accounting. "
        "Coefficients use only the original 356 development captures. No existing "
        "cohort or result has been replaced.",
        "",
        f"Completed {len(rows)} paid replays; independently reconstructed "
        f"{validation['public_decisions_reproduced']} public decisions. "
        "No new physical flights or neural inference were run.",
        "",
        "## Development diagnostics",
        "",
        "Whole-flight blocked validation holds out one lighting condition. Both conditions "
        "share a layout and route; this is not independent-scene validation. Counts are "
        "temporally correlated. Values below are macro averages over 9 models and 2 folds, "
        "not independent-trial statistical estimates. Legacy quality predicts recall times "
        "precision, so its recall loss is a target-mismatch diagnostic, "
        "not a fair calibration comparison.",
        "",
        "| Predictor | Recall log loss | Weighted frame recall MSE |",
        "|---|---:|---:|",
    ]
    for name, m in scores.items():
        lines.append(
            f"| {name} | {m['macro_binomial_log_loss']:.4f} | "
            f"{m['macro_weighted_frame_recall_mse']:.4f} |"
        )
    lines += [
        "",
        "## Paid mission outcomes",
        "",
        "Each case still has six evaluator targets and four required phases. Counts of "
        "false components differ from false confirmed reports. Strict inspection retains "
        "the original latency-infeasible classification; it is not a tunable failure to erase.",
        "",
        "| Flight / mission | Policy | Full success | Phases | "
        "Report recall | False components | Battery |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for flight in p["cases"]:
        for mission in p["missions"]:
            group = [
                r for r in baseline + rows if r["flight"] == flight and r["mission"] == mission
            ]
            for r in group:
                lines.append(
                    f"| {flight} / {mission} | {r['policy']} | {int(r['full_success'])} | "
                    f"{r['phases_succeeded']}/4 | {r['report_recall']:.3f} | "
                    f"{r['total_false_components']} | {r['final_battery']:.6f} |"
                )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "These exploratory outcomes do not establish a validated improvement. The spatial "
        "head must be compared with base features and a global recall prior; added features "
        "are not evidence of better transfer. Visible-target recall is not the correctness "
        "of a pending track, and independent Bernoulli repeat support remains a heuristic. "
        "Feature-range excursions are diagnostics, not calibrated uncertainty "
        "or an automatic fallback.",
        "",
        "The four rapid/reserve conditions retain evaluator-only paid completion witnesses; "
        "both inspection conditions remain structurally infeasible under the fixed pool. "
        "The next evidence gate is development coverage across independent layouts, followed "
        "by a locked unseen evaluation. Do not tune on this replay and call it prospective.",
        "",
    ]
    (root / "REPORT_V5.md").write_text("\n".join(lines))
    print(dict(successes=result["successes_by_policy"], development=scores), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
