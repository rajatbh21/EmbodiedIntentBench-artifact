"""Verify the original paper figure data using bundled numeric outputs and tables.

Standard library only. This checks figure provenance and reductions, not a new
physics simulation, neural inference run, or independent rerun of the Depot study.
The Lake validation experiments can be regenerated separately; see README.md.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def verify(package=HERE, validation=None):
    package = Path(package).resolve()
    validation = Path(validation).resolve() if validation else package.parent / "validation"
    data = read(package / "figure_data.json")
    manifest = read(package / "sources/manifest.json")
    checks = []
    sources = {}

    def eq(label, actual, expected, tolerance=1e-12):
        if isinstance(expected, float):
            matches = isinstance(actual, (float, int)) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=tolerance)
        else:
            matches = actual == expected
        if not matches:
            raise ValueError(f"{label}: source value differs from the preserved figure data")
        checks.append(label)

    def source(path, label, expected=None):
        digest = sha(path)
        if expected is not None:
            eq(label + " SHA-256", digest, expected)
        sources[label] = digest

    source(package / "figure_data.json", "figure_data.json", manifest["figure_data_sha256"])
    source(package / "sources/depot_summary.json", "sources/depot_summary.json", manifest["depot_summary_sha256"])
    source(package / "sources/depot_rows.json", "sources/depot_rows.json", manifest["depot_rows_sha256"])
    # Reuse the release's tables; do not duplicate its 2.6 MB runs.csv.
    used = ("runs.csv", "exp3_metrics.csv", "exp3_protocol_disagreement.csv",
            "exp3_timelines.json", "exp2_condition_classes.csv", "exp4_evidence_funnel.csv",
            "exp4_urgency_comparison.csv")
    for name in used:
        source(validation / name, "../validation/" + name, data["provenance"]["vendored"][name])
    # summary.json contains code-hash receipts which can change when release-only
    # metadata is sanitized. Its scientific cross-checks remain mandatory below.
    source(validation / "summary.json", "../validation/summary.json")
    summary = read(validation / "summary.json")

    depot = read(package / "sources/depot_summary.json")
    depot_rows = read(package / "sources/depot_rows.json")
    f4 = data["fig4"]
    eq("Fig4 original ledger digest", depot["original_ledger_sha256"], f4["ledger_sha256"])
    eq("Fig4 historical raw source digests", depot["source_hashes"], f4["raw_sha256"])
    eq("Fig4 predictor is frozen", depot["recall_diagnostic"]["evaluation_labels_fitted"], False)
    eq("Fig4 numeric row source digests", depot_rows["original_source_sha256"], f4["raw_sha256"])
    eq("Fig4 numeric rows evaluation labels not fitted", depot_rows["evaluation_labels_fitted"], False)
    eq("Fig4 numeric rows predictor unchanged", depot_rows["predictor_changed"], False)
    static = {r["policy"].removeprefix("static:"): r for r in depot["policy_summary"]}
    recall = {r["config"]: r for r in depot["recall_diagnostic"]["rows"]}
    eq("Fig4 complete model pool", set(static), {r["config"] for r in f4["rows"]})
    eq("Fig4 recall model pool", set(recall), set(static))
    eq("Fig4 model count", len(static), 9)
    cases, frames = depot_rows["policy_cases"], depot_rows["recall_rows"]
    eq("Fig4 original fixed-policy case count", len(cases), 144)
    eq("Fig4 original frame-configuration count", len(frames), 2160)
    eq("Fig4 case model pool", {r["policy"].removeprefix("static:") for r in cases}, set(static))
    eq("Fig4 frame model pool", {r["config"] for r in frames}, set(static))
    all_conditions = {(r["flight"], r["mission"], r["power_w"]) for r in cases}
    all_captures = {(r["flight"], r["capture_id"]) for r in frames}
    eq("Fig4 unique condition count", len(all_conditions), f4["conditions"])
    eq("Fig4 unique capture count", len(all_captures), f4["captures"])
    eq("Fig4 original recording count", len({r["flight"] for r in frames}), 4)
    for row in f4["rows"]:
        name = row["config"]
        eq(name + " successes", static[name]["successes"], row["successes"])
        eq(name + " conditions", static[name]["cases"], row["conditions"])
        eq(name + " panel conditions", row["conditions"], f4["conditions"])
        for source_key, figure_key in (("observed_recall", "observed_recall"),
                                       ("mean_predicted_recall", "predicted_recall"),
                                       ("frames", "captures"), ("visible_observations", "visible_observations")):
            eq(name + " " + source_key, recall[name][source_key], row.get(figure_key, f4.get(figure_key)))
        selected = [r for r in cases if r["policy"] == "static:" + name]
        observed = [r for r in frames if r["config"] == name]
        eq(name + " original case denominator", len(selected), row["conditions"])
        eq(name + " paired unique conditions", {(r["flight"], r["mission"], r["power_w"]) for r in selected}, all_conditions)
        eq(name + " original case successes", sum(r["full_success"] for r in selected), row["successes"])
        eq(name + " original capture denominator", len(observed), f4["captures"])
        eq(name + " paired unique captures", {(r["flight"], r["capture_id"]) for r in observed}, all_captures)
        trials = sum(r["trials"] for r in observed)
        eq(name + " original visible observations", trials, f4["visible_observations"])
        eq(name + " recall reduced from original rows", sum(r["hits"] for r in observed) / trials, row["observed_recall"])
        eq(name + " prediction reduced from original rows", sum(r["predicted"] * r["trials"] for r in observed) / trials, row["predicted_recall"])

    runs = table(validation / "runs.csv")
    ref = [r for r in runs if r["experiment"] == "reference" and r["contract"] == "confirm"
           and r["protocol"] == "causal" and float(r["latency_multiplier"]) == 1.0]
    eq("Reference row count", len(ref), 648)
    eq("Reference unique run count", len({r["run_id"] for r in ref}), 648)
    transfer = [r for r in ref if r["split"] == "transfer"]

    def aggregate(rows, policy, **where):
        chosen = [r for r in rows if r["policy"] == policy and all(r[k] == v for k, v in where.items())]
        if not chosen:
            raise ValueError("Empty policy-condition selection")
        return {"runs": len(chosen), "successes": sum(r["success"] == "True" for r in chosen),
                "deadline_misses": sum(int(r["perception_deadline_misses"]) for r in chosen),
                "mean_nonflight_energy_j": sum(float(r["nonflight_energy_j"]) for r in chosen) / len(chosen)}

    f5 = data["fig5"]
    for policy, expected in f5["urgency"].items():
        actual = aggregate(transfer, policy)
        for key, value in actual.items():
            eq("Fig5 urgency " + policy + " " + key, value, expected[key])
    for panel, column in (("link", "link"), ("battery", "energy_condition")):
        for policy, config in f5[panel].items():
            for level, expected in config["levels"].items():
                for key, value in aggregate(transfer, policy, **{column: level}).items():
                    eq("Fig5 " + panel + " " + policy + " " + level + " " + key, value, expected[key])
    (urgency,) = [r for r in table(validation / "exp4_urgency_comparison.csv")
                 if r["split"] == "transfer" and r["schedule"] == "all"
                 and r["aware"] == "risk_current" and r["blind"] == "risk_initial"]
    for policy, side in (("risk_current", "aware"), ("risk_initial", "blind")):
        for key in ("successes", "deadline_misses", "mean_nonflight_energy_j"):
            eq("Fig5 independent urgency table " + side + " " + key,
               float(urgency[side + "_" + key]), f5["urgency"][policy][key],
               tolerance=5e-3 if key == "mean_nonflight_energy_j" else 1e-12)

    f6 = data["fig6"]
    metrics = table(validation / "exp3_metrics.csv")
    disagreement = {float(r["multiplier"]): r for r in table(validation / "exp3_protocol_disagreement.csv")}
    for protocol, points in f6["sweep"].items():
        for point in points:
            multiplier = point["multiplier"]
            selected = [r for r in metrics if r["protocol"] == protocol and float(r["multiplier"]) == multiplier]
            eq(f"Fig6 {protocol} {multiplier} denominator", sum(int(r["runs"]) for r in selected), point["runs"])
            eq(f"Fig6 {protocol} {multiplier} successes", sum(int(r["successes"]) for r in selected), point["successes"])
            dis = disagreement[multiplier]
            eq(f"Fig6 {protocol} {multiplier} paired count", sum(int(v) for k, v in dis.items() if k != "multiplier"), 648)
            eq(f"Fig6 {protocol} {multiplier} paired successes", int(dis["both_succeed"]) + int(dis["only_" + protocol + "_succeeds"]), point["successes"])
    for policy, values in f6["ranking"].items():
        for protocol in ("causal", "no_skip"):
            (row,) = [r for r in metrics if r["policy"] == policy and r["protocol"] == protocol and float(r["multiplier"]) == 1.0]
            eq("Fig6 ranking " + policy + " " + protocol, int(row["successes"]), values[protocol])
            eq("Fig6 ranking denominator " + policy + " " + protocol, int(row["runs"]), 72)
            if protocol == "causal":
                eq("Fig6 reference ranking " + policy, aggregate(ref, policy)["successes"], values[protocol])
    timeline = f6["timeline"]
    cases = [r for r in read(validation / "exp3_timelines.json") if r["run_id"] == timeline["run_id"]]
    eq("Fig6 timeline protocols", sorted(r["protocol"] for r in cases), ["causal", "no_skip"])
    for case in cases:
        eq("Fig6 timeline predeclaration " + case["protocol"], case["selection"], "(predeclared)")
        focal = case["focal"]
        converted = {"success": case["success"], "captures": [
            {"t": c["t"], "processed": c["processed"], "done": c["done"],
             "matched": [focal.index(v) + 1 for v in c["matched"] if v in focal],
             "visible": [focal.index(v) + 1 for v in c["visible"] if v in focal]} for c in case["captures"]],
            "reports": [{"done": r["done"], "person": focal.index(r["target"]) + 1,
                         "evidence": r["evidence"]} for r in case["reports"]]}
        eq("Fig6 full timeline " + case["protocol"], converted, timeline[case["protocol"]])

    f7 = data["fig7"]
    classes = Counter(r["feasibility_class"] for r in table(validation / "exp2_condition_classes.csv") if r["contract"] == "confirm")
    expected_classes = {r["cls"]: r["conditions"] for r in f7["feasibility"]}
    eq("Fig7 complete feasibility partition", dict(classes), expected_classes)
    eq("Fig7 condition denominator", sum(classes.values()), 72)
    eq("Fig7 independent summary partition", {r["feasibility_class"]: r["conditions"] for r in summary["exp2"]["class_counts"] if r["contract"] == "confirm"}, expected_classes)
    (funnel,) = [r for r in table(validation / "exp4_evidence_funnel.csv") if r["policy"] == "risk_current"]
    (summary_funnel,) = [r for r in summary["exp4"]["funnel"] if r["policy"] == "risk_current"]
    eq("Fig7 person-condition denominator", int(funnel["targets"]), f7["funnel"]["instances"])
    for stage in f7["funnel"]["stages"]:
        eq("Fig7 funnel " + stage["stage"], int(funnel[stage["stage"]]), stage["count"])
        eq("Fig7 summary funnel " + stage["stage"], summary_funnel[stage["stage"]], stage["count"])

    return {"passed": True, "checks_passed": len(checks), "checks": checks,
            "source_sha256": sources, "verifier_sha256": sha(__file__),
            "scope": {"fig4": "Original Depot numeric case/frame rows with ledger cross-check; no raw Depot rerun or original full-file rehash.",
                      "fig5_to_fig7": "Reductions of the release validation tables, with cross-table consistency checks.",
                      "new_simulation": False, "new_inference": False, "new_policy_execution": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, help="Alternative directory of regenerated validation tables")
    parser.add_argument("--output", type=Path, help="Optionally save the verification receipt")
    args = parser.parse_args(argv)
    try:
        result = verify(validation=args.validation)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, "Source verification failed: " + str(error) + "\n")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": result["passed"], "checks_passed": result["checks_passed"], "scope": result["scope"]}, indent=2))


if __name__ == "__main__":
    main()
