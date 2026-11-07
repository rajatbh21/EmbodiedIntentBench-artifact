"""Score the frozen RGB recall predictor on the complete cohort, without fitting anything."""

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.protocol import verify
from experiments.mission_coverage_repair.study import load_profiles, scenario_for
from experiments.mission_routing_v2.features import encode
from experiments.mission_routing_v5.quality import predict
from experiments.v4_benchmark.backends import ReplayBackend


def summary(rows):
    trials = sum(r["trials"] for r in rows)
    if not trials:
        raise ValueError("Visible-target denominator required")
    result = dict(
        frames=len(rows),
        visible_observations=trials,
        observed_recall=sum(r["hits"] for r in rows) / trials,
        mean_predicted_recall=sum(r["trials"] * r["predicted"] for r in rows) / trials,
        brier=sum(
            r["hits"] * (1 - r["predicted"]) ** 2 + (r["trials"] - r["hits"]) * r["predicted"] ** 2
            for r in rows
        )
        / trials,
        global_reference_brier=sum(
            r["hits"] * (1 - r["global_prior"]) ** 2
            + (r["trials"] - r["hits"]) * r["global_prior"] ** 2
            for r in rows
        )
        / trials,
        mean_fraction_features_outside_training_range=float(np.mean([r["outside"] for r in rows])),
    )
    bins = []
    for i in range(5):
        members = [r for r in rows if min(int(r["predicted"] * 5), 4) == i]
        n = sum(r["trials"] for r in members)
        if n:
            bins.append(
                dict(
                    bin_index=i,
                    trials=n,
                    prediction=sum(r["trials"] * r["predicted"] for r in members) / n,
                    recall=sum(r["hits"] for r in members) / n,
                )
            )
    result["reliability_bins"] = bins
    return result


def run(root):
    p = verify(root)
    cohort = Path(p["cohort"])
    profile = load_profiles(p)["recall"]
    diagnostics = read(cohort / "feasibility_diagnostics.json")
    rows, hashes = [], {}
    seen = set()
    for flight, case in p["cases"].items():
        source_flight = case["source_flight"]
        if source_flight in seen:
            continue
        seen.add(source_flight)
        path = cohort / source_flight / "diagnostic_outcomes.json"
        expected = diagnostics["flights"][source_flight]["outcomes_sha256"]
        if sha(path) != expected:
            raise ValueError("Changed raw model outcomes")
        hashes[str(path)] = expected
        lookup = {(r["capture_id"], r["config_id"]): r for r in read(path)}
        scenario = scenario_for(root, p, flight, "rapid_survey", 15)
        backend = ReplayBackend(scenario, cohort / source_flight / "recording")
        try:
            for i, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                features = encode(backend.capture().rgb)
                for config, model in profile["models"].items():
                    predicted, outside = predict(model["base"], features)
                    outcome = lookup[i, config]
                    rows.append(
                        dict(
                            flight=source_flight,
                            layout=case["layout_group"],
                            config=config,
                            capture_id=i,
                            trials=len(outcome["visible_target_ids"]),
                            hits=len(outcome["matched_target_ids"]),
                            predicted=predicted,
                            global_prior=model["base"]["global_mean"],
                            outside=outside,
                        )
                    )
        finally:
            backend.close()
    if len(rows) != 3240:
        raise ValueError("Complete prediction grid required")
    summaries = [
        dict(config=c, **summary([r for r in rows if r["config"] == c])) for c in profile["models"]
    ]
    strata = [
        dict(
            config=c,
            layout=layout,
            **summary([r for r in rows if r["config"] == c and r["layout"] == layout]),
        )
        for c in profile["models"]
        for layout in sorted({r["layout"] for r in rows})
    ]
    save(
        root / "recall_transfer.json",
        dict(
            complete=True,
            posthoc=True,
            evaluation_labels_fitted=False,
            neural_calls_added=0,
            scope="Descriptive held-layout evaluation of the frozen per-visible-target recall head",
            predictor_changed=False,
            mission_success_probability_calibrated=False,
            rows=rows,
            summary=summaries,
            by_layout=strata,
            input_sha256=hashes,
            protocol_sha256=sha(root / "protocol.json"),
            source_sha256=sha(Path(__file__)),
            independent_layout_groups=3,
            confidence_intervals_claimed=False,
        ),
    )
    lines = [
        "# Frozen recall predictor: transfer diagnostic",
        "",
        "The same current-RGB recall head used by the coverage policies is evaluated "
        "against all 3,240 cached frame/configuration outcomes. No parameter, feature, "
        "bin boundary or policy is fitted to these labels. The five reliability bins "
        "are fixed equal-width intervals. Brier scores weight visible target-observations; "
        "repeated targets, frames and lighting mates are dependent. No confidence "
        "interval or population calibration claim is made.",
        "",
        "| Configuration | Predicted recall | Observed recall | Brier | Global-prior Brier |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {r['config']} | {r['mean_predicted_recall']:.3f} | {r['observed_recall']:.3f} | "
            f"{r['brier']:.3f} | {r['global_reference_brier']:.3f} |"
            for r in summaries
        ],
        "",
        "The global-prior comparator here is a recall predictor, not the "
        "`coverage_global` policy (which collapses complementarity prototypes while "
        "retaining context). This diagnostic does not evaluate calibrated mission "
        "completion probability, counterfactual reporting, or the cause of an individual "
        "policy failure. Whole-layout summaries and feature-range checks are retained "
        "in JSON so pooled numbers do not replace the three-group structure.",
        "",
    ]
    (root / "RECALL_TRANSFER.md").write_text("\n".join(lines))
    print(summaries, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
