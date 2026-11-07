"""Evaluate the unchanged RGB recall head on every newly acquired map frame."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_recall_check import summary
from experiments.mission_routing_v2.features import encode
from experiments.mission_routing_v5.quality import predict
from experiments.mission_transfer_posthoc_integrity import (
    check_diagnostics,
    checked_prediction,
    implementation_sha256,
)
from experiments.v4_benchmark.backends import ReplayBackend


def run(root):
    from experiments.mission_transfer_comparison_v4.protocol import verify
    from experiments.mission_transfer_comparison_v4.study import load_profiles, scenario_for

    p = verify(root)
    if (root / "recall_transfer.json").exists():
        raise FileExistsError("Preserve completed recall diagnostic")
    cohort = Path(p["cohort"])
    profile = load_profiles(p)["recall"]
    diagnostics = read(cohort / "feasibility_diagnostics.json")
    check_diagnostics(diagnostics, p, root, cohort)
    rows, hashes, expected_pairs = [], {}, 0
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
        if len(lookup) != len(read(path)):
            raise ValueError("Duplicate frame/configuration diagnostic outcome")
        scenario = scenario_for(root, p, flight, "rapid_survey", 15)
        checked_prediction(cohort, source_flight, p, scenario)
        manifest_path = cohort / source_flight / "predictions/manifest.json"
        if sha(manifest_path) != diagnostics["flights"][source_flight]["source_manifest_sha256"]:
            raise ValueError("Changed raw diagnostic prediction bank")
        hashes[str(manifest_path)] = sha(manifest_path)
        backend = ReplayBackend(scenario, cohort / source_flight / "recording")
        try:
            expected = {
                (i, config) for i in range(len(backend.frames)) for config in profile["models"]
            }
            if set(lookup) != expected:
                raise ValueError("Incomplete raw frame/configuration grid")
            expected_pairs += len(expected)
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
    if len(rows) != expected_pairs or seen != {c["source_flight"] for c in p["cases"].values()}:
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
            scope="Descriptive new-map evaluation after disclosed pre-inference geometry revision",
            predictor_changed=False,
            mission_success_probability_calibrated=False,
            rows=rows,
            summary=summaries,
            by_layout=strata,
            input_sha256=hashes,
            protocol_sha256=sha(root / "protocol.json"),
            source_sha256=sha(Path(__file__)),
            helper_sha256=implementation_sha256(),
            routing_admission_sha256=p["routing_admission_sha256"],
            independent_layout_groups=p["independent_evaluation_layouts"],
            independent_maps=1,
            confidence_intervals_claimed=False,
        ),
    )
    lines = [
        "# Frozen recall predictor: transfer diagnostic",
        "",
        "The same current-RGB recall head used by the coverage policies is evaluated "
        f"against all {len(rows):,} cached frame/configuration outcomes. No parameter, feature, "
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
        "in JSON so pooled numbers do not replace the within-map spatial structure. "
        "This map was not used for predictor training, but raw sensor observations were "
        "used to repair geometry and observation opportunity before model evaluation.",
        "",
    ]
    (root / "RECALL_TRANSFER.md").write_text("\n".join(lines))
    print(summaries, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
