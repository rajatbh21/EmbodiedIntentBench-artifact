"""Report actual coverage, model complementarity and layout-blocked recall errors."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.routing_development_matrix.cohort import verify
from experiments.routing_development_matrix.score import verify_labels
from experiments.routing_development_matrix.train import verify_training

BINS = ("1-63 px", "64-255 px", "256-1023 px", "1024+ px")


def size_bin(pixels):
    if pixels <= 0:
        raise ValueError("Only visible target masks have a size stratum")
    return BINS[0 if pixels < 64 else 1 if pixels < 256 else 2 if pixels < 1024 else 3]


def summarize_coverage(observations):
    visible = [o for o in observations if o["visible"]]
    by_target = {}
    for target in sorted({o["target_id"] for o in observations}):
        rows = [o for o in observations if o["target_id"] == target]
        found = [o for o in rows if o["visible"]]
        by_target[target] = dict(
            visible_captures=len(found),
            captures=len(rows),
            max_visible_pixels=max(o["visible_pixels"] for o in rows),
        )
    return dict(
        target_capture_opportunities=len(observations),
        visible_target_observations=len(visible),
        missing_target_observations=len(observations) - len(visible),
        targets_ever_visible=sum(v["visible_captures"] > 0 for v in by_target.values()),
        targets=len(by_target),
        per_target=by_target,
        visible_mask_pixel_quantiles=np.percentile(
            [o["visible_pixels"] for o in visible], [0, 25, 50, 75, 100]
        ).tolist()
        if visible
        else None,
        visible_target_range_m_quantiles=np.percentile(
            [o["camera_to_target_origin_m"] for o in visible], [0, 25, 50, 75, 100]
        ).tolist()
        if visible
        else None,
        size_strata=dict(Counter(size_bin(o["visible_pixels"]) for o in visible)),
        occlusion_fraction_measured=False,
    )


def run(root, training):
    p = verify(root)
    verify_training(training)
    validation = read(root / "validation.json")
    if (
        not validation["complete"]
        or validation["protocol_sha256"] != sha(root / "protocol.json")
        or validation["validator_sha256"] != sha(Path(__file__).with_name("validate.py"))
    ):
        raise ValueError("Missing independent dataset validation")
    sensor = read(root / "observability.json")
    if sensor["protocol_sha256"] != sha(root / "protocol.json") or sensor["source_sha256"] != sha(
        Path(__file__).with_name("observability.py")
    ):
        raise ValueError("Changed sensor coverage assessment")
    cases, label_hashes = {}, {}
    for flight in p["cases"]:
        folder = verify_labels(root, flight)
        if validation["metadata"][flight]["labels_completion_sha256"] != sha(
            folder / "completion.json"
        ):
            raise ValueError("Changed independently validated labels")
        label_hashes[flight] = sha(folder / "completion.json")
        observations = read(folder / "visibility.json")["observations"]
        rows = read(folder / "outcomes.json")
        models = {}
        by_capture = defaultdict(list)
        for row in rows:
            by_capture[row["capture_id"]].append(row)
        obs = {(o["capture_id"], o["target_id"]): o for o in observations}
        for config in sorted({r["config_id"] for r in rows}):
            selected = [r for r in rows if r["config_id"] == config]
            hits = sum(len(r["matched_target_ids"]) for r in selected)
            trials = sum(len(r["visible_target_ids"]) for r in selected)
            strata = {b: dict(hits=0, trials=0) for b in BINS}
            for row in selected:
                for target in row["visible_target_ids"]:
                    key = size_bin(obs[row["capture_id"], target]["visible_pixels"])
                    strata[key]["trials"] += 1
                    strata[key]["hits"] += target in row["matched_target_ids"]
            models[config] = dict(
                matched_targets=hits,
                visible_targets=trials,
                visible_recall=hits / trials if trials else None,
                false_components=sum(r["false_positive_components"] for r in selected),
                size_strata=strata,
            )
        cases[flight] = dict(
            coverage=summarize_coverage(observations),
            models=models,
            best_fixed_matched_observations=max(m["matched_targets"] for m in models.values()),
            free_best_per_frame_matches=sum(
                max(len(r["matched_target_ids"]) for r in rows) for rows in by_capture.values()
            ),
            free_union_matches=sum(
                len(set().union(*(set(r["matched_target_ids"]) for r in rows)))
                for rows in by_capture.values()
            ),
        )
    diagnostics = read(training / "blocked_validation.json")
    by_group = {}
    for group in read(training / "plan.json")["folds"]:
        folds = [f for f in diagnostics["folds"] if f["heldout_group"] == group]
        by_group[group] = {
            mode: dict(
                scored_models=sum(f["supported"] for f in folds),
                macro_log_loss=float(
                    np.mean(
                        [f["scores"][mode]["binomial_log_loss"] for f in folds if f["supported"]]
                    )
                )
                if any(f["supported"] for f in folds)
                else None,
            )
            for mode in ("spatial", "base", "global_recall")
        }
    result = dict(
        schema_version="development_matrix_report_v1",
        complete=True,
        scope="development_only",
        new_physical_flights=validation["new_physical_flights"],
        new_captures=validation["new_captures"],
        actual_neural_calls=validation["actual_neural_calls"],
        new_frame_config_pairs=validation["new_frame_config_pairs"],
        full_training=read(training / "completion.json"),
        coverage=cases,
        layout_blocked_errors=by_group,
        sensor_coverage_and_intent_bounds=sensor["cases"],
        sensor_assessment_sha256=sha(root / "observability.json"),
        new_policy_missions_run=0,
        prospective_evaluation=False,
        independent_maps=0,
        validation_sha256=sha(root / "validation.json"),
        protocol_sha256=sha(root / "protocol.json"),
        training_completion_sha256=sha(training / "completion.json"),
        labels_sha256=label_hashes,
        training=str(training.resolve()),
        source_sha256=sha(Path(__file__)),
        caveat=(
            "Coverage and free GT model complementarity ignore paid scheduling; "
            "not achieved routing performance"
        ),
    )
    save(root / "analysis.json", result)
    lines = [
        "# Expanded development matrix",
        "",
        "This is development data, not a fresh policy test. Two new authored layouts on the "
        "known lake are each recorded under paired day and dusk/haze. Actual poses are identical "
        "within each pair. The original development group is retained as a third training block.",
        "",
        f"New data: {validation['new_physical_flights']} physical flights, "
        f"{validation['new_captures']} "
        f"captures, {validation['new_frame_config_pairs']} frame/config pairs, "
        f"{validation['actual_neural_calls']} actual neural calls. Every dense label and RGB "
        "feature is reconstructed from verified physical frames and prediction caches.",
        "",
        "## Observed coverage",
        "",
        "A missing target can be outside the camera or hidden by scenery; these are not isolated "
        "occlusion measurements. Pixel area is the visible object-ID mask, not bounding-box area. "
        "All collected captures and configurations remain included regardless of quality.",
        "",
        "| Flight | Targets ever visible | Visible target observations | "
        "Mask pixels min / median / max |",
        "|---|---:|---:|---|",
    ]
    for f, c in cases.items():
        v = c["coverage"]
        q = v["visible_mask_pixel_quantiles"]
        lines.append(
            f"| {f} | {v['targets_ever_visible']}/{v['targets']} | "
            f"{v['visible_target_observations']} | "
            + (f"{q[0]:.0f} / {q[2]:.0f} / {q[4]:.0f} |" if q else "none |")
        )
    lines += [
        "",
        "## Intent-specific sensor bound",
        "",
        "All-target coverage is a diagnostic, not a universal mission requirement. "
        "The same RGB stream can fail strict inspection yet leave rapid search unresolved. "
        "The bound below grants perfect perception and zero latency; a failure rules out "
        "the specified fixed-flight contract, while a pass is not a policy witness. "
        "The sixteen-pixel coverage screen is separate and was defined post hoc.",
        "",
        "| Flight | Rapid | Inspection | Reserve |",
        "|---|---|---|---|",
    ]
    for flight, case in sensor["cases"].items():
        bounds = case["sensor_contract_bounds"]
        lines.append(
            "| "
            + flight
            + " | "
            + " | ".join(
                bounds[m]["conclusion"]
                for m in ("rapid_survey", "verified_inspection", "reserve_patrol")
            )
            + " |"
        )
    lines += [
        "",
        "## Model complementarity (evaluator-only)",
        "",
        "These are dense matched target observations. The free best-per-frame and union bounds "
        "use hidden outcomes and ignore latency, energy and repeated report evidence. They are "
        "not executable routing baselines or mission success scores.",
        "",
        "| Flight | Best fixed | Free best per frame | Free union |",
        "|---|---:|---:|---:|",
    ]
    for f, c in cases.items():
        lines.append(
            f"| {f} | {c['best_fixed_matched_observations']} | "
            f"{c['free_best_per_frame_matches']} | {c['free_union_matches']} |"
        )
    lines += [
        "",
        "## Leave one entire layout block out",
        "",
        "Both lighting flights remain together. No layout or route crosses folds. The estimator "
        "and its fixed penalty are inherited from v5; there is no hyperparameter selection. "
        "Values are macro working recall log losses across configurations, without confidence "
        "intervals because frames/targets are correlated. Lower is better.",
        "",
        "| Held-out layout | Spatial RGB | Base RGB | Global recall |",
        "|---|---:|---:|---:|",
    ]
    for group, scores in by_group.items():
        vals = [scores[m]["macro_log_loss"] for m in ("spatial", "base", "global_recall")]
        lines.append(
            "| "
            + group
            + " | "
            + " | ".join(f"{v:.4f}" if v is not None else "unsupported" for v in vals)
            + " |"
        )
    lines += [
        "",
        "The full-data profile is ready for a later locked evaluation, but training and "
        "blocked prediction errors do not establish mission improvement. Risk heads and mission "
        "contracts have not been retuned. This expansion remains one map, static people, ideal "
        "pose and an authored scenario; it does not supply real UAV logs or controlled occlusion.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(
        dict(
            coverage={f: c["coverage"]["visible_target_observations"] for f, c in cases.items()},
            blocked=by_group,
        ),
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.training)
