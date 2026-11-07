"""Development-only, whole-layout recall audit; never changes deployed profiles.

Prepare pins the candidate and evidence before run computes any held-out score.
The sole new candidate clips training-range extrapolation and shrinks toward the
same training-only global prior used by the existing base recall model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.mission_routing_v5.quality import BASE_NAMES, NAMES, fit, predict

METHODS = ("global_prior", "base_recall", "range_shrink")
SPEC = {
    "schema_version": "development_recall_robustness_spec_v1",
    "methods": list(METHODS),
    "existing_base": "mission_routing_v5.quality.fit/predict, first 12 RGB features",
    "alpha": 10.0,
    "prior": "(training target hits + 1) / (training visible target trials + 2)",
    "candidate": {
        "name": "range_shrink",
        "rule": "prior + (clipped_base_prediction - prior) / (1 + max_excess)",
        "clipping": "Clip varying non-intercept features to fold-training min/max",
        "max_excess": "Maximum distance outside those ranges / fold-training scale",
        "constant_features": "Set constant non-intercept features to training mean; exclude excess",
        "intercept": "Retain required intercept 1; exclude clipping and excess",
        "in_range_weight": 1.0,
        "tuned_parameters": [],
    },
    "split": "Leave one complete spatial layout out; both lighting mates stay together",
    "fit_visibility": "Exclude zero-visible frames, using the existing fitter",
    "score_visibility": (
        "Zero-visible frames retained in predictions but excluded from losses/regret"
    ),
    "primary_aggregation": (
        "Equal mean over held-out layouts; each fold pools target trials/configurations"
    ),
    "secondary_aggregation": (
        "Pooled held-out target observations, with all configurations retained"
    ),
    "metrics": ["binary_target_brier", "binomial_log_loss", "configuration_ranking_regret"],
    "log_loss_probability_clip": 1e-9,
    "ranking": "Choose largest predicted recall; lexical configuration ID breaks exact ties",
    "ranking_regret": (
        "Sum(frame best observed hits - selected observed hits) / sum(frame visible trials)"
    ),
    "ranking_costs_accounted": False,
    "candidate_specified_after_prior_test_diagnostics": True,
    "new_evaluation_labels_used": False,
    "hyperparameter_search": False,
    "automatic_policy_promotion": False,
    "mission_superiority_claimed": False,
    "confidence_intervals": False,
    "p_values": False,
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def check_hashes(inventory):
    for path, expected in inventory.items():
        if sha(path) != expected:
            raise ValueError(f"Changed pinned input or source: {path}")


def validate_groups(flights):
    groups = {f["group"] for f in flights}
    if len(groups) < 3:
        raise ValueError("At least three complete layout groups required")
    if len({f["name"] for f in flights}) != len(flights):
        raise ValueError("Duplicate flight name")
    recordings = [f["lineage"]["recording"] for f in flights]
    if len(set(recordings)) != len(recordings):
        raise ValueError("Duplicate recording")
    for group in groups:
        members = [f for f in flights if f["group"] == group]
        if len(members) != 2 or {f["lighting"] for f in members} != {"day", "dusk_haze"}:
            raise ValueError("Keep exactly both lighting mates in each layout")
    for i, left in enumerate(flights):
        for right in flights[i + 1 :]:
            if left["group"] != right["group"] and any(
                left["lineage"][k] == right["lineage"][k] for k in ("layout", "route")
            ):
                raise ValueError("Shared layout or route would cross a held-out fold")


def label_arrays(rows, captures, configs):
    """Check a complete dense table; counts come from actual cached predictions."""
    index = {}
    for row in rows:
        key = (row["capture_id"], row["config_id"])
        if key in index:
            raise ValueError("Duplicate capture/configuration outcome")
        visible, matched = row["visible_target_ids"], row["matched_target_ids"]
        if len(set(visible)) != len(visible) or len(set(matched)) != len(matched):
            raise ValueError("Repeated target identity in count labels")
        if not set(matched) <= set(visible):
            raise ValueError("Matched target is not visible")
        index[key] = row
    if set(index) != {(i, c) for i in range(captures) for c in configs}:
        raise ValueError("Incomplete dense outcome grid")
    for i in range(captures):
        reference = index[(i, configs[0])]
        for c in configs:
            row = index[(i, c)]
            if (
                set(row["visible_target_ids"]) != set(reference["visible_target_ids"])
                or row["capture_time_s"] != reference["capture_time_s"]
            ):
                raise ValueError("Configuration-dependent visibility or capture time")
    return {
        c: {
            "hits": [len(index[(i, c)]["matched_target_ids"]) for i in range(captures)],
            "trials": [len(index[(i, c)]["visible_target_ids"]) for i in range(captures)],
        }
        for c in configs
    }


def load_development(training):
    """Read only inputs already bound by the existing development training plan.

    No evaluation recording/labels and no old CV scores are opened. The existing
    independently validated dense outcome tables avoid repeated neural inference.
    """
    training = Path(training).resolve()
    plan, complete = read(training / "plan.json"), read(training / "completion.json")
    if (
        plan["schema_version"] != "anonymous_coverage_training_v1"
        or plan["evaluation_labels_used"]
        or not complete["complete"]
        or complete["plan_sha256"] != sha(training / "plan.json")
        or complete["profile_sha256"] != sha(training / "profile.json")
    ):
        raise ValueError("Require the unchanged completed development-only inventory")
    check_hashes(plan["input_sha256"])
    check_hashes(plan["source_sha256"])
    inputs = dict(plan["input_sha256"])
    for name in ("plan.json", "completion.json", "profile.json"):
        path = training / name
        inputs[str(path)] = sha(path)
    original = Path(plan["original_training"])
    old_complete = read(original / "completion.json")
    if sha(original / "plan.json") != old_complete["plan_sha256"]:
        raise ValueError("Changed original development plan")
    inputs[str(original / "plan.json")] = sha(original / "plan.json")
    screen = Path(read(original / "plan.json")["screen"])
    cohort = Path(plan["cohort"])
    profile = read(training / "profile.json")
    configs = sorted(profile["config_ids"])
    if len(configs) != 9 or len(set(configs)) != 9:
        raise ValueError("This audit fixes the existing nine-configuration pool")
    flights = []
    for meta in plan["inventory"]:
        if meta["group"] == "original_development":
            feature_path = original / f"features_{meta['lighting']}.json"
            label_path = screen / meta["lighting"] / "outcomes.json"
        else:
            folder = cohort / meta["name"] / "development_labels"
            feature_path, label_path = folder / "features.json", folder / "outcomes.json"
        if any(str(p) not in plan["input_sha256"] for p in (feature_path, label_path)):
            raise ValueError("Unbound feature/label path outside the development inventory")
        features = read(feature_path)
        x = np.asarray(features["values"], float)
        if (
            features["names"] != list(NAMES)
            or x.ndim != 2
            or x.shape[1] != len(NAMES)
            or not len(x)
            or not np.isfinite(x).all()
            or not np.all(x[:, 0] == 1)
        ):
            raise ValueError("Invalid aligned development feature matrix")
        flights.append(
            dict(
                meta,
                features=x[:, : len(BASE_NAMES)].tolist(),
                labels=label_arrays(read(label_path), len(x), configs),
            )
        )
    validate_groups(flights)
    if sum(len(f["features"]) for f in flights) != complete["captures"]:
        raise ValueError("Development capture count changed")
    return flights, configs, inputs, profile["signatures"]


def arrays(flights, config):
    return (
        np.vstack([f["features"] for f in flights]),
        np.concatenate([f["labels"][config]["hits"] for f in flights]),
        np.concatenate([f["labels"][config]["trials"] for f in flights]),
    )


def range_shrink(model, features):
    x = np.asarray(features, float)
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    low, high = np.asarray(model["train_min"]), np.asarray(model["train_max"])
    if x.shape != mean.shape or not np.isfinite(x).all() or x[0] != 1:
        raise ValueError("Finite base features with unit intercept required")
    varying = high > low
    varying[0] = False
    clipped = x.copy()
    constants = ~varying
    constants[0] = False
    clipped[constants] = mean[constants]
    clipped[varying] = np.clip(x[varying], low[varying], high[varying])
    excess = np.maximum(low[varying] - x[varying], x[varying] - high[varying])
    distance = float(np.max(np.maximum(excess, 0) / scale[varying], initial=0))
    weight = 1 / (1 + distance)
    clipped_p = predict(model, clipped)[0]
    prior = model["global_mean"]
    return float(prior + weight * (clipped_p - prior)), distance, weight


def evaluate_fold(train, test, configs):
    """Fit exclusively on train; test labels are copied only into scored rows."""
    if {f["group"] for f in train} & {f["group"] for f in test}:
        raise ValueError("Training and held-out layouts overlap")
    models, rows = {}, []
    for c in configs:
        x, hits, trials = arrays(train, c)
        model = fit(x, hits, trials)
        models[c] = model
        for flight in test:
            for i, features in enumerate(flight["features"]):
                candidate, excess, weight = range_shrink(model, features)
                rows.append(
                    dict(
                        flight=flight["name"],
                        group=flight["group"],
                        lighting=flight["lighting"],
                        capture_id=i,
                        config_id=c,
                        hits=int(flight["labels"][c]["hits"][i]),
                        trials=int(flight["labels"][c]["trials"][i]),
                        predictions={
                            "global_prior": model["global_mean"],
                            "base_recall": predict(model, features)[0],
                            "range_shrink": candidate,
                        },
                        max_standardized_excess=excess,
                        conditional_weight=weight,
                    )
                )
    return models, rows


def loss_metrics(rows, method):
    p = np.asarray([r["predictions"][method] for r in rows], float)
    h, n = np.asarray([r["hits"] for r in rows]), np.asarray([r["trials"] for r in rows])
    if not len(rows) or not n.sum():
        raise ValueError("Visible target trials required for loss scoring")
    if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("Finite probabilities required")
    q = np.clip(p, SPEC["log_loss_probability_clip"], 1 - SPEC["log_loss_probability_clip"])
    return {
        "frame_config_pairs": len(rows),
        "zero_visible_pairs": int(np.sum(n == 0)),
        "target_config_trials": int(n.sum()),
        "binary_target_brier": float(np.sum(h * (1 - p) ** 2 + (n - h) * p**2) / n.sum()),
        "binomial_log_loss": float(-np.sum(h * np.log(q) + (n - h) * np.log1p(-q)) / n.sum()),
        "observed_recall": float(h.sum() / n.sum()),
        "predicted_recall": float(np.sum(n * p) / n.sum()),
    }


def ranking_metrics(rows, method):
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[(row["flight"], row["capture_id"])].append(row)
    oracle, selected, trials, active, optimal = 0, 0, 0, 0, 0
    configs = {r["config_id"] for r in rows}
    for members in by_frame.values():
        if len(members) != len(configs) or {r["config_id"] for r in members} != configs:
            raise ValueError("Incomplete configuration choice set")
        if len({r["trials"] for r in members}) != 1:
            raise ValueError("Configuration-dependent ranking denominator")
        n = members[0]["trials"]
        if not n:
            continue
        members = sorted(members, key=lambda r: r["config_id"])
        chosen = max(members, key=lambda r: r["predictions"][method])
        best = max(r["hits"] for r in members)
        oracle += best
        selected += chosen["hits"]
        trials += n
        active += 1
        optimal += chosen["hits"] == best
    if not trials:
        raise ValueError("Visible targets required for ranking regret")
    return {
        "captures": len(by_frame),
        "visible_captures": active,
        "zero_visible_captures": len(by_frame) - active,
        "visible_target_trials": trials,
        "oracle_hits": oracle,
        "selected_hits": selected,
        "oracle_tie_optimal_captures": optimal,
        "configuration_ranking_regret": (oracle - selected) / trials,
        "selected_recall": selected / trials,
        "oracle_recall": oracle / trials,
        "costs_accounted": False,
    }


def summarize(rows, configs):
    folds = {}
    for group in sorted({r["group"] for r in rows}):
        held = [r for r in rows if r["group"] == group]
        folds[group] = {m: {**loss_metrics(held, m), **ranking_metrics(held, m)} for m in METHODS}
    keys = SPEC["metrics"]
    macro = {m: {k: float(np.mean([folds[g][m][k] for g in folds])) for k in keys} for m in METHODS}
    comparisons = {}
    for comparator in METHODS[:2]:
        comparisons[comparator] = {
            k: {
                "macro_difference_candidate_minus_comparator": macro["range_shrink"][k]
                - macro[comparator][k],
                "heldout_differences": {
                    g: folds[g]["range_shrink"][k] - folds[g][comparator][k] for g in folds
                },
                "smaller_is_better": True,
            }
            for k in keys
        }
    return {
        "folds": folds,
        "primary_equal_layout_mean": macro,
        "secondary_pooled": {
            m: {**loss_metrics(rows, m), **ranking_metrics(rows, m)} for m in METHODS
        },
        "by_config": {
            c: {m: loss_metrics([r for r in rows if r["config_id"] == c], m) for m in METHODS}
            for c in configs
        },
        "candidate_paired_differences": comparisons,
        "inference_added": False,
        "evaluation_labels_used": False,
        "mission_outcomes_evaluated": False,
        "automatic_policy_promotion": False,
        "confidence_intervals": False,
        "p_values": False,
    }


def prepare(training, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve any prior plan or attempt; use a new output")
    flights, configs, inputs, signatures = load_development(training)
    own = Path(__file__).resolve()
    sources = [
        own,
        own.parent / "mission_routing_v5/quality.py",
        own.parent / "mission_routing_v2/features.py",
        own.parent.parent / "tests/test_mission_recall_robustness.py",
    ]
    output.mkdir(parents=True)
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    snapshot_files = {}
    for i, path in enumerate(sources):
        target = snapshots / f"{i}_{path.name}"
        shutil.copyfile(path, target)
        snapshot_files[str(target.relative_to(output))] = sha(target)
    plan = {
        "schema_version": "development_recall_robustness_plan_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "training": str(Path(training).resolve()),
        "specification": SPEC,
        "config_ids": configs,
        "model_signatures": signatures,
        "folds": sorted({f["group"] for f in flights}),
        "inventory": [
            {
                **{k: v for k, v in f.items() if k not in ("features", "labels")},
                "captures": len(f["features"]),
            }
            for f in flights
        ],
        "feature_names": list(BASE_NAMES),
        "input_sha256": inputs,
        "source_sha256": {str(p): sha(p) for p in sources},
        "source_snapshot_sha256": snapshot_files,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "thread_environment": {
                name: os.environ.get(name)
                for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
            },
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "metrics_computed_at_prepare": False,
        "deployment_profiles_modified": False,
        "scope": (
            "Development-only audit after prior benchmark diagnostics; not a new untouched test"
        ),
    }
    save(output / "plan.json", plan)
    return plan


def verify_plan(output):
    output = Path(output)
    plan = read(output / "plan.json")
    if plan["specification"] != SPEC:
        raise ValueError("Changed predeclared candidate specification")
    check_hashes(plan["input_sha256"])
    check_hashes(plan["source_sha256"])
    check_hashes({str(output / p): h for p, h in plan["source_snapshot_sha256"].items()})
    if plan["runtime"]["numpy"] != np.__version__:
        raise ValueError("Changed fitting NumPy version")
    return plan


def report(summary):
    lines = [
        "# Development recall robustness audit",
        "",
        "Three complete held-out layout groups, both lighting mates retained. "
        "The candidate rule was pinned before this audit's held-out metrics were computed.",
        "",
        "## Primary: equal mean over layout folds",
        "",
        "| Predictor | Binary-target Brier | Log loss | Configuration-ranking regret |",
        "|---|---:|---:|---:|",
    ]
    for method, metrics in summary["primary_equal_layout_mean"].items():
        lines.append(
            f"| {method} | " + " | ".join(f"{metrics[k]:.6f}" for k in SPEC["metrics"]) + " |"
        )
    lines += [
        "",
        "## Every held-out layout",
        "",
        "| Layout | Predictor | Brier | Log loss | Ranking regret |",
        "|---|---|---:|---:|---:|",
    ]
    for group, methods in summary["folds"].items():
        for method, values in methods.items():
            lines.append(
                f"| {group} | {method} | "
                + " | ".join(f"{values[k]:.6f}" for k in SPEC["metrics"])
                + " |"
            )
    lines += [
        "",
        "## Scope",
        "",
        "The global prior and base logistic model are refitted using only the two training "
        "layouts in each fold. The candidate shares those coefficients and prior, clips "
        "varying features to training ranges, and shrinks by the maximum standardized "
        "extrapolation. No parameter or candidate was selected from these scores.",
        "",
        "Brier is a binary-target score computed from actual matched/visible counts, not "
        "squared error on frame-average recall. Losses exclude zero-visible frames. "
        "Ranking regret compares the selected configuration to the best observed recall "
        "in the same frame; it ignores latency, energy, false positives, reporting and "
        "repeated-evidence mission constraints. It is not mission regret or an oracle router.",
        "",
        "These are three authored layout blocks on one known map, with correlated frames "
        "and repeated targets. The longer original development flights contribute more "
        "to pooled secondary totals; the primary summary weights layout folds equally. "
        "No confidence interval, significance test or population superiority is claimed.",
        "",
        "This follow-up was motivated by already observed benchmark weaknesses. It is "
        "development-only, not a fresh test of generalization; no Lake evaluation or Depot "
        "labels are read by this script. All folds, including worse outcomes, are retained. "
        "No neural inference or physics is repeated, no deployed profile or frozen study "
        "is modified, and no candidate is automatically promoted.",
        "",
    ]
    return "\n".join(lines)


def run(output):
    output = Path(output)
    plan = verify_plan(output)
    if (output / "attempt.json").exists():
        raise FileExistsError("Preserve completed or failed fitting attempt")
    save(
        output / "attempt.json",
        {
            "started_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha(output / "plan.json"),
        },
    )
    flights, configs, inputs, _ = load_development(plan["training"])
    if inputs != plan["input_sha256"] or configs != plan["config_ids"]:
        raise ValueError("Changed development inputs after specification freeze")
    predictions, fitted = [], {}
    for heldout in plan["folds"]:
        train = [f for f in flights if f["group"] != heldout]
        test = [f for f in flights if f["group"] == heldout]
        models, rows = evaluate_fold(train, test, configs)
        fitted[heldout] = {
            "training_flights": [f["name"] for f in train],
            "heldout_flights": [f["name"] for f in test],
            "models": models,
        }
        predictions.extend(rows)
    summary = summarize(predictions, configs)
    save(output / "fold_models.json", fitted)
    save(output / "out_of_fold_predictions.json", predictions)
    save(output / "summary.json", summary)
    with (output / "REPORT.md").open("x") as handle:
        handle.write(report(summary))
    files = (
        "attempt.json",
        "fold_models.json",
        "out_of_fold_predictions.json",
        "summary.json",
        "REPORT.md",
    )
    save(
        output / "completion.json",
        {
            "complete": True,
            "completed_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": sha(output / "plan.json"),
            "artifacts_sha256": {name: sha(output / name) for name in files},
            "folds": len(plan["folds"]),
            "model_fits": len(plan["folds"]) * len(configs),
            "flights": len(flights),
            "captures": sum(len(f["features"]) for f in flights),
            "frame_config_pairs": len(predictions),
            "prediction_values": len(predictions) * len(METHODS),
            "evaluation_labels_used": False,
            "new_neural_calls": 0,
            "new_physical_flights": 0,
            "deployment_profiles_modified": False,
        },
    )
    verify(output)
    return summary


def verify(output):
    output = Path(output)
    plan = verify_plan(output)
    complete = read(output / "completion.json")
    if not complete["complete"] or complete["plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("Changed completed audit plan")
    check_hashes({str(output / n): h for n, h in complete["artifacts_sha256"].items()})
    rows = read(output / "out_of_fold_predictions.json")
    if summarize(rows, plan["config_ids"]) != read(output / "summary.json"):
        raise ValueError("Summary does not reconstruct from saved held-out predictions")
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "verify"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training", type=Path, default=Path("results/mission_coverage_training_v1_20261025")
    )
    args = parser.parse_args()
    result = (
        prepare(args.training, args.output)
        if args.action == "prepare"
        else globals()[args.action](args.output)
    )
    print(
        json.dumps(
            result.get("primary_equal_layout_mean", {"action": args.action, "complete": True}),
            indent=2,
        )
    )
