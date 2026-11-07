"""Fit only original development false-component counts, with flight-blocked checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import validate_table
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.features import NAMES
from experiments.mission_routing_v4.risk import estimate, fit_count


def run(screen, output):
    if output.exists():
        raise FileExistsError("Preserve previous risk training")
    protocol, quality = read(screen / "protocol.json"), read(screen / "calibration.json")
    complete = read(screen / "training_complete.json")
    if complete["calibration_sha256"] != sha(screen / "calibration.json"):
        raise ValueError("Development calibration changed")
    inputs = {}
    groups = {}
    for env in protocol["environments"]:
        features = read(screen / env / "features.json")
        rows = read(screen / env / "outcomes.json")
        if features["names"] != list(NAMES):
            raise ValueError("Feature schema changed")
        times = sorted({r["capture_time_s"] for r in rows})
        validate_table(rows, times, list(quality["models"]))
        if len(times) != len(features["values"]):
            raise ValueError("Missing feature/label pairs")
        by_config = {}
        for c in quality["models"]:
            selected = sorted(
                (r for r in rows if r["config_id"] == c), key=lambda r: r["capture_id"]
            )
            if [r["capture_id"] for r in selected] != list(range(len(times))):
                raise ValueError("Noncontiguous development capture IDs")
            by_config[c] = [r["false_positive_components"] for r in selected]
        groups[env] = (features["values"], by_config)
        for name in ("features.json", "outcomes.json"):
            inputs[f"{env}/{name}"] = sha(screen / env / name)
    if len(groups) < 2:
        raise ValueError("At least two flights required for blocked diagnostics")
    output.mkdir(parents=True)
    save(
        output / "plan.json",
        dict(
            schema_version="risk_training_plan_v1",
            screen=str(screen.resolve()),
            input_sha256=inputs,
            quality_profile_sha256=sha(screen / "calibration.json"),
            source_sha256={n: sha(Path(__file__).with_name(n)) for n in ("train.py", "risk.py")},
            alpha=1.0,
            pseudocount=1.0,
            validation="Leave one complete development flight out; no random frame split",
            tuning="None; blocked diagnostics are descriptive, not hyperparameter selection",
            evaluation_labels_used=False,
        ),
    )
    models, diagnostics = {}, []
    x = [f for features, _ in groups.values() for f in features]
    labels = [env for env, (features, _) in groups.items() for _ in features]
    for c, quality_model in quality["models"].items():
        y = [v for _, counts in groups.values() for v in counts[c]]
        models[c] = dict(signature=quality_model["signature"], risk=fit_count(x, y, labels))
        for heldout, (features, counts) in groups.items():
            train_x = [f for env, (fs, _) in groups.items() if env != heldout for f in fs]
            train_y = [v for env, (_, cs) in groups.items() if env != heldout for v in cs[c]]
            train_groups = [env for env, (fs, _) in groups.items() if env != heldout for _ in fs]
            head = fit_count(train_x, train_y, train_groups)
            predicted = np.array([estimate(head, f) for f in features])
            truth = np.asarray(counts[c])
            diagnostics.append(
                dict(
                    heldout_flight=heldout,
                    config_id=c,
                    samples=len(truth),
                    observed_total=int(truth.sum()),
                    estimated_total=float(predicted.sum()),
                    mean_absolute_error=float(np.abs(predicted - truth).mean()),
                    signed_mean_error=float((predicted - truth).mean()),
                    actual_count_exceeds_estimate=int((truth > predicted).sum()),
                )
            )
    profile = dict(
        schema_version="mission_false_component_risk_v1",
        feature_names=list(NAMES),
        models=models,
        training_lineage=quality["training_lineage"],
        training_plan_sha256=sha(output / "plan.json"),
        semantics="Estimated false component counts, not false reports or calibrated probability",
        method="Nonnegative ridge count plus add-one floor at worst development-flight mean",
        caveat="Heuristic floor is not a confidence bound; temporal dependence and shift remain",
    )
    save(output / "risk_profile.json", profile)
    save(
        output / "blocked_validation.json", dict(folds=diagnostics, hyperparameters_selected=False)
    )
    save(
        output / "completion.json",
        dict(
            complete=True,
            plan_sha256=sha(output / "plan.json"),
            profile_sha256=sha(output / "risk_profile.json"),
            diagnostics_sha256=sha(output / "blocked_validation.json"),
            development_captures=len(x),
            frame_config_pairs=len(x) * len(models),
        ),
    )
    print(f"Fitted {len(models)} risk heads on {len(x)} original development captures", flush=True)


def verify(folder):
    completion, plan = read(folder / "completion.json"), read(folder / "plan.json")
    for filename, key in (
        ("plan.json", "plan_sha256"),
        ("risk_profile.json", "profile_sha256"),
        ("blocked_validation.json", "diagnostics_sha256"),
    ):
        if not completion["complete"] or sha(folder / filename) != completion[key]:
            raise ValueError("Changed risk training artifact")
    for name, expected in plan["source_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("Changed risk training implementation")
    for name, expected in plan["input_sha256"].items():
        if sha(Path(plan["screen"]) / name) != expected:
            raise ValueError("Changed risk development input")
    profile = read(folder / "risk_profile.json")
    if profile["training_plan_sha256"] != sha(folder / "plan.json"):
        raise ValueError("Wrong risk training plan")
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.screen, args.output)
