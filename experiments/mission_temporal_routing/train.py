"""Freeze temporal heads and evaluate them with complete-layout blocked folds."""

from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v5.quality import BASE_NAMES, predict
from experiments.mission_temporal_routing.fitting import fit
from experiments.mission_temporal_routing.temporal import (
    PATTERNS,
    REQUIREMENTS,
    head_key,
    possible,
    window_labels,
)
from experiments.routing_development_matrix.train import load_flights, partition


def arrays(flights, config, requirement, pattern):
    x, hits, trials = [], [], []
    for flight in flights:
        for i, h, n in window_labels(flight["rows"][config], *requirement, pattern):
            x.append(flight["features"][i][: len(BASE_NAMES)])
            hits.append(h)
            trials.append(n)
    return np.asarray(x), np.asarray(hits), np.asarray(trials)


def run(cohort, original_training, output):
    if output.exists():
        raise FileExistsError("Preserve prior temporal training")
    flights, original, inputs = load_flights(cohort, original_training)
    screen = Path(read(original_training / "plan.json")["screen"])
    for f in flights:
        path = (
            screen / f["lighting"] / "outcomes.json"
            if f["group"] == "original_development"
            else cohort / f["name"] / "development_labels/outcomes.json"
        )
        rows = read(path)
        f["rows"] = {
            c: sorted((r for r in rows if r["config_id"] == c), key=lambda r: r["capture_id"])
            for c in original["models"]
        }
        # Denominators must not vary with which model happened to detect objects.
        reference = next(iter(f["rows"].values()))
        for config_rows in f["rows"].values():
            if [set(r["visible_target_ids"]) for r in config_rows] != [
                set(r["visible_target_ids"]) for r in reference
            ]:
                raise ValueError("Configuration-dependent visibility denominator")
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    sources = [
        *Path(__file__).parent.glob("*.py"),
        repo / "experiments/mission_routing_v5/quality.py",
        repo / "experiments/routing_development_matrix/train.py",
    ]
    plan = dict(
        schema_version="temporal_support_training_v1",
        cohort=str(cohort.resolve()),
        original_training=str(original_training.resolve()),
        input_sha256=inputs,
        source_sha256={str(p.resolve()): sha(p) for p in sources},
        inventory=[
            {k: v for k, v in f.items() if k not in ("features", "labels", "rows")} for f in flights
        ],
        requirements=REQUIREMENTS,
        capture_patterns_s=PATTERNS,
        alpha=10,
        horizon_s=8,
        feature_names=list(BASE_NAMES),
        label="Repeated GT identity support among targets visible at window start",
        split="Leave one complete layout out, keeping both lighting mates together",
        hyperparameter_search=False,
        evaluation_labels_used=False,
        fresh_test=False,
        calibrated_probability=False,
    )
    save(output / "plan.json", plan)
    models = {c: dict(signature=m["signature"], heads={}) for c, m in original["models"].items()}
    diagnostics = []
    for config in models:
        for requirement in REQUIREMENTS:
            for pattern in PATTERNS:
                if not possible(*requirement, pattern):
                    continue
                key = head_key(*requirement, pattern)
                x, h, n = arrays(flights, config, requirement, pattern)
                models[config]["heads"][key] = fit(x, h, n)
                for group in sorted({f["group"] for f in flights}):
                    training, testing = partition(flights, group)
                    head = fit(*arrays(training, config, requirement, pattern))
                    tx, th, tn = arrays(testing, config, requirement, pattern)
                    ps = np.asarray([predict(head, v)[0] for v in tx])
                    active = tn > 0
                    weights = tn[active]
                    y = th[active] / weights
                    diagnostics.append(
                        dict(
                            heldout_group=group,
                            config_id=config,
                            head=key,
                            training_flights=[f["name"] for f in training],
                            heldout_flights=[f["name"] for f in testing],
                            visible_window_targets=int(tn.sum()),
                            repeated_supports=int(th.sum()),
                            mae=float(np.average(abs(ps[active] - y), weights=weights)),
                            global_mae=float(
                                np.average(abs(head["global_mean"] - y), weights=weights)
                            ),
                        )
                    )
        print("Fitted temporal support: " + config, flush=True)
    save(
        output / "profile.json",
        dict(
            schema_version="temporal_support_v1",
            feature_names=list(BASE_NAMES),
            models=models,
            training_lineage=[f["lineage"] for f in flights],
            training_plan_sha256=sha(output / "plan.json"),
            calibrated_probability=False,
        ),
    )
    save(
        output / "blocked_validation.json",
        dict(
            folds=diagnostics,
            independent_layout_blocks=3,
            independent_maps=1,
            confidence_intervals=False,
            overlapping_windows=True,
            hyperparameter_selection=False,
        ),
    )
    save(
        output / "completion.json",
        dict(
            complete=True,
            plan_sha256=sha(output / "plan.json"),
            captures=sum(len(f["features"]) for f in flights),
            artifacts_sha256={
                n: sha(output / n) for n in ("profile.json", "blocked_validation.json")
            },
        ),
    )
    verify(output)


def verify(output):
    p, c = read(output / "plan.json"), read(output / "completion.json")
    if not c["complete"] or c["plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("Changed temporal training plan")
    for field in ("input_sha256", "source_sha256"):
        for path, expected in p[field].items():
            if sha(Path(path)) != expected:
                raise ValueError("Changed training source or input: " + path)
    for path, expected in c["artifacts_sha256"].items():
        if sha(output / path) != expected:
            raise ValueError("Changed temporal training output")
    result = read(output / "profile.json")
    if result["training_lineage"] != [f["lineage"] for f in p["inventory"]]:
        raise ValueError("Changed training lineage")
    if result["training_plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("Wrong temporal profile plan")
    return result
