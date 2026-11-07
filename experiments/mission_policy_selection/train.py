"""Fit recall and false-component risk without either lighting mate of the test layout."""

from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.flights import split_check
from experiments.mission_routing_v4.risk import estimate, fit_count
from experiments.mission_routing_v5.quality import BASE_NAMES, NAMES, fit
from experiments.routing_development_matrix.train import arrays, load_flights, partition


def attach_counts(flights, cohort, original_training):
    screen = Path(read(original_training / "plan.json")["screen"])
    for flight in flights:
        path = (
            screen / flight["lighting"] / "outcomes.json"
            if flight["group"] == "original_development"
            else cohort / flight["name"] / "development_labels/outcomes.json"
        )
        rows = read(path)
        flight["counts"] = {}
        for config in flight["labels"]:
            selected = sorted(
                (r for r in rows if r["config_id"] == config), key=lambda r: r["capture_id"]
            )
            if [r["capture_id"] for r in selected] != list(range(len(flight["features"]))):
                raise ValueError("Noncontiguous risk labels")
            flight["counts"][config] = [r["false_positive_components"] for r in selected]


def profiles(flights, original, plan_sha):
    recall, risk = {}, {}
    for config, model in original["models"].items():
        x, hits, trials = arrays(flights, config)
        recall[config] = dict(
            signature=model["signature"],
            spatial=fit(x, hits, trials),
            base=fit(x[:, : len(BASE_NAMES)], hits, trials),
        )
        counts = [v for f in flights for v in f["counts"][config]]
        # Keep the existing worst-flight floor; validation holds out whole layouts.
        groups = [f["name"] for f in flights for _ in f["features"]]
        risk[config] = dict(
            signature=model["signature"], risk=fit_count(x[:, : len(BASE_NAMES)], counts, groups)
        )
    common = dict(training_lineage=[f["lineage"] for f in flights], training_plan_sha256=plan_sha)
    return (
        dict(
            schema_version="conditional_recall_v5",
            feature_names=list(NAMES),
            models=recall,
            **common,
        ),
        dict(
            schema_version="mission_false_component_risk_v1",
            feature_names=list(BASE_NAMES),
            models=risk,
            **common,
        ),
    )


def run(cohort, original_training, output):
    if output.exists():
        raise FileExistsError("Preserve previous joint training")
    flights, original, inputs = load_flights(cohort, original_training)
    attach_counts(flights, cohort, original_training)
    output.mkdir(parents=True)
    sources = [
        Path(__file__),
        Path(__file__).parents[1] / "mission_routing_v4/risk.py",
        Path(__file__).parents[1] / "mission_routing_v5/quality.py",
        Path(__file__).parents[1] / "routing_development_matrix/train.py",
    ]
    save(
        output / "plan.json",
        dict(
            schema_version="joint_layout_training_v1",
            cohort=str(cohort.resolve()),
            original_training=str(original_training.resolve()),
            input_sha256=inputs,
            source_sha256={str(p.resolve()): sha(p) for p in sources},
            inventory=[
                {k: v for k, v in f.items() if k not in ("features", "labels", "counts")}
                for f in flights
            ],
            split="Leave one complete layout out, retaining both lighting mates",
            recall_alpha=10.0,
            risk_alpha=1.0,
            evaluation_labels_used=False,
            fresh_test=False,
            hyperparameter_tuning=False,
        ),
    )
    plan_sha = sha(output / "plan.json")
    artifacts, diagnostics = [], []
    groups = sorted({f["group"] for f in flights})
    for heldout in [*groups, "all_development"]:
        training, testing = (
            (flights, []) if heldout == "all_development" else partition(flights, heldout)
        )
        recall, risk = profiles(training, original, plan_sha)
        for flight in testing:
            split_check(recall["training_lineage"], flight["lineage"])
            split_check(risk["training_lineage"], flight["lineage"])
            for config in risk["models"]:
                pred = np.asarray(
                    [
                        estimate(risk["models"][config]["risk"], f[: len(BASE_NAMES)])
                        for f in flight["features"]
                    ]
                )
                truth = np.asarray(flight["counts"][config])
                diagnostics.append(
                    dict(
                        heldout_group=heldout,
                        flight=flight["name"],
                        config_id=config,
                        samples=len(truth),
                        observed_total=int(truth.sum()),
                        estimated_total=float(pred.sum()),
                        mean_absolute_error=float(abs(pred - truth).mean()),
                        signed_mean_error=float((pred - truth).mean()),
                        count_exceeds_estimate=int((truth > pred).sum()),
                    )
                )
        (output / heldout).mkdir()
        for name, data in (("recall.json", recall), ("risk.json", risk)):
            path = Path(heldout) / name
            save(output / path, data)
            artifacts.append(str(path))
    save(
        output / "risk_validation.json",
        dict(folds=diagnostics, confidence_bound=False, independent_maps=0),
    )
    artifacts.append("risk_validation.json")
    save(
        output / "completion.json",
        dict(
            complete=True,
            plan_sha256=plan_sha,
            captures=sum(len(f["features"]) for f in flights),
            folds=groups,
            artifacts_sha256={p: sha(output / p) for p in artifacts},
        ),
    )
    verify(output)


def verify(output):
    p, c = read(output / "plan.json"), read(output / "completion.json")
    if not c["complete"] or sha(output / "plan.json") != c["plan_sha256"]:
        raise ValueError("Changed joint training plan")
    load_flights(Path(p["cohort"]), Path(p["original_training"]))
    for field in ("input_sha256", "source_sha256"):
        for path, expected in p[field].items():
            if sha(Path(path)) != expected:
                raise ValueError("Changed training input or source")
    for path, expected in c["artifacts_sha256"].items():
        if sha(output / path) != expected:
            raise ValueError("Changed trained profile")
    for fold in c["folds"]:
        expected = [f["lineage"] for f in p["inventory"] if f["group"] != fold]
        for filename in ("recall.json", "risk.json"):
            profile = read(output / fold / filename)
            if (
                profile["training_lineage"] != expected
                or profile["training_plan_sha256"] != c["plan_sha256"]
            ):
                raise ValueError("Leaking or mismatched fold profile")
    return p
