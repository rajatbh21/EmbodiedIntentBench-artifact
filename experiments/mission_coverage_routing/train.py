"""Fit anonymous model-complementarity prototypes using development identities only."""

from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.routing_development_matrix.train import load_flights


def prototypes(flights, config_ids):
    """Equal layout/identity weighting; targets never visible cannot define a type."""
    groups = sorted({f["group"] for f in flights})
    rates, weights, counts = [], [], []
    for group in groups:
        members = [f for f in flights if f["group"] == group]
        identities = sorted(
            {
                t
                for f in members
                for r in f["rows"]
                if r["config_id"] == config_ids[0]
                for t in r["visible_target_ids"]
            }
        )
        if not identities:
            raise ValueError("A layout has no observable training identities")
        for identity in identities:
            row, trials = [], []
            for config in config_ids:
                selected = [
                    r
                    for f in members
                    for r in f["rows"]
                    if r["config_id"] == config and identity in r["visible_target_ids"]
                ]
                hits = sum(identity in r["matched_target_ids"] for r in selected)
                row.append((hits + 1) / (len(selected) + 2))
                trials.append(len(selected))
            if len(set(trials)) != 1:
                raise ValueError("Configuration-dependent visibility denominator")
            rates.append(row)
            weights.append(1 / (len(groups) * len(identities)))
            counts.append(trials[0])
    return dict(rates=rates, weights=weights, visible_trials=counts)


def run(cohort, original_training, output):
    if output.exists():
        raise FileExistsError("Preserve prior training")
    flights, original, inputs = load_flights(cohort, original_training)
    screen = Path(read(original_training / "plan.json")["screen"])
    for f in flights:
        path = (
            screen / f["lighting"] / "outcomes.json"
            if f["group"] == "original_development"
            else cohort / f["name"] / "development_labels/outcomes.json"
        )
        f["rows"] = read(path)
    ids = sorted(original["models"])
    fitted = prototypes(flights, ids)
    output.mkdir(parents=True)
    save(
        output / "plan.json",
        dict(
            schema_version="anonymous_coverage_training_v1",
            input_sha256=inputs,
            source_sha256={str(Path(__file__).resolve()): sha(Path(__file__))},
            cohort=str(cohort.resolve()),
            original_training=str(original_training.resolve()),
            inventory=[
                {k: v for k, v in f.items() if k not in ("features", "labels", "rows")}
                for f in flights
            ],
            smoothing_hits=1,
            smoothing_misses=1,
            grouping="Complete layout, both lighting mates",
            evaluation_labels_used=False,
            hyperparameter_search=False,
            semantics=(
                "Anonymous visible-target model-response prototypes; "
                "no deployable target identities"
            ),
        ),
    )
    save(
        output / "profile.json",
        dict(
            schema_version="anonymous_coverage_v1",
            config_ids=ids,
            **fitted,
            marginal=np.average(fitted["rates"], axis=0, weights=fitted["weights"]).tolist(),
            signatures={c: original["models"][c]["signature"] for c in ids},
            training_lineage=[f["lineage"] for f in flights],
            training_plan_sha256=sha(output / "plan.json"),
            calibrated_probability=False,
            runtime_identity_labels=False,
        ),
    )
    save(
        output / "completion.json",
        dict(
            complete=True,
            plan_sha256=sha(output / "plan.json"),
            profile_sha256=sha(output / "profile.json"),
            captures=sum(len(f["features"]) for f in flights),
            prototypes=len(fitted["weights"]),
            layouts=len({f["group"] for f in flights}),
        ),
    )
    verify(output)


def verify(output):
    p, c = read(output / "plan.json"), read(output / "completion.json")
    if not c["complete"] or c["plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("Changed prototype training plan")
    if c["profile_sha256"] != sha(output / "profile.json"):
        raise ValueError("Changed prototype coefficients")
    for field in ("input_sha256", "source_sha256"):
        for path, expected in p[field].items():
            if sha(Path(path)) != expected:
                raise ValueError("Changed prototype training input/source")
    result = read(output / "profile.json")
    if result["training_plan_sha256"] != c["plan_sha256"]:
        raise ValueError("Wrong prototype training plan")
    if result["training_lineage"] != [f["lineage"] for f in p["inventory"]]:
        raise ValueError("Wrong training lineage")
    return result
