"""Leave-one-layout-out recall fitting, with all lighting mates in the same fold."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import validate_table
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v5.quality import ALPHA, BASE_NAMES, NAMES, fit, predict
from experiments.mission_routing_v5.train import metrics
from experiments.mission_routing_v5.train import verify as verify_original
from experiments.routing_development_matrix.cohort import verify as verify_cohort
from experiments.routing_development_matrix.score import verify_labels


def validate_groups(flights):
    """A shared layout OR route cannot cross folds; records cannot occur twice."""
    if len({f["group"] for f in flights}) < 3:
        raise ValueError("At least three complete layout blocks required")
    recordings = [f["lineage"]["recording"] for f in flights]
    if len(recordings) != len(set(recordings)):
        raise ValueError("Repeated recording in training inventory")
    for index, left in enumerate(flights):
        for right in flights[index + 1 :]:
            if left["group"] != right["group"] and any(
                left["lineage"][field] == right["lineage"][field] for field in ("layout", "route")
            ):
                raise ValueError("Shared layout or route crosses validation folds")
    for group in {f["group"] for f in flights}:
        if {f["lighting"] for f in flights if f["group"] == group} != {"day", "dusk_haze"}:
            raise ValueError("A layout block must retain both lighting conditions")


def partition(flights, heldout):
    validate_groups(flights)
    train = [f for f in flights if f["group"] != heldout]
    test = [f for f in flights if f["group"] == heldout]
    if not test or not train:
        raise ValueError("Unknown or empty held-out layout")
    return train, test


def arrays(flights, config):
    return (
        np.vstack([f["features"] for f in flights]),
        np.concatenate([f["labels"][config]["hits"] for f in flights]),
        np.concatenate([f["labels"][config]["trials"] for f in flights]),
    )


def label_arrays(rows, features, configs):
    times = sorted({r["capture_time_s"] for r in rows})
    validate_table(rows, times, configs)
    if len(times) != len(features):
        raise ValueError("Missing RGB features")
    labels = {}
    for c in configs:
        selected = sorted((r for r in rows if r["config_id"] == c), key=lambda r: r["capture_id"])
        labels[c] = dict(
            hits=[len(r["matched_target_ids"]) for r in selected],
            trials=[len(r["visible_target_ids"]) for r in selected],
        )
    return labels


def load_flights(root, original_training):
    p, original = verify_cohort(root), verify_original(original_training)
    validation = read(root / "validation.json")
    if (
        not validation["complete"]
        or validation["protocol_sha256"] != sha(root / "protocol.json")
        or validation["validator_sha256"] != sha(Path(__file__).with_name("validate.py"))
    ):
        raise ValueError("Independent dense-label validation required before fitting")
    plan = read(original_training / "plan.json")
    screen = Path(plan["screen"])
    if screen.resolve() != Path(p["screen"]).resolve():
        raise ValueError("Wrong original development training")
    inputs = {
        str((root / "protocol.json").resolve()): sha(root / "protocol.json"),
        str((original_training / "completion.json").resolve()): sha(
            original_training / "completion.json"
        ),
    }
    inputs[str((root / "validation.json").resolve())] = sha(root / "validation.json")
    flights = []
    environments = read(screen / "protocol.json")["environments"]
    for env, lineage in zip(environments, original["training_lineage"], strict=True):
        features = read(original_training / f"features_{env}.json")
        rows = read(screen / env / "outcomes.json")
        if features["names"] != list(NAMES):
            raise ValueError("Wrong original feature schema")
        flights.append(
            dict(
                name="original_" + env,
                group="original_development",
                lighting=env,
                lineage=lineage,
                features=features["values"],
                labels=label_arrays(rows, features["values"], list(original["models"])),
            )
        )
        for path in (original_training / f"features_{env}.json", screen / env / "outcomes.json"):
            inputs[str(path.resolve())] = sha(path)
    for name, case in p["cases"].items():
        folder = verify_labels(root, name)
        if validation["metadata"][name]["labels_completion_sha256"] != sha(
            folder / "completion.json"
        ):
            raise ValueError("Changed validated development labels")
        features = read(folder / "features.json")
        if features["names"] != list(NAMES):
            raise ValueError("Wrong new feature schema")
        rows = read(folder / "outcomes.json")
        flights.append(
            dict(
                name=name,
                group=case["layout_group"],
                lighting=case["lighting"],
                lineage=case["lineage"],
                features=features["values"],
                labels=label_arrays(rows, features["values"], list(original["models"])),
            )
        )
        for filename in ("features.json", "outcomes.json", "completion.json"):
            path = folder / filename
            inputs[str(path.resolve())] = sha(path)
    validate_groups(flights)
    return flights, original, inputs


def run(root, original_training, output):
    if output.exists():
        raise FileExistsError("Preserve previous layout training")
    flights, original, inputs = load_flights(root, original_training)
    inventory = [{k: v for k, v in f.items() if k not in ("features", "labels")} for f in flights]
    groups = sorted({f["group"] for f in flights})
    sources = [
        Path(__file__),
        Path(__file__).with_name("score.py"),
        Path(__file__).parents[1] / "mission_routing_v5/quality.py",
        Path(__file__).parents[1] / "mission_routing_v5/train.py",
    ]
    output.mkdir(parents=True)
    save(
        output / "plan.json",
        dict(
            schema_version="layout_recall_training_plan_v1",
            cohort=str(root.resolve()),
            original_training=str(original_training.resolve()),
            inventory=inventory,
            input_sha256=inputs,
            source_sha256={str(path.resolve()): sha(path) for path in sources},
            folds=groups,
            alpha=ALPHA,
            feature_names=list(NAMES),
            tuning=False,
            split="Complete layout blocks, retaining both paired lighting flights",
            population_generalization=False,
            independent_maps=0,
            evaluation_labels_used=False,
            protocol_sha256=sha(root / "protocol.json"),
        ),
    )
    models, folds, predictions, original_transfer = {}, [], [], []
    for config, model in original["models"].items():
        x, hits, trials = arrays(flights, config)
        models[config] = dict(
            signature=model["signature"],
            spatial=fit(x, hits, trials),
            base=fit(x[:, : len(BASE_NAMES)], hits, trials),
        )
        for group in groups:
            train, test = partition(flights, group)
            tx, th, tn = arrays(train, config)
            x, hits, trials = arrays(test, config)
            if not tn.sum():
                folds.append(
                    dict(
                        heldout_group=group,
                        config_id=config,
                        supported=False,
                        reason="No visible training target trials",
                    )
                )
                continue
            spatial, base = fit(tx, th, tn), fit(tx[:, : len(BASE_NAMES)], th, tn)
            estimated = dict(
                spatial=[predict(spatial, f)[0] for f in x],
                base=[predict(base, f[: len(BASE_NAMES)])[0] for f in x],
                global_recall=[spatial["global_mean"]] * len(x),
            )
            folds.append(
                dict(
                    heldout_group=group,
                    config_id=config,
                    supported=bool(trials.sum()),
                    training_flights=[f["name"] for f in train],
                    heldout_flights=[f["name"] for f in test],
                    training_trials=int(tn.sum()),
                    heldout_trials=int(trials.sum()),
                    scores={
                        name: metrics(ps, hits, trials) if trials.sum() else None
                        for name, ps in estimated.items()
                    },
                )
            )
            offset = 0
            for flight in test:
                for index in range(len(flight["features"])):
                    i = offset + index
                    predictions.append(
                        dict(
                            flight=flight["name"],
                            group=group,
                            capture_id=index,
                            config_id=config,
                            hits=int(hits[i]),
                            trials=int(trials[i]),
                            estimates={name: float(ps[i]) for name, ps in estimated.items()},
                        )
                    )
                offset += len(flight["features"])
        # Frozen original fit is assessed only on newly acquired DEVELOPMENT flights.
        for f in flights:
            if f["group"] == "original_development":
                continue
            y = f["labels"][config]
            ps = [predict(model["spatial"], values)[0] for values in f["features"]]
            original_transfer.append(
                dict(
                    flight=f["name"],
                    config_id=config,
                    score=metrics(ps, y["hits"], y["trials"]) if sum(y["trials"]) else None,
                )
            )
    save(
        output / "profile.json",
        dict(
            schema_version="conditional_recall_v5",
            feature_names=list(NAMES),
            models=models,
            training_lineage=[f["lineage"] for f in flights],
            training_plan_sha256=sha(output / "plan.json"),
            semantics=(
                "Conditional visible-target recall; development-only, "
                "not calibrated track correctness"
            ),
        ),
    )
    save(
        output / "blocked_validation.json",
        dict(
            schema_version="layout_blocked_recall_v1",
            folds=folds,
            confidence_intervals=False,
            tuned=False,
            fresh_test=False,
            interpretation="Three authored layout blocks on one known map; correlated captures",
        ),
    )
    save(output / "out_of_fold_predictions.json", predictions)
    save(output / "original_profile_transfer.json", original_transfer)
    files = (
        "profile.json",
        "blocked_validation.json",
        "out_of_fold_predictions.json",
        "original_profile_transfer.json",
    )
    save(
        output / "completion.json",
        dict(
            schema_version="layout_recall_training_completion_v1",
            complete=True,
            plan_sha256=sha(output / "plan.json"),
            artifacts_sha256={n: sha(output / n) for n in files},
            layout_blocks=len(groups),
            flights=len(flights),
            captures=sum(len(f["features"]) for f in flights),
            frame_config_pairs=sum(len(f["features"]) for f in flights) * len(models),
        ),
    )
    verify_training(output)
    print("Completed layout-blocked training and diagnostics", flush=True)


def verify_training(output):
    complete, plan = read(output / "completion.json"), read(output / "plan.json")
    if not complete["complete"] or sha(output / "plan.json") != complete["plan_sha256"]:
        raise ValueError("Changed layout training plan")
    verify_cohort(Path(plan["cohort"]))
    verify_original(Path(plan["original_training"]))
    for c in verify_cohort(Path(plan["cohort"]))["cases"]:
        verify_labels(Path(plan["cohort"]), c)
    for field in ("input_sha256", "source_sha256"):
        for path, expected in plan[field].items():
            if sha(Path(path)) != expected:
                raise ValueError("Changed layout training input or implementation")
    for name, expected in complete["artifacts_sha256"].items():
        if sha(output / name) != expected:
            raise ValueError("Changed layout training artifact")
    profile = read(output / "profile.json")
    if profile["training_plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("Wrong layout training profile")
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--original-training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.cohort, args.original_training, args.output)
