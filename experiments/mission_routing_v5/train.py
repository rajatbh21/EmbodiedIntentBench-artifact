"""Train conditional recall on original development flights with blocked diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import validate_table
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.features import NAMES as BASE_NAMES
from experiments.mission_routing_v2.features import fit as ridge_fit
from experiments.mission_routing_v2.features import predict as ridge_predict
from experiments.mission_routing_v5.quality import ALPHA, NAMES, encode, fit, predict


def metrics(predicted, hits, trials):
    p, h, n = np.asarray(predicted), np.asarray(hits), np.asarray(trials)
    if not n.sum():
        raise ValueError("No visible targets to score")
    p = np.clip(p, 1e-9, 1 - 1e-9)
    # A weighted squared error on frame recall, not a binary-trial Brier score.
    active = n > 0
    return dict(
        visible_target_trials=int(n.sum()),
        matched_targets=int(h.sum()),
        observed_recall=float(h.sum() / n.sum()),
        estimated_recall=float(np.sum(n * p) / n.sum()),
        binomial_log_loss=float(-np.sum(h * np.log(p) + (n - h) * np.log1p(-p)) / n.sum()),
        weighted_frame_recall_mse=float(
            np.sum(n[active] * (p[active] - h[active] / n[active]) ** 2) / n.sum()
        ),
    )


def run(screen, output):
    if output.exists():
        raise FileExistsError("Preserve existing training")
    p, old = read(screen / "protocol.json"), read(screen / "calibration.json")
    completion = read(screen / "training_complete.json")
    if completion["calibration_sha256"] != sha(screen / "calibration.json"):
        raise ValueError("Changed development calibration")
    inputs = {
        str((screen / n).resolve()): sha(screen / n)
        for n in ("protocol.json", "calibration.json", "training_complete.json")
    }
    recordings, rows = {}, {}
    for env in p["environments"]:
        path = Path(p["source"]) / env / "recording/recording.json"
        recording = read(path)
        inputs[str(path.resolve())] = sha(path)
        recordings[env] = (path.parent, recording["frames"])
        for name in ("outcomes.json", "features.json"):
            path = screen / env / name
            inputs[str(path.resolve())] = sha(path)
        rows[env] = read(screen / env / "outcomes.json")
        times = sorted({r["capture_time_s"] for r in rows[env]})
        validate_table(rows[env], times, list(old["models"]))
        if len(times) != len(recording["frames"]):
            raise ValueError("Incomplete development flight")
    if len(recordings) < 2:
        raise ValueError("Blocked validation requires at least two complete flights")
    output.mkdir(parents=True)
    sources = [
        Path(__file__),
        Path(__file__).with_name("quality.py"),
        Path(__file__).parents[1] / "mission_routing_v2/features.py",
    ]
    save(
        output / "plan.json",
        dict(
            schema_version="conditional_recall_training_plan_v1",
            screen=str(screen.resolve()),
            input_sha256=inputs,
            source_sha256={str(s.resolve()): sha(s) for s in sources},
            alpha=ALPHA,
            intercept_pseudo_hits=1,
            intercept_pseudo_misses=1,
            feature_names=list(NAMES),
            folds=list(recordings),
            validation="Leave one complete lighting flight out; same development scene",
            target="Matched visible targets / visible targets; zero trials excluded from fitting",
            tuning=False,
            evaluation_labels_used=False,
            scope="Development-only fit; correlated target/frame counts are not independent trials",
        ),
    )
    groups = {}
    for env, (folder, frames) in recordings.items():
        features = []
        for frame in frames:
            path = folder / frame["file"]
            if sha(path) != frame["sha256"]:
                raise ValueError("Changed development RGB")
            with np.load(path, allow_pickle=False) as data:
                features.append(encode(data["rgb"]))
        features = np.asarray(features)
        # Ensure exact alignment with the original calibration capture order.
        original = read(screen / env / "features.json")
        if original["names"] != list(BASE_NAMES) or not np.array_equal(
            features[:, : len(BASE_NAMES)], np.asarray(original["values"])
        ):
            raise ValueError("Development features do not align with original captures")
        labels = {}
        for c in old["models"]:
            selected = sorted(
                (r for r in rows[env] if r["config_id"] == c), key=lambda r: r["capture_id"]
            )
            if [r["capture_id"] for r in selected] != list(range(len(features))):
                raise ValueError("Noncontiguous capture labels")
            labels[c] = dict(
                hits=[len(r["matched_target_ids"]) for r in selected],
                trials=[len(r["visible_target_ids"]) for r in selected],
                old_target=[
                    len(r["matched_target_ids"])
                    / max(1, len(r["visible_target_ids"]))
                    * r["matched_components"]
                    / max(1, r["predicted_components"])
                    for r in selected
                ],
            )
        groups[env] = (features, labels)
        save(output / f"features_{env}.json", dict(names=list(NAMES), values=features.tolist()))
    models, folds = {}, []
    for c, m in old["models"].items():
        all_x = np.vstack([x for x, _ in groups.values()])
        hits = np.concatenate([y[c]["hits"] for _, y in groups.values()])
        trials = np.concatenate([y[c]["trials"] for _, y in groups.values()])
        models[c] = dict(
            signature=m["signature"],
            spatial=fit(all_x, hits, trials),
            base=fit(all_x[:, : len(BASE_NAMES)], hits, trials),
        )
        for heldout, (x, y) in groups.items():
            train_x = np.vstack([fs for env, (fs, _) in groups.items() if env != heldout])
            train_h = np.concatenate(
                [ys[c]["hits"] for env, (_, ys) in groups.items() if env != heldout]
            )
            train_n = np.concatenate(
                [ys[c]["trials"] for env, (_, ys) in groups.items() if env != heldout]
            )
            train_old = np.concatenate(
                [ys[c]["old_target"] for env, (_, ys) in groups.items() if env != heldout]
            )
            spatial, base = (
                fit(train_x, train_h, train_n),
                fit(train_x[:, : len(BASE_NAMES)], train_h, train_n),
            )
            legacy = ridge_fit(train_x[:, : len(BASE_NAMES)], train_old)
            predictions = dict(
                spatial=[predict(spatial, f)[0] for f in x],
                base=[predict(base, f[: len(BASE_NAMES)])[0] for f in x],
                global_recall=[spatial["global_mean"]] * len(x),
                legacy_quality=[ridge_predict(legacy, f[: len(BASE_NAMES)])[0] for f in x],
            )
            folds.append(
                dict(
                    heldout=heldout,
                    config_id=c,
                    scores={
                        name: metrics(ps, y[c]["hits"], y[c]["trials"])
                        for name, ps in predictions.items()
                    },
                    legacy_target_differs=True,
                    out_of_range_feature_fraction=float(
                        np.mean([predict(spatial, f)[1] for f in x])
                    ),
                )
            )
    save(
        output / "profile.json",
        dict(
            schema_version="conditional_recall_v5",
            feature_names=list(NAMES),
            models=models,
            training_lineage=old["training_lineage"],
            training_plan_sha256=sha(output / "plan.json"),
            semantics=(
                "Conditional visible-target recall; NOT correctness of a predicted track "
                "or calibrated mission probability"
            ),
        ),
    )
    save(
        output / "blocked_validation.json",
        dict(
            schema_version="recall_blocked_validation_v1",
            folds=folds,
            tuning=False,
            independent_scenes=False,
            uncertainty_intervals=False,
        ),
    )
    artifacts = ["profile.json", "blocked_validation.json", *(f"features_{e}.json" for e in groups)]
    save(
        output / "completion.json",
        dict(
            schema_version="recall_training_completion_v1",
            complete=True,
            plan_sha256=sha(output / "plan.json"),
            artifacts_sha256={n: sha(output / n) for n in artifacts},
            captures=len(all_x),
            frame_config_pairs=len(all_x) * len(models),
        ),
    )
    verify(output)
    print(f"Trained {len(models)} model-specific recall heads on {len(all_x)} captures", flush=True)


def verify(output):
    plan, completed = read(output / "plan.json"), read(output / "completion.json")
    if not completed["complete"] or completed["plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("Changed training plan")
    for name, expected in completed["artifacts_sha256"].items():
        if sha(output / name) != expected:
            raise ValueError("Changed training artifact")
    for key in ("input_sha256", "source_sha256"):
        for path, expected in plan[key].items():
            if sha(Path(path)) != expected:
                raise ValueError("Changed training input or implementation")
    profile = read(output / "profile.json")
    if profile["training_plan_sha256"] != sha(output / "plan.json"):
        raise ValueError("Wrong training plan")
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.screen, args.output)
