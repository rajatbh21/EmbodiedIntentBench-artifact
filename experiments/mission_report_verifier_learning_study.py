"""Freeze, reconstruct, fit, and assess two learned report-verification models."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.mission_report_verifier_learning import (
    CONFIG_IDS,
    FEATURE_NAMES,
    HYPERPARAMETERS,
    MODELS,
    Model,
    fit,
    metrics,
    require,
)
from experiments.mission_report_verifier_learning_data import REPO, extract, source_inventory
from experiments.mission_runtime_profile import implementation_inventory, read, save, sha

OUTPUT = REPO / "results/mission_report_verifier_learning_v1_20261027"


def prepare(root):
    root = Path(root)
    require(not root.exists(), "Preserve earlier training attempt")
    inventory = source_inventory()
    sources = implementation_inventory()
    for path in (
        *REPO.glob("experiments/mission_report_verifier_learning*.py"),
        *REPO.glob("tests/test_mission_report_verifier_learning*.py"),
    ):
        sources[str(path.resolve())] = sha(path)
    root.mkdir(parents=True)
    plan = dict(
        schema_version="report_verifier_learning_plan_v1",
        **inventory,
        feature_names=list(FEATURE_NAMES),
        source_config_ids=list(CONFIG_IDS),
        models=list(MODELS),
        hyperparameters=HYPERPARAMETERS,
        implementation_sha256=sources,
        numpy_version=np.__version__,
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        fit_started=False,
        new_pipeline_outputs_accessed=False,
        new_joint_outputs_accessed=False,
        known_prior_evidence=(
            "Historical emitted-report class counts and candidate dedup counts were "
            "inspected to establish both-class support before protocol freeze; no "
            "hyperparameter was selected on held-out model scores. "
        ),
        split_rule=(
            "One inspected exploratory-v5 layout (two lighting recordings) for fitting; "
            "two east/west layouts (four lighting recordings) for held-out development "
            "assessment. Grouping is fixed before fit, not random frames. "
        ),
        minimum_fit_support=dict(total=20, each_class=5),
        threshold_selection=(
            "Fixed0.5 before fit; all models and failures retained; no held-out "
            "threshold/epoch/model selection "
        ),
        scope=(
            "Candidate identity-consistency classification on dependent prefixes of "
            "reused known simulator traces. Not independent field generalization or "
            "proof of improved mission routing. Current pipeline/joint outcome files are "
            "forbidden. "
        ),
    )
    save(root / "plan.json", plan)
    return plan


def verify(root):
    root = Path(root)
    plan = read(root / "plan.json")
    require(
        plan["schema_version"] == "report_verifier_learning_plan_v1", "Unknown learning protocol"
    )
    require(
        plan["feature_names"] == list(FEATURE_NAMES)
        and plan["hyperparameters"] == HYPERPARAMETERS
        and plan["models"] == list(MODELS),
        "Changed frozen model design",
    )
    for field in ("implementation_sha256", "input_sha256"):
        for path, expected in plan[field].items():
            require(sha(path) == expected, "Changed frozen training input: " + path)
    return plan


def run(root):
    root = Path(root)
    plan = verify(root)
    require(
        not (root / "candidates.json").exists() and not (root / "completion.json").exists(),
        "Preserve every attempted fit",
    )
    plan_sha = sha(root / "plan.json")
    data = extract(plan, root)
    train = [r for r in data["candidates"] if r["split"] == "train"]
    test = [r for r in data["candidates"] if r["split"] == "held_out_development"]
    require(train and test, "Nonempty disjoint candidate partitions required")
    counts = {
        split: dict(Counter(str(r["label"]) for r in rows))
        for split, rows in (("train", train), ("held_out_development", test))
    }
    x, y = (
        np.asarray([r["features"] for r in train]),
        np.asarray([r["label"] for r in train], dtype=float),
    )
    provenance = dict(
        plan_sha256=plan_sha,
        candidates_sha256=sha(root / "candidates.json"),
        training_candidate_ids=[r["candidate_id"] for r in train],
        training_lineage=plan["split_lineage"]["train"],
    )
    scores, predictions, checkpoints = {}, [], {}
    for kind in MODELS:
        checkpoint = fit(x, y, kind, provenance)
        path = root / (kind + ".json")
        save(path, checkpoint)
        model = Model.load(path, sha(path))
        checkpoints[kind] = dict(
            path=str(path),
            sha256=sha(path),
            canonical_model_sha256=model.sha256,
            parameters=checkpoint["parameter_count"],
        )
        scores[kind] = {}
        for split, rows in (("train", train), ("held_out_development", test)):
            probabilities = [model.predict(r["public"]) for r in rows]
            scores[kind][split] = metrics([r["label"] for r in rows], probabilities)
            if split == "held_out_development":
                scores[kind]["held_out_by_recording"] = {
                    flight: metrics(
                        [r["label"] for r in rows if r["flight"] == flight],
                        [
                            p
                            for r, p in zip(rows, probabilities, strict=True)
                            if r["flight"] == flight
                        ],
                    )
                    for flight in sorted({r["flight"] for r in rows})
                }
            predictions += [
                dict(
                    candidate_id=r["candidate_id"],
                    split=split,
                    model=kind,
                    probability=p,
                    label=r["label"],
                )
                for r, p in zip(rows, probabilities, strict=True)
            ]
    baselines = {
        "accept_all": metrics([r["label"] for r in test], [1.0] * len(test)),
        "training_prior": metrics([r["label"] for r in test], [float(y.mean())] * len(test)),
    }
    save(root / "predictions.json", predictions)
    summary = dict(
        complete=True,
        schema_version="report_verifier_learning_summary_v1",
        counts=counts,
        training_candidates=len(train),
        held_out_candidates=len(test),
        models=scores,
        baselines=baselines,
        checkpoints=checkpoints,
        selected_trace_updates=data["selected_trace_updates"],
        decoded_unique_selected_masks=data["decoded_unique_selected_masks"],
        raw_mask_reconstructed_historical_report_labels=data["historical_report_identity_checks"],
        provenance=provenance,
        scope=plan["scope"],
        candidate_metrics_not_mission_success=True,
        repeated_prefixes_not_independent_trials=True,
        no_current_pipeline_or_joint_outcome_files_used=True,
        held_out_data_used_for_training_or_threshold=False,
    )
    save(root / "summary.json", summary)
    verify(root)
    require(sha(root / "plan.json") == plan_sha, "Training plan changed")
    save(
        root / "completion.json",
        dict(
            complete=True,
            plan_sha256=plan_sha,
            candidates_sha256=sha(root / "candidates.json"),
            summary_sha256=sha(root / "summary.json"),
            predictions_sha256=sha(root / "predictions.json"),
            checkpoint_sha256={k: v["sha256"] for k, v in checkpoints.items()},
            actual_fit_performed=True,
            actual_raw_selected_mask_labels_reconstructed=True,
            scope=plan["scope"],
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        {"prepare": prepare, "run": run, "verify": verify}[args.command](args.output)
    except BaseException as exc:
        if args.output.exists() and not (args.output / "completion.json").exists():
            save(
                args.output / "failure.json",
                dict(error=type(exc).__name__, message=str(exc), no_automatic_refit=True),
            )
        raise


if __name__ == "__main__":
    main()
