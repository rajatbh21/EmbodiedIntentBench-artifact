"""Post-fit, read-only validation of the frozen learned-report experiment.

Mask matching, public features, forward passes, and metrics are recomputed here.
Deterministic refitting deliberately uses the frozen original optimizer. This is
not an independent reimplementation of static association or prefix enumeration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import label as connected_components

from experiments.mission_report_verifier_learning import fit


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def matches(mask, truth, targets):
    """Independent 4-connected, >=4-pixel, greedy one-to-one IoU-.3 matching."""
    require(mask.dtype == np.bool_ and mask.shape == truth.shape, "Raw array shape/dtype")
    components, count = connected_components(mask)
    areas = np.bincount(components.ravel(), minlength=count + 1)
    pairs = []
    result = {cid: None for cid in range(1, count + 1) if areas[cid] >= 4}
    for cid in result:
        component = components == cid
        for instance, target in targets.items():
            actual = truth == instance
            overlap = int(np.count_nonzero(component & actual))
            union = int(areas[cid] + np.count_nonzero(actual) - overlap)
            if overlap and overlap / union >= 0.3:
                pairs.append((-overlap / union, cid, instance, target))
    used = set()
    for _, cid, instance, target in sorted(pairs):
        if result[cid] is None and instance not in used:
            result[cid] = target
            used.add(instance)
    return result


def public_vector(public, config_ids):
    require(public["ground_truth_access"] is False, "Public feature declaration")
    forbidden = {"label", "target_id", "support_target_ids", "identity_consistent", "correct"}
    require(not forbidden.intersection(public), "Private field in public candidate")
    refs = public["evidence"]
    require(
        all(set(r) == {"capture_ns", "component_id", "source_config_id"} for r in refs),
        "Unknown feature reference field",
    )
    times = [r["capture_ns"] for r in refs]
    sources = [r["source_config_id"] for r in refs]
    require(len(times) >= 2 and times == sorted(set(times)), "Repeated causal support")
    require(set(sources) <= set(config_ids), "Unknown selected source")
    span, gap = times[-1] - times[0], times[-1] - times[-2]
    require(
        public["hits"] == len(refs)
        and public["span_ns"] == span
        and public["last_gap_ns"] == gap
        and public["current_capture_ns"] == times[-1]
        and public["source_config_ids"] == sorted(set(sources)),
        "Unbound feature summary",
    )
    require(span >= 2_000_000_000, "Candidate prefix admission")
    return [
        math.log1p(len(refs)),
        math.log1p(span / 1e9),
        math.log1p(gap / 1e9),
        len(set(sources)) / len(config_ids),
        *(sources.count(c) / len(refs) for c in config_ids),
    ]


def probability(vector, checkpoint):
    x = (np.asarray(vector) - checkpoint["normalizer"]["mean"]) / checkpoint["normalizer"]["std"]
    p = {k: np.asarray(v) for k, v in checkpoint["parameters"].items()}
    if checkpoint["model"] == "logistic":
        z = x @ p["w"] + p["b"]
    else:
        z = np.tanh(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]
    return float(1 / (1 + np.exp(-np.clip(z, -40, 40))))


def metrics(labels, probabilities):
    labels = [bool(y) for y in labels]
    accepted = [p >= 0.5 for p in probabilities]
    tp = sum(y and a for y, a in zip(labels, accepted, strict=True))
    fp = sum(not y and a for y, a in zip(labels, accepted, strict=True))
    tn = sum(not y and not a for y, a in zip(labels, accepted, strict=True))
    fn = sum(y and not a for y, a in zip(labels, accepted, strict=True))
    clipped = [min(1 - 1e-12, max(1e-12, p)) for p in probabilities]
    return dict(
        candidates=len(labels),
        positives=sum(labels),
        negatives=len(labels) - sum(labels),
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        acceptance_precision=tp / (tp + fp) if tp + fp else None,
        positive_acceptance_recall=tp / (tp + fn) if tp + fn else None,
        brier=sum((p - y) ** 2 for y, p in zip(labels, probabilities, strict=True)) / len(labels),
        log_loss=-sum(math.log(p if y else 1 - p) for y, p in zip(labels, clipped, strict=True))
        / len(labels),
        threshold=0.5,
    )


def equal_metrics(actual, expected):
    require(set(actual) == set(expected), "Metric fields")
    for key, value in actual.items():
        require(
            value == expected[key]
            if value is None
            else math.isclose(value, expected[key], abs_tol=1e-12, rel_tol=1e-12),
            "Reconstructed metric differs: " + key,
        )


def audit(root):
    root = Path(root)
    completion = read(root / "completion.json")
    for key in ("plan", "candidates", "predictions", "summary"):
        require(
            sha(root / (key + ".json")) == completion[key + "_sha256"], "Artifact changed: " + key
        )
    plan, data = read(root / "plan.json"), read(root / "candidates.json")
    summary, predictions = read(root / "summary.json"), read(root / "predictions.json")
    for field in ("input_sha256", "implementation_sha256"):
        for path, expected in plan[field].items():
            require(sha(path) == expected, "Frozen source changed: " + path)
    for key in ("recording", "layout", "world", "route"):
        require(
            not {r[key] for r in plan["split_lineage"]["train"]}
            & {r[key] for r in plan["split_lineage"]["held_out_development"]},
            "Split leakage: " + key,
        )
    require(
        not any(
            "mission_pipeline" in p or "mission_joint_constraints" in p
            for p in plan["input_sha256"]
        ),
        "Current outcome as training input",
    )

    # Bind every occurrence to its selected masks before evaluating candidate labels.
    runs, mask_cache, historical_checks = {}, {}, 0
    for index, run in enumerate(plan["runs"]):
        folder = Path(run["output"])
        scenario, intent = read(run["scenario"]), read(run["intent"])
        targets = {
            o["label"]: o["object_id"] for o in scenario["world"]["objects"] if o["is_target"]
        }
        events = read(folder / "execution/events.json")
        captures = {}
        for event in events:
            time_ns = round(event["capture_time_s"] * 1e9)
            require(time_ns not in captures, "Duplicate selected capture")
            raw = folder / "execution" / f"observation_{event['observation_id']:05d}.npz"
            raw_hash = sha(raw)
            require(raw_hash == plan["input_sha256"][str(raw.resolve())], "Changed selected mask")
            key = (raw_hash, tuple(sorted(targets.items())))
            if key not in mask_cache:
                with np.load(raw, allow_pickle=False) as archive:
                    mask_cache[key] = matches(
                        archive["prediction_mask"], archive["object_ids"], targets
                    )
            captures[time_ns] = (event, mask_cache[key])
        result = read(folder / "execution/results.json")
        require(
            result["backend"]["recording_identity"] == run["lineage"]["recording"],
            "Recording binding",
        )
        for report in result["report_quality"]["reports"]:
            support = [
                captures[round(r["capture_s"] * 1e9)][1][r["component_id"]]
                for r in report["evidence"]
            ]
            label = support[-1] is not None and len(set(support)) == 1
            require(
                support == report["support_target_ids"] and label == report["identity_consistent"],
                "Independent historical report label differs",
            )
            historical_checks += 1
        runs[run["output"]] = (run, intent, captures)
        if (index + 1) % 30 == 0:
            print(f"Independent labels: {index + 1}/{len(plan['runs'])} traces", flush=True)

    ids, vectors, occurrence_checks = set(), {}, 0
    for row in data["candidates"]:
        cid, public = row["candidate_id"], row["public"]
        require(
            cid not in ids
            and cid
            == digest(dict(recording=row["lineage"]["recording"], evidence=public["evidence"])),
            "Candidate deduplication identity",
        )
        ids.add(cid)
        vectors[cid] = public_vector(public, plan["source_config_ids"])
        require(vectors[cid] == row["features"], "Independent features differ")
        for origin in row["source_occurrences"]:
            run, intent, captures = runs[origin["run"]]
            require(
                row["split"] == run["split"] and row["lineage"] == run["lineage"],
                "Candidate source split",
            )
            require(
                origin["capture_ns"] == public["current_capture_ns"]
                and origin["component_id"] == public["current_component_id"],
                "Occurrence binding",
            )
            phase = next(
                p
                for p in intent["mission"]["phases"]
                if round(p["start_s"] * 1e9) <= origin["capture_ns"] < round(p["end_s"] * 1e9)
            )
            support = []
            for ref in public["evidence"]:
                event, matched = captures[ref["capture_ns"]]
                require(
                    event["execution"]["success"] is True
                    and event["execution"]["config_id"] == ref["source_config_id"]
                    and round(phase["start_s"] * 1e9)
                    <= ref["capture_ns"]
                    < round(phase["end_s"] * 1e9),
                    "Unselected or cross-phase evidence",
                )
                support.append(matched[ref["component_id"]])
            label = support[-1] is not None and len(set(support)) == 1
            require(
                support == row["evaluator_support_target_ids"] and label == row["label"],
                "Independent candidate label",
            )
            occurrence_checks += 1

    train = [r for r in data["candidates"] if r["split"] == "train"]
    x, y = (
        np.asarray([vectors[r["candidate_id"]] for r in train]),
        np.asarray([r["label"] for r in train]),
    )
    pred_map = {(r["model"], r["candidate_id"]): r for r in predictions}
    require(len(pred_map) == len(predictions) == len(data["candidates"]) * 2, "Prediction coverage")
    decisions = {}
    for kind in plan["models"]:
        path = root / (kind + ".json")
        require(sha(path) == completion["checkpoint_sha256"][kind], "Checkpoint binding")
        checkpoint = read(path)
        provenance = checkpoint["training_provenance"]
        require(
            provenance["training_candidate_ids"] == [r["candidate_id"] for r in train]
            and provenance["training_lineage"] == plan["split_lineage"]["train"]
            and provenance["plan_sha256"] == sha(root / "plan.json")
            and provenance["candidates_sha256"] == sha(root / "candidates.json"),
            "Training-only provenance",
        )
        require(
            checkpoint == fit(x, y, kind, provenance),
            "Deterministic original-optimizer refit differs",
        )
        require(
            checkpoint["normalizer"]["mean"] == x.mean(axis=0).tolist()
            and checkpoint["normalizer"]["std"] == np.maximum(x.std(axis=0), 1e-8).tolist(),
            "Training-only normalization differs",
        )
        probabilities = {}
        for row in data["candidates"]:
            cid = row["candidate_id"]
            p = probability(vectors[cid], checkpoint)
            saved = pred_map[kind, cid]
            require(
                saved["split"] == row["split"]
                and saved["label"] == row["label"]
                and math.isclose(p, saved["probability"], abs_tol=1e-14, rel_tol=1e-14),
                "Independent prediction differs",
            )
            probabilities[cid] = p
        for split in ("train", "held_out_development"):
            rows = [r for r in data["candidates"] if r["split"] == split]
            equal_metrics(
                metrics(
                    [r["label"] for r in rows], [probabilities[r["candidate_id"]] for r in rows]
                ),
                summary["models"][kind][split],
            )
        for flight, expected in summary["models"][kind]["held_out_by_recording"].items():
            rows = [r for r in data["candidates"] if r["flight"] == flight]
            equal_metrics(
                metrics(
                    [r["label"] for r in rows], [probabilities[r["candidate_id"]] for r in rows]
                ),
                expected,
            )
        decisions[kind] = {cid: p >= 0.5 for cid, p in probabilities.items()}
    held = [r for r in data["candidates"] if r["split"] == "held_out_development"]
    for name, prior in (("accept_all", 1.0), ("training_prior", float(y.mean()))):
        equal_metrics(
            metrics([r["label"] for r in held], [prior] * len(held)), summary["baselines"][name]
        )
    return dict(
        complete=True,
        audit_created_after_fit=True,
        source_sha256=sha(__file__),
        completion_sha256=sha(root / "completion.json"),
        candidates=len(ids),
        source_occurrences=occurrence_checks,
        independently_matched_raw_masks=len(mask_cache),
        independent_historical_report_labels=historical_checks,
        independent_candidate_labels=True,
        independent_public_feature_reconstruction=True,
        independent_forward_probabilities_and_metrics=True,
        deterministic_original_optimizer_refits=True,
        training_only_normalization_and_provenance=True,
        all_frozen_source_and_input_hashes_valid=True,
        threshold_decision_disagreements=sum(
            decisions["logistic"][cid] != decisions["mlp8"][cid] for cid in ids
        ),
        association_and_prefix_completeness_independently_reimplemented=False,
        limitations=(
            "Post-fit validator. Association and candidate-prefix enumeration still share "
            "the frozen original extractor. Candidate prefixes are dependent; these are "
            "not mission-success results."
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    path = args.root / "postfit_audit.json"
    require(not path.exists(), "Preserve previous audit")
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
