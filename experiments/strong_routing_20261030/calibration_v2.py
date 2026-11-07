"""Second development iteration: all 162 previously observed oblique frames.

The six first-prospective clips become development data after their observed
transfer failure. They remain the unchanged first evaluation; this calibration
must be tested only on separately locked fresh recordings. No frame, case,
model, matching threshold or routing score is selected from the new outcomes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from experiments.mission_task_deadlines.policy import DeadlineAwarePolicy
from experiments.strong_routing_20261030 import policy as original

SCHEMA = "oblique_pooled_reliability_20261030_v2"
POLICIES = original.POLICIES
TOTAL_FRAMES = 162
REGIME = original.REGIME
COHORTS = (
    ("historical_oblique", original.SOURCE, original.TRAIN_CASES),
    ("first_prospective_promoted_to_development", Path("results/strong_routing_native_v1_20261030"),
     tuple(f"{site}/sector_{sector}" for site in ("shoreline_loop", "wide_area_search", "sector_revisit")
           for sector in (2, 6))),
)


def _bound_sources(source, expected_cases, cohort_id):
    """Read all predeclared native outcomes, retaining empty and failed captures."""
    require, read, sha = original._require, original._read, original._sha
    source = Path(source).resolve()
    protocol, completion = read(source / "protocol.json"), read(source / "completion.json")
    bindings = read(source / "acquisition_bindings.json")
    require(completion["complete"] and completion["protocol_sha256"] == sha(source / "protocol.json")
            and completion["acquisition_bindings_sha256"] == sha(source / "acquisition_bindings.json"),
            "Complete hash-bound development cohort required")
    finished = {row["case_id"]: row for row in completion["cases"]}
    bound = {row["case_id"]: row for row in bindings["cases"]}
    require(len(protocol["cases"]) == len(completion["cases"]) == len(bindings["cases"]) == len(expected_cases)
            and {row["case_id"] for row in protocol["cases"]} == set(finished) == set(bound) == set(expected_cases),
            "Every declared development case is required exactly once")
    signatures = protocol["model_signatures"]
    require(set(signatures) == set(original.MODELS), "All three unchanged native models required")
    hashes = {str(source / name): sha(source / name) for name in (
        "protocol.json", "completion.json", "acquisition_bindings.json")}
    scores, training_frames = [], []
    for case_id in expected_cases:
        score_path = source / "cases" / case_id / "score.json"
        score, binding = read(score_path), bound[case_id]
        require(finished[case_id]["status"] == "completed" and binding["status"] == "valid"
                and finished[case_id]["score_sha256"] == sha(score_path), "Valid native result required")
        require(score["model_signatures"] == signatures, "Development model identity changed")
        indices = [row["capture_id"] for row in score["frames"]]
        require(indices == score["scored_capture_ids"] == list(range(18))
                and indices == [row["capture_id"] for row in binding["frames"]],
                "All 18 motion-only captures per case must be retained")
        recording_path = Path(binding["recording_path"])
        require(sha(recording_path) == binding["recording_sha256"], "Training recording changed")
        require(read(recording_path)["world"]["camera_pitch_rad"] == REGIME["camera_pitch_rad"],
                "Only the fixed declared public oblique sensor regime is supported")
        hashes[str(score_path)] = sha(score_path)
        for path, expected in score["input_sha256"].items():
            require(sha(path) == expected, "Changed native training input")
            require(path not in hashes or hashes[path] == expected, "Conflicting input hashes")
            hashes[path] = expected
        for frame in binding["frames"]:
            path = recording_path.parent / frame["source_file"]
            require(hashes[str(path)] == frame["source_sha256"], "Native training RGB binding differs")
            with np.load(path, allow_pickle=False) as archive:
                require(archive["rgb"].shape == (REGIME["image_height_px"], REGIME["image_width_px"], 3),
                        "Wrong development RGB sensor dimensions")
            training_frames.append(dict(cohort_id=cohort_id, case_id=case_id,
                capture_id=frame["capture_id"], recording_identity=binding["recording_identity"],
                source_sha256=frame["source_sha256"]))
        scores.append((cohort_id + ":" + case_id, score["frames"]))
    return signatures, hashes, training_frames, scores


def train_calibration(output):
    require, sha = original._require, original._sha
    output = Path(output)
    require(not output.exists(), "Preserve the prior calibration artifact")
    signatures, inputs, frames, scores = None, {}, [], []
    for cohort_id, source, cases in COHORTS:
        new_signatures, new_inputs, new_frames, new_scores = _bound_sources(source, cases, cohort_id)
        require(signatures is None or signatures == new_signatures, "Model identities differ across development cohorts")
        signatures = new_signatures
        require(all(path not in inputs or inputs[path] == value for path, value in new_inputs.items()),
                "Conflicting pooled input hashes")
        inputs.update(new_inputs)
        frames.extend(new_frames)
        scores.extend(new_scores)
    require(len(frames) == TOTAL_FRAMES and len(scores) == 9, "All 162 frames from nine clips required")
    document = dict(schema_version=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        sensor_regime=dict(REGIME), models=original.aggregate_counts(scores, signatures),
        training_frame_count=TOTAL_FRAMES, training_case_count=9,
        training_cohorts=[dict(cohort_id=key, source=str(path.resolve()), case_ids=list(cases),
            frame_count=18*len(cases)) for key, path, cases in COHORTS],
        training_frames=frames, input_sha256=inputs,
        implementation_sha256={str(Path(__file__).resolve()): sha(__file__),
                               str(Path(original.__file__).resolve()): sha(original.__file__)},
        formulas=dict(recall="(matched visible-target trials + 1) / (visible-target trials + 2)",
                      false_components_per_capture="(false components + 1) / (162 scored captures + 1)"),
        selection="All 54 historical oblique frames plus all 108 first-prospective frames. Equal per-capture aggregation. No failed/empty frame, case or model is omitted.",
        iteration_awareness="The first prospective cohort is explicitly promoted to development after observed calibration-transfer failure. Its original outcomes remain unchanged; it is not a held-out test of this updated policy.",
        evaluation_requirement="Freeze this artifact and unchanged score/guard definitions before inspecting separately acquired sector-3/7 evaluation predictions.",
        runtime="Only global scalar priors and model signatures survive into the policy. No training identities, test labels, future link or evaluation update.",
        limitations="Smoothed development point estimates, not confidence bounds or calibrated mission-success probabilities. Adjacent frames are dependent and the map assets are known.")
    verify_calibration(document, verify_files=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(document, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    return document


def verify_calibration(document, verify_files=True):
    require = original._require
    require(isinstance(document, dict) and document.get("schema_version") == SCHEMA,
            "Frozen pooled 162-frame calibration required")
    require(document.get("sensor_regime") == REGIME and document.get("training_frame_count") == TOTAL_FRAMES
            and document.get("training_case_count") == 9 and document.get("models"),
            "Fixed sensor regime and all nine development clips are required")
    for row in document["models"].values():
        require(isinstance(row.get("signature"), str) and row["signature"], "Model signature required")
        require(all(type(row.get(key)) is int and row[key] >= 0 for key in (
            "frames", "visible_trials", "hits", "false_components")), "Integer pooled counts required")
        require(row["frames"] == TOTAL_FRAMES and row["hits"] <= row["visible_trials"],
                "Invalid pooled denominator")
        require(row.get("recall") == (row["hits"]+1)/(row["visible_trials"]+2)
                and row.get("false_components_per_capture") == (row["false_components"]+1)/(TOTAL_FRAMES+1),
                "Pooled estimates differ from the predeclared formulas")
    if verify_files:
        expected_cohorts = [dict(cohort_id=key, source=str(path.resolve()), case_ids=list(cases),
            frame_count=18*len(cases)) for key,path,cases in COHORTS]
        require(document.get("training_cohorts") == expected_cohorts, "Pooled development cohort set changed")
        frames = document.get("training_frames", [])
        expected_frames = {(key,case,index) for key,_,cases in COHORTS for case in cases for index in range(18)}
        require(len(frames) == TOTAL_FRAMES and {(f["cohort_id"],f["case_id"],f["capture_id"]) for f in frames}
                == expected_frames, "Every development capture must be included exactly once")
        require(bool(document.get("input_sha256")) and bool(document.get("implementation_sha256")),
                "Pooled source bindings required")
        for path, expected in {**document["input_sha256"], **document["implementation_sha256"]}.items():
            require(original._sha(path) == expected, "Frozen pooled dependency changed: " + path)
    return document


class PoolCalibratedPolicy(original.CalibratedPolicy):
    """Same public routing rule and guard, with a truthfully versioned 162-frame fit."""
    def __init__(self, name, actions, costs, profile, calibration):
        original._require(name in POLICIES, "Unknown pooled routing policy")
        verify_calibration(calibration, verify_files=False)
        actions = tuple(actions)
        signatures = {a.source_config_id: a.source_signature for a in actions}
        original._require(signatures == {key:row["signature"] for key,row in calibration["models"].items()},
                          "Pooled calibration and action identities differ")
        self.name, self.calibration_sha256 = name, original._digest(calibration)
        self.models = {key:{field:row[field] for field in (
            "signature", "recall", "false_components_per_capture")}
            for key,row in calibration["models"].items()}
        quality, risk = original._constant_heads(self.models)
        self.forecaster = DeadlineAwarePolicy("deadline_aware", actions, costs, profile, quality, risk)


def make_policy(name, actions, costs, profile, quality, risk, learned_profile=None):
    """Drop-in paid factory; old contextual fits are intentionally not read."""
    return PoolCalibratedPolicy(name, actions, costs, profile, learned_profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train_calibration(args.output)
