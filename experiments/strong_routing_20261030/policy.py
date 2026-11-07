"""Frozen oblique-sensor reliability baselines with causal resource guards.

Training uses every scored frame of the three earlier oblique development clips,
including the geometry-infeasible revisit clip. These point estimates are not
statistical confidence bounds or calibrated mission-success probabilities.
Runtime keeps only per-model scalars; it never reads training files, site IDs,
candidate predictions, evaluator labels, or a future link trace.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from experiments.mission_routing_v2.features import NAMES as RGB_NAMES
from experiments.mission_routing_v5.quality import NAMES as QUALITY_NAMES
from experiments.mission_task_deadlines.policy import DeadlineAwarePolicy

SCHEMA = "oblique_model_reliability_20261030_v1"
POLICIES = ("fastest_feasible", "calibrated_quality", "calibrated_evidence_rate")
SOURCE = Path("results/wide_area_viewpoint_native_probe_v1_20261028")
TRAIN_CASES = tuple(f"{site}/offset_14/speed_1.5" for site in (
    "shoreline_loop", "wide_area_search", "sector_revisit"))
MODELS = ("CFG_MASKRCNN_1280", "CFG_YOLO11S_960_ROT0", "CFG_CABINET_UAVID_1280")
REGIME = dict(camera_pitch_rad=1.2, image_width_px=960, image_height_px=720)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def aggregate_counts(scores, signatures):
    """Aggregate complete all-model frames, without filtering empty/failed cases."""
    counts = {name: Counter(frames=0, visible_trials=0, hits=0, false_components=0)
              for name in signatures}
    seen = set()
    for case_id, frames in scores:
        for frame in frames:
            key = (case_id, frame["capture_id"])
            _require(key not in seen, "Duplicate calibration frame")
            seen.add(key)
            _require(set(frame["scores"]) == set(signatures) | {"perfect_gt"},
                     "Every model and perfect-GT row must be present")
            truth = frame["scores"]["perfect_gt"]["visible_target_ids"]
            _require(len(truth) == len(set(truth)), "Unique visible IDs required")
            for name in signatures:
                row = frame["scores"][name]
                visible, hits = row["visible_target_ids"], row["matched_target_ids"]
                false = row["false_positive_components"]
                _require(set(visible) == set(truth) and len(visible) == len(set(visible))
                         and len(hits) == len(set(hits)) and set(hits) <= set(visible),
                         "Consistent all-world calibration labels required")
                _require(type(false) is int and false >= 0,
                         "Nonnegative integer false components required")
                counts[name].update(frames=1, visible_trials=len(visible), hits=len(hits),
                                    false_components=false)
    return {name: dict(signature=signatures[name], **row,
                       recall=(row["hits"] + 1) / (row["visible_trials"] + 2),
                       false_components_per_capture=(row["false_components"] + 1)
                           / (row["frames"] + 1))
            for name, row in counts.items()}


def train_calibration(output, source=SOURCE):
    """Write one immutable training artifact; only the historical cohort is read."""
    output, source = Path(output), Path(source).resolve()
    _require(not output.exists(), "Preserve the prior calibration artifact")
    protocol, completion = _read(source / "protocol.json"), _read(source / "completion.json")
    bindings = _read(source / "acquisition_bindings.json")
    _require(completion["complete"] and completion["protocol_sha256"] == _sha(source / "protocol.json")
             and completion["acquisition_bindings_sha256"] == _sha(source / "acquisition_bindings.json"),
             "Complete hash-bound native development cohort required")
    _require({c["case_id"] for c in protocol["cases"]} == set(TRAIN_CASES)
             and len(protocol["cases"]) == len(completion["cases"]) == len(bindings["cases"]) == 3,
             "All three original development cases are required")
    finished = {row["case_id"]: row for row in completion["cases"]}
    bound = {row["case_id"]: row for row in bindings["cases"]}
    _require(set(finished) == set(bound) == set(TRAIN_CASES), "Complete development bindings required")
    signatures = protocol["model_signatures"]
    _require(set(signatures) == set(MODELS), "Exactly the three original native models are required")
    input_hashes = {str(source / name): _sha(source / name) for name in (
        "protocol.json", "completion.json", "acquisition_bindings.json")}
    scores, training_frames = [], []
    for case_id in TRAIN_CASES:
        path = source / "cases" / case_id / "score.json"
        score, binding = _read(path), bound[case_id]
        _require(finished[case_id]["status"] == "completed"
                 and finished[case_id]["score_sha256"] == _sha(path)
                 and binding["status"] == "valid", "Valid scored development case required")
        _require(score["model_signatures"] == signatures, "Development model identity changed")
        indices = [frame["capture_id"] for frame in score["frames"]]
        _require(indices == score["scored_capture_ids"] == list(range(18))
                 and indices == [frame["capture_id"] for frame in binding["frames"]],
                 "All 18 predeclared motion captures must be retained")
        recording_path = Path(binding["recording_path"])
        _require(_sha(recording_path) == binding["recording_sha256"], "Development recording changed")
        recording = _read(recording_path)
        _require(recording["world"]["camera_pitch_rad"] == REGIME["camera_pitch_rad"],
                 "Calibration is specific to the declared public oblique sensor regime")
        input_hashes[str(path)] = _sha(path)
        for input_path, expected in score["input_sha256"].items():
            _require(_sha(input_path) == expected, "Changed native training input")
            _require(input_path not in input_hashes or input_hashes[input_path] == expected,
                     "Conflicting development input hashes")
            input_hashes[input_path] = expected
        for frame in binding["frames"]:
            source_path = recording_path.parent / frame["source_file"]
            _require(input_hashes[str(source_path)] == frame["source_sha256"],
                     "Training RGB is not bound to its native score")
            with np.load(source_path, allow_pickle=False) as archive:
                _require(archive["rgb"].shape == (REGIME["image_height_px"],
                    REGIME["image_width_px"], 3), "Wrong development sensor resolution")
            training_frames.append(dict(case_id=case_id, capture_id=frame["capture_id"],
                recording_identity=binding["recording_identity"],
                source_sha256=frame["source_sha256"]))
        scores.append((case_id, score["frames"]))
    document = dict(schema_version=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        sensor_regime=dict(REGIME), models=aggregate_counts(scores, signatures),
        training_frame_count=len(training_frames), training_case_ids=list(TRAIN_CASES),
        training_frames=training_frames, input_sha256=input_hashes,
        implementation_sha256={str(Path(__file__).resolve()): _sha(__file__)},
        formulas=dict(recall="(matched visible-target trials + 1) / (visible-target trials + 2)",
                      false_components_per_capture="(false components + 1) / (all scored frames + 1)"),
        selection="Every one of the 54 predeclared oblique frames, including empty observations and the failed revisit case; no model or case selection.",
        runtime="Constant per-model priors under a fixed public sensor regime; no scene ID, evaluator label, or future information.",
        limitations="Development point estimates with smoothing, not confidence bounds or calibrated mission-success probabilities. Adjacent frames are dependent; new scenes are required for evaluation.")
    verify_calibration(document, verify_files=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return document


def verify_calibration(document, verify_files=True):
    _require(isinstance(document, dict) and document.get("schema_version") == SCHEMA,
             "Frozen oblique calibration required")
    _require(document.get("sensor_regime") == REGIME, "Wrong public sensor regime")
    _require(document.get("training_frame_count") == 54 and document.get("models"),
             "Complete 54-frame calibration required")
    for row in document["models"].values():
        _require(isinstance(row.get("signature"), str) and row["signature"], "Model signature required")
        _require(all(type(row.get(k)) is int and row[k] >= 0 for k in (
            "frames", "visible_trials", "hits", "false_components")), "Integer training counts required")
        _require(row["frames"] == 54 and row["hits"] <= row["visible_trials"], "Invalid calibration denominator")
        expected_q = (row["hits"] + 1) / (row["visible_trials"] + 2)
        expected_risk = (row["false_components"] + 1) / (row["frames"] + 1)
        _require(row.get("recall") == expected_q
                 and row.get("false_components_per_capture") == expected_risk,
                 "Calibration estimates must equal the declared count formulas")
    if verify_files:
        _require(document.get("training_case_ids") == list(TRAIN_CASES), "Training case set changed")
        frames = document.get("training_frames", [])
        _require(len(frames) == 54 and {(f["case_id"], f["capture_id"]) for f in frames}
                 == {(c, i) for c in TRAIN_CASES for i in range(18)}, "Training frame set changed")
        _require(bool(document.get("input_sha256")) and bool(document.get("implementation_sha256")),
                 "Training source bindings required")
        for path, expected in {**document["input_sha256"], **document["implementation_sha256"]}.items():
            _require(_sha(path) == expected, "Frozen calibration dependency changed: " + path)
    return document


def _constant_heads(models):
    # The shared guard expects historical head schemas. Exact constant heads
    # preserve its tested guard and cadence semantics without consulting stale fits.
    quality, risk = {}, {}
    for name, model in models.items():
        zero = [0.0] * len(RGB_NAMES)
        head = dict(mean=zero[:], scale=[1.0] * len(RGB_NAMES), coefficient=zero[:])
        q_head = {key: value[:] for key, value in head.items()}
        q = model["recall"]
        q_head["coefficient"][0] = math.log(q / (1 - q))
        q_head.update(train_min=[-1e6] * len(RGB_NAMES), train_max=[1e6] * len(RGB_NAMES))
        quality[name] = dict(signature=model["signature"], base=q_head)
        rate = model["false_components_per_capture"]
        risk[name] = dict(signature=model["signature"],
                          risk=dict(head=head, floor=rate, pseudocount_per_capture=rate))
    return (dict(schema_version="conditional_recall_v5", feature_names=list(QUALITY_NAMES), models=quality),
            dict(schema_version="mission_false_component_risk_v1", feature_names=list(RGB_NAMES), models=risk))


def _tie(row):
    return row["predicted_usable_finish_ns"], row["estimate"]["energy_j"], row["action_id"]


class CalibratedPolicy:
    def __init__(self, name, actions, costs, profile, calibration):
        _require(name in POLICIES, "Unknown calibrated routing policy")
        verify_calibration(calibration, verify_files=False)
        actions = tuple(actions)
        signatures = {a.source_config_id: a.source_signature for a in actions}
        _require(signatures == {k: v["signature"] for k, v in calibration["models"].items()},
                 "Calibration and action model identities differ")
        self.name, self.calibration_sha256 = name, _digest(calibration)
        self.models = {key: {k: row[k] for k in (
            "signature", "recall", "false_components_per_capture")}
            for key, row in calibration["models"].items()}
        quality, risk = _constant_heads(self.models)
        self.forecaster = DeadlineAwarePolicy("deadline_aware", actions, costs, profile, quality, risk)

    def select(self, **observation):
        decision = self.forecaster.select(**observation)
        admitted = []
        hits = decision["intent_requirements"]["min_hits"]
        allowance = decision["intent_requirements"]["max_false_components"]
        for row in decision["candidates"]:
            prior = self.models[row["source_config_id"]]
            row["predicted_recall"] = prior["recall"]
            row["predicted_false_components"] = prior["false_components_per_capture"]
            row["out_of_training_range_fraction"] = None
            row["score"] = None
            if row["feasible"]:
                if self.name == "fastest_feasible":
                    row["score"] = -(row["predicted_usable_finish_ns"] - observation["release_ns"]) / 1e9
                elif self.name == "calibrated_quality":
                    row["score"] = prior["recall"]
                else:
                    row["score"] = (prior["recall"] ** hits / row["evidence_window_s"]
                        * math.exp(-prior["false_components_per_capture"] / (1 + allowance)))
                admitted.append(row)
        selected = min(admitted, key=lambda r: (-r["score"], *_tie(r))) if admitted else None
        decision.update(policy=self.name, action_id=selected["action_id"] if selected else None,
            reason=(self.name + "_under_common_constraints") if selected else "no_feasible_action",
            common_resource_guard="deadline_aware_v1", calibration_sha256=self.calibration_sha256,
            calibration_scope="Fixed oblique sensor regime, development-only constant priors",
            evaluation_learning_updates=0, stale_quality_risk_heads_used=False,
            score_interpretation="Routing utility, not a mission-success probability")
        return decision


def make_policy(name, actions, costs, profile, quality, risk, learned_profile=None):
    """Drop-in paid factory; historical quality/risk arguments are intentionally unused."""
    return CalibratedPolicy(name, actions, costs, profile, learned_profile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    train_calibration(args.output)
