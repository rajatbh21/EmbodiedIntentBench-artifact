"""Reconstruct selected-arm training from immutable inputs and logged feedback.

This is an artifact/parameter reconstruction audit, not an independent perception
scorer: it reuses the pinned public context encoder, training emulator and reward
definition. It does not invoke either fitting routine. Bandit parameters are
recovered by fresh batch linear solves rather than the learner's rank-one updates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.mission_routing_v2.flights import split_check
from experiments.perception_router_suite.learning import FEATURES, SCHEMA_VERSION, context_vector
from experiments.perception_router_suite.train import _emulate, reward_proxy


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(training_dir):
    """Read and verify training artifacts; return evidence without modifying inputs."""
    root = Path(training_dir)
    profile_path, trace_path = root / "profile.json", root / "chosen_arm_trace.json"
    profile, trace, completion = map(_read, (profile_path, trace_path, root / "completion.json"))
    _require(completion.get("complete") is True, "Training is incomplete")
    _require(
        completion.get("profile_sha256") == _sha(profile_path)
        and completion.get("trace_sha256") == _sha(trace_path),
        "Training artifact hash differs from completion",
    )
    _require(
        profile.get("schema_version") == SCHEMA_VERSION
        and profile.get("feature_names") == list(FEATURES)
        and trace.get("schema_version") == "selected_arm_training_trace_v1",
        "Unexpected learned-profile or selected-arm trace schema",
    )
    metadata = profile["training"]
    for group in ("input_sha256", "source_sha256"):
        _require(bool(metadata[group]), "Empty training provenance hashes")
        for filename, expected in metadata[group].items():
            _require(_sha(filename) == expected, "Changed training provenance: " + filename)
    inputs = [Path(p) for p in metadata["input_sha256"]]
    calibrations = [p for p in inputs if p.name == "calibration.json"]
    parent_plans = [p for p in inputs if p.name == "plan.json"]
    _require(len(calibrations) == len(parent_plans) == 1, "Ambiguous training input provenance")
    screen = calibrations[0].parent
    protocol, calibration, parent = (
        _read(screen / "protocol.json"),
        _read(calibrations[0]),
        _read(parent_plans[0]),
    )
    _require(
        profile["training_lineage"] == calibration["training_lineage"],
        "Training lineage differs from original calibration",
    )
    for case in parent["cases"].values():
        split_check(profile["training_lineage"], case["lineage"])
    action_list = parent["actions_by_scale"]["1.0"]
    actions = {a["action_id"]: a for a in action_list}
    _require(
        len(actions) == len(action_list) == 18
        and profile["action_signatures"]
        == {a: value["source_signature"] for a, value in actions.items()}
        and set(profile["bandit"]) == set(actions)
        and set(profile["supervised"]) == set(actions),
        "Learned action pool or signatures differ from frozen action descriptors",
    )
    captures = []
    for environment in protocol["environments"]:
        rgb = _read(screen / environment / "features.json")["values"]
        outcomes = _read(screen / environment / "outcomes.json")
        rows = {}
        for row in outcomes:
            values = rows.setdefault(row["capture_id"], {})
            _require(row["config_id"] not in values, "Duplicate original outcome label")
            values[row["config_id"]] = row
        _require(set(rows) == set(range(len(rgb))), "Original capture alignment changed")
        for i, values in enumerate(rgb):
            _require(
                set(rows[i]) == set(calibration["models"]), "Incomplete original outcome table"
            )
            captures.append((environment, values, rows[i]))
    descriptions, decisions = trace["context_description"], trace["trace"]
    total = len(descriptions)
    _require(
        len(captures) == metadata["original_captures"] == 356
        and len(decisions) == total == metadata["augmented_contexts"]
        and total == len(captures) * metadata["resource_augmentations_per_capture"]
        and trace["seed"] == metadata["seed"],
        "Training context or feedback count mismatch",
    )
    _require(
        sorted(row["context_index"] for row in decisions) == list(range(total)),
        "Feedback trace does not visit every context exactly once",
    )
    _require(
        Counter((d["capture_index"], d["augmentation"]) for d in descriptions)
        == Counter(
            (c, a)
            for c in range(len(captures))
            for a in range(metadata["resource_augmentations_per_capture"])
        ),
        "Synthetic descriptions duplicate or omit a capture/augmentation pair",
    )
    matrices, vectors = {}, {}
    for action in actions:
        head = profile["bandit"][action]
        _require(head["ridge"] > 0, "Invalid training ridge penalty")
        matrices[action] = head["ridge"] * np.eye(len(FEATURES))
        vectors[action] = np.zeros(len(FEATURES))
    counts = Counter()
    positive_rewards = 0
    for step, row in enumerate(decisions):
        action = row["action_id"]
        _require(action in actions, "Selected feedback arm outside the declared pool")
        counts[action] += 1
        _require(
            row["step"] == step
            and row["feedback_arms"] == 1
            and row["action_update_count"] == counts[action],
            "Selected-arm update chronology/count is inconsistent",
        )
        description = descriptions[row["context_index"]]
        environment, rgb, outcomes = captures[description["capture_index"]]
        _require(environment == description["environment"], "Capture environment mismatch")
        candidate = _emulate(actions[action], description["state"])
        _require(candidate["feasible"], "Training trace selected an infeasible action")
        reward = reward_proxy(outcomes[candidate["source_config_id"]], candidate)
        _require(reward == row["reward"], "Selected-arm reward differs from original label")
        positive_rewards += int(reward > 0)
        feature = context_vector(rgb, candidate, description["state"])
        head = profile["bandit"][action]
        mean, scale = np.asarray(head["mean"]), np.asarray(head["scale"])
        _require(
            mean.shape == scale.shape == (len(FEATURES),)
            and np.isfinite(mean).all()
            and np.isfinite(scale).all()
            and (scale > 0).all(),
            "Invalid frozen standardization parameters",
        )
        z = (feature - mean) / scale
        matrices[action] += np.outer(z, z)
        vectors[action] += reward * z
    coefficient_errors, inverse_errors = {}, {}
    for action in actions:
        head = profile["bandit"][action]
        _require(head["updates"] == counts[action], "Frozen arm update count differs from trace")
        coefficient = np.linalg.solve(matrices[action], vectors[action])
        inverse = np.linalg.solve(matrices[action], np.eye(len(FEATURES)))
        stored_coefficient, stored_inverse = (
            np.asarray(head["coefficient"]),
            np.asarray(head["inverse"]),
        )
        _require(
            stored_coefficient.shape == coefficient.shape
            and stored_inverse.shape == inverse.shape
            and np.isfinite(stored_coefficient).all()
            and np.isfinite(stored_inverse).all(),
            "Invalid frozen bandit parameters",
        )
        coefficient_errors[action] = float(np.max(np.abs(coefficient - stored_coefficient)))
        inverse_errors[action] = float(np.max(np.abs(inverse - stored_inverse)))
    _require(
        max(coefficient_errors.values()) < 1e-8 and max(inverse_errors.values()) < 1e-8,
        "Frozen bandit parameters cannot be reconstructed from selected feedback",
    )
    _require(
        metadata["bandit_updates_by_action"] == dict(counts)
        and metadata["supervised_reward_queries"] == total * len(actions)
        and metadata["bandit_reward_queries"] == total
        and metadata["bandit_parameter_updates"] == total
        and trace["reward_queries"] == trace["parameter_updates"] == total,
        "Training audit metadata does not match actual feedback trace",
    )
    return dict(
        schema_version="perception_router_training_audit_v1",
        all_passed=True,
        profile_sha256=_sha(profile_path),
        trace_sha256=_sha(trace_path),
        completion_sha256=_sha(root / "completion.json"),
        auditor_source_path=str(Path(__file__).resolve()),
        auditor_source_sha256=_sha(__file__),
        input_hashes_verified=len(metadata["input_sha256"]),
        training_source_hashes_verified=len(metadata["source_sha256"]),
        original_captures=len(captures),
        reconstructed_selected_rewards=total,
        feasible_selected_actions=total,
        positive_selected_rewards=positive_rewards,
        reconstructed_parameter_updates=sum(counts.values()),
        updates_by_action=dict(sorted(counts.items())),
        coefficient_max_abs_error=max(coefficient_errors.values()),
        inverse_max_abs_error=max(inverse_errors.values()),
        training_evaluation_lineages_disjoint=True,
        scope=(
            "Reconstructs original chosen rewards and batch A,b parameters; validates trace "
            "chronology, counts, signatures and hashes. Reuses pinned context encoder, training "
            "emulator and reward definition. Does not independently rerun neural predictions, "
            "verify optimal UCB action choices, or audit supervised fitted coefficients."
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.training)
    destination = args.output or args.training / "training_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"Verified {result['reconstructed_parameter_updates']} selected-arm updates", flush=True)
