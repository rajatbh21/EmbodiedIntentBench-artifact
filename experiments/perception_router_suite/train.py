"""Development-only supervised and selected-arm contextual-bandit training.

Only the original 356 recorded RGB contexts supply labels. Random resource
augmentations are a declared training emulator, not additional measured flights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.mission_routing_v2.features import NAMES as RGB_FEATURES
from experiments.mission_routing_v2.flights import split_check
from experiments.perception_router_suite.learning import (
    FEATURES,
    SCHEMA_VERSION,
    context_vector,
    fit_bandit,
    fit_supervised,
)

REPO = Path(__file__).resolve().parents[2]
SCREEN = REPO / "results/mission_routing_v2_screen_20261017"
PARENT = REPO / "results/mission_joint_constraints_v1_20261027"
SEED = 104729
AUGMENTATIONS = 12


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def reward_proxy(outcome, candidate):
    """Timely positive-evidence precision/recall per capture cadence, not mission success."""
    visible, matched = outcome["visible_target_ids"], outcome["matched_target_ids"]
    counts = (outcome["matched_components"], outcome["predicted_components"])
    if (
        type(visible) is not list
        or type(matched) is not list
        or any(type(v) is not str for v in (*visible, *matched))
        or len(set(visible)) != len(visible)
        or len(set(matched)) != len(matched)
        or not set(matched) <= set(visible)
        or any(type(v) is not int or v < 0 for v in counts)
        or counts[0] > counts[1]
    ):
        raise ValueError(
            "Valid development target matches and nonnegative component counts required"
        )
    if not candidate["feasible"]:
        return 0.0
    recall = len(outcome["matched_target_ids"]) / max(1, len(outcome["visible_target_ids"]))
    precision = outcome["matched_components"] / max(1, outcome["predicted_components"])
    cadence = max(2.0, 2.0 * math.ceil(candidate["latency_s"] / 2.0))
    return float(np.clip(recall * precision * 2.0 / cadence, 0, 1))


def _emulate(action, state):
    """Stationary link and configured costs; never reads a future trace."""
    switched = state["resident"] != action["action_id"]
    load_s = action["load_latency_s"] if switched else 0.0
    load_j = action["load_energy_j"] if switched else 0.0
    if action["placement"] == "local":
        latency = action["latency_s"]
        energy, communication, delivered = action["energy_j"], 0.0, True
    else:
        up_s = state["payload_mb"] * 8 / state["upload_mbps"]
        down_s = state["response_mb"] * 8 / state["download_mbps"]
        transport = up_s + down_s + state["rtt_s"] + action["latency_s"]
        latency = transport + 0.01
        energy = 0.15 + 2.0 * (up_s + down_s) + 0.5 * (state["rtt_s"] + action["latency_s"])
        communication = state["payload_mb"] + state["response_mb"]
        delivered = not state["outage"] and transport <= 8.0
    # 20 ms preparation, 50 ms routing, loaded model, and the 10 ms simulation grid.
    latency = math.ceil((latency + 0.07 + load_s) / 0.01) * 0.01
    required = energy + load_j + 2.85 + state["remaining_flight_s"] * 120.0
    feasible = (
        delivered
        and latency <= state["perception_budget_s"]
        and required <= state["reserve_available_j"]
        and action["memory_mb"] + 68 <= 4096
    )
    return dict(
        action_id=action["action_id"],
        source_config_id=action["source_config_id"],
        placement=action["placement"],
        latency_s=latency,
        reserve_required_j=required,
        estimate=dict(energy_j=energy, communication_mb=communication),
        feasible=feasible,
    )


def train(output_dir, *, screen=SCREEN, parent=PARENT):
    output_dir, screen, parent = Path(output_dir), Path(screen), Path(parent)
    if output_dir.exists():
        raise FileExistsError("Preserve the existing training artifact")
    protocol = _read(screen / "protocol.json")
    calibration = _read(screen / "calibration.json")
    completion = _read(screen / "training_complete.json")
    parent_plan = _read(parent / "plan.json")
    if completion["calibration_sha256"] != _sha(screen / "calibration.json"):
        raise ValueError("Original development calibration changed")
    lineage = calibration["training_lineage"]
    for case in parent_plan["cases"].values():
        split_check(lineage, case["lineage"])
    actions = sorted(parent_plan["actions_by_scale"]["1.0"], key=lambda a: a["action_id"])
    action_ids = [a["action_id"] for a in actions]
    if (
        len(actions) != 18
        or len(set(action_ids)) != len(actions)
        or {(a["source_config_id"], a["placement"]) for a in actions}
        != {(c, p) for c in calibration["models"] for p in ("local", "remote")}
        or any(
            calibration["models"][a["source_config_id"]]["signature"] != a["source_signature"]
            for a in actions
        )
    ):
        raise ValueError("Exactly 18 unchanged local/remote development actions required")
    inputs = {
        str((screen / n).resolve()): _sha(screen / n)
        for n in ("protocol.json", "calibration.json", "training_complete.json")
    }
    inputs[str((parent / "plan.json").resolve())] = _sha(parent / "plan.json")
    captures = []
    for environment in protocol["environments"]:
        features = _read(screen / environment / "features.json")
        outcomes = _read(screen / environment / "outcomes.json")
        if features["names"] != list(RGB_FEATURES):
            raise ValueError("Changed development RGB feature schema")
        for name in ("features.json", "outcomes.json"):
            path = screen / environment / name
            inputs[str(path.resolve())] = _sha(path)
        recording = Path(protocol["source"]) / environment / "recording/recording.json"
        inputs[str(recording.resolve())] = _sha(recording)
        by_capture = {}
        for row in outcomes:
            if row["config_id"] in by_capture.setdefault(row["capture_id"], {}):
                raise ValueError("Duplicate development capture/configuration label")
            by_capture[row["capture_id"]][row["config_id"]] = row
        if set(by_capture) != set(range(len(features["values"]))):
            raise ValueError("Incomplete development RGB/label alignment")
        for i, rgb in enumerate(features["values"]):
            if set(by_capture[i]) != set(calibration["models"]):
                raise ValueError("Incomplete full-information development table")
            captures.append(
                dict(environment=environment, capture_id=i, rgb=rgb, outcomes=by_capture[i])
            )
    if len(captures) != 356:
        raise ValueError("This frozen design requires the original 356 development captures")
    rng = np.random.default_rng(SEED)
    contexts, candidates, eligible, features_by_action = [], [], [], {a: [] for a in action_ids}
    for capture_index, capture in enumerate(captures):
        for augmentation in range(AUGMENTATIONS):
            battery = float(rng.uniform(0.28, 0.85))
            state = dict(
                perception_budget_s=float(rng.choice([1.0, 2.0, 4.0, 6.0, 8.0])),
                battery_frac=battery,
                reserve_available_j=(battery - 0.2) * 108000,
                min_hits=int(rng.choice([1, 2, 3])),
                min_evidence_span_s=float(rng.choice([0, 2, 4, 8])),
                max_false_components=int(rng.choice([0, 1, 3, 8])),
                upload_mbps=float(rng.choice([1, 5, 20, 50])),
                download_mbps=float(rng.choice([5, 20, 100])),
                rtt_s=float(rng.choice([0.03, 0.1, 0.3])),
                outage=bool(rng.random() < 0.15),
                payload_mb=float(rng.uniform(0.1, 2.5)),
                response_mb=0.12,
                resident=action_ids[int(rng.integers(len(action_ids)))],
                remaining_flight_s=float(rng.uniform(0, 50)),
            )
            rows = {a["action_id"]: _emulate(a, state) for a in actions}
            admitted = [a for a in action_ids if rows[a]["feasible"]]
            if not admitted:
                raise ValueError("Declared synthetic ranges must permit a local continuation")
            contexts.append(
                dict(
                    capture_index=capture_index,
                    augmentation=augmentation,
                    environment=capture["environment"],
                    state=state,
                )
            )
            candidates.append(rows)
            eligible.append(admitted)
            for action in action_ids:
                features_by_action[action].append(
                    context_vector(capture["rgb"], rows[action], state)
                )

    def reward_for(index, action):
        capture = captures[contexts[index]["capture_index"]]
        row = candidates[index][action]
        return reward_proxy(capture["outcomes"][row["source_config_id"]], row)

    supervised = fit_supervised(features_by_action, reward_for)
    bandit, trace = fit_bandit(features_by_action, eligible, reward_for, seed=SEED)
    update_counts = Counter(row["action_id"] for row in trace)
    if sum(m["updates"] for m in bandit.values()) != len(contexts) or any(
        m["updates"] != update_counts[a] for a, m in bandit.items()
    ):
        raise ValueError("Chosen-arm trace and bandit update counts disagree")
    output_dir.mkdir(parents=True)
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("learning.py"),
        REPO / "experiments/mission_routing_v2/features.py",
    )
    metadata = dict(
        seed=SEED,
        original_captures=len(captures),
        original_labeled_frame_config_pairs=3204,
        resource_augmentations_per_capture=AUGMENTATIONS,
        augmented_contexts=len(contexts),
        supervised_reward_queries=len(contexts) * len(actions),
        bandit_reward_queries=len(trace),
        bandit_parameter_updates=len(trace),
        bandit_updates_by_action=dict(sorted(update_counts.items())),
        input_sha256=inputs,
        source_sha256={str(p.resolve()): _sha(p) for p in source_paths},
        original_recordings_disjoint_from_evaluation=True,
        evaluation_labels_used=False,
        evaluation_feedback=False,
        validation="Original two lighting flights share one scene; no independent-scene claim",
        supervised_method="Disjoint full-information ridge reward regression, alpha=10",
        bandit_method="Disjoint LinUCB, ridge=1, alpha=0.5; selected-arm feedback only",
        evaluation_method="Frozen greedy reward mean for both learned selectors",
        reward="On-time recall * component precision * 2 / max(2, ceil(latency/2)*2)",
        empty_visible_frames="Zero positive-evidence reward, not a detection failure label",
        reward_is_mission_success_probability=False,
        source_of_action_costs="Configured descriptors in frozen joint-constraints parent plan",
        emulator_scope=(
            "356 original RGB contexts with independent seeded synthetic resource states; "
            "stationary snapshot links; fixed 30Wh/120W, 10ms grid, 50ms router, 20ms preparation. "
            "Reserve includes remaining flight and current call/downstream, omits runtime's "
            "cheap future inference continuation. No private future links or mission outcomes. "
            "Synthetic contexts are not additional independent captures or real measurements."
        ),
        limitations=(
            "The proxy does not optimize repeated mission evidence. Intent fields are public "
            "covariates, but the reward itself is a generic per-call perception utility. "
            "No hyperparameter tuning or claim of published-system reproduction."
        ),
    )
    profile = dict(
        schema_version=SCHEMA_VERSION,
        feature_names=list(FEATURES),
        action_signatures={a["action_id"]: a["source_signature"] for a in actions},
        action_sources={a["action_id"]: a["source_config_id"] for a in actions},
        training_lineage=lineage,
        supervised=supervised,
        bandit=bandit,
        training=metadata,
    )
    _save(output_dir / "profile.json", profile)
    _save(
        output_dir / "chosen_arm_trace.json",
        dict(
            schema_version="selected_arm_training_trace_v1",
            seed=SEED,
            trace=trace,
            reward_queries=len(trace),
            parameter_updates=len(trace),
            context_description=contexts,
        ),
    )
    _save(
        output_dir / "completion.json",
        dict(
            complete=True,
            profile_sha256=_sha(output_dir / "profile.json"),
            trace_sha256=_sha(output_dir / "chosen_arm_trace.json"),
        ),
    )
    return profile


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = train(args.output)
    print(
        f"Trained on {result['training']['original_captures']} original RGB contexts; "
        f"{result['training']['bandit_parameter_updates']} selected-arm updates",
        flush=True,
    )
