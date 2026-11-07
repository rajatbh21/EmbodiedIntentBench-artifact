"""Reconstruct dense training labels from physical frames and real neural masks."""

import argparse
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.flights import load_scenario
from experiments.mission_routing_v2.models import CONTROL, ENSEMBLE, KIND, ROTATED
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_routing_v5.quality import encode
from experiments.routing_development_matrix.cohort import verify
from experiments.routing_development_matrix.score import verify_labels, visibility
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame


def validate_inference_grid(measured, expected, specs):
    keys = [(r["capture_id"], r["config_id"]) for r in measured]
    if len(set(keys)) != len(keys) or set(keys) != set(expected):
        raise ValueError("Incomplete actual inference grid")
    expected_calls = {
        spec.config_id: (0 if spec.config_id == ENSEMBLE else 4 if spec.kind == KIND else 1)
        for spec in specs
    }
    if any(
        r["actual_neural_calls"] != expected_calls[r["config_id"]]
        or r["measured_wall_clock_s"] is None
        or not np.isfinite(r["measured_wall_clock_s"])
        or r["measured_wall_clock_s"] <= 0
        for r in measured
    ):
        raise ValueError("Incorrect actual neural-call accounting or timing")
    return sum(r["actual_neural_calls"] for r in measured)


def run(root):
    p = verify(root)
    from experiments.mission_routing_v2_route_validation import audit

    if not audit(root)["all_passed"]:
        raise ValueError("Invalid physical route")
    records, metadata, captures, pairs, calls = {}, {}, 0, 0, 0
    for flight, case in p["cases"].items():
        folder = verify_labels(root, flight)
        rows = read(folder / "outcomes.json")
        expected = {(r["capture_id"], r["config_id"]): r for r in rows}
        features = read(folder / "features.json")["values"]
        observed_visibility = read(folder / "visibility.json")["observations"]
        scenario = load_scenario(
            root / flight / "rapid_survey/scenario.json", case["environment"], Path(p["assets"])
        )
        if split_fingerprint(scenario, scenario.scenery_sha256) != case["lineage"]:
            raise ValueError("Declared lineage differs from actual scenario")
        pred = root / flight / "predictions"
        manifest = checked_manifest(pred, root / "protocol.json")
        masks = {c: VerifiedMasks(pred, m) for c, m in manifest["models"].items()}
        measured = read(pred / "inference.json")
        calls += validate_inference_grid(measured, set(expected), scenario.executors)
        scorers = {
            c: MissionEvaluator(
                scenario.matching_iou_threshold,
                len(scenario.targets),
                scenario.observation_interval_s,
            )
            for c in masks
        }
        backend = ReplayBackend(scenario, root / flight / "recording")
        reconstructed_visibility = []
        try:
            for index, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                t = (timestamp - scenario.schedule.epoch_ns) / 1e9
                if not np.array_equal(encode(frame.rgb), features[index]):
                    raise ValueError("RGB feature reconstruction differs")
                reconstructed_visibility.extend(
                    visibility(frame, scenario.world["objects"], index, t)
                )
                key = rgb_key(frame.rgb)
                if not np.array_equal(
                    masks[ENSEMBLE][key], masks[CONTROL][key] | masks[ROTATED][key]
                ):
                    raise ValueError("Derived ensemble differs from its two charged neural views")
                truth = evaluation_frame(frame, index, t, scenario)
                for config, scorer in scorers.items():
                    row = dict(
                        capture_id=index,
                        config_id=config,
                        **scorer.update(truth, masks[config][rgb_key(frame.rgb)]).to_dict(),
                    )
                    if row != expected[(index, config)]:
                        raise ValueError("Canonical dense labels differ")
                    pairs += 1
                captures += 1
        finally:
            backend.close()
        if reconstructed_visibility != observed_visibility:
            raise ValueError("Visibility reconstruction differs")
        recording = read(root / flight / "recording/recording.json")
        records[flight] = [(f["timestamp_ns"], f["body_pose"]) for f in recording["frames"]]
        metadata[flight] = dict(
            labels_completion_sha256=sha(folder / "completion.json"),
            recording_sha256=sha(root / flight / "recording/recording.json"),
            manifest_sha256=sha(pred / "manifest.json"),
        )
    for group in p["layout_groups"]:
        if records[group + "_day"] != records[group + "_dusk_haze"]:
            raise ValueError("Paired lighting trajectories differ")
    result = dict(
        schema_version="development_matrix_validation_v1",
        complete=True,
        new_physical_flights=len(records),
        new_captures=captures,
        new_frame_config_pairs=pairs,
        actual_neural_calls=calls,
        paired_physical_poses_identical=True,
        canonical_labels_reconstructed=True,
        rgb_features_reconstructed=True,
        visibility_reconstructed=True,
        metadata=metadata,
        protocol_sha256=sha(root / "protocol.json"),
        validator_sha256=sha(Path(__file__)),
    )
    save(root / "validation.json", result)
    print(
        {
            k: result[k]
            for k in ("complete", "new_captures", "new_frame_config_pairs", "actual_neural_calls")
        },
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
