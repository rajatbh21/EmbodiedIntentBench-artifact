"""Evaluator-only dense labels and observed target-size/visibility coverage.

Visibility loss conflates field of view, scenery occlusion and target orientation;
mask fill ratios are shape descriptors, never ground-truth occlusion fractions.
None of these fields enter the policy RGB feature extractor.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import validate_table
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.flights import load_scenario
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_routing_v5.quality import NAMES, encode
from experiments.routing_development_matrix.cohort import verify
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame


def visibility(frame, objects, index, time_s):
    rows = []
    for obj in objects:
        if not obj["is_target"]:
            continue
        ys, xs = np.where(frame.object_ids == obj["label"])
        count = len(xs)
        width = int(xs.max() - xs.min() + 1) if count else 0
        height = int(ys.max() - ys.min() + 1) if count else 0
        rows.append(
            dict(
                capture_id=index,
                capture_time_s=time_s,
                target_id=obj["object_id"],
                visible=count > 0,
                visible_pixels=count,
                bbox_width_px=width,
                bbox_height_px=height,
                bbox_fill_ratio=count / (width * height) if count else None,
                image_fraction=count / frame.object_ids.size,
                camera_to_target_origin_m=math.dist(
                    frame.body_pose["position_m"], obj["position_m"]
                ),
            )
        )
    return rows


def run(root, flight):
    p = verify(root)
    folder = root / flight
    destination = folder / "development_labels"
    if destination.exists():
        raise FileExistsError("Preserve previous development labels")
    case = p["cases"][flight]
    scenario = load_scenario(
        folder / "rapid_survey/scenario.json", case["environment"], Path(p["assets"])
    )
    prediction_folder = folder / "predictions"
    manifest = checked_manifest(prediction_folder, root / "protocol.json")
    if manifest["recording_identity"] != scenario.recording_identity or manifest[
        "recording_sha256"
    ] != sha(folder / "recording/recording.json"):
        raise ValueError("Wrong physical recording")
    masks, scorers = {}, {}
    for spec in scenario.executors:
        model = manifest["models"][spec.config_id]
        if model["signature"] != model_signature(spec):
            raise ValueError("Wrong model signature")
        masks[spec.config_id] = VerifiedMasks(prediction_folder, model)
        scorers[spec.config_id] = MissionEvaluator(
            scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
        )
    destination.mkdir()
    save(
        destination / "plan.json",
        dict(
            schema_version="development_label_plan_v1",
            scope="evaluator_only",
            source_sha256=sha(Path(__file__)),
            protocol_sha256=sha(root / "protocol.json"),
            manifest_sha256=sha(prediction_folder / "manifest.json"),
            recording_sha256=sha(folder / "recording/recording.json"),
            feature_source_sha256=sha(Path(__file__).parents[1] / "mission_routing_v5/quality.py"),
        ),
    )
    rows, features, coverage, times = [], [], [], []
    backend = ReplayBackend(scenario, folder / "recording")
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            frame = backend.capture()
            t = (timestamp - scenario.schedule.epoch_ns) / 1e9
            times.append(t)
            features.append(encode(frame.rgb).tolist())
            truth = evaluation_frame(frame, index, t, scenario)
            coverage.extend(visibility(frame, scenario.world["objects"], index, t))
            for config, scorer in scorers.items():
                rows.append(
                    dict(
                        capture_id=index,
                        config_id=config,
                        **scorer.update(truth, masks[config][rgb_key(frame.rgb)]).to_dict(),
                    )
                )
    finally:
        backend.close()
    validate_table(rows, times, list(masks))
    save(destination / "outcomes.json", rows)
    save(destination / "features.json", dict(names=list(NAMES), values=features))
    save(
        destination / "visibility.json",
        dict(
            schema_version="target_visibility_coverage_v1",
            gt_aware=True,
            occlusion_fraction_measured=False,
            observations=coverage,
        ),
    )
    save(
        destination / "completion.json",
        dict(
            schema_version="development_label_completion_v1",
            complete=True,
            plan_sha256=sha(destination / "plan.json"),
            artifacts_sha256={
                name: sha(destination / name)
                for name in ("features.json", "outcomes.json", "visibility.json")
            },
            captures=len(features),
            frame_config_pairs=len(rows),
        ),
    )
    print(f"Scored {flight}: {len(features)} captures, {len(rows)} frame/config pairs", flush=True)


def verify_labels(root, flight):
    destination = root / flight / "development_labels"
    plan, complete = read(destination / "plan.json"), read(destination / "completion.json")
    if not complete["complete"] or complete["plan_sha256"] != sha(destination / "plan.json"):
        raise ValueError("Incomplete development labels")
    if (
        plan["source_sha256"] != sha(Path(__file__))
        or plan["feature_source_sha256"]
        != sha(Path(__file__).parents[1] / "mission_routing_v5/quality.py")
        or plan["protocol_sha256"] != sha(root / "protocol.json")
        or plan["manifest_sha256"] != sha(root / flight / "predictions/manifest.json")
        or plan["recording_sha256"] != sha(root / flight / "recording/recording.json")
    ):
        raise ValueError("Changed development scoring input or implementation")
    checked_manifest(root / flight / "predictions", root / "protocol.json")
    for name, expected in complete["artifacts_sha256"].items():
        if sha(destination / name) != expected:
            raise ValueError("Changed development label artifact")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight", required=True)
    args = parser.parse_args()
    run(args.output, args.flight)
