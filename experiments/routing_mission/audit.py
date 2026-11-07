"""Check physical evidence, source integrity and policy-only paired comparisons."""

import math
from itertools import pairwise

import numpy as np
from PIL import Image, ImageDraw

from experiments.routing_mission.environment import load
from experiments.routing_mission.experiment import freeze, read, specification
from experiments.routing_mission.inference import checksum
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.mission_audit import segment_distance


def audit(root):
    spec = specification(root)
    freeze(root)
    evidence = {}
    pose_sets = {}
    samples = []
    for name, environment in spec["environments"].items():
        scenario = load(root / "scenario.json", environment)
        folder = root / name / "recording"
        backend = ReplayBackend(scenario, folder)
        poses = []
        visibility = {
            str(label): dict(visible_frames=0, max_pixels=0) for label in scenario.targets
        }
        frames = read(folder / "recording.json")["frames"]
        indices = {round(i * (len(frames) - 1) / 3) for i in range(4)}
        brightness = []
        for index, metadata in enumerate(frames):
            backend.advance_to(metadata["timestamp_ns"])
            frame = backend.capture()
            if frame.rgb.shape != (environment["height_px"], environment["width_px"], 3):
                raise ValueError("sensor resolution differs from experiment")
            if not set(np.unique(frame.object_ids)) <= {
                0,
                *[o["label"] for o in scenario.world["objects"]],
            }:
                raise ValueError("unmapped simulator labels")
            poses.append(frame.body_pose["position_m"])
            brightness.append(float(frame.rgb.mean()))
            for label in scenario.targets:
                pixels = int(np.count_nonzero(frame.object_ids == label))
                visibility[str(label)]["visible_frames"] += int(pixels > 0)
                visibility[str(label)]["max_pixels"] = max(
                    visibility[str(label)]["max_pixels"], pixels
                )
            if index in indices:
                samples.append((name, index, frame.rgb))
        backend.close()
        points = [scenario.world["initial_position_m"]] + [
            w["position_m"] for w in scenario.world["waypoints"]
        ]
        errors = [min(segment_distance(p, a, b) for a, b in pairwise(points)) for p in poses]
        pose_sets[name] = poses
        evidence[name] = dict(
            captures=len(frames),
            duration_s=scenario.duration_s,
            shape=[environment["height_px"], environment["width_px"]],
            path_length_m=sum(math.dist(a, b) for a, b in pairwise(poses)),
            nearest_route_error_p95_m=float(np.percentile(errors, 95)),
            altitude_range_m=[min(p[2] for p in poses), max(p[2] for p in poses)],
            visibility=visibility,
            mean_rgb=float(np.mean(brightness)),
            world_sdf_sha256=checksum(folder / "world.sdf"),
            recording_sha256=checksum(folder / "recording.json"),
        )
        log = (folder / "gazebo.log").read_text()
        if "not defined in SDF" in log:
            raise ValueError("unsupported SDF parameters in recorded world")
    reference = next(iter(pose_sets.values()))
    deltas = {
        name: max(math.dist(a, b) for a, b in zip(reference, poses, strict=True))
        for name, poses in pose_sets.items()
    }
    # Lighting/noise do not change flight forces. Record observed differences, not an assumption.
    report = dict(
        physical_evidence=evidence,
        cross_environment_max_pose_difference_m=deltas,
        fixed_flight_within_each_environment=True,
        limitations=[
            "sampled route error is not continuous collision validation",
            "physical rotor simulation, ideal-state controller, configured energy",
            "report correctness available only to evaluator",
        ],
    )
    if (root / "comparison.json").exists():
        results = read(root / "comparison.json")
        report["policy_runs"] = len(results["rows"])
        report["all_components_completed"] = results["complete"]
        for row in results["rows"]:
            if row["recording_sha256"] != evidence[row["environment"]]["recording_sha256"]:
                raise ValueError("comparison is not linked to audited recording")
        report["every_policy_uses_audited_recording"] = True
    save_json(root / "physical_audit.json", report)
    canvas = Image.new("RGB", (1280, 500 * len(spec["environments"])), "#171b22")
    for slot, (name, index, rgb) in enumerate(samples):
        x = (slot % 4) * 320
        y = (slot // 4) * 500
        canvas.paste(Image.fromarray(rgb).resize((320, 240)), (x, y + 30))
        ImageDraw.Draw(canvas).text((x + 5, y + 8), f"{name}: capture {index}", fill="white")
    draw = ImageDraw.Draw(canvas)
    for n, (_name, poses) in enumerate(pose_sets.items()):
        origin = (640, n * 500 + 385)

        def xy(p, origin=origin):
            return (origin[0] + p[0] * 2, origin[1] - p[1] * 2)

        draw.line([xy(p) for p in poses], fill="#54d6db", width=3)
        draw.text(
            (20, n * 500 + 300),
            "Actual sampled flight XY (cyan); fixed across policies",
            fill="white",
        )
    canvas.save(root / "physical_preview.jpg")
    return report
