"""Evaluator-only geometry/visibility audit and shareable mission map.

Actual odometry is compared with planned segments; geometric coverage is not
inferred from the waypoints. No output from this tool is a router observation.
"""

from __future__ import annotations

import argparse
import math
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.model_probe import target_visibility
from experiments.v4_benchmark.scenario import load_scenario


def segment_distance(point, start, end):
    p, a, b = map(lambda v: np.asarray(v, dtype=float), (point, start, end))
    delta = b - a
    norm = float(delta @ delta)
    fraction = min(1, max(0, float((p - a) @ delta) / norm)) if norm else 0
    return float(np.linalg.norm(p - (a + fraction * delta)))


def audit(scenario_path: Path, recording: Path, output: Path) -> dict:
    scenario = load_scenario(scenario_path)
    backend = ReplayBackend(scenario, recording)
    output.mkdir(parents=True, exist_ok=False)
    captures, poses, rgbs = [], [], []
    times = list(
        range(scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns)
    )
    preview_ids = {round(i * (len(times) - 1) / 5) for i in range(6)}
    try:
        for index, timestamp in enumerate(times):
            backend.advance_to(timestamp)
            frame = backend.capture()
            poses.append(frame.body_pose["position_m"])
            captures.append(
                {
                    "observation_id": index,
                    "mission_time_s": (timestamp - scenario.schedule.epoch_ns) / 1e9,
                    "targets": target_visibility(frame.object_ids, scenario.targets),
                }
            )
            if index in preview_ids:
                rgbs.append((index, frame.rgb.copy()))
    finally:
        backend.close()
    goals = scenario.world["waypoints"]
    segments = [(a["position_m"], b["position_m"]) for a, b in pairwise(goals)]
    if not segments:
        segments = [(goals[0]["position_m"], goals[0]["position_m"])]
    errors = [min(segment_distance(p, a, b) for a, b in segments) for p in poses]
    goal_errors = []
    for index, goal in enumerate(goals):
        next_time = goals[index + 1]["start_s"] if index + 1 < len(goals) else scenario.duration_s
        samples = [
            p
            for p, c in zip(poses, captures, strict=True)
            if goal["start_s"] <= c["mission_time_s"] < next_time
        ]
        goal_errors.append(
            {
                "goal_index": index,
                "start_s": goal["start_s"],
                "last_sample_error_m": math.dist(samples[-1], goal["position_m"])
                if samples
                else None,
            }
        )
    targets = []
    for target in scenario.targets.values():
        rows = [
            c
            for c in captures
            if any(t["target_id"] == target and t["visible_pixels"] > 0 for t in c["targets"])
        ]
        areas = [
            t["visible_pixels"] for c in captures for t in c["targets"] if t["target_id"] == target
        ]
        targets.append(
            {
                "target_id": target,
                "visible_capture_count": len(rows),
                "first_visible_s": rows[0]["mission_time_s"] if rows else None,
                "last_visible_s": rows[-1]["mission_time_s"] if rows else None,
                "max_visible_pixels": max(areas),
            }
        )
    report = {
        "schema_version": "0.1",
        "gt_aware": True,
        "scope": "geometry and visibility diagnostic, not mission scoring",
        "recording_identity": scenario.recording_identity,
        "capture_count": len(captures),
        "nearest_planned_segment_error_p95_m": float(np.percentile(errors, 95)),
        "nearest_planned_segment_error_max_m": max(errors),
        "sampled_path_length_m": sum(math.dist(a, b) for a, b in pairwise(poses)),
        "sampled_speed_max_mps": max(
            (math.dist(a, b) / scenario.observation_interval_s for a, b in pairwise(poses)),
            default=0.0,
        ),
        "altitude_range_m": [min(p[2] for p in poses), max(p[2] for p in poses)],
        "goal_errors": goal_errors,
        "targets": targets,
        "limitations": (
            "Nearest-segment error does not prove area coverage; speed is sampled, "
            "not a continuous bound. No collision/contact validation."
        ),
    }
    save_json(output / "audit.json", report)
    save_json(output / "visibility.json", captures)
    render_map(scenario.world, poses, output / "map.png")
    sheet = Image.new("RGB", (640, 3 * 270), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, (index, rgb) in enumerate(rgbs):
        x, y = (slot % 2) * 320, (slot // 2) * 270
        draw.text(
            (x + 5, y + 5),
            f"Capture {index}, mission {index * scenario.observation_interval_s:.1f}s",
            fill="black",
        )
        sheet.paste(Image.fromarray(rgb).resize((320, 240)), (x, y + 25))
    sheet.save(output / "rgb_samples.png")
    return report


def render_map(world: dict, poses: list, output: Path) -> None:
    image = Image.new("RGB", (1000, 900), "#f6f5ef")
    draw = ImageDraw.Draw(image)

    def xy(p):
        return (500 + 18 * p[0], 450 - 18 * p[1])

    draw.rectangle([xy([-24, 20]), xy([24, -20])], fill="#a7b08b", outline="#333333", width=2)
    for obj in world["objects"]:
        if obj["is_target"]:
            continue
        x, y, _z = obj["position_m"]
        w, d, _h = obj["size_m"]
        bounds = [xy([x - w / 2, y + d / 2]), xy([x + w / 2, y - d / 2])]
        color = tuple(round(v * 255) for v in obj["color_rgba"][:3])
        if obj.get("shape") in ("cylinder", "ellipsoid"):
            draw.ellipse(bounds, fill=color)
        else:
            draw.rectangle(bounds, fill=color)
    planned = [xy(g["position_m"]) for g in world["waypoints"]]
    draw.line(planned, fill="#e59400", width=5)
    draw.line([xy(p) for p in poses], fill="#136cdb", width=2)
    for index, p in enumerate(planned):
        draw.text((p[0] + 5, p[1] + 5), str(index), fill="black")
    for obj in world["objects"]:
        if obj["is_target"]:
            x, y = xy(obj["position_m"])
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="#e3242b")
            draw.text((x + 7, y - 12), obj["object_id"], fill="#9b0010")
    draw.text((45, 25), "PARK SEARCH | 48 x 40 m | north is +Y | metres", fill="black")
    draw.text(
        (45, 45),
        "Orange: planned route | Blue: recorded flight | Red: evaluator-only target locations",
        fill="black",
    )
    draw.text(
        (45, 840),
        "Procedural development map; not a real site, flight approval or photoreal simulation.",
        fill="black",
    )
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("scenario", "recording", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(audit(args.scenario, args.recording, args.output))


if __name__ == "__main__":
    main()
