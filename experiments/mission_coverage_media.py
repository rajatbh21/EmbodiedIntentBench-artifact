"""Render a causal side-by-side replay of two predeclared policies after validation.

Presentation only: each box comes from that policy's completed prediction on its
own capture. Never draw an old mask on a newer frame or reveal it before completion.
Run with PYTHONNOUSERSITE=1 /usr/bin/python3 for the system OpenCV H264 encoder.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG, FG, MUTED, GREEN = "#101923", "#edf3f9", "#acbed0", "#77ff85"
POLICIES = ("coverage_tail", "coverage_no_battery")
SIZE, FPS, SPEED = (1280, 800), 12, 4


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rgb_key(rgb):
    return hashlib.sha256(
        str(rgb.shape).encode() + str(rgb.dtype).encode() + rgb.tobytes()
    ).hexdigest()


def text(draw, xy, value, size=19, color=FG):
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    draw.text(xy, value, fill=color, font=font)


def panel(rgb, mask):
    im = Image.fromarray(rgb).resize((640, 480))
    draw = ImageDraw.Draw(im)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    sx, sy = 640 / rgb.shape[1], 480 / rgb.shape[0]
    for x, y, w, h, area in stats[1:count]:
        if area < 4:
            continue
        bounds = (
            max(0, x * sx - 2),
            max(0, y * sy - 2),
            min(639, (x + w) * sx + 2),
            min(479, (y + h) * sy + 2),
        )
        draw.rectangle(bounds, outline=GREEN, width=2)
    return im


def run(root, flight, mission):
    sources = {}

    def checked(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Changed media source: " + str(path))
        sources[str(path.resolve())] = digest
        return read(path)

    protocol = checked(root / "protocol.json")
    validation = checked(root / f"validation_{flight}.json")
    if not validation["complete"] or validation["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Independent condition audit required")
    name = f"summary_{flight}_15.json"
    summary = checked(root / name, validation["source_summaries_sha256"][name])
    source = Path(protocol["cohort"])
    source_flight = protocol["cases"][flight]["source_flight"]
    records = checked(source / source_flight / "recording/recording.json")
    folder = source / source_flight / "predictions"
    completion = checked(folder / "completion.json")
    manifest = checked(folder / "manifest.json", completion["manifest_sha256"])
    if manifest["protocol_sha256"] != sha(source / "protocol.json") or manifest[
        "recording_sha256"
    ] != sha(source / source_flight / "recording/recording.json"):
        raise ValueError("Incompatible source predictions")
    scenarios = checked(root / flight / mission / "scenario.json")
    intent = checked(root / flight / mission / "intent.json")
    duration = scenarios["mission"]["duration_s"]
    runs = []
    for policy in POLICIES:
        row = next(r for r in summary["cases"] if r["mission"] == mission and r["policy"] == policy)
        events = checked(root / row["path"] / "execution/events.json", row["events_sha256"])
        checked(root / row["path"] / "outcome.json", row["outcome_sha256"])
        result = checked(root / row["path"] / "execution/results.json", row["results_sha256"])
        views, reports, total = [], [], 0
        for event in events:
            frame = records["frames"][event["observation_id"]]
            path = source / source_flight / "recording" / frame["file"]
            if (
                sha(path) != frame["sha256"]
                or frame["timestamp_ns"] != event["capture_timestamp_ns"]
            ):
                raise ValueError("Wrong physical capture")
            sources[str(path.resolve())] = frame["sha256"]
            with np.load(path, allow_pickle=False) as data:
                rgb = data["rgb"]
            config = event["execution"]["config_id"]
            reference = manifest["models"][config]["predictions"][rgb_key(rgb)]
            mask_path = folder / reference["file"]
            if sha(mask_path) != reference["sha256"]:
                raise ValueError("Changed neural mask")
            sources[str(mask_path.resolve())] = reference["sha256"]
            with np.load(mask_path, allow_pickle=False) as data:
                mask = data["prediction_mask"]
            if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
                raise ValueError("Invalid segmentation mask")
            views.append(panel(rgb, mask))
            total += len(event["emitted_reports"])
            reports.append(total)
        runs.append(
            dict(
                policy=policy,
                row=row,
                events=events,
                views=views,
                reports=reports,
                completion=[e["completion_time_s"] for e in events],
                end=result["final_time_s"],
            )
        )
    end = max(duration, *(r["end"] for r in runs))

    def render(t, finished=False):
        im = Image.new("RGB", SIZE, BG)
        draw = ImageDraw.Draw(im)
        text(draw, (20, 12), "FIXED FLIGHT / Causal model-routing replay", 27)
        text(draw, (975, 18), f"SIM {min(t, end):05.1f} s  |  {SPEED}x", 21)
        phase = next(
            (p["phase_id"] for p in intent["mission"]["phases"] if p["start_s"] <= t < p["end_s"]),
            "mission complete",
        )
        text(
            draw,
            (20, 55),
            f"{flight}  |  {mission}  |  {phase}",
            19,
            MUTED,
        )
        for i, run in enumerate(runs):
            x = i * 640
            text(draw, (x + 16, 91), run["policy"].replace("static:CFG_", "fixed: "), 19)
            index = bisect.bisect_right(run["completion"], t) - 1
            if index < 0:
                text(draw, (x + 100, 315), "Waiting for first completed prediction", 19, MUTED)
                continue
            event = run["events"][index]
            im.paste(run["views"][index], (x, 126))
            config = event["execution"]["config_id"].removeprefix("CFG_")
            text(draw, (x + 16, 620), config, 19, GREEN)
            text(
                draw,
                (x + 16, 650),
                f"Captured {event['capture_time_s']:.0f}s  |  "
                f"Ready {event['completion_time_s']:.2f}s  |  "
                f"Age {t - event['capture_time_s']:.1f}s",
                17,
            )
            text(
                draw,
                (x + 16, 677),
                f"Completed calls {index + 1}  |  Predicted reports {run['reports'][index]}",
                17,
                MUTED,
            )
            if finished:
                row = run["row"]
                text(
                    draw,
                    (x + 16, 706),
                    f"FINAL: success={row['full_success']}  "
                    f"phases={row['phases_succeeded']}/4  recall={row['report_recall']:.2f}",
                    17,
                )
        text(
            draw,
            (20, 759),
            "Boxes: completed model predictions on their own capture; "
            "not ground truth. Sampled replay, not live onboard inference.",
            17,
            MUTED,
        )
        return im

    destination = root / "media"
    destination.mkdir(exist_ok=True)
    path = destination / f"{flight}_{mission}.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError("H264 encoder unavailable; use system Python/OpenCV")
    try:
        for i in range(int(end / SPEED * FPS) + 1):
            writer.write(cv2.cvtColor(np.asarray(render(i / FPS * SPEED)), cv2.COLOR_RGB2BGR))
        last = render(end, True)
        for _ in range(3 * FPS):
            writer.write(cv2.cvtColor(np.asarray(last), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    last.save(destination / f"{flight}_{mission}_summary.png")
    provenance = dict(
        schema_version="mission_coverage_media_v1",
        flight=flight,
        mission=mission,
        policies=POLICIES,
        selection="fixed presentation pair; no ranking on this cohort",
        protocol_scope=protocol["scope"],
        gt_boxes_used=False,
        show_prediction_only_after_completion=True,
        playback_speed=SPEED,
        source_sha256=sources,
        video_sha256=sha(path),
        implementation_sha256=sha(Path(__file__)),
    )
    (destination / f"{flight}_{mission}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--mission", choices=("rapid_survey", "verified_inspection"), required=True)
    args = parser.parse_args()
    run(args.output, args.flight, args.mission)
