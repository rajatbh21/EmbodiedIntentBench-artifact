"""Create a sampled day/dusk flythrough from actual recorded Gazebo frames.

Run with PYTHONNOUSERSITE=1 /usr/bin/python3 (system OpenCV H264 encoder).
No new image content, frame interpolation, or invented inference is generated.
"""

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from experiments.lake_media_english import english_missions

ROOT = Path(__file__).resolve().parents[1]
FPS, SPEED = 12, 8
BG, FG, MUTED, CYAN = "#101923", "#edf3f9", "#a6b7c9", "#52e4df"


@lru_cache(None)
def font(size):
    return ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size, index=1
    )


def text(draw, x, y, value, size=20, color=FG):
    draw.text((x, y), value, font=font(size), fill=color)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def render(root):
    output = root / "media_en"
    output.mkdir(exist_ok=True)
    scene = json.loads((root / "scenario.json").read_text())
    missions = english_missions(json.loads((root / "missions.json").read_text())["missions"])
    manifests, frames, sources = {}, {}, {}
    for env in ("day", "dusk_haze"):
        folder = root / env / "recording"
        sources[str(folder / "recording.json")] = sha(folder / "recording.json")
        manifests[env] = json.loads((folder / "recording.json").read_text())
        frames[env] = []
        for row in manifests[env]["frames"]:
            p = folder / row["file"]
            assert sha(p) == row["sha256"]
            with np.load(p, allow_pickle=False) as data:
                frames[env].append(Image.fromarray(data["rgb"]).resize((640, 480)))
    assert len(frames["day"]) == len(frames["dusk_haze"])
    # Both environmental variants must preserve the same capture schedule.
    assert [f["timestamp_ns"] for f in manifests["day"]["frames"]] == [
        f["timestamp_ns"] for f in manifests["dusk_haze"]["frames"]
    ]
    times = [
        (f["timestamp_ns"] / 1e9 - scene["mission"]["warmup_s"]) for f in manifests["day"]["frames"]
    ]
    duration = scene["mission"]["duration_s"]

    def frame(t):
        index = max(i for i, time_s in enumerate(times) if time_s <= t)
        im = Image.new("RGB", (1280, 800), BG)
        d = ImageDraw.Draw(im)
        text(d, 24, 12, "LAKESIDE / Recorded Gazebo flight", 29)
        text(d, 1000, 17, f"SIM {t:05.1f} s", 25, CYAN)
        text(d, 24, 59, "Day: terrain, buildings, trees and occlusion", 20)
        text(d, 664, 59, "Dusk: low light and light haze", 20)
        im.paste(frames["day"][index], (0, 100))
        im.paste(frames["dusk_haze"][index], (640, 100))
        for i, env in enumerate(("day", "dusk_haze")):
            xyz = manifests[env]["frames"][index]["body_pose"]["position_m"]
            text(
                d,
                24 + i * 640,
                587,
                f"Capture {times[index]:.0f}s  |  "
                f"Position ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) m",
                18,
                MUTED,
            )
        for i, mission in enumerate(missions):
            phase = next((p for p in mission["phases"] if p["start_s"] <= t < p["end_s"]), None)
            name = phase["instruction"] if phase else "Transit / waiting for next stage"
            text(
                d,
                24,
                628 + i * 36,
                f"{i + 1}. {mission['title']}  →  {name}",
                20,
                CYAN if phase else MUTED,
            )
        text(
            d,
            24,
            755,
            "2s sensor samples | 8x playback | Frames held between samples | Not continuous video",
            19,
            MUTED,
        )
        return im

    def card(title, lines):
        im = Image.new("RGB", (1280, 800), BG)
        d = ImageDraw.Draw(im)
        text(d, 48, 80, title, 36, CYAN)
        for i, line in enumerate(lines):
            text(d, 48, 200 + 80 * i, line, 27)
        return im

    intro = card(
        "A textured lakeside simulation environment",
        [
            "Public Gazebo Harmonic terrain, vegetation, house and water assets.",
            "Day and dusk observations along the same nominal flight path.",
            "Missions: sector search, perimeter inspection and battery-limited revisits.",
            "The active stage of each mission is shown below the camera views.",
            "This video shows recorded observations, without model predictions.",
        ],
    )
    outro = card(
        "One environment, different mission requirements",
        [
            "Each stage evaluates deadlines, repeated evidence and false alarms.",
            "Revisit verification requires fresh evidence; battery reserves also apply.",
            "All policies use the same recording within each environment.",
            "People are static meshes. Walking and crowd evacuation are not modeled.",
            "Scenery: OpenRobotics / Gazebo Fuel / CC BY 4.0",
        ],
    )
    path = output / "lakeside_flight.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), FPS, (1280, 800))
    if not writer.isOpened():
        raise RuntimeError("H264 encoder unavailable")
    count = 0
    timeline = []

    def emit(im):
        nonlocal count
        writer.write(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
        count += 1

    try:
        for _ in range(3 * FPS):
            emit(intro)
        for i in range(math.ceil(duration / SPEED * FPS)):
            t = i * SPEED / FPS
            emit(frame(t))
            timeline.append(dict(video_frame=count - 1, simulation_time_s=t))
        for _ in range(4 * FPS):
            emit(outro)
    finally:
        writer.release()
    frame(40).save(output / "comparison_preview.jpg", quality=94)
    intro.save(output / "intro.jpg", quality=92)
    cap = cv2.VideoCapture(str(path))
    decoded = 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    assert decoded == count
    data = dict(
        video_sha256=sha(path),
        script_sha256=sha(Path(__file__)),
        language="en",
        labels_sha256=sha(Path(__file__).with_name("lake_media_english.py")),
        sources=sources,
        fps=FPS,
        speed=SPEED,
        duration_s=count / FPS,
        frames=count,
        decoded_frames=decoded,
        frame_interpolation=False,
        timeline=timeline,
    )
    (output / "provenance.json").write_text(json.dumps(data, indent=2) + "\n")
    print(path, count / FPS, "seconds", decoded, "decoded frames")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    render(parser.parse_args().root)
