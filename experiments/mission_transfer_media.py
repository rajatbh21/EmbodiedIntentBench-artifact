"""Posthoc illustration of a condition selected before policy outcomes exist.

Freeze selection first; render only after the independent condition audit exists.
Run with PYTHONNOUSERSITE=1 /usr/bin/python3 for the system OpenCV H264 encoder.
Repository verification runs separately in the Python 3.12 project environment.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from itertools import pairwise
from pathlib import Path

FLIGHT, MISSION = "east_bay_day_b340", "rapid_survey"
PROTOCOL_SCHEMA = "mission_map_transfer_comparison_v3"
POLICIES = ("coverage_tail", "approxdet_eq5_adapter", "static:CFG_YOLO11S_TILED_640")
TITLES = (
    ("Coverage-tail", "Mission-aware proposed policy"),
    ("ApproxDet Eq. 5 adapter", "Accuracy under a latency limit"),
    ("Fixed tiled YOLO11s", "One configuration throughout"),
)
SIZE, FPS, SPEED = (1920, 960), 12, 4
PANEL_WIDTH, VIEW_HEIGHT, VIEW_TOP = 640, 480, 156
BG, FG, MUTED, GREEN = "#101923", "#edf3f9", "#acbed0", "#77ff85"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_key(rgb):
    return hashlib.sha256(
        str(rgb.shape).encode() + str(rgb.dtype).encode() + rgb.tobytes()
    ).hexdigest()


def selection_document():
    return dict(
        schema_version="mission_transfer_media_selection_v1",
        flight=FLIGHT,
        mission=MISSION,
        policies=list(POLICIES),
        rationale="Fixed low-battery, daylight, east-bay illustration; not outcome-selected",
        population_claim=False,
        selection_uses_outcomes=False,
        implementation_sha256=sha(Path(__file__)),
    )


def freeze_selection(root):
    root = Path(root).resolve()
    destination = root / "media_transfer_v1"
    path = destination / "selection.json"
    expected = dict(selection_document(), protocol_sha256=sha(root / "protocol.json"))
    if path.exists():
        saved = read(path)
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Changed existing media selection")
        return path
    if any(root.rglob("outcome.json")) or any(root.glob("validation_*.json")):
        raise ValueError("Illustrative condition must be selected before policy outcomes")
    protocol = read(root / "protocol.json")
    if (
        protocol["schema_version"] != PROTOCOL_SCHEMA
        or FLIGHT not in protocol["cases"]
        or MISSION not in protocol["missions"]
        or not set(POLICIES) <= set(protocol["policies"])
    ):
        raise ValueError("Preselected illustration does not belong to this comparison")
    destination.mkdir(exist_ok=False)
    path.write_text(
        json.dumps(
            # Python 3.10 is required for the system H264 encoder.
            dict(expected, selected_at_utc=datetime.now(timezone.utc).isoformat()),  # noqa: UP017
            indent=2,
        )
        + "\n"
    )
    return path


def completed_index(completions, time_s):
    """No selected output is visible before its own recorded completion."""
    if not math.isfinite(time_s):
        raise ValueError("Finite display time required")
    if any(not math.isfinite(t) for t in completions) or any(
        b < a for a, b in pairwise(completions)
    ):
        raise ValueError("Finite nondecreasing completion times required")
    return bisect.bisect_right(completions, time_s) - 1


def displayed_budget(events, index, time_s, *, final_time_s, flight_power_w, initial, capacity_j):
    """Follow the runtime's completed debits plus elapsed flight power, not host time."""
    charged_time = min(time_s, final_time_s)
    if index < 0:
        energy = charged_time * flight_power_w
    else:
        event = events[index]
        if event["completion_time_s"] > time_s:
            raise ValueError("Cannot reveal a future completed debit")
        energy = (
            event["cumulative_energy_j"]
            + (charged_time - event["completion_time_s"]) * flight_power_w
        )
    return energy, max(0, initial - energy / capacity_j)


def wrap_words(value, measure, width):
    """Wrap all text without hiding constraints; split long individual tokens."""
    if width <= 0:
        raise ValueError("Positive text width required")
    lines, line = [], ""
    for word in str(value).split():
        if line and measure(line + " " + word) > width:
            lines.append(line)
            line = ""
        if measure(word) > width:
            if line:
                lines.append(line)
                line = ""
            for char in word:
                if line and measure(line + char) > width:
                    lines.append(line)
                    line = ""
                line += char
        else:
            line = line + " " + word if line else word
    if line:
        lines.append(line)
    return lines or [""]


@lru_cache(None)
def font(size):
    from PIL import ImageFont

    return ImageFont.truetype(FONT, size)


def text(draw, xy, value, size=19, color=FG):
    draw.text(xy, value, fill=color, font=font(size))


def text_block(draw, xy, value, *, width, height, size=17, color=FG):
    for chosen in range(size, 10, -1):
        lines = wrap_words(value, lambda s, size=chosen: draw.textlength(s, font=font(size)), width)
        if len(lines) * (chosen + 4) <= height:
            for index, line in enumerate(lines):
                text(draw, (xy[0], xy[1] + index * (chosen + 4)), line, chosen, color)
            return
    raise ValueError("Display text cannot fit without omitting content")


def panel(rgb, mask):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    im = Image.fromarray(rgb).resize((PANEL_WIDTH, VIEW_HEIGHT))
    draw = ImageDraw.Draw(im)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    sx, sy = PANEL_WIDTH / rgb.shape[1], VIEW_HEIGHT / rgb.shape[0]
    for x, y, w, h, area in stats[1:count]:
        if area < 4:
            continue
        draw.rectangle(
            (
                max(0, x * sx - 2),
                max(0, y * sy - 2),
                min(PANEL_WIDTH - 1, (x + w) * sx + 2),
                min(VIEW_HEIGHT - 1, (y + h) * sy + 2),
            ),
            outline=GREEN,
            width=2,
        )
    return im


def verify_with_project_python(root):
    """Use the new study verifier/loaders without importing Python 3.12 code in 3.10."""
    repo = Path(__file__).resolve().parents[1]
    script = """
import json, sys
from pathlib import Path
from experiments.lake_missions.assets import sha
from experiments.mission_transfer_comparison_v3.protocol import verify
from experiments.mission_environment_transfer_v3.scenario import load_scenario
from experiments.mission_environment_transfer_v3.sensor import verify_gate
root, flight, mission = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
p = verify(root)
verify_gate(Path(p['cohort']))
s = load_scenario(root / flight / mission / 'scenario.json',
                  p['cases'][flight]['environment'], Path(p['assets']))
print(json.dumps(dict(complete=True, protocol_sha256=sha(root / 'protocol.json'),
    recording_identity=s.recording_identity, duration_s=s.duration_s,
    validator_sha256=sha(Path('experiments/mission_transfer_comparison_v3/validate.py')))))
"""
    env = dict(os.environ, PYTHONPATH=str(repo), OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [str(repo / ".venv/bin/python"), "-c", script, str(root), FLIGHT, MISSION],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    return json.loads(result.stdout)


def run(root):
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    root = Path(root).resolve()
    destination = root / "media_transfer_v1"
    selection_path = destination / "selection.json"
    if not selection_path.exists():
        raise ValueError("Freeze the predefined illustration before any policy outcomes")
    selection = read(selection_path)
    if any(selection.get(k) != v for k, v in selection_document().items()):
        raise ValueError("Preselected media implementation or condition changed")
    verified = verify_with_project_python(root)
    sources = {}

    def checked(path, expected=None):
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Changed media source: " + str(path))
        sources[str(path.resolve())] = digest
        return read(path)

    checked(selection_path)
    protocol = checked(root / "protocol.json", selection["protocol_sha256"])
    validation = checked(root / f"validation_{FLIGHT}.json")
    if (
        not validation["complete"]
        or validation["condition"] != FLIGHT
        or validation["protocol_sha256"] != verified["protocol_sha256"]
        or validation["validator_sha256"] != verified["validator_sha256"]
    ):
        raise ValueError("Independent matching condition audit required")
    name = f"summary_{FLIGHT}_15.json"
    summary = checked(root / name, validation["source_summaries_sha256"][name])
    source = Path(protocol["cohort"])
    source_flight = protocol["cases"][FLIGHT]["source_flight"]
    records_path = source / source_flight / "recording/recording.json"
    records = checked(records_path)
    folder = source / source_flight / "predictions"
    completion = checked(folder / "completion.json")
    manifest = checked(folder / "manifest.json", completion["manifest_sha256"])
    if (
        not completion["complete"]
        or manifest["protocol_sha256"] != sha(source / "protocol.json")
        or manifest["recording_sha256"] != sha(records_path)
        or manifest["recording_identity"] != verified["recording_identity"]
    ):
        raise ValueError("Incompatible physical recording or predictions")
    checked(source / "protocol.json", protocol["cohort_protocol_sha256"])
    scenario = checked(root / FLIGHT / MISSION / "scenario.json")
    intent = checked(root / FLIGHT / MISSION / "intent.json")
    duration = verified["duration_s"]
    runs = []
    for policy in POLICIES:
        matches = [r for r in summary["cases"] if r["mission"] == MISSION and r["policy"] == policy]
        if len(matches) != 1:
            raise ValueError("Exactly one audited policy condition required")
        row = matches[0]
        events = checked(root / row["path"] / "execution/events.json", row["events_sha256"])
        outcome = checked(root / row["path"] / "outcome.json", row["outcome_sha256"])
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
                raise ValueError("Changed selected neural output")
            sources[str(mask_path.resolve())] = reference["sha256"]
            with np.load(mask_path, allow_pickle=False) as data:
                mask = data["prediction_mask"]
            if mask.dtype != np.bool_ or mask.shape != rgb.shape[:2]:
                raise ValueError("Invalid selected segmentation output")
            views.append(panel(rgb, mask))
            total += len(event["emitted_reports"])
            reports.append(total)
        runs.append(
            dict(
                policy=policy,
                row=row,
                outcome=outcome,
                events=events,
                views=views,
                reports=reports,
                completion=[e["completion_time_s"] for e in events],
                end=result["final_time_s"],
            )
        )
    end = max(duration, *(r["end"] for r in runs))

    def render(time_s, finished=False):
        t = min(time_s, end)
        im = Image.new("RGB", SIZE, BG)
        draw = ImageDraw.Draw(im)
        text(draw, (20, 14), "DEPOT MAP TRANSFER / Matched physical flight", 29)
        text(draw, (1650, 21), f"SIM {t:05.1f}s | {SPEED}x", 21)
        phase = next(
            (p for p in intent["mission"]["phases"] if p["start_s"] <= t < p["end_s"]), None
        )
        label = (
            "Mission window complete"
            if phase is None
            else (
                f"{phase['phase_id']} | {phase['min_hits']} hits, "
                f"{phase['min_evidence_span_s']:g}s span, {phase['required_recall']:.0%} coverage"
            )
        )
        text(draw, (20, 59), f"{FLIGHT} | {MISSION} | {label}", 19, MUTED)
        for i, run in enumerate(runs):
            x = i * PANEL_WIDTH
            text(draw, (x + 16, 100), TITLES[i][0], 24)
            text(draw, (x + 16, 130), TITLES[i][1], 16, MUTED)
            index = completed_index(run["completion"], t)
            if index >= 0:
                event = run["events"][index]
                im.paste(run["views"][index], (x, VIEW_TOP))
                text_block(
                    draw,
                    (x + 16, 646),
                    event["execution"]["config_id"].removeprefix("CFG_"),
                    width=608,
                    height=27,
                    size=20,
                    color=GREEN,
                )
                text(
                    draw,
                    (x + 16, 678),
                    f"Capture {event['capture_time_s']:.0f}s | "
                    f"Ready {event['completion_time_s']:.2f}s | "
                    f"Age {t - event['capture_time_s']:.1f}s",
                    17,
                )
                text(
                    draw,
                    (x + 16, 705),
                    f"Completed calls {index + 1} | Predicted reports {run['reports'][index]}",
                    17,
                    MUTED,
                )
            else:
                text_block(
                    draw,
                    (x + 60, 355),
                    "Waiting for the first completed prediction",
                    width=520,
                    height=80,
                    size=24,
                    color=MUTED,
                )
            energy, battery = displayed_budget(
                run["events"],
                index,
                t,
                final_time_s=run["end"],
                flight_power_w=scenario["mission"]["flight_power_w"],
                initial=scenario["mission"]["initial_battery_frac"],
                capacity_j=scenario["mission"]["battery_capacity_wh"] * 3600,
            )
            reserve = intent["limits"]["min_final_battery_frac"]
            text(
                draw,
                (x + 16, 733),
                f"Declared debit {energy / 1000:.2f}kJ | Battery {battery:.2%} (min {reserve:.0%})",
                16,
                MUTED,
            )
            if finished:
                outcome = run["outcome"]
                status = "PASS" if outcome["mission_success"] else "FAIL"
                nph = sum(p["success"] for p in outcome["phases"])
                text(
                    draw,
                    (x + 16, 773),
                    f"FINAL {status} | Phases {nph}/{len(outcome['phases'])} | "
                    f"Report recall {outcome['reporting']['recall_by_deadline']:.2f}",
                    18,
                )
                failures = ", ".join(outcome["failed_constraints"]) or "None"
                text_block(
                    draw,
                    (x + 16, 805),
                    "Failed constraints: " + failures,
                    width=608,
                    height=87,
                    size=16,
                    color=MUTED,
                )
        text(
            draw,
            (20, 914),
            "Boxes are completed selected predictions on their own capture; "
            "no ground-truth boxes. Sampled replay, not live onboard inference.",
            17,
            MUTED,
        )
        text(
            draw,
            (20, 938),
            "One preselected illustration; the complete paired grid, not this video, "
            "supports the comparison. Costs are configured, not measured energy.",
            14,
            MUTED,
        )
        return im

    base = destination / f"{FLIGHT}_{MISSION}"
    video, still, provenance_path = [
        Path(str(base) + suffix) for suffix in (".mp4", "_summary.png", "_provenance.json")
    ]
    if any(path.exists() for path in (video, still, provenance_path)):
        raise FileExistsError("Preserve previous media render and provenance")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"avc1"), FPS, SIZE)
    if not writer.isOpened():
        raise RuntimeError("H264 encoder unavailable; use the system Python/OpenCV runtime")
    try:
        for index in range(int(end / SPEED * FPS) + 1):
            writer.write(cv2.cvtColor(np.asarray(render(index / FPS * SPEED)), cv2.COLOR_RGB2BGR))
        last = render(end, True)
        for _ in range(3 * FPS):
            writer.write(cv2.cvtColor(np.asarray(last), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    last.save(still)
    provenance_path.write_text(
        json.dumps(
            dict(
                schema_version="mission_transfer_media_v1",
                flight=FLIGHT,
                mission=MISSION,
                policies=list(POLICIES),
                selection_sha256=sha(selection_path),
                presentation=(
                    "Posthoc illustration of one condition preselected before policy outcomes"
                ),
                protocol_scope=protocol["scope"],
                population_claim=False,
                gt_boxes_used=False,
                show_prediction_only_after_completion=True,
                draw_mask_on_its_original_capture_only=True,
                displayed_component_min_area_px=4,
                configured_costs_not_energy_measurements=True,
                playback_speed=SPEED,
                fps=FPS,
                dimensions_px=SIZE,
                source_sha256=sources,
                independent_verification=verified,
                video_sha256=sha(video),
                summary_sha256=sha(still),
                implementation_sha256=sha(Path(__file__)),
            ),
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(dict(video=str(video), summary=str(still), provenance=str(provenance_path))))
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "render"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(freeze_selection(args.output)) if args.mode == "freeze" else run(args.output)
