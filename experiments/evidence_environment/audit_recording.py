"""Validate manifest-selected frames; never count raw transport duplicate files."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.scenario import load_scenario


def audit(scenario_path, recording):
    scenario = load_scenario(scenario_path)
    manifest = json.loads((recording / "recording.json").read_text())
    backend = ReplayBackend(scenario, recording)
    rows = []
    try:
        for index, entry in enumerate(manifest["frames"]):
            backend.advance_to(entry["timestamp_ns"])
            frame = backend.capture()
            rows.append(
                dict(
                    file=entry["file"],
                    visible_target_pixels={
                        str(label): int(np.count_nonzero(frame.object_ids == label))
                        for label in scenario.targets
                    },
                )
            )
            if index == len(manifest["frames"]) // 2:
                Image.fromarray(frame.rgb).save(recording / "rgb_midpoint.png")
    finally:
        backend.close()
    report = dict(
        frames=len(rows),
        target_visible_frames={
            str(label): sum(r["visible_target_pixels"][str(label)] > 0 for r in rows)
            for label in scenario.targets
        },
        max_visible_pixels={
            str(label): max(r["visible_target_pixels"][str(label)] for r in rows)
            for label in scenario.targets
        },
        scope="simulation GT visibility diagnostic, not perception success",
        rows=rows,
    )
    (recording / "visibility_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.scenario, args.recording)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
