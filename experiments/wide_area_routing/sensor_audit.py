"""Evaluator-only sensor diagnostics for the declared wide-area pilot windows.

The observation rule is geometry-only: retain an original capture if its mission
time lies within eight seconds of an unsnapped nominal sector pass. This module
does not read neural predictions or change the window after inspecting labels.
Perfect target-union masks are an optimistic, zero-cost sensor diagnostic, not a
deployable router, a tracker feasibility certificate, or measured model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.v4_benchmark.runner import EvaluationFrame


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_indices(times_s, nominal_pass_s, half_width_s=8.0):
    """Inclusive raw-time interval; never round the nominal pass to a capture."""
    if not math.isfinite(nominal_pass_s) or not math.isfinite(half_width_s) or half_width_s <= 0:
        raise ValueError("Finite pass time and positive window half width required")
    if any(not math.isfinite(t) or t < 0 for t in times_s) or any(
        right <= left for left, right in zip(times_s, times_s[1:])
    ):
        raise ValueError("Capture times must be finite, nonnegative, and strictly increasing")
    return [i for i, time_s in enumerate(times_s) if abs(time_s - nominal_pass_s) <= half_width_s]


def enough_evidence(times_s, min_hits=2, min_span_s=2.0):
    times = sorted(set(times_s))
    return len(times) >= min_hits and times[-1] - times[0] >= min_span_s


def perfect_union_score(object_ids, targets, index, time_s, evaluator):
    """Use the unchanged connected-component IoU matcher on an ideal union mask.

    Touching target masks may form one component and cannot count as two matched
    targets. Small components remain subject to the canonical four-pixel filter.
    """
    if object_ids.ndim != 2 or not np.issubdtype(object_ids.dtype, np.integer):
        raise ValueError("A two-dimensional integer instance image is required")
    present = set(map(int, np.unique(object_ids))) - {0}
    if not present <= set(targets):
        raise ValueError("Unknown object label in evaluator image")
    union = np.isin(object_ids, list(targets))
    observation = EvaluationFrame(
        index,
        time_s,
        union.astype(np.uint8),
        object_ids,
        tuple(sorted(targets[label] for label in present)),
        {"instance_object_ids": {str(label): name for label, name in targets.items()}},
    )
    return evaluator.update(observation, union)


def audit_case(folder, *, matching_iou_threshold=0.3):
    folder = Path(folder)
    scenario, design, recording = (
        read(folder / "scenario.json"),
        read(folder / "mission_design.json"),
        read(folder / "recording/recording.json"),
    )
    if not recording["provenance"]["full_declared_mission"] or not recording["provenance"]["physical_route_passed"]:
        raise ValueError("A complete physically validated recording is required")
    frames, schedule = recording["frames"], recording["capture_schedule"]
    expected = list(range(schedule["epoch_ns"], schedule["end_ns"], schedule["interval_ns"]))
    if [frame["timestamp_ns"] for frame in frames] != expected:
        raise ValueError("Recording does not cover its exact declared capture clock")
    times = [(frame["timestamp_ns"] - schedule["epoch_ns"]) / 1e9 for frame in frames]
    targets = {obj["label"]: obj["object_id"] for obj in scenario["world"]["objects"]}
    if len(targets) != len(scenario["world"]["objects"]) or len(set(targets.values())) != len(targets):
        raise ValueError("Unique target labels and identities are required")
    evaluator = MissionEvaluator(matching_iou_threshold, len(targets), schedule["interval_ns"] / 1e9)
    sectors, seen_indices, seen_targets, frame_hashes = [], set(), set(), {}
    for sector in design["sectors"]:
        target_ids = sector["evaluator_target_ids"]
        if not target_ids or len(set(target_ids)) != len(target_ids) or not set(target_ids) <= set(targets.values()):
            raise ValueError("Each sector needs unique known evaluator targets")
        if seen_targets.intersection(target_ids):
            raise ValueError("This diagnostic requires disjoint sector target groups")
        seen_targets.update(target_ids)
        indices = selected_indices(times, sector["nominal_pass_time_s"])
        if not indices or seen_indices.intersection(indices):
            raise ValueError("Each declared sector must have a nonempty disjoint capture window")
        seen_indices.update(indices)
        visibility = {
            target: dict(nonzero_capture_times_s=[], at_least_4px_capture_times_s=[],
                         perfect_union_matched_times_s=[], peak_pixels=0)
            for target in target_ids
        }
        scores = []
        for index in indices:
            frame = frames[index]
            path = (folder / "recording" / frame["file"]).resolve()
            if path.parent != (folder / "recording").resolve() or sha(path) != frame["sha256"]:
                raise ValueError("Changed or escaped acquisition frame")
            if frame["body_pose"]["timestamp_ns"] != frame["timestamp_ns"]:
                raise ValueError("Capture and body pose timestamps differ")
            frame_hashes[frame["file"]] = frame["sha256"]
            with np.load(path, allow_pickle=False) as arrays:
                ids = arrays["object_ids"]
                score = perfect_union_score(ids, targets, index, times[index], evaluator)
                unique, counts = np.unique(ids, return_counts=True)
                pixels = {targets[int(label)]: int(count) for label, count in zip(unique, counts)
                          if int(label) in targets}
            for target, row in visibility.items():
                count = pixels.get(target, 0)
                row["peak_pixels"] = max(row["peak_pixels"], count)
                if count > 0:
                    row["nonzero_capture_times_s"].append(times[index])
                if count >= 4:
                    row["at_least_4px_capture_times_s"].append(times[index])
                if target in score.matched_target_ids:
                    row["perfect_union_matched_times_s"].append(times[index])
            scores.append(score.to_dict())
        fractions = {}
        for key in ("nonzero", "at_least_4px", "perfect_union"):
            field = key + ("_matched_times_s" if key == "perfect_union" else "_capture_times_s")
            for row in visibility.values():
                row[key + "_two_hit_2s"] = enough_evidence(row[field])
            fractions[key] = sum(row[key + "_two_hit_2s"] for row in visibility.values()) / len(target_ids)
        sectors.append(dict(
            sector_id=sector["sector_id"], nominal_pass_time_s=sector["nominal_pass_time_s"],
            raw_interval_s=[sector["nominal_pass_time_s"] - 8, sector["nominal_pass_time_s"] + 8],
            capture_indices=indices, capture_times_s=[times[i] for i in indices],
            evaluator_targets=visibility, two_hit_2s_target_fractions=fractions,
            perfect_union_false_components=sum(row["false_positive_components"] for row in scores),
            perfect_union_frame_scores=scores,
        ))
    if seen_targets != set(targets.values()):
        raise ValueError("Sector groups must cover every declared target exactly once")
    return dict(
        recording_identity=recording["recording_identity"], target_count=len(targets),
        capture_count=len(frames), selected_capture_count=len(seen_indices),
        selected_capture_indices=sorted(seen_indices), sectors=sectors,
        all_targets_two_hit_2s={
            key: all(sector["two_hit_2s_target_fractions"][key] == 1 for sector in sectors)
            for key in ("nonzero", "at_least_4px", "perfect_union")
        },
        perfect_union_quality=evaluator.quality_details(scenario["duration_s"]),
        input_sha256={name: sha(folder / name) for name in (
            "scenario.json", "mission_design.json", "recording/recording.json")},
        selected_frame_sha256=frame_hashes,
    )


def audit(cohort, output):
    from experiments.wide_area_missions.cohort import verify

    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve the earlier sensor inspection; choose a new audit path")
    protocol = verify(cohort)
    report = dict(
        schema_version="wide_area_raw_window_sensor_audit_v1",
        inspected_utc=datetime.now(timezone.utc).isoformat(),
        cohort=str(cohort), cohort_protocol_sha256=sha(cohort / "protocol.json"),
        audit_source_sha256=sha(__file__),
        observation_rule="abs(original capture mission time - unsnapped nominal sector pass) <= 8 seconds",
        min_hits=2, min_evidence_span_s=2.0, matching_iou_threshold=0.3,
        evaluator_only=True, sensor_ground_truth_inspected=True,
        neural_predictions_read=False, neural_outcomes_used_for_window_or_threshold_changes=False,
        chronology="Geometry-defined windows and 2-hit/2-second requirements were chosen before this diagnostic. "
                   "Sensor visibility is known after the recorded inspection time; this is not a blind sensor holdout.",
        interpretation="Nonzero visibility is the canonical visible-target diagnostic. Four pixels is an additional "
                       "sensor diagnostic. Perfect target-union masks use the unchanged component IoU matcher at zero "
                       "service cost. These do not certify neural perception, tracker association, paid scheduling, "
                       "report correctness, battery feasibility, or full mission success.",
        cases={name: audit_case(cohort / name) for name in protocol["cases"]},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.cohort, args.output)
    print(json.dumps({name: {key: case[key] for key in (
        "selected_capture_count", "target_count", "all_targets_two_hit_2s")}
        for name, case in report["cases"].items()}, indent=2))


if __name__ == "__main__":
    main()
