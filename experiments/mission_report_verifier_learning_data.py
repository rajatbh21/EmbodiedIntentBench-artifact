"""Reconstruct causal candidate prefixes from older selected-mask traces only.

Ground truth is confined to label construction. Training features are generated
before component matching and are checked by the separate public-only model API.
"""

from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import numpy as np

from experiments.lake_tracking import (
    StaticReportTracker,
    SurfaceProjector,
    TrackingConfig,
    detections,
)
from experiments.lake_tracking_evaluation import component_matches
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_report_verifier_learning import digest, feature_vector, require
from experiments.mission_runtime_profile import read, save, sha
from experiments.v4_benchmark.runner import EvaluationFrame

REPO = Path(__file__).resolve().parents[1]
TRAIN = REPO / "results/mission_routing_v5_exploratory_20261017"
EVALUATION = REPO / "results/mission_policy_selection_v1_20261018"


def source_inventory():
    runs, inputs, lineages = [], {}, {"train": [], "held_out_development": []}
    for split, folder in (("train", TRAIN), ("held_out_development", EVALUATION)):
        protocol = read(folder / "protocol.json")
        inputs[str(folder / "protocol.json")] = sha(folder / "protocol.json")
        if (folder / "completion.json").exists():
            inputs[str(folder / "completion.json")] = sha(folder / "completion.json")
        source = Path(protocol["source"]) if split == "train" else folder
        for flight, case in protocol["cases"].items():
            lineages[split].append(dict(flight=flight, **case["lineage"]))
        for path in sorted((folder / "runs").glob("*/power_15/*/*/execution/results.json")):
            relative = path.relative_to(folder)
            flight, mission = relative.parts[1], relative.parts[3]
            scenario = source / flight / mission / "scenario.json"
            intent = source / flight / mission / "intent.json"
            events_path = path.with_name("events.json")
            events = read(events_path)
            run = dict(
                split=split,
                flight=flight,
                mission=mission,
                output=str(path.parent.parent),
                scenario=str(scenario),
                intent=str(intent),
                assets=protocol["assets"],
                lineage=protocol["cases"][flight]["lineage"],
            )
            runs.append(run)
            paths = [path, events_path, path.parent.parent / "protocol.json", scenario, intent]
            paths += [path.with_name(f"observation_{e['observation_id']:05d}.npz") for e in events]
            for item in paths:
                require(item.is_file(), "Missing historical selected evidence: " + str(item))
                inputs[str(item.resolve())] = sha(item)
    require(
        runs and {r["split"] for r in runs} == set(lineages), "Both old data partitions required"
    )
    for key in ("layout", "world", "recording", "route"):
        require(
            not {r[key] for r in lineages["train"]}
            & {r[key] for r in lineages["held_out_development"]},
            "Training/evaluation lineage overlap: " + key,
        )
    require(
        not any("mission_pipeline" in p or "mission_joint_constraints" in p for p in inputs),
        "Current experiment outcomes forbidden",
    )
    return dict(runs=runs, input_sha256=inputs, split_lineage=lineages)


def evaluator_frame(ids, targets, index, capture_s):
    visible = tuple(sorted(targets[i] for i in np.unique(ids) if i in targets))
    return EvaluationFrame(
        index,
        capture_s,
        np.isin(ids, list(targets)).astype(np.uint8),
        ids,
        visible,
        {"instance_object_ids": {str(k): v for k, v in targets.items()}},
    )


def candidate_document(track, sources, phase, cid, capture_ns):
    refs = [
        dict(
            capture_ns=round(r["capture_s"] * 1e9),
            component_id=r["component_id"],
            source_config_id=sources[round(r["capture_s"] * 1e9)],
        )
        for r in track["evidence"]
        if phase["start_s"] <= r["capture_s"] < phase["end_s"]
    ]
    if not refs:
        return None
    times = [r["capture_ns"] for r in refs]
    return dict(
        track_id="excluded_from_features",
        current_component_id=cid,
        current_capture_ns=capture_ns,
        phase_id=phase["phase_id"],
        hits=len(refs),
        span_ns=times[-1] - times[0],
        last_gap_ns=times[-1] - times[-2] if len(times) > 1 else None,
        source_config_ids=sorted({r["source_config_id"] for r in refs}),
        predicted_position_m=list(track["position"]) if track["position"] is not None else None,
        required_hits=phase["min_hits"],
        required_span_ns=round(phase["min_evidence_span_s"] * 1e9),
        phase_deadline_ns=round(phase["deadline_s"] * 1e9),
        evidence=refs,
        ground_truth_access=False,
    )


def extract(plan, output):
    rows, cache, terrains = {}, {}, {}
    old_label_checks, decoded_masks, trace_updates = 0, 0, 0
    for number, run in enumerate(plan["runs"], 1):
        root = Path(run["output"])
        result, events = read(root / "execution/results.json"), read(root / "execution/events.json")
        require(result["status"] == "completed", "Incomplete historical execution")
        require(
            not any(result[k] for k in ("router_errors", "invalid_actions", "inference_failures")),
            "Historical execution failure",
        )
        require(
            result["backend"]["recording_identity"] == run["lineage"]["recording"],
            "Recording substitution",
        )
        scenario, intent = read(run["scenario"]), read(run["intent"])
        targets = {
            o["label"]: o["object_id"] for o in scenario["world"]["objects"] if o["is_target"]
        }
        options = {f.name: result["tracking"][f.name] for f in fields(TrackingConfig)}
        options.update(min_hits=2**31, remember_reports=False)
        config = TrackingConfig(**options)
        tracker = StaticReportTracker(config)
        if run["assets"] not in terrains:
            terrains[run["assets"]] = load_public_terrain(Path(run["assets"]))[0]
        projectors = (
            {}
            if config.association == "image_iou"
            else {
                config.association: SurfaceProjector(
                    terrains[run["assets"]] if config.association == "terrain_world" else None
                )
            }
        )
        sources, labels = {}, {}
        for event in events:
            capture_s, completion_s = event["capture_time_s"], event["completion_time_s"]
            capture_ns = round(capture_s * 1e9)
            source = event["execution"]["config_id"]
            require(event["execution"]["success"] is True, "Only completed selected masks allowed")
            sources[capture_ns] = source
            raw = root / "execution" / f"observation_{event['observation_id']:05d}.npz"
            require(sha(raw) == plan["input_sha256"][str(raw.resolve())], "Historical mask changed")
            key = (run["lineage"]["recording"], event["observation_id"], source, config.association)
            if key not in cache:
                with np.load(raw, allow_pickle=False) as archive:
                    mask, ids = archive["prediction_mask"], archive["object_ids"]
                    require(
                        mask.dtype == np.bool_ and mask.shape == ids.shape, "Selected mask shape"
                    )
                    measurements = detections(
                        mask,
                        event["capture_body_pose"],
                        scenario["world"].get("camera_pitch_rad", math.pi / 2),
                        projectors,
                    )
                    truth = evaluator_frame(ids, targets, event["observation_id"], capture_s)
                    matches = component_matches(truth, mask, 0.3)
                    cache[key] = (sha(raw), measurements, matches)
                    decoded_masks += 1
            raw_sha, measurements, matches = cache[key]
            require(raw_sha == sha(raw), "Same recording/action has different selected arrays")
            require(
                not tracker.update(measurements, capture_s, completion_s),
                "Candidate generator emitted report",
            )
            trace_updates += 1
            labels.update({(capture_ns, cid): target for cid, target in matches.items()})
            phase = next(
                (p for p in intent["mission"]["phases"] if p["start_s"] <= capture_s < p["end_s"]),
                None,
            )
            if phase is None:
                continue
            for assignment in tracker.assignments:
                public = candidate_document(
                    tracker.tracks[assignment["track_id"]],
                    sources,
                    phase,
                    assignment["component_id"],
                    capture_ns,
                )
                if public["hits"] < 2 or public["span_ns"] < 2_000_000_000:
                    continue
                vector = feature_vector(public).tolist()
                support = [
                    labels.get((r["capture_ns"], r["component_id"])) for r in public["evidence"]
                ]
                label = support[-1] is not None and all(t == support[-1] for t in support)
                candidate_id = digest(
                    dict(recording=run["lineage"]["recording"], evidence=public["evidence"])
                )
                origin = dict(
                    run=run["output"],
                    capture_ns=capture_ns,
                    component_id=assignment["component_id"],
                )
                if candidate_id in rows:
                    previous = rows[candidate_id]
                    require(
                        previous["split"] == run["split"]
                        and previous["features"] == vector
                        and previous["label"] == label,
                        "Duplicate candidate changed data or partition",
                    )
                    previous["source_occurrences"].append(origin)
                else:
                    rows[candidate_id] = dict(
                        candidate_id=candidate_id,
                        split=run["split"],
                        flight=run["flight"],
                        lineage=run["lineage"],
                        public=public,
                        features=vector,
                        label=label,
                        evaluator_support_target_ids=support,
                        source_occurrences=[origin],
                    )
        for report in result["report_quality"]["reports"]:
            support = [
                labels.get((round(r["capture_s"] * 1e9), r["component_id"]))
                for r in report["evidence"]
            ]
            consistent = support[-1] is not None and all(t == support[-1] for t in support)
            require(
                support == report["support_target_ids"]
                and consistent == report["identity_consistent"],
                "Raw mask identity reconstruction disagrees with historical report",
            )
            old_label_checks += 1
        if number % 10 == 0:
            print(
                f"Reconstructed {number}/{len(plan['runs'])} older traces; ",
                f"{len(rows)} unique candidates",
                flush=True,
            )
    data = dict(
        schema_version="report_candidate_training_data_v1",
        candidates=list(rows.values()),
        decoded_unique_selected_masks=decoded_masks,
        selected_trace_updates=trace_updates,
        historical_report_identity_checks=old_label_checks,
        current_pipeline_or_joint_outcomes_used=False,
        candidate_source=(
            "All phase-local >=2-hit,>=2s prefixes on historical selected-mask "
            "schedules; not only emitted reports "
        ),
        deduplication=(
            "recording identity plus ordered phase-local (capture,component,source "
            "configuration) support "
        ),
        ground_truth_is_label_only=True,
    )
    save(Path(output) / "candidates.json", data)
    return data
