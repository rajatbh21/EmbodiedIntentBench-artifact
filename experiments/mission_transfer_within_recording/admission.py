"""Preserve the failed strict gate and admit only within-recording routing comparisons."""

from __future__ import annotations

import argparse
import math
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_environment_transfer_v3 import sensor
from experiments.mission_environment_transfer_v3.cohort import verify as verify_cohort
from experiments.mission_environment_transfer_v3.route import verify_route
from experiments.mission_environment_transfer_v3.scenario import load_scenario
from experiments.routing_development_matrix.observability import assess, contract_bound
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.scenario import digest

FILENAME = "routing_admission_v1.json"
REJECTION_SHA256 = "ba4eb5e90930324c4a4d2b67e802b1cc1f340a18971cd0cf1c780a263eecd244"
EVIDENCE = (
    "protocol.json",
    "study/protocol.json",
    "sensor_gate.json",
    "physical_route_validation.json",
    "acquisition_order.json",
    "acquisition_completion.json",
    "acquisition_failure.json",
)
SCOPE = {
    "estimand": "Routing-policy contrasts within each identical physical observation bank",
    "amendment_stage": "post_sensor_pre_model",
    "original_strict_gate_passed": False,
    "fresh_full_prospective_test": False,
    "cross_lighting_pose_equality_required": False,
    "cross_lighting_numeric_tolerance_used": False,
    "pure_lighting_causal_effect_identified": False,
    "lighting_analysis": "Descriptive strata of independently acquired physical conditions",
    "same_available_rgb_pose_timestamp_bank_across_policies": True,
    "same_processed_frame_subset_required": False,
    "policy_dependent_navigation_supported": False,
    "all_four_preselected_recordings_retained": True,
    "mission_thresholds_models_policies_and_costs_changed": False,
    "sensor_bounds_prove_deployable_feasibility": False,
}


def _rejection(root, protocol):
    note = read(root / "ADMISSION_REJECTED.json")
    if sha(root / "ADMISSION_REJECTED.json") != REJECTION_SHA256:
        raise ValueError("Changed original strict admission rejection")
    if (
        note["schema_version"] != "strict_cross_lighting_admission_rejection_v1"
        or note["original_admission_passed"] is not False
        or note["neural_predictions_started"] is not False
        or note["policy_replays_started"] is not False
        or note["physical_route_passed"] is not True
        or note["per_recording_visibility_and_intent_opportunities_passed"] is not True
        or set(note["evidence_sha256"]) != set(EVIDENCE)
        or set(note["recording_sha256"]) != set(protocol["cases"])
    ):
        raise ValueError("Unexpected original strict rejection evidence")
    for relative, expected in note["evidence_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError("Changed original acquisition evidence: " + relative)
    for flight, expected in note["recording_sha256"].items():
        if sha(root / flight / "recording/recording.json") != expected:
            raise ValueError("Changed original recording: " + flight)
    order = read(root / "acquisition_order.json")
    completion = read(root / "acquisition_completion.json")
    failure = read(root / "acquisition_failure.json")
    study = read(root / "study/protocol.json")
    if (
        order["flights"] != sorted(protocol["cases"])
        or order["cohort_protocol_sha256"] != sha(root / "protocol.json")
        or order["study_protocol_sha256"] != sha(root / "study/protocol.json")
        or order["methods_frozen_before_this_capture_run"] is not True
        or completion["acquisition_complete"] is not True
        or completion["admitted_to_model_evaluation"] is not False
        or failure["completed_recordings"] != 4
        or failure["admitted_to_model_evaluation"] is not False
        or failure["neural_inference_started"] is not False
        or study["cohort_protocol_sha256"] != sha(root / "protocol.json")
        or Path(study["cohort"]).resolve() != root.resolve()
    ):
        raise ValueError("Acquisition chronology or original comparison binding differs")
    return {
        "rejection_sha256": REJECTION_SHA256,
        "evidence_sha256": note["evidence_sha256"],
        "recording_sha256": note["recording_sha256"],
    }


def _no_outputs(root, output):
    for folder in {root.resolve(), output.resolve()}:
        for name in ("predictions", "inference.json", "events.json", "runs"):
            if any(folder.rglob(name)):
                raise ValueError("Admission must precede every model/policy attempt: " + name)


def _pose_difference(a, b):
    position = math.dist(a["position_m"], b["position_m"])
    qa, qb = (np.asarray(p["quaternion_xyzw"], dtype=float) for p in (a, b))
    # Normalize only for the angular diagnostic; preserve raw stored poses in identity checks.
    qa, qb = qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)
    chord = min(float(np.linalg.norm(qa - qb)), float(np.linalg.norm(qa + qb)))
    return position, 4 * math.asin(min(1.0, chord / 2))


def _paired_diagnostics(root, protocol, recordings):
    groups = {}
    for flight, case in protocol["cases"].items():
        groups.setdefault(case["layout_group"], []).append(flight)
    if len(groups) != 2 or any(len(v) != 2 for v in groups.values()):
        raise ValueError("Retain both physical layouts and both lighting acquisitions")
    result = {}
    for group, flights in sorted(groups.items()):
        flights.sort()
        left, right = [recordings[f]["frames"] for f in flights]
        if [f["timestamp_ns"] for f in left] != [f["timestamp_ns"] for f in right]:
            raise ValueError("Original paired capture schedules differ")
        delta, changed, identical_pose_count = [], [], 0
        for a, b in zip(left, right, strict=True):
            position, angle = _pose_difference(a["body_pose"], b["body_pose"])
            identical = a["body_pose"] == b["body_pose"]
            identical_pose_count += int(identical)
            delta.append((position, angle))
            with (
                np.load(root / flights[0] / "recording" / a["file"], allow_pickle=False) as x,
                np.load(root / flights[1] / "recording" / b["file"], allow_pickle=False) as y,
            ):
                pixels = int(np.count_nonzero(x["object_ids"] != y["object_ids"]))
            if pixels:
                changed.append({"timestamp_ns": a["timestamp_ns"], "changed_pixels": pixels})
        differing = [
            a["timestamp_ns"]
            for a, b in zip(left, right, strict=True)
            if a["body_pose"] != b["body_pose"]
        ]
        result[group] = {
            "flights": flights,
            "paired_frames": len(left),
            "timestamps_identical": True,
            "poses_bit_identical": not differing,
            "identical_pose_frames": identical_pose_count,
            "first_differing_pose_timestamp_ns": min(differing) if differing else None,
            "max_euclidean_position_difference_m": max(v[0] for v in delta),
            "max_rotation_angle_difference_rad": max(v[1] for v in delta),
            "identical_gt_frames": len(left) - len(changed),
            "changed_gt_pixels_total": sum(v["changed_pixels"] for v in changed),
            "changed_gt_frames": changed,
            "diagnostic_only_not_an_acceptance_tolerance": True,
        }
    return result


def inspect(root):
    """Reconstruct the original failed gate, without rewriting any frozen evidence."""
    protocol = verify_cohort(root)
    if len(protocol["cases"]) != 4 or len(protocol["missions"]) != 2:
        raise ValueError("All four preselected recordings and both missions are required")
    evidence = _rejection(root, protocol)
    route = verify_route(root)
    cases, recordings, banks = {}, {}, {}
    for flight, case in protocol["cases"].items():
        folder = root / flight / "recording"
        scenario = load_scenario(
            root / flight / "rapid_survey/scenario.json",
            case["environment"],
            Path(protocol["assets"]),
        )
        recording = read(folder / "recording.json")
        recordings[flight] = recording
        frames = recording["frames"]
        if len(frames) != 60 or recording["world"] != scenario.world:
            raise ValueError("Original complete 60-frame physical bank is required")
        areas = {o["object_id"]: [] for o in case["placement"] if o["is_target"]}
        backend = ReplayBackend(scenario, folder)
        try:
            for timestamp in backend.frames:
                backend.advance_to(timestamp)
                frame = backend.capture()
                pose = frame.body_pose
                vector = np.asarray(pose["position_m"] + pose["quaternion_xyzw"], dtype=float)
                if (
                    vector.shape != (7,)
                    or not np.isfinite(vector).all()
                    or not math.isclose(np.linalg.norm(vector[3:]), 1.0, abs_tol=1e-8)
                    or frame.rgb.shape[:2]
                    != (case["environment"]["height_px"], case["environment"]["width_px"])
                ):
                    raise ValueError("Invalid recorded physical pose or image dimensions")
                for obj in case["placement"]:
                    if obj["object_id"] in areas:
                        areas[obj["object_id"]].append(
                            int(np.count_nonzero(frame.object_ids == obj["label"]))
                        )
        finally:
            backend.close()
        times = [(f["timestamp_ns"] - frames[0]["timestamp_ns"]) / 1e9 for f in frames]
        target_times = {
            t: [s for s, a in zip(times, values, strict=True) if a > 0]
            for t, values in areas.items()
        }
        cases[flight] = dict(
            **assess(areas),
            sensor_contract_bounds={
                m: contract_bound(target_times, read(root / flight / m / "intent.json"))
                for m in protocol["missions"]
            },
            areas=areas,
            capture_times_s=times,
            recording_sha256=sha(folder / "recording.json"),
        )
        banks[flight] = {
            "recording_identity": scenario.recording_identity,
            "recording_sha256": sha(folder / "recording.json"),
            "frame_count": len(frames),
            "frame_inventory_sha256": digest(frames),
            "timestamp_pose_bank_sha256": digest(
                [(f["timestamp_ns"], f["body_pose"]) for f in frames]
            ),
            "capture_timestamps_ns": [f["timestamp_ns"] for f in frames],
        }
    diagnostics = _paired_diagnostics(root, protocol, recordings)
    same = all(g["poses_bit_identical"] for g in diagnostics.values())
    reconstructed = dict(
        schema_version="depot_transfer_sensor_gate_v1",
        cases=cases,
        passed=bool(route["all_passed"] and same and sensor.passes(cases)),
        route_passed=route["all_passed"],
        paired_poses_identical=same,
        rule_defined_before_capture=True,
        model_predictions_used=False,
        gt_aware=True,
        deployable_policy=False,
        mission_feasibility_proven=False,
        protocol_sha256=sha(root / "protocol.json"),
        source_sha256=sha(Path(sensor.__file__)),
        physical_route_validation_sha256=sha(root / "physical_route_validation.json"),
    )
    if reconstructed != read(root / "sensor_gate.json"):
        raise ValueError("Strict failed gate does not match unchanged raw-data recomputation")
    if reconstructed["passed"] or same or not route["all_passed"] or not sensor.passes(cases):
        raise ValueError("Only original cross-lighting pose equality may fail this amendment")
    return {
        "original_evidence": evidence,
        "banks": banks,
        "recomputed_sensor_cases_sha256": digest(cases),
        "physical_route_passed": True,
        "all_original_visibility_and_intent_bounds_passed": True,
        "cross_lighting_diagnostics": diagnostics,
        "raw_capture_count": sum(b["frame_count"] for b in banks.values()),
    }


def source_inventory(root):
    repo = Path(__file__).resolve().parents[2]
    files = [
        *Path(__file__).parent.glob("*.py"),
        repo / "tests/test_mission_transfer_within_recording.py",
        repo / "docs/mission_transfer_within_recording.md",
    ]
    result = dict(read(root / "protocol.json")["implementation_sha256"])
    result.update({str(f.resolve()): sha(f) for f in files})
    return result


def prepare(cohort, output):
    cohort, output = Path(cohort).resolve(), Path(output).resolve()
    if (output / FILENAME).exists():
        raise FileExistsError("Preserve the original scope-amendment certificate")
    _no_outputs(cohort, output)
    inspection = inspect(cohort)
    certificate = {
        "schema_version": "within_recording_routing_admission_v1",
        "cohort": str(cohort),
        "admitted": True,
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": SCOPE,
        "pre_model_attestation": {
            "neural_prediction_attempts": 0,
            "policy_replay_attempts": 0,
            "rgb_and_ground_truth_inspected": True,
            "model_and_policy_outcomes_inspected": False,
        },
        "implementation_sha256": source_inventory(cohort),
        **inspection,
    }
    _no_outputs(cohort, output)
    output.mkdir(parents=True, exist_ok=True)
    save(output / FILENAME, certificate)
    return verify(output)


def verify(root):
    root = Path(root)
    certificate = read(root / FILENAME)
    if (
        certificate["schema_version"] != "within_recording_routing_admission_v1"
        or certificate["admitted"] is not True
        or certificate["scope"] != SCOPE
        or certificate["pre_model_attestation"]
        != {
            "neural_prediction_attempts": 0,
            "policy_replay_attempts": 0,
            "rgb_and_ground_truth_inspected": True,
            "model_and_policy_outcomes_inspected": False,
        }
    ):
        raise ValueError("Changed admission scope or pre-model attestation")
    cohort = Path(certificate["cohort"])
    if certificate["implementation_sha256"] != source_inventory(cohort):
        raise ValueError("Changed frozen admission implementation")
    actual = inspect(cohort)
    if any(certificate[k] != value for k, value in actual.items()):
        raise ValueError("Changed admitted observation bank or evidence")
    # Outcomes are allowed after prepare; their presence cannot invalidate an honest freeze.
    return certificate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.cohort, args.output) if args.mode == "prepare" else verify(args.output)
    print({"admitted": result["admitted"], "raw_capture_count": result["raw_capture_count"]})
