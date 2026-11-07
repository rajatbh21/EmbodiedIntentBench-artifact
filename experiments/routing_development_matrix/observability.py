"""Post-hoc sensor coverage screen, independent of neural predictions.

A deliberately modest screen requires four distinct captures with at least 16 GT
mask pixels per target. It is not a detector-feasibility or mission-success proof.
The rule was added after inspecting coverage; do not call it preregistered.
"""

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.objectives import enough_evidence
from experiments.mission_routing_v2.flights import contract_document
from experiments.routing_development_matrix.cohort import verify

MIN_PIXELS = 16
MIN_CAPTURES = 4


def assess(areas):
    if not areas or any(not values for values in areas.values()):
        raise ValueError("Nonempty target coverage sequences required")
    if len({len(values) for values in areas.values()}) != 1:
        raise ValueError("Every target must cover the same canonical captures")
    if any(
        not isinstance(v, int) or isinstance(v, bool) or v < 0 for vs in areas.values() for v in vs
    ):
        raise ValueError("Pixel areas must be nonnegative integers")
    targets = {
        target: dict(
            visible_captures=sum(v > 0 for v in values),
            usable_captures=sum(v >= MIN_PIXELS for v in values),
            max_pixels=max(values),
            screen_passed=sum(v >= MIN_PIXELS for v in values) >= MIN_CAPTURES,
        )
        for target, values in areas.items()
    }
    return dict(
        all_targets_passed=all(t["screen_passed"] for t in targets.values()),
        targets=targets,
        mission_feasibility_proven=False,
    )


def contract_bound(target_times, document):
    """Optimistic zero-latency visibility bound; passing never proves feasibility."""
    targets = document["evaluator_target_ids"]
    if set(target_times) != set(targets):
        raise ValueError("Visibility must cover every contract target")
    phases = []
    for phase in document["mission"]["phases"]:
        supported = [
            target
            for target in phase["evaluator_target_ids"]
            if enough_evidence(
                [
                    t
                    for t in target_times[target]
                    if phase["start_s"] <= t < phase["end_s"] and t <= phase["deadline_s"]
                ],
                phase,
            )
        ]
        recall = len(supported) / len(phase["evaluator_target_ids"])
        phases.append(
            dict(
                phase_id=phase["phase_id"],
                evidence_recall_upper_bound=recall,
                required_recall=phase["required_recall"],
                cannot_meet_recall=recall < phase["required_recall"],
            )
        )
    report = document["reporting"]
    upper = sum(
        any(t <= report["deadline_s"] for t in target_times[target]) for target in targets
    ) / len(targets)
    impossible = any(p["cannot_meet_recall"] for p in phases) or upper < report["min_report_recall"]
    return dict(
        mission_id=document["mission"]["mission_id"],
        sensor_certified_infeasible=impossible,
        conclusion="sensor_infeasible" if impossible else "feasibility_unresolved",
        phase_bounds=phases,
        report_recall_upper_bound=upper,
        ignored_constraints=[
            "model detection errors",
            "serial latency",
            "tracking association",
            "false components and reports",
            "energy",
            "memory",
        ],
        gt_aware=True,
        deployable_policy=False,
    )


def run(root):
    p = verify(root)
    cases = {}
    for flight, c in p["cases"].items():
        folder = root / flight / "recording"
        recording = read(folder / "recording.json")
        areas = {o["object_id"]: [] for o in c["placement"] if o["is_target"]}
        for frame in recording["frames"]:
            path = folder / frame["file"]
            if sha(path) != frame["sha256"]:
                raise ValueError("Changed physical sensor capture")
            with np.load(path, allow_pickle=False) as data:
                ids = data["object_ids"]
                for obj in c["placement"]:
                    if obj["object_id"] in areas:
                        areas[obj["object_id"]].append(int((ids == obj["label"]).sum()))
        times = [
            (f["timestamp_ns"] - recording["frames"][0]["timestamp_ns"]) / 1e9
            for f in recording["frames"]
        ]
        target_times = {
            target: [t for t, area in zip(times, values, strict=True) if area > 0]
            for target, values in areas.items()
        }
        bounds = {
            mission: contract_bound(target_times, contract_document(list(areas), mission))
            for mission in ("rapid_survey", "verified_inspection", "reserve_patrol")
        }
        cases[flight] = dict(
            sensor_contract_bounds=bounds,
            **assess(areas),
            recording_sha256=sha(folder / "recording.json"),
        )
    result = dict(
        schema_version="development_observability_screen_v1",
        min_pixels=MIN_PIXELS,
        min_captures=MIN_CAPTURES,
        gt_aware=True,
        model_predictions_used=False,
        rule_defined_after_coverage_inspection=True,
        data_removed=False,
        cases=cases,
        protocol_sha256=sha(root / "protocol.json"),
        source_sha256=sha(Path(__file__)),
        scope="Development sensor coverage screen; not deployable policy or mission feasibility",
    )
    save(root / "observability.json", result)
    print({f: c["all_targets_passed"] for f, c in cases.items()}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
