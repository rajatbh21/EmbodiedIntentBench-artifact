"""Preregistered sensor screen before any new model prediction or policy replay."""

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.cohort import verify
from experiments.mission_routing_v2_route_validation import audit
from experiments.routing_development_matrix.observability import assess, contract_bound


def passes(cases):
    if not cases:
        raise ValueError("Nonempty sensor cases required")
    if any(not c["sensor_contract_bounds"] for c in cases.values()):
        raise ValueError("Nonempty intent-specific bounds required")
    return all(
        c["all_targets_passed"]
        and all(not b["sensor_certified_infeasible"] for b in c["sensor_contract_bounds"].values())
        for c in cases.values()
    )


def run(root):
    p = verify(root)
    destination = root / "sensor_gate.json"
    if destination.exists():
        raise FileExistsError("Preserve the pre-inference sensor decision")
    if any((root / f / "predictions").exists() for f in p["cases"]):
        raise ValueError("Sensor gate must precede all inference")
    route = audit(root)
    cases, paired = {}, {}
    for flight, c in p["cases"].items():
        folder = root / flight / "recording"
        recording = read(folder / "recording.json")
        frames = recording["frames"]
        areas = {o["object_id"]: [] for o in c["placement"] if o["is_target"]}
        times = [(f["timestamp_ns"] - frames[0]["timestamp_ns"]) / 1e9 for f in frames]
        for frame in frames:
            path = folder / frame["file"]
            if sha(path) != frame["sha256"]:
                raise ValueError("Changed physical capture")
            with np.load(path, allow_pickle=False) as data:
                for obj in c["placement"]:
                    if obj["object_id"] in areas:
                        areas[obj["object_id"]].append(
                            int((data["object_ids"] == obj["label"]).sum())
                        )
        target_times = {
            t: [s for s, a in zip(times, vs, strict=True) if a > 0] for t, vs in areas.items()
        }
        cases[flight] = dict(
            **assess(areas),
            sensor_contract_bounds={
                m: contract_bound(target_times, read(root / flight / m / "intent.json"))
                for m in p["missions"]
            },
            areas=areas,
            capture_times_s=times,
            recording_sha256=sha(folder / "recording.json"),
        )
        paired.setdefault(c["layout_group"], []).append(
            [(f["timestamp_ns"], f["body_pose"]) for f in frames]
        )
    same = len(paired) == 3 and all(len(v) == 2 and v[0] == v[1] for v in paired.values())
    result = dict(
        schema_version="coverage_prospective_sensor_gate_v1",
        cases=cases,
        passed=bool(route["all_passed"] and same and passes(cases)),
        route_passed=route["all_passed"],
        paired_poses_identical=same,
        rule_defined_before_capture=True,
        model_predictions_used=False,
        gt_aware=True,
        deployable_policy=False,
        mission_feasibility_proven=False,
        protocol_sha256=sha(root / "protocol.json"),
        source_sha256=sha(Path(__file__)),
    )
    save(destination, result)
    print(
        {
            "passed": result["passed"],
            "cases": {
                f: {
                    "all_targets": c["all_targets_passed"],
                    "missions": {
                        m: b["conclusion"] for m, b in c["sensor_contract_bounds"].items()
                    },
                }
                for f, c in cases.items()
            },
        },
        flush=True,
    )
    return result


def verify_gate(root):
    p = verify(root)
    result = read(root / "sensor_gate.json")
    if (
        not result["passed"]
        or not passes(result["cases"])
        or not result["paired_poses_identical"]
        or not result["route_passed"]
        or result["protocol_sha256"] != sha(root / "protocol.json")
        or result["source_sha256"] != sha(Path(__file__))
        or set(result["cases"]) != set(p["cases"])
    ):
        raise ValueError("Valid pre-inference sensor gate required")
    for flight, case in result["cases"].items():
        if case["recording_sha256"] != sha(root / flight / "recording/recording.json"):
            raise ValueError("Changed screened recording")
        if set(case["sensor_contract_bounds"]) != set(p["missions"]):
            raise ValueError("Incomplete sensor intent grid")
        if assess(case["areas"])["targets"] != case["targets"]:
            raise ValueError("Sensor area summary differs")
        folder = root / flight / "recording"
        frames = read(folder / "recording.json")["frames"]
        times = [(f["timestamp_ns"] - frames[0]["timestamp_ns"]) / 1e9 for f in frames]
        if times != case["capture_times_s"]:
            raise ValueError("Screened capture times differ")
        for index, frame in enumerate(frames):
            path = folder / frame["file"]
            if sha(path) != frame["sha256"]:
                raise ValueError("Changed screened frame")
            with np.load(path, allow_pickle=False) as data:
                for obj in p["cases"][flight]["placement"]:
                    if (
                        obj["is_target"]
                        and int((data["object_ids"] == obj["label"]).sum())
                        != case["areas"][obj["object_id"]][index]
                    ):
                        raise ValueError("Screened visibility differs from raw GT")
        target_times = {
            t: [s for s, a in zip(times, vs, strict=True) if a > 0]
            for t, vs in case["areas"].items()
        }
        for mission in p["missions"]:
            if (
                contract_bound(target_times, read(root / flight / mission / "intent.json"))
                != case["sensor_contract_bounds"][mission]
            ):
                raise ValueError("Sensor intent bound differs")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
