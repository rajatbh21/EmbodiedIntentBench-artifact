"""Transfer recorded UAV motion to Gazebo with explicit rigid translation only.

This validates source-derived trajectories. It does not reconstruct the original
water scene, people, camera motion, mission instructions, energy or perception.
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from experiments.lake_causal_router import split_fingerprint
from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.placement import collision_bounds, terrain_height
from experiments.mission_coverage_routing.cohort import verify as cohort_verify
from experiments.mission_routing_v2.flights import load_scenario
from experiments.mission_routing_v2.flights import verify as base_verify


def sample(rows, times):
    t = np.asarray([r["time_s"] for r in rows], float)
    t -= t[0]
    query = np.asarray(times, float)
    if (
        len(t) < 2
        or np.any(np.diff(t) <= 0)
        or np.max(np.diff(t)) > 1
        or np.any(query < 0)
        or np.any(query > t[-1] + 1e-9)
    ):
        raise ValueError("Continuous ordered trace without extrapolation required")
    xyz = np.asarray([r["local_east_north_altitude_delta_m"] for r in rows], float)
    xyz -= xyz[0]
    return np.column_stack([np.interp(query, t, xyz[:, i]) for i in range(3)])


def prepare(imported, cohort, root):
    if root.exists():
        raise FileExistsError("Preserve trajectory transfer attempts")
    p = read(imported / "manifest.json")
    if not p["complete"] or p["used_for_frozen_coverage_evaluation"]:
        raise ValueError("Separate completed telemetry import required")
    if p["implementation_sha256"] != sha(Path(__file__).with_name("seadronessee.py")):
        raise ValueError("Changed importer")
    for g in p["groups"]:
        if sha(imported / g["file"]) != g["sha256"]:
            raise ValueError("Changed normalized source")
    old = cohort_verify(cohort)
    assets = Path(old["assets"])
    bounds = collision_bounds(assets)
    template = cohort / "central_slope_day/rapid_survey"
    root.mkdir(parents=True)
    cases = {}
    frozen = {}
    for wi, window in enumerate(p["candidate_windows"]):
        group = read(imported / (window["group_id"] + ".json"))
        ids = set(window["source_frame_ids"])
        rows = [r for r in group["frames"] if r["source_image_id"] in ids]
        times = [*np.arange(0, 120, 2), window["duration_s"]]
        relative = sample(rows, times)
        center = np.asarray([(-150, 50), (-40, 50)][wi], float)
        dense = sample(rows, np.linspace(0, window["duration_s"], 481))
        offset_z = 0.0
        for delta in dense:
            x, y = delta[:2] + center
            ceiling = terrain_height(assets, x, y) + 12
            for _, low, high in bounds:
                if low[0] - 2 <= x <= high[0] + 2 and low[1] - 2 <= y <= high[1] + 2:
                    ceiling = max(ceiling, high[2] + 5)
            offset_z = max(offset_z, ceiling - delta[2])
        translation = np.array([*center, round(offset_z + 1, 2)])
        desired = relative + translation
        name = "real_" + group["source_identity"]["drone"]
        folder = root / name / "rapid_survey"
        folder.mkdir(parents=True)
        doc = copy.deepcopy(read(template / "scenario.json"))
        max_speed = max(
            1.2, float(np.max(np.linalg.norm(np.diff(desired, axis=0), axis=1) / np.diff(times)))
        )
        requested_speed = max_speed
        max_speed = min(3.0, max_speed)
        doc["scenario_id"] = "REAL_TELEMETRY_TRANSFER_" + name
        doc["world"].update(
            initial_position_m=desired[0].tolist(),
            max_speed_mps=max_speed,
            waypoints=[
                dict(start_s=float(times[i]), position_m=desired[i + 1].tolist())
                for i in range(len(times) - 1)
            ],
        )
        save(folder / "scenario.json", doc)
        for n in ("contract.json", "intent.json"):
            (folder / n).write_bytes((template / n).read_bytes())
        for path in folder.glob("*.json"):
            frozen[str(path.relative_to(root))] = sha(path)
        environment = dict(old["cases"]["central_slope_day"]["environment"], seed=2801 + wi)
        scenario = load_scenario(folder / "scenario.json", environment, assets)
        cases[name] = dict(
            environment=environment,
            lineage=split_fingerprint(scenario, scenario.scenery_sha256),
            source_group=window["group_id"],
            source_identity=group["source_identity"],
            source_window=window,
            translation_m=translation.tolist(),
            rotation_deg=0,
            scale=1,
            reference_times_s=[float(t) for t in times],
            reference_positions_m=desired.tolist(),
            max_speed_mps=max_speed,
            requested_reference_speed_mps=requested_speed,
            platform_speed_cap_mps=3.0,
            reference_timing_rescaled=False,
            geometry_clearance_m=12,
        )
    sourcefiles = [
        Path(__file__),
        Path(__file__).with_name("seadronessee.py"),
        Path(__file__).parents[2] / "tests/test_real_mission_telemetry.py",
    ]
    result = {
        k: old[k]
        for k in ("assets", "screen", "calibration_sha256", "costs_sha256", "source_sha256")
    }
    result.update(
        schema_version="real_telemetry_physics_transfer_v1",
        scope="motion_transfer_only_not_mission_evaluation",
        cases=cases,
        missions=["rapid_survey"],
        frozen_files=frozen,
        imported=str(imported.resolve()),
        import_manifest_sha256=sha(imported / "manifest.json"),
        source_files_sha256={str(f.resolve()): sha(f) for f in sourcefiles},
        declared_before_capture=True,
        source_training_split=True,
        physical_duration_s=120,
        source_scene_reconstructed=False,
        source_camera_motion_reconstructed=False,
        actors="Unchanged synthetic placeholders, not reconstructed source people",
        mission_scores_reported=False,
        pose_metrics=(
            "Reference versus actual at the same capture timestamp; "
            "no pass threshold tuned after capture"
        ),
        vertical_datum="Added scenery-derived translation; source altitude changes preserved",
    )
    save(root / "protocol.json", result)
    verify(root)
    return result


def verify(root):
    p = base_verify(root)
    if p["schema_version"] != "real_telemetry_physics_transfer_v1":
        raise ValueError("Wrong transfer")
    if sha(Path(p["imported"]) / "manifest.json") != p["import_manifest_sha256"]:
        raise ValueError("Changed telemetry input")
    for f, expected in p["source_files_sha256"].items():
        if sha(Path(f)) != expected:
            raise ValueError("Changed transfer source")
    return p


def audit(root):
    p = verify(root)
    results = {}
    for f, c in p["cases"].items():
        r = root / f / "recording"
        doc = read(r / "recording.json")
        frames = doc["frames"]
        for frame in frames:
            if sha(r / frame["file"]) != frame["sha256"]:
                raise ValueError("Changed physics frame")
        times = np.array([(fr["timestamp_ns"] - frames[0]["timestamp_ns"]) / 1e9 for fr in frames])
        reference = np.asarray(c["reference_positions_m"])
        desired = np.column_stack(
            [np.interp(times, c["reference_times_s"], reference[:, i]) for i in range(3)]
        )
        actual = np.asarray([fr["body_pose"]["position_m"] for fr in frames])
        error = np.linalg.norm(actual - desired, axis=1)
        results[f] = dict(
            captures=len(frames),
            source_identity=c["source_identity"],
            rmse_position_m=float(np.sqrt(np.mean(error**2))),
            max_position_error_m=float(max(error)),
            median_position_error_m=float(np.median(error)),
            reference_times_s=times.tolist(),
            reference_positions_m=desired.tolist(),
            actual_positions_m=actual.tolist(),
            recording_sha256=sha(r / "recording.json"),
            complete=len(frames) == 60,
        )
    save(
        root / "validation.json",
        dict(
            complete=all(c["complete"] for c in results.values()),
            cases=results,
            protocol_sha256=sha(root / "protocol.json"),
            source_sha256=sha(Path(__file__)),
            source_motion_transferred=True,
            mission_success_claimed=False,
        ),
    )
    print(
        {
            f: {k: c[k] for k in ("captures", "rmse_position_m", "max_position_error_m")}
            for f, c in results.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "record", "audit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imported", type=Path)
    parser.add_argument("--cohort", type=Path)
    a = parser.parse_args()
    if a.mode == "prepare":
        prepare(a.imported, a.cohort, a.output)
    elif a.mode == "record":
        from experiments.mission_routing_v2.flights import record

        for f in verify(a.output)["cases"]:
            record(a.output, f)
        audit(a.output)
    else:
        audit(a.output)
