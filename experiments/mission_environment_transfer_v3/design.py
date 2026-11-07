"""Transparent geometry revision and nominal camera opportunity checks."""

import math
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import sha


def waypoints(design):
    return [
        dict(start_s=time, position_m=design[which])
        for time, which in (
            (0, "start"),
            (8, "end"),
            (30, "end"),
            (38, "start"),
            (60, "end"),
            (90, "start"),
        )
    ]


def nominal_opportunities(designs):
    """A conservative constant-speed/frustum calculation, not a sensor guarantee."""
    speed, pitch = 0.40, 1.2
    tx, ty = math.tan(1.2 / 2), 0.75 * math.tan(1.2 / 2)
    rows = {}
    for group, design in designs.items():
        position = np.asarray(design["start"], dtype=float)
        schedule = waypoints(design)
        observations = [[] for _ in design["targets"]]
        for tick in range(1200):
            time = tick / 10
            target = next(w["position_m"] for w in reversed(schedule) if w["start_s"] <= time)
            if tick % 20 == 0:
                for i, xy in enumerate(design["targets"]):
                    dx, dy = np.asarray(xy) - position[:2]
                    dz = 0.9 - (position[2] - 0.1)
                    forward = math.cos(pitch) * dx - math.sin(pitch) * dz
                    vertical = math.sin(pitch) * dx + math.cos(pitch) * dz
                    if forward > 0 and abs(dy) <= forward * tx and abs(vertical) <= forward * ty:
                        observations[i].append(time)
            delta = np.asarray(target) - position
            distance = np.linalg.norm(delta)
            if distance:
                position += delta * min(1, speed * 0.1 / distance)
        phases = []
        for phase, indices in enumerate(((0, 1), (2, 3), (4, 5), (0, 1))):
            captures = [
                [t for t in observations[i] if phase * 30 <= t < (phase + 1) * 30] for i in indices
            ]
            passed = all(len(times) >= 3 and times[-1] - times[0] >= 4 for times in captures)
            phases.append(
                dict(
                    phase_id=f"sector_{phase + 1}",
                    target_indices=list(indices),
                    capture_times_s=captures,
                    strict_nominal_opportunity=passed,
                )
            )
        rows[group] = dict(
            phases=phases, passed=all(p["strict_nominal_opportunity"] for p in phases)
        )
    if not all(row["passed"] for row in rows.values()):
        raise ValueError("Nominal geometry lacks strict observation opportunities")
    return dict(
        passed=True,
        cases=rows,
        nominal_translation_speed_mps=speed,
        actual_v2_free_flight_speed_approximately_mps=0.408,
        commanded_speed_cap_mps=0.45,
        anchor_height_m=0.9,
        ignored_effects=[
            "rigid-body acceleration and pitch",
            "source geometry occlusion",
            "image formation",
            "model errors",
            "serial inference latency",
        ],
        sensor_gate_still_required=True,
        feasibility_proven=False,
        source_sha256=sha(Path(__file__)),
    )


def revision(rejected):
    rejected = Path(rejected)
    if any(rejected.rglob("inference.json")) or any(rejected.rglob("predictions")):
        raise ValueError("This engineering revision requires no model outcomes")
    if any(rejected.rglob("events.json")):
        raise ValueError("This engineering revision requires no policy replays")
    sensor = read(rejected / "sensor_gate.json")
    if sensor["passed"]:
        raise ValueError("A failed physical/sensor gate is required")
    captures = sum(
        len(read(rejected / flight / "recording/recording.json")["frames"])
        for flight in sensor["cases"]
    )
    if captures != 240:
        raise ValueError("Expected four retained 60-frame rejected recordings")
    return dict(
        rejected_cohort=str(rejected.resolve()),
        protocol_sha256=sha(rejected / "protocol.json"),
        sensor_gate_sha256=sha(rejected / "sensor_gate.json"),
        physical_route_validation_sha256=sha(rejected / "physical_route_validation.json"),
        admission_rejected_sha256=sha(rejected / "ADMISSION_REJECTED.json"),
        captures_seen=captures,
        rgb_and_ground_truth_seen=True,
        model_predictions=0,
        policy_replays=0,
        policy_change=False,
        model_weights_change=False,
        mission_threshold_change=False,
        resource_cost_change=False,
        strict_unmodified_prospective_test=False,
        reason="Installed Gazebo ignored collision submesh selection; one phase lacked visibility",
        interventions=[
            "Standalone collision mesh matches original visible source geometry",
            "Shorten route legs from 12 to 8m",
            "Hold initial position for8s in phases1 and2 before each transit",
        ],
    )
