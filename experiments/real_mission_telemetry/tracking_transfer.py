"""Versioned fixed-reference controller; preserve the slower waypoint experiment."""

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

from experiments.lake_causal_router import split_fingerprint
from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.scenario import LakeScenario
from experiments.mission_routing_v2.flights import load_scenario
from experiments.real_mission_telemetry.reference import validate
from experiments.real_mission_telemetry.transfer import verify as verify_original
from experiments.v4_benchmark.backends import record_world
from experiments.v4_benchmark.scenario import digest


@dataclass(frozen=True)
class ReferenceScenario(LakeScenario):
    reference: dict | None = None
    controller_sha256: str = ""

    @property
    def recording_identity(self):
        return digest(
            dict(
                base=super().recording_identity,
                reference=self.reference,
                controller_sha256=self.controller_sha256,
            )
        )


def load(root, p, flight):
    c = p["cases"][flight]
    base = load_scenario(
        root / flight / "rapid_survey/scenario.json", c["environment"], Path(p["assets"])
    )
    reference = dict(
        times_s=c["reference_times_s"],
        positions_m=c["reference_positions_m"],
        warmup_s=base.warmup_s,
        speed_limit_mps=8.0,
    )
    validate(reference["times_s"], reference["positions_m"])
    return ReferenceScenario(
        **vars(base), reference=reference, controller_sha256=digest(p["controller_sources_sha256"])
    )


def prepare(original, root):
    p = copy.deepcopy(verify_original(original))
    if root.exists():
        raise FileExistsError("Preserve reference-controller attempts")
    root.mkdir(parents=True)
    for name, expected in p["frozen_files"].items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((original / name).read_bytes())
        if sha(path) != expected:
            raise ValueError("Changed physical scene")
    directory = Path(__file__).parent
    repo = directory.parents[1]
    names = ("reference.py", "transport.py", "backend.py", "worker.py", "tracking_transfer.py")
    sources = [directory / n for n in names] + [repo / "tests/test_telemetry_controller.py"]
    p.update(
        controller_sources_sha256={str(s.resolve()): sha(s) for s in sources},
        original_transfer=str(original.resolve()),
        original_protocol_sha256=sha(original / "protocol.json"),
        controller=(
            "Feedforward source-reference velocity plus 0.8/s position correction, "
            "vector clipped at 8 m/s"
        ),
        posthoc_motion_controller_revision=True,
        source_timestamps_changed=False,
        source_positions_changed=False,
        source_scene_reconstructed=False,
        source_camera_motion_reconstructed=False,
        benchmark_policy_used=False,
        unchanged_simulator_acceleration_limit_mps2=2,
    )
    for flight, c in p["cases"].items():
        c.update(
            legacy_waypoint_speed_field_mps=3.0,
            controller_speed_limit_mps=8.0,
            platform_speed_cap_mps=8.0,
        )
        scenario = load(root, p, flight)
        c["lineage"] = split_fingerprint(scenario, scenario.scenery_sha256)
    save(root / "protocol.json", p)
    verify(root)


def verify(root):
    p = verify_original(root)
    original = Path(p["original_transfer"])
    verify_original(original)
    if sha(original / "protocol.json") != p["original_protocol_sha256"]:
        raise ValueError("Changed original transfer")
    for source, expected in p["controller_sources_sha256"].items():
        if sha(Path(source)) != expected:
            raise ValueError("Changed reference controller")
    for flight, c in p["cases"].items():
        scenario = load(root, p, flight)
        if split_fingerprint(scenario, scenario.scenery_sha256) != c["lineage"]:
            raise ValueError("Changed reference lineage")
    return p


def record(root):
    from experiments.real_mission_telemetry.backend import LiveBackend
    from experiments.real_mission_telemetry.transfer import audit

    p = verify(root)
    for flight in p["cases"]:
        scenario = load(root, p, flight)
        backend = LiveBackend(scenario, root / flight / "live")
        try:
            record_world(scenario, backend, root / flight / "recording")
        finally:
            backend.close()
        print("Recorded " + flight + " with the reference controller", flush=True)
    audit(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "record"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original", type=Path)
    a = parser.parse_args()
    if a.mode == "prepare":
        prepare(a.original, a.output)
    else:
        record(a.output)
