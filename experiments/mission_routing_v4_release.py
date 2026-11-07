"""Check completed v4 evidence links without claiming general research readiness."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_routing_v4.flights import verify
from experiments.mission_routing_v4_report import run as report


def run(root):
    p, analysis = verify(root), report(root)
    validation = read(root / "validation.json")
    route = read(root / "physical_route_validation.json")
    if not route["all_passed"]:
        raise ValueError("Physical gate failed")
    total_calls, pairs, poses = 0, 0, []
    for flight in p["cases"]:
        folder = root / flight / "predictions"
        manifest = checked_manifest(folder, root / "protocol.json")
        records = read(root / flight / "recording/recording.json")
        if manifest["recording_sha256"] != sha(root / flight / "recording/recording.json"):
            raise ValueError("Wrong physical recording")
        measurements = read(folder / "inference.json")
        calls = sum(r["actual_neural_calls"] for r in measurements)
        if calls != read(folder / "completion.json")["actual_neural_calls"]:
            raise ValueError("Wrong neural call count")
        expected = {(i, c) for i in range(len(records["frames"])) for c in manifest["models"]}
        actual = [(r["capture_id"], r["config_id"]) for r in measurements]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Incomplete actual inference grid")
        total_calls += calls
        pairs += len(actual)
        poses.append([(f["timestamp_ns"], f["body_pose"]) for f in records["frames"]])
    if not all(pose == poses[0] for pose in poses):
        raise ValueError("Paired lighting physical trajectories differ")
    transfer = read(root / "risk_transfer.json")
    if transfer["protocol_sha256"] != sha(root / "protocol.json") or transfer["models_retrained"]:
        raise ValueError("Invalid risk transfer provenance")
    if transfer["diagnostics_sha256"] != sha(root / "feasibility_diagnostics.json"):
        raise ValueError("Changed dense evaluator diagnostic")
    if transfer["source_sha256"] != sha(Path(__file__).with_name("mission_routing_v4_transfer.py")):
        raise ValueError("Changed transfer implementation")
    timing = read(root / "router_timing.json")
    if (
        timing["validation_sha256"] != sha(root / "validation.json")
        or timing["configured_costs_changed"]
    ):
        raise ValueError("Invalid timing provenance")
    if timing["source_sha256"] != sha(Path(__file__).with_name("mission_routing_v4_timing.py")):
        raise ValueError("Changed timing implementation")
    for figure_name in ("mission_outcomes", "risk_tradeoff"):
        figure = read(root / (figure_name + "_provenance.json"))
        if (
            figure["figure_sha256"] != sha(root / (figure_name + ".png"))
            or figure["status_sha256"] != sha(root / "analysis_v4.json")
            or figure["implementation_sha256"]
            != sha(Path(__file__).with_name("mission_routing_v4_plot.py"))
        ):
            raise ValueError("Changed figure, implementation or source results")
    plan = read(root / "media_plan.json")
    if plan["source_sha256"] != sha(
        Path(__file__).with_name("mission_routing_v4_media.py")
    ) or plan["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Changed predeclared media plan")
    decoded = read(root / "media/verification.json")
    for flight in p["cases"]:
        meta = read(root / "media" / f"{flight}_provenance.json")
        video = root / "media" / f"{flight}_rapid_survey.mp4"
        if (
            meta["policies"] != plan["policies"]
            or meta["implementation_sha256"] != plan["source_sha256"]
        ):
            raise ValueError("Media pair or implementation differs from plan")
        if meta["video_sha256"] != sha(video) or decoded[flight]["sha256"] != sha(video):
            raise ValueError("Changed decoded video")
        if not all(decoded[flight][k] for k in ("first_frame_decoded", "last_frame_decoded")):
            raise ValueError("Video cannot be decoded")
        for name, expected in meta["source_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError("Changed media source: " + name)
    groups = {}
    for row in analysis["cases"]:
        groups.setdefault((row["flight"], row["mission"]), []).append(row["full_success"])
    result = dict(
        schema_version="v4_completion_audit_v1",
        complete=True,
        executions=analysis["executions"],
        reproduced_public_decisions=validation["public_decisions_reproduced"],
        actual_neural_calls=total_calls,
        frame_configuration_pairs=pairs,
        paired_physical_poses_identical=True,
        cases_with_full_success_discrimination=sum(any(v) and not all(v) for v in groups.values()),
        cases=len(groups),
        evaluator_witnesses=analysis["evaluator_witnesses"],
        artifact_sha256={
            name: sha(root / name)
            for name in (
                "protocol.json",
                "validation.json",
                "analysis_v4.json",
                "router_timing.json",
                "risk_transfer.json",
                "physical_route_validation.json",
                "mission_outcomes_provenance.json",
                "risk_tradeoff_provenance.json",
                "media/verification.json",
            )
        },
        source_sha256=sha(Path(__file__)),
        scope="Completed known-map experiment, not real-mission or generalization certification",
    )
    save(root / "completion_audit.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
