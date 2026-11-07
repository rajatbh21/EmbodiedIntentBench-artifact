"""Verify a complete fresh-layout artifact without certifying research readiness."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_routing_v6.flights import verify
from experiments.mission_routing_v6.sensor import verify_gate


def run(root):
    p = verify(root)
    verify_gate(root)
    analysis = read(root / "analysis_v6.json")
    validation = read(root / "validation.json")
    timing = read(root / "router_timing.json")
    figure = read(root / "policy_comparison_provenance.json")
    preview = read(root / "physical_preview_provenance.json")
    diagnostics = read(root / "feasibility_diagnostics.json")
    if (
        not analysis["complete"]
        or not validation["complete"]
        or analysis["new_paid_replays"] != 90
        or validation["cases"] != 90
        or analysis["reproduced_public_decisions"] != validation["events"]
        or validation["events"] != validation["public_decisions_reproduced"]
        or analysis["actual_neural_calls"] != validation["actual_neural_calls"]
        or analysis["actual_neural_calls"] != 2040
        or validation["frame_config_pairs"] != 1080
        or analysis["protocol_sha256"] != sha(root / "protocol.json")
        or analysis["validation_sha256"] != sha(root / "validation.json")
        or timing["protocol_sha256"] != sha(root / "protocol.json")
        or timing["validation_sha256"] != sha(root / "validation.json")
        or figure["analysis_sha256"] != sha(root / "analysis_v6.json")
        or figure["image_sha256"] != sha(root / "policy_comparison.png")
        or preview["protocol_sha256"] != sha(root / "protocol.json")
        or preview["image_sha256"] != sha(root / "physical_preview.png")
    ):
        raise ValueError("Incomplete or mismatched evaluation artifacts")
    for artifact, key, filename in (
        (analysis, "reporter_sha256", "mission_routing_v6_report.py"),
        (timing, "source_sha256", "mission_routing_v6_timing.py"),
        (figure, "source_sha256", "mission_routing_v6_plot.py"),
        (preview, "source_sha256", "mission_routing_v6_preview.py"),
    ):
        if artifact[key] != sha(Path(__file__).with_name(filename)):
            raise ValueError("Changed artifact implementation")
    for key, filename in (
        ("sensor_gate_sha256", "sensor_gate.json"),
        ("diagnostics_sha256", "feasibility_diagnostics.json"),
        ("witness_summary_sha256", "witnesses/summary.json"),
    ):
        if analysis[key] != sha(root / filename):
            raise ValueError("Changed evaluated evidence")
    for filename, expected in analysis["summaries_sha256"].items():
        if (
            sha(root / filename) != expected
            or validation["source_summaries_sha256"][filename] != expected
        ):
            raise ValueError("Changed validated summary")
        for row in read(root / filename)["cases"]:
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                if sha(root / row["path"] / name) != row[key]:
                    raise ValueError("Changed validated execution")
    files = [
        "protocol.json",
        "sensor_gate.json",
        "physical_route_validation.json",
        "validation.json",
        "analysis_v6.json",
        "REPORT_V6.md",
        "feasibility_diagnostics.json",
        "FEASIBILITY.md",
        "witnesses/protocol.json",
        "witnesses/summary.json",
        "WITNESSES.md",
        "router_timing.json",
        "policy_comparison.png",
        "policy_comparison_provenance.json",
        "physical_preview.png",
        "physical_preview_provenance.json",
    ]
    for flight in p["cases"]:
        pred = root / flight / "predictions"
        manifest = checked_manifest(pred, root / "protocol.json")
        if len(manifest["models"]) != 9 or manifest["capture_count"] != 60:
            raise ValueError("Incomplete actual inference grid")
        if diagnostics["flights"][flight]["source_manifest_sha256"] != sha(pred / "manifest.json"):
            raise ValueError("Changed evaluated prediction manifest")
        for model in manifest["models"].values():
            for prediction in model["predictions"].values():
                if sha(pred / prediction["file"]) != prediction["sha256"]:
                    raise ValueError("Changed neural prediction file")
        files.extend(
            [
                f"{flight}/predictions/{n}"
                for n in ("manifest.json", "completion.json", "inference.json")
            ]
        )
        files.append(f"{flight}/recording/recording.json")
        media = read(root / f"media/{flight}_provenance.json")
        if (
            media["implementation_sha256"]
            != sha(Path(__file__).with_name("mission_routing_v6_media.py"))
            or media["video_sha256"] != sha(root / f"media/{flight}_rapid_survey.mp4")
            or media["gt_boxes_used"]
            or not media["show_prediction_only_after_completion"]
            or media["policies"] != ["base_recall", "original_base_recall"]
        ):
            raise ValueError("Changed or mislabeled causal media")
        for name, expected in media["source_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError("Changed replay media source")
        files.extend(
            f"media/{flight}_{suffix}"
            for suffix in ("provenance.json", "rapid_survey.mp4", "summary.png")
        )
    result = dict(
        schema_version="fresh_layout_completion_audit_v6",
        complete=True,
        scope=p["scope"],
        new_physical_flights=2,
        new_captures=120,
        actual_neural_calls=2040,
        frame_config_pairs=1080,
        new_paid_replays=90,
        additional_evaluator_witness_replays=6,
        reproduced_public_decisions=validation["events"],
        primary_policy=p["primary_policy"],
        successes_by_policy=analysis["successes_by_policy"],
        feasibility_by_case=analysis["feasibility_by_case"],
        research_readiness_certified=False,
        population_generalization=False,
        artifacts_sha256={n: sha(root / n) for n in files},
        source_sha256=sha(Path(__file__)),
    )
    save(root / "completion_audit.json", result)
    print(
        {k: result[k] for k in ("complete", "new_paid_replays", "reproduced_public_decisions")},
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
