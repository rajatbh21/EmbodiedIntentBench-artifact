"""Audit completion of the exploratory revision without claiming fresh evaluation."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v5.protocol import verify


def run(root):
    p = verify(root)
    analysis, validation = read(root / "analysis_v5.json"), read(root / "validation.json")
    timing, figure = (
        read(root / "router_timing.json"),
        read(root / "recall_tradeoff_provenance.json"),
    )
    if (
        not analysis["complete"]
        or not validation["complete"]
        or analysis["fresh_test"]
        or analysis["new_paid_replays"] != validation["cases"]
        or validation["events"] != validation["public_decisions_reproduced"]
        or analysis["reproduced_public_decisions"] != validation["events"]
        or analysis["protocol_sha256"] != sha(root / "protocol.json")
        or analysis["validation_sha256"] != sha(root / "validation.json")
        or timing["validation_sha256"] != sha(root / "validation.json")
        or timing["protocol_sha256"] != sha(root / "protocol.json")
        or figure["analysis_sha256"] != sha(root / "analysis_v5.json")
        or figure["image_sha256"] != sha(root / "recall_tradeoff.png")
    ):
        raise ValueError("Incomplete or mismatched exploratory artifacts")
    for artifact, key, filename in (
        (analysis, "reporter_sha256", "mission_routing_v5_report.py"),
        (timing, "source_sha256", "mission_routing_v5_timing.py"),
        (figure, "source_sha256", "mission_routing_v5_plot.py"),
    ):
        if artifact[key] != sha(Path(__file__).with_name(filename)):
            raise ValueError("Changed analysis implementation")
    for filename, expected in analysis["summaries_sha256"].items():
        if (
            sha(root / filename) != expected
            or validation["source_summaries_sha256"][filename] != expected
        ):
            raise ValueError("Changed independently validated summary")
        for row in read(root / filename)["cases"]:
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                if sha(root / row["path"] / name) != row[key]:
                    raise ValueError("Changed validated execution")
    files = (
        "protocol.json",
        "analysis_v5.json",
        "validation.json",
        "router_timing.json",
        "REPORT_V5.md",
        "recall_tradeoff.png",
        "recall_tradeoff_provenance.json",
    )
    result = dict(
        schema_version="conditional_recall_completion_audit_v1",
        complete=True,
        scope=p["scope"],
        fresh_test=False,
        new_physical_recordings=0,
        new_neural_calls=0,
        new_paid_replays=validation["cases"],
        reproduced_public_decisions=validation["events"],
        all_new_policies_successes=analysis["successes_by_policy"],
        readiness=analysis["readiness"],
        artifacts_sha256={n: sha(root / n) for n in files},
        source_sha256=sha(Path(__file__)),
    )
    save(root / "completion_audit.json", result)
    print(
        {
            k: result[k]
            for k in ("complete", "fresh_test", "new_paid_replays", "reproduced_public_decisions")
        }
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
