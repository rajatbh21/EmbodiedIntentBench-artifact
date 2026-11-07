"""Check development selection and resource artifacts without certifying research readiness."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_policy_selection.protocol import verify
from experiments.mission_policy_selection.study import POLICIES
from experiments.mission_policy_selection_report import rank


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError("Artifact checksum mismatch: " + str(path))


def run(root, resource):
    protocol = verify(root)
    complete = read(root / "completion.json")
    if (
        not complete["complete"]
        or complete["fresh_test"]
        or complete["research_readiness_certified"]
    ):
        raise ValueError("Incorrect selection completion scope")
    for filename, key in (
        ("protocol.json", "protocol_sha256"),
        ("analysis.json", "analysis_sha256"),
        ("REPORT.md", "report_sha256"),
        ("validation.json", "validation_sha256"),
    ):
        checked(root / filename, complete[key])
    checked(
        Path(__file__).with_name("mission_policy_selection_report.py"), complete["source_sha256"]
    )
    analysis = read(root / "analysis.json")
    validation = read(root / "validation.json")
    if analysis["schema_version"] != "mission_policy_selection_analysis_v2":
        raise ValueError("Exact arithmetic report required")
    checked(
        Path(__file__).with_name("mission_policy_selection_report.py"),
        analysis["report_implementation_sha256"],
    )
    if analysis["ranking"] != rank(
        analysis["cases"], protocol["cases"], protocol["missions"], POLICIES
    ):
        raise ValueError("Ranking does not reproduce")
    if analysis["selection_result"]["selected_policy"] != analysis["ranking"][0]["policy"]:
        raise ValueError("Selected policy differs from declared ranking")
    events = 0
    for row in analysis["cases"]:
        destination = root / row["path"]
        for filename, key in (
            ("outcome.json", "outcome_sha256"),
            ("execution/events.json", "events_sha256"),
            ("execution/results.json", "results_sha256"),
        ):
            checked(destination / filename, row[key])
        events += len(read(destination / "execution/events.json"))
    if (
        events != validation["public_decisions_reproduced"]
        or len(analysis["cases"]) != validation["cases"]
    ):
        raise ValueError("Validated grid count differs")
    for name, expected in validation["source_summaries_sha256"].items():
        checked(root / name, expected)
    envelope = read(resource / "envelope.json")
    resource_complete = read(resource / "completion.json")
    if (
        not envelope["complete"]
        or envelope["mission_feasibility_proven"]
        or envelope["prediction_labels_used"]
        or envelope["new_policy_runs"]
    ):
        raise ValueError("Wrong resource analysis scope")
    for path, expected in envelope["source_sha256"].items():
        checked(Path(path), expected)
    checked(resource / "envelope.json", resource_complete["envelope_sha256"])
    checked(resource / "REPORT.md", resource_complete["report_sha256"])
    checked(root / "protocol.json", envelope["selection_protocol_sha256"])
    images = [(root, "policy_comparison"), (resource, "resource_envelope")]
    for folder, name in images:
        provenance = read(folder / f"{name}.provenance.json")
        checked(Path(provenance["source"]), provenance["source_sha256"])
        checked(folder / f"{name}.png", provenance["image_sha256"])
        checked(
            Path(__file__).with_name("mission_policy_selection_plot.py"),
            provenance["plotter_sha256"],
        )
    artifacts = [
        root / n
        for n in (
            "protocol.json",
            "analysis.json",
            "REPORT.md",
            "validation.json",
            "completion.json",
            "policy_comparison.png",
            "policy_comparison.provenance.json",
        )
    ]
    artifacts += [
        resource / n
        for n in (
            "envelope.json",
            "REPORT.md",
            "completion.json",
            "resource_envelope.png",
            "resource_envelope.provenance.json",
        )
    ]
    result = dict(
        complete=True,
        scope="development_selection_and_cost_screen",
        fresh_test=False,
        research_readiness_certified=False,
        policy_replays=len(analysis["cases"]),
        public_decisions=events,
        new_physical_flights=0,
        new_neural_calls=0,
        exact_ranking_reproduced=True,
        selected_policy=analysis["ranking"][0]["policy"],
        resource_conditions=len(envelope["grid"]),
        artifacts_sha256={str(f.resolve()): sha(f) for f in artifacts},
        release_source_sha256=sha(Path(__file__)),
    )
    save(root / "release_audit.json", result)
    print({k: v for k, v in result.items() if k != "artifacts_sha256"}, flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resource", type=Path, required=True)
    args = p.parse_args()
    run(args.output, args.resource)
