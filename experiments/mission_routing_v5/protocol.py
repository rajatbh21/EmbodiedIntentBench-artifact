"""Lock exploratory reuse of an inspected cohort; never label it prospective."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v4.flights import verify as verify_source
from experiments.mission_routing_v5.train import verify as verify_training


def implementation():
    return {p.name: sha(p) for p in Path(__file__).parent.glob("*.py")}


def prepare(source, training, output):
    if output.exists():
        raise FileExistsError("Preserve existing exploratory cohort")
    p = verify_source(source)
    verify_training(training)
    p.update(
        schema_version="conditional_recall_exploratory_v5",
        scope="Exploratory reuse of inspected v4 physical recordings; NOT fresh test",
        source=str(source.resolve()),
        source_protocol_sha256=sha(source / "protocol.json"),
        recall_training=str(training.resolve()),
        recall_profile_sha256=sha(training / "profile.json"),
        new_implementation_sha256=implementation(),
        policies=["spatial_recall", "base_recall", "global_recall"],
        evaluation_labels_used_for_fit=False,
        design_informed_by_inspected_v4_failures=True,
        active_cpu_power_sensitivity_w=[15],
        new_physical_recordings=False,
        new_neural_inference=False,
    )
    output.mkdir(parents=True)
    save(output / "protocol.json", p)
    verify(output)


def verify(output):
    p = read(output / "protocol.json")
    if p["schema_version"] != "conditional_recall_exploratory_v5":
        raise ValueError("Wrong exploratory protocol")
    if implementation() != p["new_implementation_sha256"]:
        raise ValueError("Changed v5 implementation")
    source, training = Path(p["source"]), Path(p["recall_training"])
    upstream = verify_source(source)
    for field in ("cases", "missions", "screen", "risk_training", "assets"):
        if p[field] != upstream[field]:
            raise ValueError("Changed inherited experiment condition")
    if p["policies"] != ["spatial_recall", "base_recall", "global_recall"]:
        raise ValueError("Changed policy grid")
    verify_training(training)
    if (
        sha(source / "protocol.json") != p["source_protocol_sha256"]
        or sha(training / "profile.json") != p["recall_profile_sha256"]
    ):
        raise ValueError("Changed source cohort or recall profile")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.training, args.output)
