"""Correct reporting references without changing any frozen latency measurement."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.mission_execution_profile import OUTPUT as SOURCE
from experiments.mission_execution_profile import verify
from experiments.mission_runtime_profile import REPO, read, save, sha, summarize

OUTPUT = REPO / "results/mission_execution_profile_reporting_v1_20261027"
REFERENCES = {"declared_s", "above_declared_count", "above_declared_fraction"}


def corrected(values, original, reference):
    result = summarize(values, declared_s=reference)
    if any(result[k] != v for k, v in original.items() if k not in REFERENCES):
        raise ValueError("Reporting correction would change measured statistics")
    return result


def derive(profile, rows, tracking):
    result = {}
    for name, item in profile["profiles"].items():
        warm = [r for r in rows if r["config_id"] == name and r["stage"] == "warm"]
        result[name] = {
            domain: corrected([r[key] for r in warm], item[domain], item["declared_latency_s"])
            for domain, key in (("warm_wall", "wall_s"), ("warm_process", "process_s"))
        }
    warm = [r for trace in tracking for r in trace["samples"] if not r["instance_cold"]]
    tracking_result = {
        domain: corrected([r[key] for r in warm], profile["tracking"][domain], 0.1)
        for domain, key in (("warm_wall", "wall_s"), ("warm_process", "process_s"))
    }
    return dict(profiles=result, tracking=tracking_result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new reporting directory; original artifacts stay unchanged")
    profile = verify(args.source)
    filenames = (
        "profile.json",
        "inference_samples.json",
        "tracking_samples.json",
        "plan.json",
        "input_lock.json",
        "completion.json",
    )
    inputs = {str(args.source / name): sha(args.source / name) for name in filenames}
    report = dict(
        schema_version="mission_execution_profile_reference_erratum_v1",
        source=str(args.source.resolve()),
        inputs_sha256=inputs,
        generator_sha256={str(Path(__file__).resolve()): sha(__file__)},
        cause="The reused summary helper defaulted to the routing reference of 0.05 seconds.",
        correction="Recompute only declared_s and above_declared_* reference fields using "
        "each configuration declared_latency_s and the configured tracking latency of 0.1 s.",
        interpretation="Wall reference counts compare to the configured cost, not a deadline. "
        "Process-time counts are arithmetic comparisons of summed CPU time; they are not "
        "elapsed-latency or deadline failures.",
        no_remeasurement=True,
        original_artifacts_unchanged=True,
        latency_s_unchanged=profile["latency_s"],
        tracking_p95_unchanged=profile["tracking"]["warm_p95_s"],
        corrected=derive(
            profile,
            read(args.source / "inference_samples.json"),
            read(args.source / "tracking_samples.json"),
        ),
    )
    if any(sha(path) != expected for path, expected in inputs.items()):
        raise ValueError("Original artifacts changed while deriving report")
    save(args.output / "report.json", report)
    print(str(args.output / "report.json"))


if __name__ == "__main__":
    main()
