"""Export a relocatable, standard-library-only endpoint review capsule.

Raw images, weights and simulator binaries are intentionally outside this capsule.
The full workspace audit reconstructs their scores. This capsule checks immutable
exported events and recomputes the exact canonical endpoint and table denominators.
"""

import argparse
import json
import shutil
from pathlib import Path


def build(study, output):
    from experiments.lake_feasibility_inference import read
    from experiments.lake_missions.assets import save, sha
    from experiments.mission_coverage_repair.protocol import verify

    p = verify(study)
    completion = read(study / "completion.json")
    if not completion["complete"]:
        raise ValueError("Study is incomplete")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(study / name) != expected:
            raise ValueError("Changed completed study")
    analysis = read(study / "analysis.json")
    witnesses = read(study / "witness_validation.json")
    if not witnesses["complete"]:
        raise ValueError("Witness validation incomplete")
    output.mkdir(parents=True, exist_ok=False)
    files = {}

    def copy(source, relative, expected=None):
        digest = sha(source)
        if expected is not None and digest != expected:
            raise ValueError("Changed audited input: " + str(source))
        dest = output / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        if sha(dest) != digest:
            raise ValueError("Export byte mismatch")
        files[relative] = digest

    repo = Path(__file__).resolve().parents[1]
    copy(Path(__file__), "review.py")
    for name in (
        "experiments/__init__.py",
        "experiments/lake_intent_contract.py",
        "experiments/lake_missions/__init__.py",
        "experiments/lake_missions/objectives.py",
    ):
        copy(repo / name, name)
    for name in (
        "protocol.json",
        "completion.json",
        "paper_release.json",
        "validation.json",
        "analysis.json",
        "REPORT.md",
        "interpretation.json",
        "INTERPRETATION.md",
        "energy_robustness.json",
        "host_timing.json",
        "HOST_TIMING.md",
        "recall_transfer.json",
        "RECALL_TRANSFER.md",
        "input_controls.json",
        "witness_validation.json",
        "comparison.png",
        "comparison.pdf",
        "success_matrix.png",
        "figures/benchmark_architecture.pdf",
        "figures/benchmark_architecture.png",
    ):
        copy(study / name, "study/" + name)
    cohort = Path(p["cohort"])
    cohort_protocol = read(cohort / "protocol.json")
    for name in (
        "protocol.json",
        "environment.json",
        "runtime_inventory.json",
        "powercap_probe.json",
        "sensor_gate.json",
        "physical_route_validation.json",
        "FEASIBILITY.md",
        "feasibility_diagnostics.json",
        "ERROR_SEQUENCES.md",
        "error_sequences.json",
    ):
        copy(cohort / name, "cohort/" + name)
    frozen_sources = {**cohort_protocol["implementation_sha256"], **p["repair_source_sha256"]}
    for name, expected in frozen_sources.items():
        source = Path(name)
        relative = source.relative_to(repo).as_posix()
        copy(source, "source_snapshot/" + relative, expected)
    for flight in p["cases"]:
        name = f"validation_{flight}.json"
        copy(study / name, "study/" + name)
        name = f"summary_{flight}_15.json"
        copy(study / name, "study/" + name)
        name = f"witnesses/{flight}/validation.json"
        copy(study / name, "study/" + name, witnesses["independent_audits_sha256"][flight])
    exported = []
    for role, rows in (("policy", analysis["policy_cases"]), ("witness", witnesses["cases"])):
        for row in rows:
            if "path" not in row:
                continue
            relative = f"{role}/{row['flight']}/{row['mission']}/{row['policy'].replace(':', '_')}"
            source = study / row["path"]
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
                ("intent.json", None),
            ):
                copy(source / name, relative + "/" + name, row[key] if key else None)
            exported.append(
                dict(
                    role=role,
                    flight=row["flight"],
                    mission=row["mission"],
                    policy=row["policy"],
                    relative=relative,
                    success=row["full_success"],
                )
            )
    for index, failed in enumerate(p["failed_executions"]):
        path = Path(failed["path"])
        copy(path, f"preserved_failures/{index}/results.json", failed["results_sha256"])
        copy(
            path.with_name("events.json"),
            f"preserved_failures/{index}/events.json",
            failed["events_sha256"],
        )
    for doc in (
        "mission_coverage_routing.md",
        "mission_coverage_reproduction.md",
        "mission_coverage_results.md",
        "paper_evidence.md",
    ):
        copy(repo / "docs" / doc, "docs/" + doc)
    manifest = dict(
        schema_version="mission_endpoint_review_capsule_v1",
        files_sha256=files,
        policy_count=21,
        condition_count=24,
        execution_count=504,
        witness_count=24,
        independent_layouts=3,
        maps=1,
        rows=exported,
        source_study_protocol_sha256=sha(study / "protocol.json"),
        source_completion_sha256=sha(study / "completion.json"),
        scope="Exact endpoint recomputation from audited exported events; "
        "not raw-image, neural, policy-decision or physics reproduction",
    )
    save(output / "manifest.json", manifest)
    (output / "README.txt").write_text(
        "Mission Routing Benchmark endpoint review capsule\n\n"
        "Run with Python 3.11 or later, no installed packages required:\n"
        "    python3 -I -S review.py check\n\n"
        "The capsule can be moved to a different directory. It verifies all content hashes, "
        "recomputes every canonical mission endpoint from exported events, checks the "
        "21-policy x 24-condition denominator, and checks 24 evaluator-only witnesses.\n"
        "It does not execute models, regenerate physical images, reselect policy actions "
        "or independently reconstruct raw image matching. Those stronger checks are "
        "recorded in the complete workspace study and require its assets/runtime.\n"
        "The disclosed numerical repair and four original failures are preserved.\n"
        "The frozen runtime sources and repaired policy are in source_snapshot/. "
        "Their inclusion does not supply the omitted weights or scene assets.\n"
    )
    print(dict(files=len(files), bytes=sum((output / f).stat().st_size for f in files)), flush=True)


def check(root):
    # Keep this path self-contained: isolated Python must not import workspace packages.
    import hashlib
    import sys

    sys.path.insert(0, str(root))

    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != "mission_endpoint_review_capsule_v1":
        raise ValueError("Wrong capsule")
    for name, expected in manifest["files_sha256"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe relative artifact path")
        actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError("Changed capsule artifact: " + name)
    from experiments.lake_intent_contract import IntentContract, evaluate

    scorer_path = Path(sys.modules[IntentContract.__module__].__file__).resolve()
    if not scorer_path.is_relative_to(root.resolve()):
        raise ValueError("Run the capsule's review.py in a fresh isolated Python process")

    rows = manifest["rows"]
    policy_rows = [r for r in rows if r["role"] == "policy"]
    witness_rows = [r for r in rows if r["role"] == "witness"]
    policies = {r["policy"] for r in policy_rows}
    cells = {(r["flight"], r["mission"]) for r in policy_rows}
    keys = [(r["flight"], r["mission"], r["policy"]) for r in policy_rows]
    if (
        len(policy_rows) != 504
        or len(policies) != 21
        or len(cells) != 24
        or len(set(keys)) != 504
        or set(keys) != {(f, m, p) for f, m in cells for p in policies}
    ):
        raise ValueError("Incomplete policy grid")
    if len(witness_rows) != 24 or {(r["flight"], r["mission"]) for r in witness_rows} != cells:
        raise ValueError("Incomplete witness grid")
    summary, witness_successes = {}, 0
    for row in rows:
        relative = Path(row["relative"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe execution path")
        directory = root / row["relative"]
        contract = IntentContract(json.loads((directory / "intent.json").read_text()))
        events = json.loads((directory / "execution/events.json").read_text())
        runtime = json.loads((directory / "execution/results.json").read_text())
        outcome = json.loads((directory / "outcome.json").read_text())
        recomputed = evaluate(contract, events, runtime)
        for key in ("mission_success", "constraints", "failed_constraints", "phases", "reporting"):
            if recomputed[key] != outcome[key]:
                raise ValueError("Canonical endpoint mismatch: " + row["relative"])
        if recomputed["mission_success"] != row["success"]:
            raise ValueError("Exported success mismatch")
        if row["role"] == "policy":
            summary.setdefault(row["policy"], 0)
            summary[row["policy"]] += recomputed["mission_success"]
        else:
            witness_successes += recomputed["mission_success"]
    recorded = json.loads((root / "study/analysis.json").read_text())
    expected = {r["policy"]: r["successes"] for r in recorded["policy_summary"]}
    if summary != expected:
        raise ValueError("Published summary mismatch")
    result = dict(
        complete=True,
        portable_endpoint_reproduction=True,
        policy_executions=504,
        witnesses=24,
        witness_successes=witness_successes,
        policy_successes=summary,
        raw_image_reconstruction=False,
        neural_inference_repeated=False,
        physics_repeated=False,
        policy_decisions_repeated=False,
        isolated_python=bool(sys.flags.isolated),
        site_packages_disabled=bool(sys.flags.no_site),
        scorer_loaded_from=str(scorer_path),
    )
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("build", "check"))
    parser.add_argument("--study", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "build":
        build(args.study, args.output)
    else:
        check(args.output or Path(__file__).resolve().parent)
