"""Run the artifact smoke check or reproduce all released validation tables.

CPU-only stored-prediction replay. The original algorithms, contracts, costs,
and reference artifacts are unchanged. Every invocation uses a fresh output
directory; supplied examples and prior executions are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STAGES = ("parity", "bounds", "reruns", "causal", "hindsight", "verify", "sweep")
REPORT_AGGREGATION = {
    "method": "sequential_addition_matching_python_3_10",
    "scope": "Only experiments.validation.analysis.sum during report generation",
    "reason": "Python 3.12 changed floating-point built-in sum; preserve original report aggregation exactly.",
    "runtime_and_policy_algorithms_changed": False,
    "comparison_tolerance": 0,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declared_source_changes(repo=REPO):
    """Allow only the explicitly documented presentation-source transformation."""
    path = Path(repo) / "docs/release_source_changes.json"
    if not path.exists():
        return {}
    document = json.loads(path.read_text())
    if document.get("schema_version") != "release_source_changes_v1":
        raise ValueError("Unknown release-source provenance schema")
    changes = document["files"]
    if set(changes) != {"figures.py"}:
        raise ValueError("Only the declared figure-presentation source may differ")
    for row in changes.values():
        if set(row) != {"original_sha256", "released_sha256", "reason"} or not row["reason"]:
            raise ValueError("Incomplete release-source provenance")
        for key in ("original_sha256", "released_sha256"):
            if len(row[key]) != 64 or any(c not in "0123456789abcdef" for c in row[key]):
                raise ValueError("Invalid release-source digest")
    return changes


def verify_reference_sources(repo=REPO):
    """Bind the receipt to original sources or the exact declared presentation edit."""
    repo = Path(repo)
    reference = repo / "examples/validation"
    summary = json.loads((reference / "summary.json").read_text())
    source = repo / "experiments/validation"
    changes = declared_source_changes(repo)
    if not set(changes) <= set(summary["source_sha256"]):
        raise ValueError("Release provenance has no original source binding")
    if digest(source / "config.json") != summary["config_sha256"]:
        raise ValueError("Validation configuration differs from the published reference")
    for name, expected in summary["source_sha256"].items():
        if name in changes:
            change = changes[name]
            if expected != change["original_sha256"]:
                raise ValueError("Original source digest differs from release provenance: " + name)
            expected = change["released_sha256"]
        if Path(name).name != name or digest(source / name) != expected:
            raise ValueError("Validation source differs from the published reference: " + name)
    paths = sorted(reference.glob("*.csv")) + [
        reference / "exp3_timelines.json", reference / "summary.json",
    ]
    if not any(p.name == "runs.csv" for p in paths):
        raise ValueError("Missing published validation run table")
    return {p.name: digest(p) for p in paths}


def compare_reports(reference, generated, source_changes=None):
    """Require every published CSV and JSON table to regenerate exactly."""
    reference, generated = Path(reference), Path(generated)
    names = sorted(p.name for p in reference.glob("*.csv"))
    differences = []
    expected_csv = set(names)
    actual_csv = {p.name for p in generated.glob("*.csv")}
    for name in sorted(expected_csv ^ actual_csv):
        differences.append({"file": name, "reason": "CSV file set differs"})
    for name in names + ["exp3_timelines.json", "summary.json"]:
        path = generated / name
        if not path.exists():
            differences.append({"file": name, "reason": "Missing generated table"})
            continue
        if name.endswith(".json"):
            expected = json.loads((reference / name).read_text())
            actual = json.loads(path.read_text())
            provenance_valid = True
            if name == "summary.json":
                for source_name, change in (source_changes or {}).items():
                    if (expected.get("source_sha256", {}).get(source_name) == change["original_sha256"]
                            and actual.get("source_sha256", {}).get(source_name) == change["released_sha256"]):
                        expected["source_sha256"][source_name] = change["released_sha256"]
                    else:
                        provenance_valid = False
            same = provenance_valid and expected == actual
        else:
            same = (reference / name).read_bytes() == path.read_bytes()
        if not same:
            differences.append({"file": name, "reason": "Generated values differ"})
    return differences


def sequential_sum(values, start=0):
    """Match the original report's Python 3.10 ordered numeric addition."""
    total = start
    for value in values:
        total += value
    return total


def build_reports(raw, report):
    """Keep report arithmetic portable without changing any execution module."""
    from experiments.validation import analysis

    absent = object()
    original = vars(analysis).get("sum", absent)
    analysis.sum = sequential_sum
    try:
        analysis.build(raw, report)
    finally:
        if original is absent:
            del analysis.sum
        else:
            analysis.sum = original


def recheck(existing, output, execution_driver_sha256):
    """Resume reporting from a complete execution; preserve its original receipt."""
    from tools import replay

    existing, output = Path(existing).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory; prior results are preserved")
    if (len(execution_driver_sha256) != 64
            or any(c not in "0123456789abcdef" for c in execution_driver_sha256)):
        raise ValueError("Provide the original execution driver SHA-256")
    references, changes = verify_reference_sources(), declared_source_changes()
    original_path = existing / "artifact_validation.json"
    if not original_path.exists():
        original_path = existing / "reviewer_validation.json"  # Compatibility with earlier runs.
    original = replay.read(original_path)
    if (original["mode"] != "full" or set(original["stages"]) != set(STAGES) | {"analyze"}
            or original["reference_table_sha256"] != references
            or original["release_source_changes"] != changes
            or original["baseline"]["runs"] != 648
            or original["baseline"]["exact_scientific_parity"] is not True):
        raise ValueError("Existing execution is incomplete or its reference bindings changed")
    baseline_root, raw = existing / "replay", existing / "validation"
    replay.verify_data(baseline_root)
    baseline = replay.read(baseline_root / "validation.json")
    sources = replay.source_hashes()
    if (baseline["source_hashes"] != sources or not baseline["complete"]
            or not baseline["exact_scientific_parity"] or baseline["runs"] != 648
            or baseline["plan_sha256"] != digest(baseline_root / "plan.json")
            or baseline["rows_sha256"] != digest(baseline_root / "rows.json")
            or baseline["bundle_sha256"] != digest(baseline_root / "manifest.json")
            or digest(baseline_root / "manifest.json") != digest(replay.DEFAULT_DATA / "manifest.json")
            or replay.compare_rows(replay.read(baseline_root / "rows.json"),
                                   replay.read(baseline_root / "expected_rows.json"))):
        raise ValueError("Existing baseline evidence or execution sources changed")
    parity = replay.read(raw / "parity.json")
    witness = replay.read(raw / "witness_verification.json")
    sweep = replay.read(raw / "sweep_receipt.json")
    if (parity["runs"] != 648 or not all(parity[k] for k in
            ("exact_row_parity", "outcome_parity", "leakage_passed"))
            or parity["downstream_execution_failures"] != 0
            or not witness["all_reproduced"]
            or witness["witnesses"] != len(witness["checks"])
            or not all(c["identical_row"] and c["success"] for c in witness["checks"])
            or sweep["runs"] != 648 or sweep["settings"] != 144
            or not all(sweep["consistency"].values())):
        raise ValueError("Existing validation stage checks did not pass")
    evidence = [original_path, baseline_root / "validation.json"] + sorted(raw.glob("*"))
    hashes = {str(p.relative_to(existing)): digest(p) for p in evidence if p.is_file()}
    receipt = dict(
        mode="recheck", complete=False,
        operation="Resumed reporting after all scientific stages completed; no experiment rerun",
        original_execution_receipt_sha256=digest(original_path),
        original_execution_complete=original["complete"],
        execution_driver_sha256=execution_driver_sha256,
        reporting_driver_sha256=digest(__file__),
        reference_table_sha256=references, release_source_changes=changes,
        execution_source_sha256=sources, preserved_evidence_sha256=hashes,
        report_aggregation=REPORT_AGGREGATION, experiment_executions=0,
        original_execution_elapsed_host_s=original["elapsed_host_s"],
        original_execution_stages=original["stages"], baseline=original["baseline"],
        hindsight={k: witness[k] for k in ("witnesses", "all_reproduced")},
    )
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        build_reports(raw, output)
        differences = compare_reports(REPO / "examples/validation", output, changes)
        receipt.update(exact_validation_table_parity=not differences, table_differences=differences)
        if differences:
            raise ValueError("Regenerated validation reports differ")
        if (hashes != {str(p.relative_to(existing)): digest(p) for p in evidence if p.is_file()}
                or sources != replay.source_hashes() or references != verify_reference_sources()
                or changes != declared_source_changes()):
            raise ValueError("Evidence or sources changed during reporting")
        receipt["generated_table_sha256"] = {name: digest(output / name) for name in references}
        receipt["complete"] = True
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["elapsed_host_s"] = time.perf_counter() - started
        replay.save(output / "artifact_recheck.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                     ("mode", "complete", "exact_validation_table_parity", "elapsed_host_s")}, indent=2))
    return receipt


def run(mode, output=None, workers=2, figures=False):
    from tools import replay

    if mode not in {"verify", "smoke", "full"}:
        raise ValueError("Choose verify, smoke or full")
    if type(workers) is not int or workers < 1:
        raise ValueError("Workers must be a positive integer")
    if figures and mode != "full":
        raise ValueError("Figures require the full validation tables")
    if output is None and mode != "verify":
        raise ValueError("Smoke and full runs require an output directory")
    output = Path(output).resolve() if output is not None else None
    if output is not None and output.exists():
        raise ValueError("Use a new output directory; prior results are preserved")
    references = verify_reference_sources()
    changes = declared_source_changes()
    plan = replay.verify_data(replay.DEFAULT_DATA)
    if mode == "verify":
        receipt = dict(mode=mode, complete=True, declared_runs=len(plan["grid"]),
                       reference_table_sha256=references, release_source_changes=changes,
                       experiment_executions=0)
        if output is not None:
            output.mkdir(parents=True, exist_ok=False)
            replay.save(output / "artifact_validation.json", receipt)
        print("Verified replay inputs, 648-cell grid and validation sources; inventoried reference tables")
        return receipt
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    receipt = dict(
        mode=mode, complete=False, reference_table_sha256=references,
        release_source_changes=changes,
        fresh_neural_inference=False, new_simulator_flight=False,
        stages={},
    )
    try:
        baseline = replay.run(
            replay.DEFAULT_DATA, output / "replay", all_runs=mode == "full", workers=workers,
        )
        receipt["baseline"] = {
            key: baseline[key] for key in ("runs", "exact_scientific_parity", "elapsed_host_s")
        }
        if mode == "full":
            from experiments.validation import study

            raw, report = output / "validation", output / "tables"
            raw.mkdir()
            # Only the input location changes. The original study functions,
            # frozen config, runtime, scoring and search limits stay unchanged.
            original_data_root = study.data_root
            study.data_root = lambda: output / "replay"
            try:
                for stage in STAGES:
                    mark = time.perf_counter()
                    getattr(study, "stage_" + stage)(raw, workers)
                    receipt["stages"][stage] = {"elapsed_host_s": time.perf_counter() - mark}
                    print(f"Completed {stage}", flush=True)
                mark = time.perf_counter()
                build_reports(raw, report)
                receipt["report_aggregation"] = REPORT_AGGREGATION
                receipt["stages"]["analyze"] = {"elapsed_host_s": time.perf_counter() - mark}
            finally:
                study.data_root = original_data_root
            differences = compare_reports(REPO / "examples/validation", report, changes)
            receipt.update(exact_validation_table_parity=not differences, table_differences=differences)
            if differences:
                raise ValueError("Validation tables differ; inspect artifact_validation.json")
            witness = json.loads((raw / "witness_verification.json").read_text())
            receipt["hindsight"] = {key: witness[key] for key in ("witnesses", "all_reproduced")}
            if not witness["all_reproduced"]:
                raise ValueError("Hindsight witness verification failed")
            if figures:
                subprocess.run(
                    [sys.executable, "-m", "experiments.validation.figures", "--report", str(report)],
                    cwd=REPO, check=True,
                )
        if verify_reference_sources() != references or declared_source_changes() != changes:
            raise ValueError("Published reference artifacts changed during validation")
        receipt["complete"] = True
    except BaseException as exc:
        receipt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        receipt["elapsed_host_s"] = time.perf_counter() - started
        replay.save(output / "artifact_validation.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("mode", "complete", "elapsed_host_s")}, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("verify", "smoke", "full", "recheck"))
    parser.add_argument("--output", type=Path, help="New output directory; required for smoke/full")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--figures", action="store_true", help="Also render figures; requires matplotlib")
    parser.add_argument("--existing-run", type=Path, help="Completed full execution; recheck only")
    parser.add_argument("--execution-driver-sha256", help="Recorded original execution-driver SHA-256; recheck only")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.mode != "verify" and args.output is None:
        parser.error("--output is required for smoke/full")
    if args.mode == "recheck":
        if args.existing_run is None or args.execution_driver_sha256 is None or args.figures:
            parser.error("recheck requires --existing-run and --execution-driver-sha256; no --figures")
        recheck(args.existing_run, args.output, args.execution_driver_sha256)
    else:
        if args.existing_run is not None or args.execution_driver_sha256 is not None:
            parser.error("Existing execution options require recheck mode")
        run(args.mode, args.output, args.workers, args.figures)


if __name__ == "__main__":
    main()
