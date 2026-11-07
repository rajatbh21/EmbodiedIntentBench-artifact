"""Freeze and execute a fresh factorial replay with isolated parallel workers.

The existing task-deadline runner, scoring, and receipts are reused unchanged.
Only disjoint mission cases run concurrently; within each case, all decisions,
costs, and observations keep their original causal simulation order. Host wall
time is a throughput diagnostic and is never substituted for simulated costs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from experiments.mission_runtime_profile import REPO, read, save, sha
from experiments.mission_task_deadlines import study

OUTPUT = REPO / "results/mission_factor_sweeps_v1_20261028"
CONFIG = Path(__file__).with_name("config.json")
MANIFEST = "orchestration_manifest.json"
_WORKER_STATE = None


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(document):
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_workers(workers):
    require(type(workers) is int and 1 <= workers <= 4, "Use one to four isolated workers")


def validate_analysis_design(config, design):
    """Reject a mislabeled intervention before freezing any input or outcome."""
    require(
        design.get("schema_version") == "mission_factor_sweep_analysis_spec_v1",
        "Unknown analysis design schema",
    )
    factors = ["deadline", "network", "battery", "intent"]
    levels = dict(
        deadline=[profile.name for profile in config.deadline_profiles],
        network=list(config.links),
        battery=list(config.batteries),
        intent=list(config.missions),
    )
    require(design.get("factors") == factors, "Analysis factors differ from execution")
    require(design.get("levels") == levels, "Analysis levels differ from execution")
    require(design.get("flights") == list(config.flights), "Analysis flights differ from execution")
    require(
        design.get("policies") == list(config.policies), "Analysis policies differ from execution"
    )
    require(
        type(design.get("expected_runs")) is int
        and design["expected_runs"] == study.specification(config)["expected_runs"],
        "Analysis expected run count differs from execution",
    )
    baseline = design.get("baseline")
    require(
        isinstance(baseline, dict)
        and set(baseline) == set(factors)
        and all(baseline[factor] in levels[factor] for factor in factors),
        "Analysis baseline must belong to every executed factor",
    )
    perception_budgets = {
        "perception_2s": 2_000_000_000,
        "perception_4s": 4_000_000_000,
        "perception_8s": 8_000_000_000,
    }
    for profile in config.deadline_profiles:
        require(
            profile.name in perception_budgets
            and profile.perception_ns == perception_budgets[profile.name],
            "Perception deadline label does not match the declared invocation budget",
        )
        require(
            profile.association_ns == 150_000_000 and profile.verification_ns == 30_000_000,
            "Only perception deadline varies; downstream budgets must remain fixed",
        )


def canonical_rows(grid, rows):
    """Validate paired cells and restore planned order after unordered execution."""
    require(len(rows) == len(grid), "Incomplete execution grid")
    expected = {item["run_id"]: item for item in grid}
    require(len(expected) == len(grid), "Duplicate planned run ID")
    indexed = {}
    for row in rows:
        run_id = row.get("run_id")
        require(run_id in expected and run_id not in indexed, "Unknown or duplicate result")
        require(
            all(row.get(key) == value for key, value in expected[run_id].items()),
            "Result does not match its planned factorial cell",
        )
        indexed[run_id] = row
    return [indexed[item["run_id"]] for item in grid]


def prepare(root=OUTPUT, *, config_path=CONFIG, workers=4, analysis_design):
    """Freeze all design and orchestration sources before any execution output."""
    validate_workers(workers)
    require(
        isinstance(analysis_design, dict) and analysis_design, "Explicit analysis design required"
    )
    # Validate JSON serializability before creating a frozen study directory.
    design_hash = digest(analysis_design)
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = study.StudyConfig.from_dict(read(config_path))
    validate_analysis_design(config, analysis_design)
    require(config_path.is_relative_to(REPO), "Factorial configuration must be inside repository")
    plan = study.prepare(root, config)
    sources = {
        *Path(__file__).parent.glob("*.py"),
        CONFIG,
        config_path,
        *REPO.glob("tests/test_factor_sweep*.py"),
    }
    implementation, snapshots = {}, {}
    for source in sorted(sources):
        source = source.resolve()
        require(source.is_relative_to(REPO), "Orchestration sources must be inside repository")
        expected = sha(source)
        relative = Path("orchestration_source_snapshot") / source.relative_to(REPO)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        require(sha(destination) == expected, "Source changed during orchestration freeze")
        implementation[str(source)] = expected
        snapshots[str(relative)] = expected
    save(root / "analysis_design.json", analysis_design)
    save(
        root / MANIFEST,
        dict(
            schema_version="factor_sweep_orchestration_v1",
            plan_sha256=sha(root / "plan.json"),
            config_path=str(config_path),
            config_sha256=sha(config_path),
            expected_runs=plan["expected_runs"],
            worker_count=workers,
            process_start_method="fork" if workers > 1 else "in_process",
            ordering="Cases execute independently; the final index follows the frozen grid.",
            fresh_execution=True,
            historical_outputs_reused=False,
            shared_inputs="Verified read-only RGB, encoded payload, terrain, and manifests.",
            analysis_design=analysis_design,
            analysis_design_sha256=design_hash,
            analysis_design_file_sha256=sha(root / "analysis_design.json"),
            implementation_sha256=implementation,
            snapshot_sha256=snapshots,
            prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            outcome_files_exist_at_freeze=False,
        ),
    )
    return verify(root)


def verify(root):
    root = Path(root).resolve()
    plan = study.verify(root)
    manifest = read(root / MANIFEST)
    require(manifest["schema_version"] == "factor_sweep_orchestration_v1", "Unknown orchestrator")
    require(manifest["plan_sha256"] == sha(root / "plan.json"), "Changed execution plan")
    require(manifest["expected_runs"] == plan["expected_runs"], "Changed expected run count")
    validate_workers(manifest["worker_count"])
    require(
        manifest["process_start_method"]
        == ("fork" if manifest["worker_count"] > 1 else "in_process"),
        "Changed process isolation method",
    )
    require(
        manifest["fresh_execution"] is True
        and manifest["historical_outputs_reused"] is False
        and manifest["outcome_files_exist_at_freeze"] is False,
        "Incorrect prospective fresh-execution declaration",
    )
    require(
        sha(manifest["config_path"]) == manifest["config_sha256"]
        and study.StudyConfig.from_dict(read(manifest["config_path"])).to_dict() == plan["factors"],
        "Changed factorial configuration",
    )
    require(
        digest(manifest["analysis_design"]) == manifest["analysis_design_sha256"]
        and read(root / "analysis_design.json") == manifest["analysis_design"]
        and sha(root / "analysis_design.json") == manifest["analysis_design_file_sha256"],
        "Changed analysis design",
    )
    validate_analysis_design(
        study.StudyConfig.from_dict(plan["factors"]), manifest["analysis_design"]
    )
    for path, expected in manifest["implementation_sha256"].items():
        require(sha(path) == expected, "Changed orchestration source: " + path)
    for relative, expected in manifest["snapshot_sha256"].items():
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root) and sha(path) == expected,
            "Changed or escaped orchestration source snapshot",
        )
    if (root / "execution_started.json").exists():
        started = read(root / "execution_started.json")
        require(
            started["orchestration_manifest_sha256"] == sha(root / MANIFEST),
            "Manifest changed after start",
        )
    if (root / "execution_receipt.json").exists():
        receipt = read(root / "execution_receipt.json")
        require(
            receipt["complete"] is True
            and receipt["fresh_runs"] == plan["expected_runs"]
            and receipt["historical_outputs_reused"] is False
            and receipt["worker_count"] == manifest["worker_count"]
            and receipt["wall_time_used_for_simulated_costs"] is False,
            "Changed completed execution declaration",
        )
        for name, expected in receipt["artifacts_sha256"].items():
            path = (root / name).resolve()
            require(
                path.is_relative_to(root) and sha(path) == expected,
                "Changed execution receipt binding",
            )
    return plan, manifest


def load_shared_inputs(plan):
    """Validate recordings once, before workers inherit immutable input banks."""
    parent_root = Path(plan["parent_root"])
    parent = read(parent_root / "plan.json")
    terrain, _ = study.load_public_terrain(Path(parent["assets"]))
    banks, manifests = {}, {}
    for item in plan["grid"]:
        flight = item["flight"]
        if flight in banks:
            continue
        scenario = study.scenario_for(parent_root, parent, item["cell"])
        banks[flight] = study.frame_bank(scenario, Path(parent["cohort"]) / flight / "recording")
        manifests[flight] = study.checked_manifest(
            Path(parent["cohort"]) / flight / "predictions",
            Path(parent["cohort"]) / "protocol.json",
        )
    return terrain, banks, manifests


def _execute_item(item):
    require(_WORKER_STATE is not None, "Missing inherited immutable execution inputs")
    root, plan, terrain, banks, manifests = _WORKER_STATE
    return study.execute_case(
        root, plan, item, terrain, banks[item["flight"]], manifests[item["flight"]]
    )


def execute(root=OUTPUT):
    """Run every planned case freshly; failed or partial attempts are preserved."""
    global _WORKER_STATE
    root = Path(root).resolve()
    plan, manifest = verify(root)
    for name in (
        "runs",
        "run_index.json",
        "execution_started.json",
        "execution_failure.json",
        "execution_receipt.json",
        "completion.json",
        "progress.json",
    ):
        require(not (root / name).exists(), "Preserve previous execution attempt: " + name)
    workers = manifest["worker_count"]
    if workers > 1:
        require(
            "fork" in multiprocessing.get_all_start_methods(),
            "This frozen plan requires fork workers",
        )
    started = time.perf_counter()
    save(
        root / "execution_started.json",
        dict(
            orchestration_manifest_sha256=sha(root / MANIFEST),
            plan_sha256=sha(root / "plan.json"),
            worker_count=workers,
            started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        ),
    )
    rows = []

    def record(row):
        rows.append(row)
        save(
            root / "progress.json",
            dict(
                completed=len(rows),
                expected=plan["expected_runs"],
                completed_run_ids=sorted(r["run_id"] for r in rows),
                elapsed_wall_s=time.perf_counter() - started,
            ),
        )
        print(f"{len(rows)}/{plan['expected_runs']} {row['run_id']}", flush=True)

    try:
        terrain, banks, manifests = load_shared_inputs(plan)
        _WORKER_STATE = (root, plan, terrain, banks, manifests)
        if workers == 1:
            for item in plan["grid"]:
                record(_execute_item(item))
        else:
            context = multiprocessing.get_context("fork")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
                futures = [pool.submit(_execute_item, item) for item in plan["grid"]]
                try:
                    for future in as_completed(futures):
                        record(future.result())
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        rows = canonical_rows(plan["grid"], rows)
        verify(root)
        save(
            root / "run_index.json",
            dict(plan_sha256=sha(root / "plan.json"), cases=rows, complete_grid=True),
        )
        save(
            root / "execution_receipt.json",
            dict(
                schema_version="factor_sweep_execution_receipt_v1",
                complete=True,
                fresh_runs=len(rows),
                historical_outputs_reused=False,
                worker_count=workers,
                wall_time_s=time.perf_counter() - started,
                wall_time_used_for_simulated_costs=False,
                artifacts_sha256={
                    name: sha(root / name)
                    for name in ("plan.json", MANIFEST, "execution_started.json", "run_index.json")
                },
            ),
        )
        return rows
    except BaseException as error:
        save(
            root / "execution_failure.json",
            dict(
                complete=False,
                exception_type=type(error).__name__,
                message=str(error),
                collected_run_ids=sorted(r["run_id"] for r in rows),
                note="Every run directory and partial artifact is preserved; "
                "do not retry in place.",
            ),
        )
        raise
    finally:
        _WORKER_STATE = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "summarize", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        from experiments.mission_factor_sweeps.analysis import specification

        prepare(
            args.output,
            config_path=args.config,
            workers=args.workers,
            analysis_design=specification(),
        )
    elif args.action == "execute":
        execute(args.output)
    elif args.action == "summarize":
        verify(args.output)
        study.summarize(args.output)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
