"""Command-line entry point for the four validation experiments.

Stages (run in this order; each writes under ``--output``, default
``results/validation``; raw outputs are local-only):

  parity     648 reference cells through the instrumented runtime: exact row
             parity with expected_rows.json, frozen-outcome parity, leakage
             audit, and evaluation-only rescoring under every contract variant
             and relaxation (Experiments 1a and 4).
  bounds     Per-capture GT/model tables and feasibility bounds (Experiment 2A).
  reruns     648 cells x each runtime-supported non-reference contract, with the
             contract as public input (Experiment 1b).
  causal     648 cells x {causal, no_skip} x latency multipliers (Experiment 3).
  hindsight  Witness search for conditions no released policy satisfied
             (Experiment 2B).
  verify     Re-execute every witness in fresh workers; must reproduce exactly.
  sweep      Sensitivity map: rescore reference histories under 144 declared
             threshold settings (Experiment 1c).
  analyze    Tables, CSVs and figures-data under ``--report`` (committed).

Usage: python -m experiments.validation.study <stage> [--workers N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.validation import contracts, execution  # noqa: E402

CONFIG = contracts.CONFIG
DEFAULT_OUTPUT = REPO / "results/validation"
DEFAULT_REPORT = REPO / "examples/validation"


def _progress(label):
    started = time.perf_counter()

    def report(done, total):
        if done == total or done % 200 < 12:
            print(f"[{label}] {done}/{total} ({time.perf_counter() - started:.0f}s)", flush=True)

    return report


def data_root():
    return REPO / CONFIG["data_root"]


def grid():
    return json.loads((data_root() / "plan.json").read_text())["grid"]


def condition_key(item):
    return "/".join(
        (item["flight"], item["intent_schedule"], item["link"], item["energy_condition"])
    )


def stage_parity(out, workers):
    from tools import replay

    jobs = [
        dict(job_id=i["run_id"], item=i, outcome_parity=True, rescore=True, observe=True)
        for i in grid()
    ]
    rows = execution.execute(data_root(), jobs, workers=workers, progress=_progress("parity"))
    references = json.loads((data_root() / "expected_rows.json").read_text())
    differences = replay.compare_rows([r["row"] for r in rows], references)
    outcome_diffs = [r["job_id"] for r in rows if r["outcome_parity_differences"]]
    leaks = [
        dict(run_id=r["job_id"], violations=r["leakage"]["violations"])
        for r in rows
        if not r["leakage"]["passed"]
    ]
    from experiments.validation.leakage import calibration_disjointness

    plan = json.loads((data_root() / "plan.json").read_text())
    calibration = json.loads((data_root() / plan["calibration_path"]).read_text())
    receipt = dict(
        runs=len(rows),
        exact_row_parity=not differences,
        row_differences=differences,
        outcome_parity=not outcome_diffs,
        outcome_differences=outcome_diffs,
        leakage_passed=not leaks,
        leakage_violations=leaks,
        decisions_audited=sum(r["leakage"]["decisions"] for r in rows),
        prediction_reads_audited=sum(r["leakage"]["prediction_reads"] for r in rows),
        downstream_execution_failures=sum(r["pipeline_failures"] for r in rows),
        calibration=calibration_disjointness(plan, calibration),
    )
    execution.write_jsonl(out / "reference.jsonl", rows)
    (out / "parity.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "runs",
                    "exact_row_parity",
                    "outcome_parity",
                    "leakage_passed",
                    "decisions_audited",
                )
            },
            indent=2,
        )
    )
    if differences or outcome_diffs or leaks:
        raise SystemExit("Parity or leakage check failed; see parity.json")


def _bounds_worker(flight):
    from experiments.validation import bounds

    return bounds.capture_table(execution._WS, flight)


def stage_bounds(out, workers):
    from experiments.validation import bounds

    flights = sorted({i["flight"] for i in grid()})
    with ProcessPoolExecutor(
        max_workers=min(workers, len(flights)),
        initializer=execution._init,
        initargs=(str(data_root()),),
    ) as pool:
        tables = list(pool.map(_bounds_worker, flights))
    plan = json.loads((data_root() / "plan.json").read_text())
    results = []
    for table in tables:
        case = plan["cases"][table["flight"]]
        for name in contracts.VARIANTS:
            doc = contracts.contract_for(case, 0.95, name).document()
            for kind in ("sensor", "detectable", "model_union"):
                results.append(
                    dict(
                        flight=table["flight"],
                        contract=name,
                        **contracts_bound(bounds, table, doc, kind),
                    )
                )
    (out / "bounds.json").write_text(
        json.dumps(dict(tables=tables, bounds=results), indent=2, sort_keys=True) + "\n"
    )
    for r in results:
        if r["kind"] == "model_union" or not r["passed"]:
            print(r["flight"], r["contract"], r["kind"], r["passed"])


def contracts_bound(bounds, table, doc, kind):
    return bounds.bound_result(table, doc, kind)


def rerun_contracts():
    return [
        n
        for n, spec in contracts.VARIANTS.items()
        if n != contracts.REFERENCE and spec.get("rerun_supported", True)
    ]


def stage_reruns(out, workers):
    jobs = [
        dict(job_id=name + "|" + i["run_id"], item=i, contract=name)
        for name in rerun_contracts()
        for i in grid()
    ]
    rows = execution.execute(data_root(), jobs, workers=workers, progress=_progress("reruns"))
    execution.write_jsonl(out / "reruns.jsonl", rows)


def stage_causal(out, workers):
    spec = CONFIG["causal_ablation"]
    jobs = [
        dict(job_id=f"{p}|{m}|{i['run_id']}", item=i, protocol=p, multiplier=m)
        for p in spec["protocols"]
        for m in spec["latency_multipliers"]
        for i in grid()
    ]
    rows = execution.execute(data_root(), jobs, workers=workers, progress=_progress("causal"))
    execution.write_jsonl(out / "causal.jsonl", rows)


def stage_hindsight(out, workers):
    from experiments.validation import hindsight

    spec = CONFIG["feasibility"]["hindsight_search"]
    plan = json.loads((data_root() / "plan.json").read_text())
    bounds = json.loads((out / "bounds.json").read_text())
    tables = {t["flight"]: t for t in bounds["tables"]}
    union = {
        (b["flight"], b["contract"]): b["passed"]
        for b in bounds["bounds"]
        if b["kind"] == "model_union"
    }
    histories = {}
    for r in execution.read_jsonl(out / "reference.jsonl"):
        histories[(contracts.REFERENCE, r["job_id"])] = r
    for r in execution.read_jsonl(out / "reruns.jsonl"):
        name, run_id = r["job_id"].split("|", 1)
        histories[(name, run_id)] = r
    by_condition = {}
    for item in plan["grid"]:
        by_condition.setdefault(condition_key(item), []).append(item)
    actions = [a["action_id"] for a in plan["actions"]]
    jobs, skipped = [], []
    for name in [contracts.REFERENCE, *rerun_contracts()]:  # survey cannot be a runtime input
        for key, items in sorted(by_condition.items()):
            runs = [histories[(name, i["run_id"])] for i in items]
            flight = items[0]["flight"]
            if any(r["summary"]["success"] for r in runs):
                skipped.append(
                    dict(condition=key, contract=name, reason="released_policy_succeeded")
                )
                continue
            if not union[(flight, name)]:
                skipped.append(
                    dict(condition=key, contract=name, reason="model_union_bound_failed")
                )
                continue
            release = {c["index"]: c["release_ns"] for c in tables[flight]["captures"]}
            starts = [
                dict(
                    source="policy:" + i["policy"],
                    script=hindsight.script_from_events(r["events"], release),
                )
                for i, r in zip(items, runs, strict=True)
            ]
            starts += [
                dict(source="constant:" + str(a), script={r: a for r in release.values()})
                for a in actions
            ]
            carrier = next(i for i in items if i["policy"] == "fixed_local_yolo")
            jobs.append(
                dict(
                    job_id=name + "|" + key,
                    condition=key,
                    contract=name,
                    item=carrier,
                    starts=starts,
                    actions=actions,
                    release_ns_by_index=release,
                    include_abstain=spec["actions_include_abstain"],
                    max_evaluations=spec["max_evaluations"],
                    max_sweeps=spec["max_sweeps"],
                )
            )
    print(f"[hindsight] {len(jobs)} searches; {len(skipped)} conditions need none", flush=True)
    results, started = [], time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=execution._init, initargs=(str(data_root()),)
    ) as pool:
        for res in pool.map(hindsight._search_chunk, [[j] for j in jobs]):
            results.extend(res)
            if len(results) % 10 == 0 or len(results) == len(jobs):
                found = sum(r["success"] for r in results)
                print(
                    f"[hindsight] {len(results)}/{len(jobs)} "
                    f"({time.perf_counter() - started:.0f}s; witnesses {found})",
                    flush=True,
                )
    execution.write_jsonl(out / "hindsight.jsonl", results)
    (out / "hindsight_skipped.json").write_text(
        json.dumps(skipped, indent=2, sort_keys=True) + "\n"
    )


def stage_sweep(out, workers):
    """Sensitivity map: rescore the 648 reference histories under all 144 declared settings."""
    settings = contracts.sweep_settings()
    jobs = [dict(job_id=i["run_id"], item=i, sweep=True, keep_events=False) for i in grid()]
    rows = execution.execute(data_root(), jobs, workers=workers, progress=_progress("sweep"))
    reference = {r["job_id"]: r for r in execution.read_jsonl(out / "reference.jsonl")}
    index = {tuple(s.values()): n for n, s in enumerate(settings)}
    checks = {
        "reference": (index[(2, 2, 2, True, 44.0)], lambda r: r["summary"]["success"]),
        "time_critical": (
            index[(2, 2, 2, True, 36.0)],
            lambda r: r["rescore"]["time_critical"]["success"],
        ),
        "confirm_strict": (
            index[(2, 3, 2, True, 44.0)],
            lambda r: r["rescore"]["confirm_strict"]["success"],
        ),
    }
    failures = {
        name: [
            r["job_id"] for r in rows if (r["sweep"][n] == "1") != bool(get(reference[r["job_id"]]))
        ]
        for name, (n, get) in checks.items()
    }
    receipt = dict(
        settings=len(settings),
        runs=len(rows),
        consistency={k: not v for k, v in failures.items()},
        mismatches=failures,
    )
    execution.write_jsonl(
        out / "sweep.jsonl",
        [dict(job_id=r["job_id"], row=r["row"], sweep=r["sweep"]) for r in rows],
    )
    (out / "sweep_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["consistency"]))
    if any(failures.values()):
        raise SystemExit("Sensitivity sweep failed a declared consistency check")


def stage_verify(out, workers):
    """Re-execute every hindsight witness in fresh workers; results must be identical."""
    from tools import replay

    found = [h for h in execution.read_jsonl(out / "hindsight.jsonl") if h["success"]]
    plan = json.loads((data_root() / "plan.json").read_text())
    carriers = {condition_key(i): i for i in plan["grid"] if i["policy"] == "fixed_local_yolo"}
    jobs = [
        dict(
            job_id=h["job_id"],
            item=carriers[h["condition"]],
            contract=h["contract"],
            script=h["script"],
            label="hindsight_witness",
        )
        for h in found
    ]
    rows = {r["job_id"]: r for r in execution.execute(data_root(), jobs, workers=workers, chunk=1)}
    checks = []
    for h in found:
        r = rows[h["job_id"]]
        diff = replay.compare_rows([r["row"]], [h["row"]])
        checks.append(
            dict(job_id=h["job_id"], success=r["summary"]["success"], identical_row=not diff)
        )
    receipt = dict(
        witnesses=len(found),
        all_reproduced=all(c["success"] and c["identical_row"] for c in checks),
        checks=checks,
    )
    (out / "witness_verification.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: receipt[k] for k in ("witnesses", "all_reproduced")}))
    if not receipt["all_reproduced"]:
        raise SystemExit("A hindsight witness did not reproduce")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "stage",
        choices=(
            "parity",
            "bounds",
            "reruns",
            "causal",
            "hindsight",
            "verify",
            "sweep",
            "analyze",
            "all",
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    stages = (
        ("parity", "bounds", "reruns", "causal", "hindsight", "verify", "sweep", "analyze")
        if args.stage == "all"
        else (args.stage,)
    )
    for stage in stages:
        started = time.perf_counter()
        if stage == "analyze":
            from experiments.validation import analysis

            analysis.build(args.output, args.report)
        else:
            globals()["stage_" + stage](args.output, args.workers)
        print(f"== {stage} done in {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
