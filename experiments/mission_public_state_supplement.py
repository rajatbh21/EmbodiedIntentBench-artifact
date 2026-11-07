"""Read-only supplementary full-public-state/resource audit of frozen main studies.

No model, policy or tracking execution is repeated. Existing logged predictions and
scores remain the inputs to canonical endpoint evaluation. Only new sidecars are written.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from experiments.mission_runtime_profile import REPO, implementation_inventory, read, save, sha

OUTPUT = REPO / "results/mission_public_state_supplement_v1_20261027"
STUDIES = {
    "depot": (
        "mission_environment_transfer_v3_20261026/study_paired_scope_v1",
        "mission_transfer_comparison_v4",
        352,
    ),
    "lake": (
        "mission_coverage_prospective_v1_20261025/study_repair_v1",
        "mission_coverage_repair",
        504,
    ),
}
FLAGS = (
    "memory_constraint_success",
    "deadline_success",
    "battery_constraint_success",
    "communication_constraint_success",
    "privacy_constraint_success",
)


def bound(path, expected, bindings):
    actual = sha(path)
    if actual != expected:
        raise ValueError("Changed source binding: " + str(path))
    bindings[str(path.resolve())] = actual
    return read(path)


def complete_grid(rows, protocol, expected_count):
    expected = {
        (f, m, c, power)
        for f in protocol["cases"]
        for m in protocol["missions"]
        for c in protocol["policies"]
        for power in protocol["active_cpu_power_sensitivity_w"]
    }
    actual = [(r["flight"], r["mission"], r["policy"], r["power_w"]) for r in rows]
    if len(expected) != expected_count or len(actual) != expected_count or set(actual) != expected:
        raise ValueError("Missing, duplicate or unexpected original-study cell")
    if len({r["path"] for r in rows}) != expected_count:
        raise ValueError("Multiple cells share an execution path")


def prepare(output):
    if output.exists():
        raise ValueError("Refuse to overwrite supplement")
    bundles, bindings, copies = {}, {}, {}
    for cohort, (relative, module, count) in STUDIES.items():
        root = REPO / "results" / relative
        api = importlib.import_module("experiments." + module + ".study")
        protocol = api.verify(root)
        completion = read(root / "completion.json")
        if not completion["complete"]:
            raise ValueError("Original study not complete")
        bindings[str(root / "completion.json")] = sha(root / "completion.json")
        for name in ("protocol.json", "analysis.json", "validation.json"):
            bound(root / name, completion["artifacts_sha256"][name], bindings)
        validation = read(root / "validation.json")
        if not validation["complete"] or validation["protocol_sha256"] != sha(
            root / "protocol.json"
        ):
            raise ValueError("Original validation binding differs")
        rows = []
        for flight in protocol["cases"]:
            name = f"validation_{flight}.json"
            condition = bound(root / name, validation["condition_audits_sha256"][name], bindings)
            if not condition["complete"] or condition["condition"] != flight:
                raise ValueError("Original condition incomplete")
            for summary_name, expected in condition["source_summaries_sha256"].items():
                summary = bound(root / summary_name, expected, bindings)
                if not summary["complete"] or summary["protocol_sha256"] != sha(
                    root / "protocol.json"
                ):
                    raise ValueError("Original summary binding differs")
                rows.extend(summary["cases"])
        complete_grid(rows, protocol, count)
        for row in rows:
            destination = root / row["path"]
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                bound(destination / name, row[key], bindings)
            outcome = read(destination / "outcome.json")
            bound(destination / "protocol.json", outcome["protocol_sha256"], bindings)
        for flight in protocol["cases"]:
            for mission in protocol["missions"]:
                for name in ("scenario.json", "contract.json", "intent.json"):
                    relative = str(Path(flight) / mission / name)
                    source = root / relative
                    bound(source, protocol["frozen_files"][relative], bindings)
                    target = output / "inputs" / cohort / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                    copies[str(target.relative_to(output))] = sha(target)
        bundles[cohort] = dict(root=root, api=api, protocol=protocol, rows=rows)
    plan = dict(
        schema_version="mission_public_state_supplement_plan_v1",
        original_studies={
            k: dict(
                root=str(v["root"]),
                count=len(v["rows"]),
                protocol_sha256=sha(v["root"] / "protocol.json"),
            )
            for k, v in bundles.items()
        },
        original_inputs_sha256=bindings,
        copied_inputs_sha256=copies,
        implementation_sha256=implementation_inventory(),
        affinity=sorted(os.sched_getaffinity(0)),
        environment={
            k: os.environ[k] for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        },
        exact_public_state_equality=True,
        reconstructed_resource_flags=list(FLAGS),
        original_tracking_cost=True,
        model_policy_tracking_rerun=False,
        scope="Reconstruct complete RuntimeState and paid chronology/resources from original "
        "scenarios and action sequences. Re-evaluate canonical endpoints on original logged "
        "scores/reports. This supplement does not independently recompute masks, reports, "
        "policy choices, or sensor ground truth.",
    )
    save(output / "plan.json", plan)
    return plan, bundles


def verify_bindings(output, plan):
    for path, expected in plan["original_inputs_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Original artifact changed: " + path)
    for path, expected in plan["copied_inputs_sha256"].items():
        if sha(output / path) != expected:
            raise ValueError("Consumed copied mission changed: " + path)
    for path, expected in plan["implementation_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Audit implementation changed: " + path)


def run(output):
    from experiments.lake_intent_contract import IntentContract, evaluate
    from experiments.lake_intent_runner import bind_scenario
    from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST
    from experiments.mission_measured_cost_audit_v2 import check_execution, compare_outcome, require
    from experiments.v4_benchmark.scenario import digest

    plan, bundles = prepare(output)
    rows, counts = [], {}
    for cohort, bundle in bundles.items():
        root, api, protocol = bundle["root"], bundle["api"], bundle["protocol"]
        scenarios = {}
        for row in bundle["rows"]:
            key = row["flight"], row["mission"], row["power_w"]
            if key not in scenarios:
                flight, mission, power = key
                contract = IntentContract(
                    read(output / "inputs" / cohort / flight / mission / "intent.json")
                )
                scenario = bind_scenario(
                    api.scenario_for(output / "inputs" / cohort, protocol, flight, mission, power),
                    contract,
                )
                scenarios[key] = scenario, contract
            scenario, contract = scenarios[key]
            folder = root / row["path"]
            events, result, outcome, execution_protocol = (
                read(folder / name)
                for name in (
                    "execution/events.json",
                    "execution/results.json",
                    "outcome.json",
                    "protocol.json",
                )
            )
            checked = dict(
                cohort=cohort,
                flight=row["flight"],
                mission=row["mission"],
                policy=row["policy"],
                path=str(folder),
                decisions=len(events),
                input_sha256={
                    name: sha(folder / name)
                    for name in (
                        "execution/events.json",
                        "execution/results.json",
                        "outcome.json",
                        "protocol.json",
                    )
                },
            )
            try:
                require(
                    execution_protocol["scenario_sha256"]
                    == result["scenario_sha256"]
                    == digest(scenario.source),
                    "Original scenario binding differs",
                )
                require(
                    execution_protocol["contract_sha256"] == contract.sha256
                    and execution_protocol["public_contract_sha256"] == contract.public_sha256,
                    "Original intent binding differs",
                )
                require(
                    execution_protocol["recording_identity"]
                    == scenario.recording_identity
                    == result["backend"]["recording_identity"],
                    "Original recording identity differs",
                )
                require(execution_protocol["router"] == row["policy"], "Original policy differs")
                require(
                    outcome["events_sha256"] == row["events_sha256"]
                    and outcome["execution_sha256"] == row["results_sha256"],
                    "Outcome inputs differ",
                )
                canonical = evaluate(contract, events, result)
                reconstructed = check_execution(
                    scenario, events, result, row, canonical, DEFAULT_TRACKING_COST
                )
                compare_outcome(canonical, outcome)
                checked.update(
                    passed=True,
                    reconstruction=reconstructed,
                    resource_flags={k: result["constraints"][k] for k in FLAGS},
                    canonical_mission_success=canonical["mission_success"],
                )
            except (ValueError, KeyError, AssertionError) as error:
                checked.update(passed=False, error_type=type(error).__name__, error=str(error))
            rows.append(checked)
        own = [r for r in rows if r["cohort"] == cohort]
        counts[cohort] = dict(
            executions=len(own),
            decisions=sum(r["decisions"] for r in own),
            passed=sum(r["passed"] for r in own),
            failed=sum(not r["passed"] for r in own),
        )
        print(cohort, counts[cohort], flush=True)
    verify_bindings(output, plan)
    report = dict(
        schema_version="mission_public_state_supplement_v1",
        complete=True,
        passed=all(r["passed"] for r in rows),
        plan_sha256=sha(output / "plan.json"),
        cohorts=counts,
        executions=len(rows),
        decisions=sum(r["decisions"] for r in rows),
        tracking_cost=asdict(DEFAULT_TRACKING_COST),
        cases=rows,
        scope=plan["scope"],
    )
    save(output / "audit.json", report)
    save(
        output / "completion.json",
        dict(
            complete=True,
            passed=report["passed"],
            plan_sha256=sha(output / "plan.json"),
            audit_sha256=sha(output / "audit.json"),
        ),
    )
    if not report["passed"]:
        raise ValueError("Supplemental failures preserved; original studies untouched")
    return report


def main():
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    os.sched_setaffinity(0, {max(os.sched_getaffinity(0))})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)


if __name__ == "__main__":
    main()
