"""Paired fixed-flight diagnostics, with editable cases and no population inference.

Each case is validated through the ordinary scenario loader. Routers/executors are
fresh per run; reports restate the runner's metrics, never recompute mission success.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path
from typing import Any

from missionbench.schemas.loading import DocumentReader, read_json_object
from missionbench.v2.executors import build_executors
from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.calibrated_router import bind_router
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import finite_tree, load_scenario


def merge(base: dict, changes: dict) -> dict:
    """Merge objects recursively, replace arrays whole; schema validation follows."""
    result = copy.deepcopy(base)
    for key, value in changes.items():
        result[key] = (
            merge(result[key], value)
            if key != "mission_requirements"
            and isinstance(result.get(key), dict)
            and isinstance(value, dict)
            else copy.deepcopy(value)
        )
    return result


def run_suite(
    spec_path: Path,
    recording: Path,
    output: Path,
    *,
    case_recordings: dict[str, Path] | None = None,
    feasibility_budget: int = 0,
    prediction_cache: Path | None = None,
) -> dict[str, Any]:
    if type(feasibility_budget) is not int or feasibility_budget < 0:
        raise ValueError("feasibility_budget must be a nonnegative integer")
    data = read_json_object(spec_path)
    finite_tree(data)
    spec = DocumentReader(
        data,
        context=str(spec_path),
        allowed_fields=(
            "suite_schema_version",
            "base_scenario",
            "routers",
            "cases",
        ),
    )
    if spec.get_str("suite_schema_version") != "0.1":
        raise ValueError("unsupported suite version")
    base_path = spec_path.parent / spec.get_str("base_scenario")
    base = read_json_object(base_path)
    contract = read_json_object(base_path.parent / base["contract_path"])
    routers = spec.get_object_list("routers", allowed_fields=("name", "parameters"))
    cases = spec.get_object_list(
        "cases",
        allowed_fields=(
            "case_id",
            "description",
            "scenario_overrides",
            "contract_overrides",
        ),
    )
    if not routers or not cases:
        raise ValueError("suite needs routers and cases")
    names = [r.get_str("name") for r in routers]
    ids = [c.get_str("case_id") for c in cases]
    case_recordings = case_recordings or {}
    if set(case_recordings) - set(ids):
        raise ValueError("recording overrides name unknown cases")
    if len(set(names)) != len(names) or len(set(ids)) != len(ids):
        raise ValueError("duplicate router or case")
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", key) for key in ids):
        raise ValueError("case IDs must contain only letters, numbers, underscores or hyphens")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "suite.json", data)
    prepared = []
    # Preflight every case before any evaluation. Persist complete, reloadable inputs.
    for case, case_id in zip(cases, ids, strict=True):
        description = case.get_str("description")
        changes = case.get_passthrough("scenario_overrides")
        contract_changes = case.get_passthrough("contract_overrides")
        if not isinstance(changes, dict) or not isinstance(contract_changes, dict):
            raise ValueError("overrides must be objects")
        if "contract_path" in changes:
            raise ValueError("use contract_overrides instead of overriding contract_path")
        folder = output / case_id
        folder.mkdir()
        scenario_data = merge(base, changes)
        scenario_data["scenario_id"] = f"{base['scenario_id']}__{case_id}"
        scenario_data["contract_path"] = "contract.json"
        for obj in scenario_data["world"]["objects"]:
            if "mesh" in obj:
                obj["mesh"]["manifest"] = str(
                    (base_path.parent / obj["mesh"]["manifest"]).resolve()
                )
        save_json(folder / "contract.json", merge(contract, contract_changes))
        save_json(folder / "scenario.json", scenario_data)
        scenario = load_scenario(folder / "scenario.json")
        case_recording = case_recordings.get(case_id, recording)
        replay = ReplayBackend(scenario, case_recording)
        replay.close()
        for router, name in zip(routers, names, strict=True):
            params = router.get_passthrough("parameters")
            if not isinstance(params, dict):
                raise ValueError("router parameters must be an object")
            candidate = build_router(
                name, scenario.profiles, scenario.observation_interval_s, params
            )
            bind_router(candidate, scenario)
        prepared.append((case_id, description, folder, scenario, case_recording))
    rows = []
    feasibility_reports = []
    for case_id, description, folder, scenario, case_recording in prepared:
        feasibility = None
        if feasibility_budget:
            from experiments.v4_benchmark.feasibility import analyze

            feasibility = analyze(
                scenario, case_recording, folder / "feasibility", max_schedules=feasibility_budget
            )
            feasibility_reports.append({"case_id": case_id, **feasibility})
        for index, (router, name) in enumerate(zip(routers, names, strict=True)):
            backend = ReplayBackend(scenario, case_recording)
            try:
                result = run_mission(
                    scenario,
                    backend,
                    build_router(
                        name,
                        scenario.profiles,
                        scenario.observation_interval_s,
                        router.get_passthrough("parameters"),
                    ),
                    (
                        _cached_executors(scenario, prediction_cache)
                        if prediction_cache is not None
                        else build_executors(scenario.executors)
                    ),
                    router_name=name,
                    output=folder / f"router_{index:02d}",
                )
            finally:
                backend.close()
            if feasibility and feasibility["status"] == "infeasible" and result["mission_success"]:
                raise RuntimeError("Mission success contradicts evaluator infeasibility proof")
            rows.append(
                {
                    "case_id": case_id,
                    "description": description,
                    "router": name,
                    "status": result["status"],
                    "feasibility_status": feasibility["status"] if feasibility else None,
                    "failure_attribution": (
                        "component_error"
                        if result["status"] == "error"
                        else None
                        if result["mission_success"]
                        else "failed_despite_verified_feasible_schedule"
                        if feasibility and feasibility["status"] == "feasible"
                        else "infeasible_under_fixed_flight_and_executor_pool"
                        if feasibility and feasibility["status"] == "infeasible"
                        else "feasibility_unknown"
                    ),
                    "error": result["error"],
                    "mission_success": result["mission_success"],
                    "mission_requirements": result["mission_requirements"],
                    "first_correct_report_s": result["first_correct_report_s"],
                    "termination_reason": result["termination_reason"],
                    "failed_constraints": [k for k, v in result["constraints"].items() if not v],
                    "target_recall": result["quality"]["target_recall"],
                    "detection_precision": result["quality"]["detection_precision"],
                    "energy_j": result["cumulative_energy_j"],
                    "communication_mb": result["cumulative_communication_mb"],
                    "final_battery_frac": result["final_battery_frac"],
                    "final_time_s": result["final_time_s"],
                    "processed_observations": result["processed_observations"],
                    "skipped_observations": len(result["skipped_observation_ids"]),
                    "config_switches": result["config_switches"],
                    "config_selection_history": result["config_selection_history"],
                    "privacy_violations": result["privacy_violations"],
                    "inference_failures": result["inference_failures"],
                    "backend": result["backend"],
                    "result_path": f"{case_id}/router_{index:02d}/results.json",
                }
            )
    report = {
        "suite_schema_version": "0.1",
        "evaluation_purpose": "integration_diagnostic",
        "interpretation": "Controlled sensitivity cases; no success-rate or CI claim.",
        "rows": rows,
    }
    save_json(output / "comparison.json", report)
    if feasibility_reports:
        save_json(
            output / "feasibility_summary.json",
            {
                "schema_version": "0.1",
                "gt_aware": True,
                "is_policy": False,
                "cases": feasibility_reports,
            },
        )
        summary = [
            "# GT-aware feasibility (not a policy ranking)",
            "",
            "Unknown means the bounded search did not settle feasibility. "
            "Bounds apply only to this fixed flight and supported executor pool.",
            "",
            "| Case | Status | Reason | Verified schedules |",
            "|---|---|---|---:|",
        ]
        for item in feasibility_reports:
            summary.append(
                f"| {item['case_id']} | {item['status']} | {item['reason']} "
                f"| {item['evaluated_schedules']} |"
            )
        (output / "feasibility_summary.md").write_text("\n".join(summary) + "\n")
    lines = [
        "# V4 paired mission diagnostics",
        "",
        report["interpretation"],
        "",
        "Synthetic rendered perception; configured costs and network. All failures retained.",
        "",
        "| Case | Router | Success | Recall | Precision | Energy J | MB "
        "| Frames / skipped | Failure |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        failure = ", ".join(row["failed_constraints"])
        if row["termination_reason"] != "mission_window_complete":
            failure = f"{row['termination_reason']}; {failure}"
        lines.append(
            f"| {row['case_id']} | {row['router']} | {row['mission_success']} "
            f"| {row['target_recall']:.3f} | {row['detection_precision']:.3f} "
            f"| {row['energy_j']:.1f} | {row['communication_mb']:.2f} "
            f"| {row['processed_observations']} / {row['skipped_observations']} | {failure} |"
        )
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    audit = decision_audit(output, rows, ids[0])
    save_json(output / "decision_audit.json", audit)
    audit_lines = [
        "# Router decision audit",
        "",
        f"Reference case: {ids[0]}",
        "",
        audit["interpretation"],
        "",
        "Action differences are counted only at shared capture IDs. "
        "Different capture schedules remain in JSON. Pressure counts can overlap.",
        "",
        "| Case | Router | Same recording | Changed / shared actions "
        "| Deadline pressure | Battery pressure | Explanation mismatches |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for pair in audit["pairs"]:
        audit_lines.append(
            f"| {pair['case_id']} | {pair['router']} | {pair['same_recording']} "
            f"| {len(pair['changed_action_capture_ids'])} / {pair['common_capture_count']} "
            f"| {pair['deadline_pressure_decisions']} | {pair['battery_pressure_decisions']} "
            f"| {pair['explanation_mismatches']} |"
        )
    (output / "decision_audit.md").write_text("\n".join(audit_lines) + "\n")
    return report


def decision_audit(output: Path, rows: list[dict], reference_case: str) -> dict:
    """Compare actions at shared capture IDs, avoiding sequence-index misalignment."""
    references = {r["router"]: r for r in rows if r["case_id"] == reference_case}
    pairs = []
    for row in rows:
        reference = references[row["router"]]

        def events(item: dict) -> dict:
            path = (output / item["result_path"]).parent / "events.json"
            return {e["observation_id"]: e for e in json.loads(path.read_text())}

        before, after = events(reference), events(row)
        common = sorted(before.keys() & after.keys())
        changed = [
            i for i in common if before[i]["action"]["config_id"] != after[i]["action"]["config_id"]
        ]
        decisions = [e["router_decision"] for e in after.values() if e.get("router_decision")]
        pairs.append(
            {
                "case_id": row["case_id"],
                "router": row["router"],
                "same_recording": reference["backend"]["recording_sha256"]
                == row["backend"]["recording_sha256"],
                "common_capture_count": len(common),
                "changed_action_capture_ids": changed,
                "reference_only_capture_ids": sorted(before.keys() - after.keys()),
                "case_only_capture_ids": sorted(after.keys() - before.keys()),
                "deadline_pressure_decisions": sum(
                    d.get("deadline_pressure", False) for d in decisions
                ),
                "battery_pressure_decisions": sum(
                    d.get("battery_pressure", False) for d in decisions
                ),
                "explanation_mismatches": sum(
                    d.get("explanation_matches_selection") is False for d in decisions
                ),
                "reference_success": reference["mission_success"],
                "case_success": row["mission_success"],
            }
        )
    return {
        "audit_schema_version": "0.1",
        "reference_case": reference_case,
        "interpretation": "Different actions indicate response, not correctness or superiority.",
        "pairs": pairs,
    }


def _cached_executors(scenario, directory):
    from experiments.v4_benchmark.probe_cache import build_cached

    return build_cached(scenario.executors, directory, scenario.recording_identity)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--recording", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--prediction-cache",
        type=Path,
        help="Explicit frozen real-prediction replay; no new inference",
    )
    parser.add_argument(
        "--feasibility-budget",
        type=int,
        default=0,
        help="Optional GT-aware analysis per case; max schedules, 0 disables",
    )
    parser.add_argument(
        "--case-recordings",
        type=Path,
        help="JSON object mapping case IDs to recording paths relative to that file",
    )
    args = parser.parse_args()
    recordings = None
    if args.case_recordings:
        mapping = read_json_object(args.case_recordings)
        if any(not isinstance(v, str) for v in mapping.values()):
            raise ValueError("recording paths must be strings")
        recordings = {k: args.case_recordings.parent / v for k, v in mapping.items()}
    report = run_suite(
        args.suite,
        args.recording,
        args.output,
        case_recordings=recordings,
        feasibility_budget=args.feasibility_budget,
        prediction_cache=args.prediction_cache,
    )
    print(f"Saved {len(report['rows'])} paired runs to {args.output / 'comparison.md'}")
    return 2 if any(row["status"] == "error" for row in report["rows"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
