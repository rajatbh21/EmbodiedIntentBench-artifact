"""Post-lock model evidence diagnostic, never an input to routing or selection.

Score every frozen native model mask on the predeclared capture opportunities.
An optimistic cross-model bound unions matched target IDs at each capture; it
does not union masks, create an executable policy, or claim paid feasibility.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_missions.objectives import enough_evidence
from experiments.mission_runtime_profile import read, save, sha
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.wide_area_routing.adapter import load_bank, load_case


def phase_evidence(phases, scores):
    """Canonical phase-local two-hit/span semantics at optimistic zero service cost."""
    rows = []
    for phase in phases:
        selected = [s for s in scores if phase["start_s"] <= s["capture_time_s"] < phase["end_s"]]
        evidence = {
            target: sorted({s["capture_time_s"] for s in selected
                            if target in s["matched_target_ids"]})
            for target in phase["evaluator_target_ids"]
        }
        supported = [target for target, times in evidence.items() if enough_evidence(times, phase)]
        false = sum(s["false_positive_components"] for s in selected)
        recall = len(supported) / len(evidence)
        rows.append(dict(
            phase_id=phase["phase_id"], capture_count=len(selected),
            capture_ids=[s["observation_id"] for s in selected],
            required_targets=phase["evaluator_target_ids"], supported_targets=supported,
            supported_target_count=len(supported), target_count=len(evidence),
            evidence_capture_times=evidence, evidence_recall=recall,
            required_recall=phase["required_recall"], min_hits=phase["min_hits"],
            min_evidence_span_s=phase["min_evidence_span_s"],
            evidence_requirement_met=recall >= phase["required_recall"],
            false_components=false, max_false_components=phase["max_false_components"],
            zero_cost_phase_requirement_met=(recall >= phase["required_recall"]
                                             and false <= phase["max_false_components"]),
        ))
    return rows


def cross_model_bound(phases, model_scores):
    """Necessary evidence condition; a failed bound rules out any selected policy.

    Every admitted selected mask's canonical matched IDs are a subset of this
    per-capture union. Taking several models at the same capture is still one
    evidence hit per target, never several temporal observations. False outputs,
    time/energy costs, exclusive actions and downstream association are relaxed.
    """
    if not model_scores:
        raise ValueError("A nonempty model pool is required")
    first = next(iter(model_scores.values()))
    expected = [(s["observation_id"], s["capture_time_s"]) for s in first]
    if len({index for index, _ in expected}) != len(expected):
        raise ValueError("Each model must score unique original capture IDs")
    if any([(s["observation_id"], s["capture_time_s"]) for s in scores] != expected
           for scores in model_scores.values()):
        raise ValueError("Every model must have the same canonical capture clock")
    union = [dict(observation_id=index, capture_time_s=time_s,
                  matched_target_ids=sorted({target for scores in model_scores.values()
                                             for target in scores[position]["matched_target_ids"]}),
                  false_positive_components=0)
             for position, (index, time_s) in enumerate(expected)]
    rows = phase_evidence(phases, union)
    for row in rows:
        # This bound deliberately ignores false positives; do not expose its
        # synthetic zero count as a measurement or a full phase-success result.
        for key in ("false_components", "max_false_components", "zero_cost_phase_requirement_met"):
            del row[key]
    failed = [row["phase_id"] for row in rows if not row["evidence_requirement_met"]]
    return dict(
        evaluator_only=True, executable_policy=False,
        construction="Union of native per-model canonical matched target IDs at each original capture; no mask union",
        relaxed_constraints=["Perception latency and task deadlines", "One selected action per capture",
                             "False components", "Downstream association and verification",
                             "Report correctness and timing", "Battery and network costs"],
        all_phase_evidence_requirements_met=not failed,
        failed_phase_ids=failed, phases=rows, union_capture_scores=union,
        conclusion=(
            "Model-pool evidence infeasible for this locked intent and these scheduled opportunities: "
            "at least one phase lacks sufficient repeated target evidence even under the all-model union."
            if failed else
            "The necessary evidence bound passes. Paid one-action routing, downstream reporting and "
            "resource feasibility are not established."
        ),
    )


def diagnostic_sources():
    repo = Path(__file__).resolve().parents[2]
    names = (
        "experiments/wide_area_routing/diagnostics.py",
        "experiments/wide_area_routing/adapter.py",
        "experiments/lake_hindsight_witness.py",
        "experiments/lake_missions/objectives.py",
        "experiments/v4_benchmark/runner.py",
        "missionbench/v2/evaluation.py",
        "missionbench/v2/executors.py",
    )
    return {str(repo / name): sha(repo / name) for name in names}


def verify_saved(report):
    for category in ("input_sha256", "implementation_sha256"):
        for path, expected in report[category].items():
            if sha(path) != expected:
                raise ValueError(f"Changed diagnostic {category}: {path}")


def audit_flight(root, plan, flight):
    """Read only complete native inference; preserve every existing diagnostic."""
    root = Path(root).resolve()
    output = root / "model_diagnostics" / (flight + ".json")
    if output.exists():
        previous = read(output)
        verify_saved(previous)
        if previous["implementation_sha256"] != diagnostic_sources():
            raise ValueError("Diagnostic implementation changed; preserve previous output")
        return previous
    folder = root / "predictions" / flight
    completion = read(folder / "completion.json")
    manifest = read(folder / "manifest.json")
    if not completion["complete"] or completion["manifest_sha256"] != sha(folder / "manifest.json"):
        raise ValueError("Complete, hash-bound native predictions required")
    item = next(i for i in plan["grid"] if i["flight"] == flight)
    case = load_case(root, plan, item)
    scenario, contract = case["scenario"], case["contract"]
    bank = load_bank(root, plan, flight)
    if (manifest["recording_identity"] != scenario.recording_identity
            or manifest["recording_sha256"] != bank["provenance"]["recording_sha256"]):
        raise ValueError("Diagnostic predictions and original RGB do not match")
    expected_signatures = {a.source_config_id: a.source_signature for a in case["actions"]}
    if {name: m["signature"] for name, m in manifest["models"].items()} != expected_signatures:
        raise ValueError("Diagnostic model pool differs from the frozen study")
    indices = bank["opportunity_indices"]
    if manifest["capture_count"] != len(indices):
        raise ValueError("Incomplete declared observation set")
    phases = contract.document()["mission"]["phases"]
    truths = {
        i: evaluation_frame(bank["frames"][i], i,
                            (scenario.schedule.at(i) - scenario.schedule.epoch_ns) / 1e9, scenario)
        for i in indices
    }
    input_hashes = {str(path.resolve()): sha(path) for path in (
        root / "plan.json", folder / "manifest.json", folder / "completion.json",
        folder / "selection.json", folder.parent / "specification.json",
        root / plan["cases"][flight]["scenario_path"],
        root / plan["cases"][flight]["mission_design_path"],
        Path(plan["cohort"]) / flight / "recording/recording.json",
    )}
    rows, model_scores = {}, {}
    for name, model in manifest["models"].items():
        masks = VerifiedMasks(folder, model)
        evaluator = MissionEvaluator(scenario.matching_iou_threshold, len(scenario.targets),
                                     scenario.observation_interval_s)
        scores = []
        for index in indices:
            rgb_key = bank["payloads"][index].rgb_sha256
            mask = masks[rgb_key]
            scores.append(evaluator.update(truths[index], mask).to_dict())
            path = folder / model["predictions"][rgb_key]["file"]
            input_hashes[str(path.resolve())] = model["predictions"][rgb_key]["sha256"]
        evidence = phase_evidence(phases, scores)
        model_scores[name] = scores
        rows[name] = dict(
            source_signature=model["signature"], quality=evaluator.quality_details(scenario.duration_s),
            supported_target_count=sum(p["supported_target_count"] for p in evidence),
            evidence_sufficient_phases=sum(p["evidence_requirement_met"] for p in evidence),
            zero_cost_phase_requirements_met=sum(p["zero_cost_phase_requirement_met"] for p in evidence),
            phases=evidence, frame_scores=scores,
        )
    report = dict(
        schema_version="wide_area_post_lock_model_evidence_v1", flight=flight,
        inspected_utc=datetime.now(timezone.utc).isoformat(),
        evaluator_only=True, used_to_modify_study=False, routing_outcome=False,
        opportunity_indices=indices, opportunity_count=len(indices),
        model_count=len(rows), target_count=len(scenario.targets), phase_count=len(phases),
        matching_iou_threshold=scenario.matching_iou_threshold, min_component_px=4,
        contract_sha256=contract.sha256,
        interpretation="All native model masks at all locked opportunities, with zero execution cost. "
                       "Recall and false components are diagnostics; these are not paid router outcomes.",
        models=rows, optimistic_cross_model_evidence_bound=cross_model_bound(phases, model_scores),
        implementation_sha256=diagnostic_sources(), input_sha256=input_hashes,
    )
    # load_bank already checks each selected frame against the acquisition
    # manifest. Bind those exact source files explicitly for portable auditing.
    acquisition = read(Path(plan["cohort"]) / flight / "recording/recording.json")
    for index in indices:
        entry = acquisition["frames"][index]
        path = Path(plan["cohort"]) / flight / "recording" / entry["file"]
        report["input_sha256"][str(path.resolve())] = entry["sha256"]
    verify_saved(report)
    save(output, report)
    return report


def audit(root):
    from experiments.wide_area_routing.study import verify

    root = Path(root).resolve()
    plan = verify(root)  # Must exist and remain frozen before any mask is read.
    cases, pending = {}, []
    for flight in plan["cases"]:
        if not (root / "predictions" / flight / "completion.json").exists():
            pending.append(flight)
            continue
        cases[flight] = audit_flight(root, plan, flight)
        print(f"{flight}: {cases[flight]['optimistic_cross_model_evidence_bound']['conclusion']}", flush=True)
    summary = dict(
        schema_version="wide_area_post_lock_model_diagnostics_v1", complete=not pending,
        pending_flights=pending, evaluator_only=True, used_to_modify_study=False,
        plan_sha256=sha(root / "plan.json"), implementation_sha256=diagnostic_sources(),
        cases=cases, case_artifact_sha256={
            flight: sha(root / "model_diagnostics" / (flight + ".json")) for flight in cases},
    )
    if not pending:
        output = root / "model_diagnostics.json"
        if output.exists() and read(output) != summary:
            raise ValueError("Preserve existing completed diagnostic")
        if not output.exists():
            save(output, summary)
    else:
        save(root / "model_diagnostics" / "progress.json", dict(
            completed_flights=list(cases), pending_flights=pending,
            plan_sha256=summary["plan_sha256"], case_artifact_sha256=summary["case_artifact_sha256"]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    audit(args.root)


if __name__ == "__main__":
    main()
