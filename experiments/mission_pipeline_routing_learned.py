"""Separate checkpoint-bound learned report-verification development study.

Reuses the frozen task runtime and chronology with explicit public-only factories.
The two real fitted models are always retained; no threshold or policy is selected
from these outcomes. All new downstream costs are authored assumptions.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from functools import partial
from pathlib import Path

import numpy as np

from experiments.mission_pipeline_routing import audit, study
from experiments.mission_pipeline_routing.catalog import (
    Algorithm,
    Capability,
    Catalog,
    Costs,
    Task,
    default_catalog,
)
from experiments.mission_pipeline_routing.runtime import Pipeline, Selection
from experiments.mission_pipeline_routing.verification import RuleVerdict, VerificationFactory
from experiments.mission_report_verifier_learning import Model
from experiments.mission_report_verifier_learning_study import OUTPUT as TRAINING
from experiments.mission_report_verifier_learning_study import verify as verify_training
from experiments.mission_runtime_profile import REPO, implementation_inventory, read, save, sha

OUTPUT = REPO / "results/mission_pipeline_routing_learned_v1_20261027"
POLICIES = ("fixed_logistic", "fixed_mlp8", "intent_resource")
IDS = {"logistic": "VERIFY_LEARNED_LOGISTIC", "mlp8": "VERIFY_LEARNED_MLP8"}


def catalog(checkpoints):
    items = list(default_catalog().algorithms)
    for kind, service_ns, memory, load_ns in (
        ("logistic", 1_000_000, 4, 5_000_000),
        ("mlp8", 3_000_000, 8, 10_000_000),
    ):
        items.append(
            Algorithm(
                algorithm_id=IDS[kind],
                task=Task.REPORT_VERIFICATION,
                implementation="fitted_" + kind + "_identity_consistency_gate_v1",
                requires=frozenset({Capability.TRACK_EVIDENCE}),
                provides=frozenset(),
                costs=Costs(service_ns, service_ns / 1e9 * 15, memory, load_ns, load_ns / 1e9 * 15),
                kind="learned_evidence_model",
                checkpoint_sha256=checkpoints[kind]["sha256"],
            )
        )
    return Catalog(items)


class LearnedRule:
    def __init__(self, checkpoint_path, expected_sha256):
        self.model = Model.load(checkpoint_path, expected_sha256)

    def decide(self, features):
        public = dict(features)
        public["source_config_ids"] = list(features["source_config_ids"])
        probability = self.model.predict(public)
        return RuleVerdict(
            probability >= 0.5, "fitted_identity_consistency_probability", probability
        )


def make_pipeline(root, plan, *, terrain):
    available = catalog(plan["checkpoints"])
    factories = {}
    for kind, checkpoint in plan["checkpoints"].items():
        path = str((Path(root) / checkpoint["path"]).resolve())
        descriptor = available.resolve(
            IDS[kind], Task.REPORT_VERIFICATION, {Capability.TRACK_EVIDENCE}
        )
        factories[IDS[kind]] = VerificationFactory(
            descriptor,
            partial(LearnedRule, path, checkpoint["sha256"]),
            ((path, checkpoint["sha256"]),),
        )
    return Pipeline(terrain=terrain, catalog=available, verification_factories=factories)


def select_policy(name, context, history):
    if name == "fixed_logistic":
        kind = "logistic"
    elif name == "fixed_mlp8":
        kind = "mlp8"
    elif name == "intent_resource":
        kind = (
            "mlp8"
            if context.required_hits >= 3
            and context.battery_frac >= 0.3
            and context.deadline_ns - context.now_ns >= 8_000_000_000
            else "logistic"
        )
    else:
        raise ValueError("Unknown predeclared learned pipeline policy")
    return Selection("ASSOC_TERRAIN_WORLD", IDS[kind])


def specification(checkpoints):
    result = study.specification()
    result.update(
        schema_version="learned_task_pipeline_spec_v1",
        policies=list(POLICIES),
        grid=[
            dict(flight=f, mission=m, policy=p)
            for f in study.FLIGHTS
            for m in study.MISSIONS
            for p in POLICIES
        ],
        expected_runs=12,
        catalog=catalog(checkpoints).document(),
        association_config_id="ASSOC_TERRAIN_WORLD",
        learned_downstream_models=True,
        new_neural_inference=True,
        cached_perception_reexecuted=False,
        learned_verifier_forward_executed_when_guard_admissible=True,
        report_verifier_training=False,
        checkpoint_training="separate completed old-layout candidate study, no fitting here",
        decision_threshold=0.5,
        costs="authored downstream service/load/memory and 15W energy assumptions; "
        "fixed perception uses previous verified host warm-p95 latency",
        policy_rule="MLP8 iff public required_hits>=3, battery>=.3 and remaining>=8s; "
        "logistic otherwise; fixed terrain association",
        known_before_freeze="both fitted models have identical threshold decisions on "
        "the earlier candidate assessment; no current learned mission outcomes inspected",
        scope="development task-routing demonstration with two actual learned evidence "
        "models and fixed learned perception; no held-out field generalization, model "
        "or adaptive superiority, onboard latency, or electrical measurement claim",
    )
    return result


def prepare(root):
    root = Path(root)
    study.require(
        root.resolve() == OUTPUT.resolve() and not root.exists(), "Unused output required"
    )
    classical = study.verify(study.OUTPUT)
    training = verify_training(TRAINING)
    complete = read(TRAINING / "completion.json")
    study.require(complete["complete"] and complete["actual_fit_performed"], "Actual fitted models")
    for name in ("plan", "candidates", "summary", "predictions"):
        study.require(
            sha(TRAINING / (name + ".json")) == complete[name + "_sha256"],
            "Changed fitted artifact: " + name,
        )
    postfit = read(TRAINING / "postfit_audit.json")
    postfit_source = REPO / "experiments/mission_report_verifier_learning_audit.py"
    study.require(
        postfit["complete"]
        and postfit["completion_sha256"] == sha(TRAINING / "completion.json")
        and postfit["source_sha256"] == sha(postfit_source)
        and postfit["independent_candidate_labels"]
        and postfit["independent_public_feature_reconstruction"]
        and postfit["independent_forward_probabilities_and_metrics"],
        "Completed bound independent fitted-checkpoint audit required",
    )
    checkpoints = {
        k: dict(path="checkpoints/" + k + ".json", sha256=complete["checkpoint_sha256"][k])
        for k in IDS
    }
    for kind, row in checkpoints.items():
        Model.load(TRAINING / (kind + ".json"), row["sha256"])
    root.mkdir(parents=True)
    inputs = dict(classical["input_sha256"])
    inputs[str(study.OUTPUT / "plan.json")] = sha(study.OUTPUT / "plan.json")
    copied = {}

    def copy(path, relative):
        path = Path(path)
        inputs[str(path.resolve())] = sha(path)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        study.require(sha(target) == sha(path), "Changed copy")
        copied[str(relative)] = sha(target)

    for flight in study.FLIGHTS:
        for mission in study.MISSIONS:
            for name in ("scenario.json", "intent.json", "contract.json"):
                relative = Path("missions") / flight / mission / name
                copy(study.OUTPUT / relative, relative)
    for name in ("plan", "completion", "summary", "candidates", "predictions", "postfit_audit"):
        inputs[str(TRAINING / (name + ".json"))] = sha(TRAINING / (name + ".json"))
    for kind, row in checkpoints.items():
        copy(TRAINING / (kind + ".json"), row["path"])
    sources = dict(classical["source_sha256"])
    sources.update(training["implementation_sha256"])
    sources.update(implementation_inventory())
    sources[str(postfit_source)] = sha(postfit_source)
    for path in (Path(__file__), REPO / "tests/test_mission_pipeline_routing_learned.py"):
        sources[str(path.resolve())] = sha(path)
    for path, expected in sources.items():
        study.require(sha(path) == expected, "Changed inherited source")
        relative = Path("source_snapshot") / Path(path).relative_to(REPO)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied[str(relative)] = expected
    spec = specification(checkpoints)
    save(root / "specification.json", spec)
    plan = dict(
        spec,
        schema_version="learned_task_pipeline_study_v1",
        classical_plan_sha256=sha(study.OUTPUT / "plan.json"),
        source=classical["source"],
        assets=classical["assets"],
        cases=classical["cases"],
        calibration=classical["calibration"],
        calibration_sha256=classical["calibration_sha256"],
        upstream_signature=classical["upstream_signature"],
        upstream_latency_ns=classical["upstream_latency_ns"],
        upstream_energy_j=classical["upstream_energy_j"],
        checkpoints=checkpoints,
        training_plan_sha256=sha(TRAINING / "plan.json"),
        source_sha256=sources,
        input_sha256=inputs,
        copied_sha256=copied,
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        learned_mission_outcomes_existed_at_freeze=False,
    )
    save(root / "plan.json", plan)
    verify(root)
    return plan


def verify(root):
    root = Path(root)
    plan = read(root / "plan.json")
    study.require(plan["schema_version"] == "learned_task_pipeline_study_v1", "Unknown study")
    spec = specification(plan["checkpoints"])
    study.require(read(root / "specification.json") == spec, "Changed learned specification")
    for key, value in spec.items():
        if key != "schema_version":
            study.require(plan[key] == value, "Changed learned study design: " + key)
    for field in ("source_sha256", "input_sha256"):
        for path, expected in plan[field].items():
            study.require(sha(path) == expected, "Changed learned source/input: " + path)
    for relative, expected in plan["copied_sha256"].items():
        path = (root / relative).resolve()
        study.require(
            path.is_relative_to(root.resolve()) and sha(path) == expected, "Changed learned copy"
        )
    classical = study.verify(study.OUTPUT)
    study.require(
        sha(study.OUTPUT / "plan.json") == plan["classical_plan_sha256"], "Changed original plan"
    )
    for key in (
        "source",
        "assets",
        "cases",
        "calibration",
        "calibration_sha256",
        "upstream_signature",
        "upstream_latency_ns",
        "upstream_energy_j",
    ):
        study.require(plan[key] == classical[key], "Changed original condition: " + key)
    for flight in study.FLIGHTS:
        for mission in study.MISSIONS:
            for name in ("scenario.json", "intent.json", "contract.json"):
                relative = Path("missions") / flight / mission / name
                study.require(
                    sha(root / relative) == sha(study.OUTPUT / relative), "Changed mission"
                )
    completion = read(TRAINING / "completion.json")
    for kind, row in plan["checkpoints"].items():
        study.require(
            row["sha256"] == completion["checkpoint_sha256"][kind]
            and sha(root / row["path"]) == row["sha256"],
            "Changed fitted checkpoint",
        )
    return plan


def execute(root):
    root = Path(root)
    plan = verify(root)
    study.require(not (root / "run_index.json").exists(), "Preserve learned executions")
    terrain, _ = study.load_public_terrain(Path(plan["assets"]))
    factory = partial(make_pipeline, root, plan)
    rows = []
    for item in plan["grid"]:
        verify(root)
        rows.append(
            study.execute_case(
                root, plan, item, terrain, pipeline_factory=factory, policy_selector=select_policy
            )
        )
        save(root / "progress.json", dict(completed=len(rows), expected=12))
        print(f"Completed learned {len(rows)}/12", flush=True)
    save(
        root / "run_index.json",
        dict(complete_grid=True, plan_sha256=sha(root / "plan.json"), cases=rows),
    )
    return rows


def validate(root):
    plan = verify(root)
    return audit.run(
        root,
        verify_plan=verify,
        catalog_document=catalog(plan["checkpoints"]).document(),
        pipeline_factory=partial(make_pipeline, root, plan),
        policy_selector=select_policy,
    )


def finalize(root):
    root = Path(root)
    plan = verify(root)
    index, validation = read(root / "run_index.json"), read(root / "validation.json")
    rows = index["cases"]
    keys = ("flight", "mission", "policy")
    study.require(
        not (root / "completion.json").exists()
        and validation["complete"]
        and validation["plan_sha256"] == sha(root / "plan.json")
        and validation["run_index_sha256"] == sha(root / "run_index.json")
        and validation["auditor_sha256"]
        == plan["source_sha256"][str(Path(audit.__file__).resolve())]
        and validation["runs"] == plan["expected_runs"] == len(rows) == 12
        and validation["decisions"]
        == validation["independent_cost_checks"]
        == sum(r["calls"] for r in rows)
        and {tuple(r[k] for k in keys) for r in rows}
        == {tuple(r[k] for k in keys) for r in plan["grid"]},
        "Complete independently reconstructed learned grid required",
    )
    study.require(
        all(
            validation[k] is True
            for k in (
                "all_public_decisions_reselected",
                "raw_selected_masks_and_reports_reconstructed",
                "clock_resources_and_terminal_flags_reconstructed",
                "original_canonical_endpoint_verified",
                "no_mission_runner_called",
            )
        ),
        "Complete selected checkpoint reconstruction required",
    )
    for relative, expected in validation["output_sha256"].items():
        study.require(sha(root / relative) == expected, "Changed audited learned output")
    forward_calls = {name: 0 for name in IDS.values()}
    paid_stage_calls = {name: 0 for name in IDS.values()}
    for row in rows:
        for event in read(root / row["path"] / "events.json"):
            selected = event["selection"]["verification_id"]
            paid_stage_calls[selected] += 1
            for decision in event["pipeline"]["verification_decisions"]:
                verdict = decision["selected_rule"]
                if verdict is not None:
                    study.require(
                        verdict["reason"] == "fitted_identity_consistency_probability"
                        and verdict["score"] is not None,
                        "Actual selected fitted forward verdict required",
                    )
                    forward_calls[selected] += 1
    study.require(all(n > 0 for n in forward_calls.values()), "Both learned models must execute")
    summary = dict(
        complete=True,
        runs=12,
        decisions=sum(r["calls"] for r in rows),
        policies={
            p: dict(
                runs=4,
                full_success=sum(r["full_success"] for r in rows if r["policy"] == p),
                mean_report_recall=float(
                    np.mean([r["report_recall"] for r in rows if r["policy"] == p])
                ),
                false_reports=sum(r["false_reports"] for r in rows if r["policy"] == p),
                duplicate_reports=sum(r["duplicate_reports"] for r in rows if r["policy"] == p),
            )
            for p in POLICIES
        },
        all_failures=[
            {k: r[k] for k in (*keys, "failed_constraints")} for r in rows if not r["full_success"]
        ],
        learned_downstream_models=True,
        actual_neural_perception_reexecuted=False,
        actual_learned_verifier_forward_calls=forward_calls,
        paid_verification_stage_calls=paid_stage_calls,
        fresh_evaluation=False,
        scope=plan["scope"],
    )
    save(root / "summary.json", summary)
    lines = [
        "# Learned downstream report-verification demonstration",
        "",
        plan["scope"],
        "",
        "| Policy | Full success / 4 | Mean report recall | False reports | Duplicates |",
        "|---|---:|---:|---:|---:|",
    ]
    for p, row in summary["policies"].items():
        lines.append(
            f"| {p} | {row['full_success']}/4 | {row['mean_report_recall']:.4f} | "
            f"{row['false_reports']} | {row['duplicate_reports']} |"
        )
    lines.extend(
        [
            "",
            "Two development recordings and two intents are four conditions, not four "
            "independent environments. Both fitted models and all mission failures are retained. "
            "The model may reject support; it cannot lower the public evidence floor. "
            "Downstream latency, memory, and energy are assumptions. No current outcome was "
            "used to retrain, pick the threshold, or select an architecture.",
            "",
            "Actual learned forward calls (after the independent hard guard): "
            + json.dumps(forward_calls, sort_keys=True)
            + ". These differ from paid verifier-stage calls, which include guarded candidates.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    save(
        root / "completion.json",
        dict(
            complete=True,
            expected_runs=12,
            source_unchanged=True,
            artifacts_sha256={
                n: sha(root / n)
                for n in (
                    "plan.json",
                    "run_index.json",
                    "validation.json",
                    "summary.json",
                    "REPORT.md",
                )
            },
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify", "execute", "validate", "finalize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    try:
        globals()[args.action](args.output)
    except BaseException as exc:
        if args.output.exists() and not (args.output / "completion.json").exists():
            failure = args.output / (args.action + "_failure.json")
            if not failure.exists():
                save(
                    failure,
                    dict(error=type(exc).__name__, message=str(exc), no_automatic_repair=True),
                )
        raise
    print(json.dumps(dict(action=args.action, complete=True)))


if __name__ == "__main__":
    main()
