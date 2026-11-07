"""Freeze and compare adaptive perception routers on the unchanged paid runtime.

The study reuses recorded Gazebo RGB, cached actual predictions and the prior
deadline executor. It changes only the perception selection policy. Training
finishes before this plan is frozen; evaluation never updates learned heads.
Existing study sources and outcomes are retained without modification.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from experiments.lake_intent_contract import IntentContract
from experiments.mission_factor_sweeps.execution import (
    canonical_rows,
    load_shared_inputs,
    validate_workers,
)
from experiments.mission_joint_constraints import verify as verify_parent
from experiments.mission_routing_v2.flights import split_check
from experiments.mission_runtime_profile import REPO, read, save, sha
from experiments.mission_task_deadlines import study as original
from experiments.mission_task_deadlines.contract import DeadlineProfile, TaskKind
from experiments.perception_router_suite.learning import FEATURES, SCHEMA_VERSION, predict
from experiments.perception_router_suite.runner import execute_case

PARENT = original.PARENT
OUTPUT = REPO / "results/perception_router_suite_v1_20261028"
CONFIG = Path(__file__).with_name("config.json")
POLICIES = (
    "fixed_local_fast",
    "fixed_local_heavy",
    "fixed_remote_union",
    "deadline_aware",
    "resource_threshold",
    "accuracy_latency",
    "supervised_utility",
    "linucb_frozen",
)
DEFAULT_PROFILES = tuple(
    DeadlineProfile(f"perception_{seconds}s", seconds * 1_000_000_000, 150_000_000, 30_000_000)
    for seconds in (2, 4, 8)
)
TRAINING_FILES = ("profile.json", "chosen_arm_trace.json", "completion.json")
_WORKER_STATE = None


def require(value, message):
    if not value:
        raise ValueError(message)


@dataclass(frozen=True)
class StudyConfig:
    flights: tuple = ("east_shore_day", "west_rocks_day")
    missions: tuple = ("rapid_survey", "verified_inspection")
    batteries: tuple = (0.34, 0.65)
    links: tuple = ("good", "constrained", "intermittent")
    deadline_profiles: tuple = DEFAULT_PROFILES
    policies: tuple = POLICIES

    def __post_init__(self):
        choices = dict(
            flights={"east_shore_day", "west_rocks_day"},
            missions={"rapid_survey", "verified_inspection"},
            batteries={0.34, 0.65},
            links={"good", "constrained", "intermittent"},
            policies=set(POLICIES),
        )
        for name, values in vars(self).items():
            require(type(values) is tuple and bool(values), "Nonempty tuple levels required")
            require(len(set(values)) == len(values), "Duplicate factor levels")
            if name in choices:
                require(set(values) <= choices[name], "Unknown frozen input or policy level")
        require(
            all(type(v) in (int, float) and not isinstance(v, bool) for v in self.batteries),
            "Explicit numeric battery fractions required",
        )
        require(
            all(isinstance(p, DeadlineProfile) for p in self.deadline_profiles),
            "Typed deadline profiles required",
        )
        names = [p.name for p in self.deadline_profiles]
        require(len(names) == len(set(names)), "Distinct deadline profile names required")
        require(
            all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", n) for n in names),
            "Safe deadline profile names required",
        )
        for profile in self.deadline_profiles:
            require(
                profile in DEFAULT_PROFILES,
                "Use declared 2/4/8 second perception budgets with fixed downstream budgets",
            )

    def to_dict(self):
        return {
            k: [p.to_dict() for p in value] if k == "deadline_profiles" else list(value)
            for k, value in vars(self).items()
        }

    @classmethod
    def from_dict(cls, value):
        require(
            isinstance(value, dict) and set(value) == set(cls.__dataclass_fields__),
            "Missing or unknown study configuration fields",
        )
        require(all(type(v) is list for v in value.values()), "JSON factor arrays required")
        return cls(
            **{
                k: tuple(DeadlineProfile.from_dict(p) for p in v)
                if k == "deadline_profiles"
                else tuple(v)
                for k, v in value.items()
            }
        )


def specification(config=None):
    # The original builder accepts this identically shaped typed configuration.
    spec = original.specification(config or StudyConfig())
    spec.update(
        schema_version="perception_router_suite_spec_v1",
        comparison="Eight perception selection policies share the action pool, recorded "
        "trajectories, inference caches, task budgets, network, battery, intent, downstream "
        "stages and configured routing overhead. All cells are newly replayed.",
        training_protocol="Frozen development-trained supervised and chosen-arm LinUCB "
        "models; no evaluation-time labels, reward feedback or parameter updates.",
        routing_overhead="Common configured router latency and energy, including the "
        "learned policies; this study does not establish measured controller overhead parity.",
        policy_reproduction="Mechanism-level benchmark baselines, not full reproductions "
        "of ApproxDet, LiteReconfig, Chanakya or a confidence cascade.",
        analysis_protocol="Compare matched policy cells; report each intent separately, "
        "mission and phase success, deadline admission/misses, energy, bytes and selected "
        "action counts. Two reused deterministic flights do not support population claims.",
        evaluation_updates_learned_models=False,
        historical_outcomes_reused=False,
    )
    return spec


def validate_training(training, parent):
    """Verify training provenance, held-out flights and selected-arm feedback receipts."""
    training = Path(training)
    profile, trace, completion = (read(training / name) for name in TRAINING_FILES)
    require(
        completion.get("complete") is True
        and completion.get("profile_sha256") == sha(training / "profile.json")
        and completion.get("trace_sha256") == sha(training / "chosen_arm_trace.json"),
        "Incomplete or changed training artifacts",
    )
    require(
        profile.get("schema_version") == SCHEMA_VERSION
        and profile.get("feature_names") == list(FEATURES),
        "Wrong learned profile or context schema",
    )
    actions = parent["actions_by_scale"]["1.0"]
    signatures = {a["action_id"]: a["source_signature"] for a in actions}
    sources = {a["action_id"]: a["source_config_id"] for a in actions}
    require(
        profile.get("action_signatures") == signatures and profile.get("action_sources") == sources,
        "Learned action identity differs from frozen runtime",
    )
    for section in ("supervised", "bandit"):
        require(set(profile[section]) == set(signatures), "Incomplete learned action pool")
        for action in signatures:
            predict(profile[section], action, [1.0] + [0.0] * (len(FEATURES) - 1))
    metadata = profile["training"]
    require(
        metadata.get("original_recordings_disjoint_from_evaluation") is True
        and metadata.get("evaluation_labels_used") is False
        and metadata.get("evaluation_feedback") is False,
        "Evaluation labels or feedback are forbidden during training",
    )
    for case in parent["cases"].values():
        split_check(profile["training_lineage"], case["lineage"])
    entries, contexts = trace["trace"], metadata["augmented_contexts"]
    require(
        type(contexts) is int
        and contexts > 0
        and trace.get("schema_version") == "selected_arm_training_trace_v1"
        and trace.get("seed") == metadata["seed"]
        and len(entries) == contexts
        and len(trace["context_description"]) == contexts
        and trace["reward_queries"] == trace["parameter_updates"] == contexts
        and metadata["bandit_reward_queries"] == metadata["bandit_parameter_updates"] == contexts
        and metadata["supervised_reward_queries"] == contexts * len(signatures),
        "Training feedback counts differ from declared contexts",
    )
    updates, visited = Counter(), set()
    for step, row in enumerate(entries):
        action, context = row["action_id"], row["context_index"]
        require(
            action in signatures
            and row["step"] == step
            and type(context) is int
            and 0 <= context < contexts
            and context not in visited
            and type(row["feedback_arms"]) is int
            and row["feedback_arms"] == 1
            and type(row["reward"]) in (int, float)
            and math.isfinite(row["reward"])
            and 0 <= row["reward"] <= 1,
            "Invalid chosen-arm training receipt",
        )
        visited.add(context)
        updates[action] += 1
        require(row["action_update_count"] == updates[action], "Incorrect arm update counter")
    require(
        dict(updates) == metadata["bandit_updates_by_action"]
        and all(profile["bandit"][a]["updates"] == updates[a] for a in signatures),
        "Bandit heads and selected-arm feedback ledger disagree",
    )
    dependencies = {}
    for field in ("input_sha256", "source_sha256"):
        require(
            isinstance(metadata[field], dict) and metadata[field], "Missing training provenance"
        )
        for path, expected in metadata[field].items():
            require(sha(path) == expected, "Changed training dependency: " + path)
            require(
                path not in dependencies or dependencies[path] == expected, "Conflicting hashes"
            )
            dependencies[path] = expected
    return dependencies


def prepare(root=OUTPUT, config=None, *, training, workers=4, config_path=None):
    root, training = Path(root).resolve(), Path(training).resolve()
    require(not root.exists(), "Preserve existing study output")
    validate_workers(workers)
    config = config or StudyConfig()
    require(isinstance(config, StudyConfig), "Typed router study configuration required")
    if config_path is not None:
        config_path = Path(config_path).resolve()
        require(
            StudyConfig.from_dict(read(config_path)) == config,
            "Configuration file and requested factors differ",
        )
    for name in TRAINING_FILES:
        require((training / name).is_file(), "Missing finalized training input: " + name)
        read(training / name)  # Reject malformed JSON before creating output.
    parent = verify_parent(PARENT)
    require(read(PARENT / "completion.json")["complete"] is True, "Incomplete parent study")
    training_inputs = validate_training(training, parent)
    training_audit = training / "training_audit.json"
    if training_audit.exists():
        audit = read(training_audit)
        require(
            audit.get("all_passed") is True
            and audit.get("profile_sha256") == sha(training / "profile.json")
            and audit.get("trace_sha256") == sha(training / "chosen_arm_trace.json")
            and audit.get("completion_sha256") == sha(training / "completion.json")
            and sha(audit["auditor_source_path"]) == audit["auditor_source_sha256"],
            "Training reconstruction audit is stale or failed",
        )
        training_inputs[str(training_audit)] = sha(training_audit)
        training_inputs[audit["auditor_source_path"]] = audit["auditor_source_sha256"]
    spec = specification(config)
    require(all(row["cell"] in parent["cases"] for row in spec["grid"]), "Missing parent input")
    source_paths = dict(parent["implementation_sha256"])
    sources = {
        *Path(__file__).parent.glob("*.py"),
        CONFIG,
        *REPO.glob("experiments/mission_task_deadlines/*.py"),
        REPO / "experiments/mission_task_deadlines/defaults.json",
        *REPO.glob("experiments/mission_pipeline_routing/*.py"),
        REPO / "experiments/mission_factor_sweeps/execution.py",
        REPO / "experiments/mission_task_deadline_audit.py",
        *REPO.glob("tests/test_mission_task_deadline*.py"),
        *REPO.glob("tests/test_perception_router*.py"),
    }
    for source in sources:
        source_paths[str(source.resolve())] = sha(source)
    inputs = dict(parent["input_sha256"])
    for path, expected in training_inputs.items():
        require(
            path not in inputs or inputs[path] == expected, "Conflicting inherited training hash"
        )
        inputs[path] = expected
    for relative, expected in parent["copied_input_sha256"].items():
        inputs[str((PARENT / relative).resolve())] = expected
    for name in ("plan.json", "completion.json", "validation.json"):
        inputs[str((PARENT / name).resolve())] = sha(PARENT / name)
    for name in TRAINING_FILES:
        inputs[str(training / name)] = sha(training / name)
    if config_path is not None:
        inputs[str(config_path)] = sha(config_path)
    root.mkdir(parents=True)
    snapshots, copied_training = {}, {}
    for path, expected in source_paths.items():
        require(sha(path) == expected, "Changed source before freeze")
        relative = Path("source_snapshot") / Path(path).relative_to(REPO)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha(target) == expected, "Source changed during snapshot")
        snapshots[str(relative)] = expected
    for name in TRAINING_FILES:
        relative = Path("training") / name
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(training / name, target)
        require(sha(target) == inputs[str(training / name)], "Training changed during freeze")
        copied_training[str(relative)] = sha(target)
    save(root / "specification.json", spec)
    plan = dict(
        spec,
        schema_version="perception_router_suite_plan_v1",
        parent_root=str(PARENT),
        parent_plan_sha256=sha(PARENT / "plan.json"),
        prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_input_sha256=inputs,
        implementation_sha256=source_paths,
        snapshot_sha256=snapshots,
        training_source_root=str(training),
        training_artifacts_sha256=copied_training,
        learned_profile_path=str(root / "training/profile.json"),
        chosen_arm_trace_path=str(root / "training/chosen_arm_trace.json"),
        worker_count=workers,
        process_start_method="fork" if workers > 1 else "in_process",
        outcome_files_exist_at_freeze=False,
    )
    save(root / "plan.json", plan)
    return verify(root)


def verify(root=OUTPUT):
    root = Path(root).resolve()
    plan = read(root / "plan.json")
    require(plan["schema_version"] == "perception_router_suite_plan_v1", "Unknown study schema")
    spec = specification(StudyConfig.from_dict(plan["factors"]))
    for key, expected in spec.items():
        if key != "schema_version":
            require(plan[key] == expected, "Changed frozen design: " + key)
    require(read(root / "specification.json") == spec, "Changed specification")
    require(plan["outcome_files_exist_at_freeze"] is False, "Incorrect prospective freeze claim")
    validate_workers(plan["worker_count"])
    require(
        plan["process_start_method"] == ("fork" if plan["worker_count"] > 1 else "in_process"),
        "Changed process isolation method",
    )
    for field in ("source_input_sha256", "implementation_sha256"):
        for path, expected in plan[field].items():
            require(sha(path) == expected, "Changed frozen input/source: " + path)
    for field in ("snapshot_sha256", "training_artifacts_sha256"):
        for relative, expected in plan[field].items():
            path = (root / relative).resolve()
            require(
                path.is_relative_to(root) and sha(path) == expected,
                "Changed or escaped frozen copy: " + relative,
            )
    require(
        set(plan["training_artifacts_sha256"])
        == {str(Path("training") / name) for name in TRAINING_FILES},
        "Incomplete copied training inventory",
    )
    require(
        plan["learned_profile_path"] == str(root / "training/profile.json")
        and plan["chosen_arm_trace_path"] == str(root / "training/chosen_arm_trace.json"),
        "Learned input path escapes frozen training inputs",
    )
    for name in TRAINING_FILES:
        require(
            plan["training_artifacts_sha256"][str(Path("training") / name)]
            == plan["source_input_sha256"][str(Path(plan["training_source_root"]) / name)],
            "Training copy differs from source binding",
        )
    require(
        sha(Path(plan["parent_root"]) / "plan.json") == plan["parent_plan_sha256"],
        "Changed parent plan",
    )
    training_inputs = validate_training(
        root / "training", read(Path(plan["parent_root"]) / "plan.json")
    )
    require(
        all(
            plan["source_input_sha256"].get(path) == expected
            for path, expected in training_inputs.items()
        ),
        "Training dependencies missing from frozen input inventory",
    )
    if (root / "execution_started.json").exists():
        started = read(root / "execution_started.json")
        require(started["plan_sha256"] == sha(root / "plan.json"), "Plan changed after start")
    if (root / "execution_receipt.json").exists():
        receipt = read(root / "execution_receipt.json")
        require(
            receipt["complete"] is True
            and receipt["fresh_runs"] == plan["expected_runs"]
            and receipt["historical_outputs_reused"] is False
            and receipt["worker_count"] == plan["worker_count"]
            and receipt["wall_time_used_for_simulated_costs"] is False,
            "Incorrect completed execution declaration",
        )
        for name, expected in receipt["artifacts_sha256"].items():
            path = (root / name).resolve()
            require(
                path.is_relative_to(root) and sha(path) == expected,
                "Changed execution receipt binding",
            )
    if (root / "completion.json").exists():
        completion = read(root / "completion.json")
        require(
            completion["complete"] is True and completion["runs"] == plan["expected_runs"],
            "Incorrect completion declaration",
        )
        for name, expected in completion["artifacts_sha256"].items():
            path = (root / name).resolve()
            require(
                path.is_relative_to(root) and sha(path) == expected, "Changed completed artifact"
            )
    return plan


def _execute_item(item):
    require(_WORKER_STATE is not None, "Missing inherited immutable execution inputs")
    root, plan, terrain, banks, manifests = _WORKER_STATE
    return execute_case(root, plan, item, terrain, banks[item["flight"]], manifests[item["flight"]])


def execute(root=OUTPUT):
    global _WORKER_STATE
    root = Path(root).resolve()
    plan = verify(root)
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
    workers = plan["worker_count"]
    if workers > 1:
        require("fork" in multiprocessing.get_all_start_methods(), "Frozen plan requires fork")
    started = time.perf_counter()
    save(
        root / "execution_started.json",
        dict(
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
            dict(
                plan_sha256=sha(root / "plan.json"),
                cases=rows,
                complete_grid=True,
            ),
        )
        save(
            root / "execution_receipt.json",
            dict(
                schema_version="perception_router_suite_execution_receipt_v1",
                complete=True,
                fresh_runs=len(rows),
                historical_outputs_reused=False,
                worker_count=workers,
                wall_time_s=time.perf_counter() - started,
                wall_time_used_for_simulated_costs=False,
                artifacts_sha256={
                    name: sha(root / name)
                    for name in ("plan.json", "execution_started.json", "run_index.json")
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
                note="Partial artifacts are preserved; never retry an execution in place.",
            ),
        )
        raise
    finally:
        _WORKER_STATE = None


def summarize(root=OUTPUT):
    root = Path(root).resolve()
    plan = verify(root)
    require(not (root / "completion.json").exists(), "Preserve final study certificate")
    require((root / "execution_receipt.json").exists(), "Fresh execution receipt required")
    index = read(root / "run_index.json")
    require(
        index["plan_sha256"] == sha(root / "plan.json") and index["complete_grid"] is True,
        "Invalid execution index",
    )
    require(canonical_rows(plan["grid"], index["cases"]) == index["cases"], "Changed grid order")
    counts, bindings = Counter(), {}
    for row in index["cases"]:
        destination = (root / row["path"]).resolve()
        require(destination.is_relative_to(root), "Run path escapes study")
        for name, expected in row["artifacts_sha256"].items():
            path = (destination / name).resolve()
            require(
                path.is_relative_to(destination) and sha(path) == expected, "Changed run artifact"
            )
            bindings[str(path.relative_to(root))] = expected
        events, result, outcome = (
            read(destination / (n + ".json")) for n in ("events", "results", "outcome")
        )
        contract = IntentContract(read(destination / "intent.json"))
        require(
            original.evaluate_recoverable(contract, events, result) == outcome,
            "Changed canonical outcome scoring",
        )
        require(outcome["task_metrics"] == row["task_metrics"], "Changed task counts")
        require(outcome["mission_success"] == row["mission_success"], "Changed mission success")
        counts["decisions"] += len(events)
        counts["reports"] += len(result["report_quality"]["reports"])
        for task, value in outcome["task_metrics"].items():
            counts[task + "_attempts"] += value["attempts"]
            counts[task + "_deadline_misses"] += value["deadline_misses"]

    def aggregate(rows):
        return dict(
            runs=len(rows),
            mission_successes=sum(r["mission_success"] for r in rows),
            successful_phases=sum(r["successful_phases"] for r in rows),
            total_phases=4 * len(rows),
            below_reserve=sum(r["final_battery_frac"] < 0.2 for r in rows),
            abstentions=sum(r["abstentions"] for r in rows),
            task_metrics={
                task.value: {
                    key: sum(r["task_metrics"][task.value][key] for r in rows)
                    for key in rows[0]["task_metrics"][task.value]
                }
                for task in TaskKind
            },
        )

    policies = {
        policy: aggregate([r for r in index["cases"] if r["policy"] == policy])
        for policy in plan["factors"]["policies"]
    }
    by_intent = {
        mission: {
            policy: aggregate(
                [r for r in index["cases"] if r["policy"] == policy and r["mission"] == mission]
            )
            for policy in plan["factors"]["policies"]
        }
        for mission in plan["factors"]["missions"]
    }
    summary = dict(
        runs=len(index["cases"]),
        policies=policies,
        by_intent=by_intent,
        totals=dict(counts),
        scope=plan["scope"],
        strict_intent_feasibility=plan["strict_intent_feasibility"],
    )
    save(root / "summary.json", summary)
    save(
        root / "validation.json",
        dict(
            complete=True,
            output_sha256=bindings,
            complete_grid=True,
            counts=dict(counts),
            validation_scope="Artifact hashes and canonical outcome/metric reconstruction; "
            "not an independent implementation of the complete scheduler or selection policies.",
        ),
    )
    save(
        root / "completion.json",
        dict(
            complete=True,
            runs=len(index["cases"]),
            completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            artifacts_sha256={
                name: sha(root / name)
                for name in (
                    "plan.json",
                    "run_index.json",
                    "summary.json",
                    "validation.json",
                    "execution_receipt.json",
                )
            },
        ),
    )
    verify(root)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "summarize", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.training is None:
            parser.error("prepare requires --training with finalized profile and chosen-arm trace")
        prepare(
            args.output,
            StudyConfig.from_dict(read(args.config)),
            training=args.training,
            workers=args.workers,
            config_path=args.config,
        )
    elif args.action == "execute":
        execute(args.output)
    elif args.action == "summarize":
        summarize(args.output)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
