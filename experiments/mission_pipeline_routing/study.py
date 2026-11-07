"""Predeclared development-only task-routing demonstration on fixed real model masks.

The original intents, trajectory and perception checkpoint are unchanged. Classical
association and report verification are independently selected. New downstream
costs are assumptions; only the fixed upstream neural latency uses an existing
controlled host-p95 measurement. No learned verifier is claimed in this version.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_pipeline_routing.catalog import default_catalog
from experiments.mission_pipeline_routing.policy import POLICIES, select
from experiments.mission_pipeline_routing.runtime import (
    CompletedPrediction,
    Pipeline,
    PublicContext,
    PublicPose,
)
from experiments.mission_routing_v2.flights import load_scenario
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_runtime_profile import (
    DEVELOPMENT,
    FLIGHTS,
    MISSIONS,
    REPO,
    implementation_inventory,
    read,
    save,
    sha,
)
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.v4_gazebo_pilot.timing import seconds_to_ns

OUTPUT = REPO / "results/mission_pipeline_routing_v1_20261027"
CALIBRATION = REPO / "results/mission_execution_profile_v1_20261027"
MODEL = "CFG_YOLO11S_960_ROT0"


def require(value, message):
    if not value:
        raise ValueError(message)


def plain(value):
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def specification():
    return dict(
        schema_version="task_pipeline_development_spec_v1",
        flights=list(FLIGHTS),
        missions=list(MISSIONS),
        policies=list(POLICIES),
        perception_config_id=MODEL,
        grid=[
            dict(flight=f, mission=m, policy=p) for f in FLIGHTS for m in MISSIONS for p in POLICIES
        ],
        expected_runs=32,
        catalog=default_catalog().document(),
        upstream_latency="existing controlled host warm p95, rounded to integer nanoseconds",
        upstream_energy="15W times host-p95 latency; assumed, not measured electrical energy",
        upstream_loading="original scenario single-resident cold-start latency/energy/memory",
        downstream_routing_latency_ns=50_000_000,
        downstream_routing_energy_j=0.75,
        downstream_costs="authored assumptions from typed catalog; not profiled on host or onboard",
        fixed_path=True,
        navigation_action_available=False,
        unchanged_intent_contracts=True,
        report_guard="max(selected rule and active public phase hits/span); "
        "current-phase evidence only",
        switch="single resident per slot; load charge and temporal reset; independent slots",
        cancellation="paid in full; residency reset remains; "
        "no selected-mask history/report commit",
        scope="development demo; classical downstream algorithms, fixed learned perception; "
        "no heterogeneous learned-model superiority, held-out generalization, field or power claim",
        learned_downstream_models=False,
        new_neural_inference=False,
        report_verifier_training=False,
        earlier_development_outcomes_known=True,
    )


def scenario_for(root, plan, flight, mission):
    return load_scenario(
        Path(root) / "missions" / flight / mission / "scenario.json",
        plan["cases"][flight]["environment"],
        Path(plan["assets"]),
    )


def prepare(root):
    from experiments.mission_execution_profile import verify as verify_calibration
    from experiments.mission_pipeline_routing import audit
    from experiments.mission_policy_selection.protocol import verify as verify_development

    root = Path(root)
    require(
        root.resolve() == OUTPUT.resolve() and not root.exists(),
        "Separate unused versioned output required",
    )
    development = verify_development(DEVELOPMENT)
    profile = verify_calibration(CALIBRATION)
    require(
        profile["complete"] and profile["eligible_for_sensitivity"],
        "Completed compatible host profile required",
    )
    measured = profile["profiles"][MODEL]
    require(
        measured["prediction_parity"] and measured["warm_samples"] == 24,
        "Fixed upstream parity required",
    )
    source = Path(development["cohort"])
    root.mkdir(parents=True)
    save(root / "specification.json", specification())
    frozen, copied = {}, {}

    def pin(path):
        frozen[str(Path(path).resolve())] = sha(path)

    def copy(path, relative):
        pin(path)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha(target) == sha(path), "Copied input differs")
        copied[str(relative)] = sha(target)

    pin(DEVELOPMENT / "protocol.json")
    pin(source / "protocol.json")
    for name in ("profile.json", "plan.json", "completion.json"):
        pin(CALIBRATION / name)
    cases = {}
    for flight in FLIGHTS:
        case = development["cases"][flight]
        require(case["role"] == "development", "Evaluation flight forbidden")
        cases[flight] = case
        manifest = checked_manifest(source / flight / "predictions", source / "protocol.json")
        require(
            manifest["models"][MODEL]["signature"] == measured["signature"],
            "Upstream calibration identity mismatch",
        )
        for name in (
            "recording/recording.json",
            "predictions/manifest.json",
            "predictions/completion.json",
            "predictions/inference.json",
        ):
            pin(source / flight / name)
        for mission in MISSIONS:
            for name in ("scenario.json", "intent.json", "contract.json"):
                copy(
                    DEVELOPMENT / flight / mission / name,
                    Path("missions") / flight / mission / name,
                )
    _, metadata = load_public_terrain(Path(development["assets"]))
    pin(metadata["raster_path"])
    pin(Path(development["assets"]) / "harmonic_terrain/model.sdf")
    inventory = implementation_inventory()
    for path in [
        *Path(__file__).parent.glob("*.py"),
        Path(audit.__file__),
        *REPO.glob("tests/test_mission_pipeline_routing*.py"),
    ]:
        inventory[str(path.resolve())] = sha(path)
    for path, expected in inventory.items():
        relative = Path("source_snapshot") / Path(path).relative_to(REPO)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        require(sha(target) == expected, "Copied source differs")
        copied[str(relative)] = expected
    plan = dict(
        specification(),
        schema_version="task_pipeline_study_v1",
        source=str(source),
        assets=development["assets"],
        cases=cases,
        calibration=str(CALIBRATION),
        calibration_sha256=sha(CALIBRATION / "profile.json"),
        upstream_signature=measured["signature"],
        upstream_latency_ns=seconds_to_ns(measured["warm_p95_s"]),
        upstream_energy_j=15 * measured["warm_p95_s"],
        source_sha256=inventory,
        input_sha256=frozen,
        copied_sha256=copied,
        frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        outcomes_existed_at_freeze=False,
    )
    save(root / "plan.json", plan)
    verify(root)
    return plan


def verify(root):
    root = Path(root)
    plan = read(root / "plan.json")
    require(plan["schema_version"] == "task_pipeline_study_v1", "Unknown pipeline version")
    for key, value in specification().items():
        if key != "schema_version":
            require(plan[key] == value, "Changed predeclared study: " + key)
    require(read(root / "specification.json") == specification(), "Changed specification")
    for key in ("source_sha256", "input_sha256"):
        for path, expected in plan[key].items():
            require(sha(path) == expected, "Changed locked input/source: " + path)
    for path, expected in plan["copied_sha256"].items():
        target = (root / path).resolve()
        require(
            target.is_relative_to(root.resolve()) and sha(target) == expected,
            "Changed copied input/source",
        )
    development = read(DEVELOPMENT / "protocol.json")
    require(
        plan["source"] == development["cohort"]
        and plan["assets"] == development["assets"]
        and plan["cases"] == {f: development["cases"][f] for f in FLIGHTS},
        "Changed source recordings, map prior or development role",
    )
    for flight in FLIGHTS:
        require(plan["cases"][flight]["role"] == "development", "Evaluation flight forbidden")
        for mission in MISSIONS:
            for name in ("scenario.json", "intent.json", "contract.json"):
                require(
                    sha(root / "missions" / flight / mission / name)
                    == sha(DEVELOPMENT / flight / mission / name),
                    "Original mission/scenario contract changed",
                )
    measured = read(Path(plan["calibration"]) / "profile.json")["profiles"][MODEL]
    require(
        plan["upstream_signature"] == measured["signature"]
        and plan["upstream_latency_ns"] == seconds_to_ns(measured["warm_p95_s"])
        and plan["upstream_energy_j"] == 15 * measured["warm_p95_s"],
        "Changed upstream measured assignment",
    )
    return plan


def public_context(contract, capture_ns, now_ns, battery, memory):
    public = contract.public_document()
    phase = next(
        p
        for p in public["mission"]["phases"]
        if seconds_to_ns(p["start_s"], allow_zero=True) <= capture_ns < seconds_to_ns(p["end_s"])
    )
    return PublicContext(
        now_ns,
        battery,
        seconds_to_ns(public["limits"]["deadline_s"]),
        phase["phase_id"],
        seconds_to_ns(phase["start_s"], allow_zero=True),
        seconds_to_ns(phase["end_s"]),
        seconds_to_ns(phase["deadline_s"]),
        phase["min_hits"],
        seconds_to_ns(phase["min_evidence_span_s"], allow_zero=True),
        4096,
        memory,
    )


def result_from(
    scenario,
    contract,
    events,
    quality,
    reports,
    *,
    time_s,
    energy,
    upstream_j,
    upstream_load_j,
    pipeline_j,
    memory,
    termination,
    skipped,
):
    capacity = scenario.battery_capacity_wh * 3600
    battery = max(0.0, scenario.initial_battery_frac - energy / capacity)
    extra, first = scenario.requirements.evaluate(quality, events, reports)
    flags = dict(
        extra,
        memory_constraint_success=memory <= 4096,
        quality_success=scenario.contract.quality_satisfied(
            quality[scenario.contract.quality_metric]
        ),
        deadline_success=time_s <= scenario.contract.deadline_s,
        battery_constraint_success=battery >= scenario.contract.min_final_battery_frac,
        communication_constraint_success=True,
        privacy_constraint_success=True,
    )
    return dict(
        schema_version="task_pipeline_execution_v1",
        contract=asdict(scenario.contract),
        status="completed",
        error=None,
        termination_reason=termination,
        mission_completed=termination == "mission_window_complete",
        final_time_s=time_s,
        final_battery_frac=battery,
        cumulative_energy_j=energy,
        cumulative_communication_mb=0.0,
        inference_energy_j=upstream_j,
        model_load_energy_j=upstream_load_j,
        pipeline_energy_j=pipeline_j,
        flight_energy_j=time_s * scenario.flight_power_w,
        peak_configured_memory_mb=memory,
        processed_observations=len(events),
        skipped_observation_ids=skipped,
        router_errors=0,
        inference_failures=0,
        invalid_actions=0,
        privacy_violations=0,
        constraints=flags,
        quality=quality,
        report_quality=reports,
        first_correct_report_s=first,
        mission_success=termination == "mission_window_complete" and all(flags.values()),
        contract_sha256=contract.sha256,
        recording_identity=scenario.recording_identity,
    )


def execute_case(root, plan, item, terrain, *, pipeline_factory=None, policy_selector=None):
    """Execute one locked condition; optional factories receive public data only.

    Versioned studies must bind injected implementations and checkpoint identities
    in their own plans. The classical CLI always uses the defaults.
    """
    flight, mission, policy = (item[k] for k in ("flight", "mission", "policy"))
    destination = Path(root) / "runs" / flight / mission / policy
    require(not destination.exists(), "Preserve an existing pipeline execution")
    destination.mkdir(parents=True)
    contract = IntentContract(read(Path(root) / "missions" / flight / mission / "intent.json"))
    scenario = bind_scenario(scenario_for(root, plan, flight, mission), contract)
    spec = next(s for s in scenario.executors if s.config_id == MODEL)
    require(
        model_signature(spec) == plan["upstream_signature"],
        "Changed actual learned upstream implementation",
    )
    source = Path(plan["source"])
    manifest = checked_manifest(source / flight / "predictions", source / "protocol.json")
    require(
        manifest["recording_identity"] == scenario.recording_identity, "Changed physical flight"
    )
    masks = VerifiedMasks(source / flight / "predictions", manifest["models"][MODEL])
    resource = scenario.resources.for_config(MODEL)
    pipeline = (pipeline_factory or Pipeline)(terrain=terrain)
    scorer = MissionEvaluator(
        scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
    )
    reporter = RunnerReportEvaluator(scenario.matching_iou_threshold, scenario.targets.values(), 2)
    backend = ReplayBackend(scenario, source / flight / "recording")
    schedule = scenario.schedule
    capacity = scenario.battery_capacity_wh * 3600
    energy = upstream_j = load_j = memory = 0.0
    current_time = 0.0
    index = 0
    events, skipped_ids = [], []
    termination = "mission_window_complete"
    try:
        while schedule.at(index) < schedule.end_ns:
            capture = schedule.at(index)
            capture_relative = capture - schedule.epoch_ns
            if scenario.initial_battery_frac - energy / capacity <= 0:
                termination = "battery_depleted"
                break
            if (
                current_time > scenario.contract.deadline_s
                or capture_relative / 1e9 > scenario.contract.deadline_s
            ):
                termination = "deadline_exceeded"
                break
            backend.advance_to(capture)
            frame = backend.capture()
            key = rgb_key(frame.rgb)
            prediction = masks[key]
            cold = not events
            load_ns = seconds_to_ns(resource.load_latency_s, allow_zero=True) if cold else 0
            load_energy = resource.load_energy_j if cold else 0.0
            upstream_ready = capture_relative + load_ns + plan["upstream_latency_ns"]
            upstream_j += plan["upstream_energy_j"]
            load_j += load_energy
            energy = (
                upstream_ready / 1e9 * scenario.flight_power_w
                + upstream_j
                + load_j
                + pipeline.cumulative_energy_j
            )
            battery = max(0.0, scenario.initial_battery_frac - energy / capacity)
            context = public_context(
                contract, capture_relative, upstream_ready, battery, resource.memory_mb
            )
            history = plain(pipeline.snapshot(upstream_ready))
            choice = (policy_selector or select)(policy, context, history)
            completed = CompletedPrediction(
                MODEL,
                capture_relative,
                upstream_ready,
                prediction,
                PublicPose(
                    tuple(frame.body_pose["position_m"]),
                    tuple(frame.body_pose["quaternion_xyzw"]),
                    scenario.world.get("camera_pitch_rad", math.pi / 2),
                ),
            )
            pending = pipeline.dispatch(choice, completed, context)
            completion, next_index, skipped = schedule.complete(
                index, pending.ready_ns - capture_relative
            )
            relative_completion = completion - schedule.epoch_ns
            for missed in skipped:
                backend.advance_to(schedule.at(missed))
                missed_frame = backend.capture()
                truth = evaluation_frame(
                    missed_frame, missed, (schedule.at(missed) - schedule.epoch_ns) / 1e9, scenario
                )
                scorer.note_skipped_visibility(truth.visible_target_ids)
                reporter.note_visibility(truth)
            backend.advance_to(completion)
            output = pipeline.complete(pending.job_id, relative_completion)
            current_time = relative_completion / 1e9
            energy = (
                current_time * scenario.flight_power_w
                + upstream_j
                + load_j
                + pipeline.cumulative_energy_j
            )
            memory = max(memory, pending.resident_memory_mb)
            truth = evaluation_frame(frame, index, capture_relative / 1e9, scenario)
            score = scorer.update(truth, prediction)
            reporter.update(truth, prediction, output["emitted_reports"])
            events.append(
                dict(
                    observation_id=index,
                    capture_timestamp_ns=capture,
                    capture_time_s=capture_relative / 1e9,
                    completion_time_s=current_time,
                    capture_body_pose=frame.body_pose,
                    source_rgb_key=key,
                    selected_mask_sha256=hashlib.sha256(prediction.tobytes()).hexdigest(),
                    upstream_available_ns=upstream_ready,
                    upstream_load_latency_ns=load_ns,
                    upstream_load_energy_j=load_energy,
                    upstream_latency_ns=plan["upstream_latency_ns"],
                    upstream_energy_j=plan["upstream_energy_j"],
                    public_context=asdict(context),
                    public_history=history,
                    selection=asdict(choice),
                    pipeline=output,
                    execution=dict(
                        config_id=MODEL, success=True, predicted_pixels=int(prediction.sum())
                    ),
                    score=score.to_dict(),
                    emitted_reports=output["emitted_reports"],
                    battery_after=max(0.0, scenario.initial_battery_frac - energy / capacity),
                    cumulative_energy_j=energy,
                    skipped_observation_ids=list(skipped),
                )
            )
            skipped_ids.extend(skipped)
            index = next_index
        if termination == "mission_window_complete":
            current_time = max(current_time, scenario.duration_s)
            energy = (
                current_time * scenario.flight_power_w
                + upstream_j
                + load_j
                + pipeline.cumulative_energy_j
            )
            if scenario.initial_battery_frac - energy / capacity <= 0:
                termination = "battery_depleted"
    finally:
        backend.close()
    result = result_from(
        scenario,
        contract,
        events,
        scorer.quality_details(current_time),
        reporter.summary(current_time),
        time_s=current_time,
        energy=energy,
        upstream_j=upstream_j,
        upstream_load_j=load_j,
        pipeline_j=pipeline.cumulative_energy_j,
        memory=memory,
        termination=termination,
        skipped=skipped_ids,
    )
    result.update(
        policy=policy,
        selected_executor_calls=pipeline.executed_calls,
        association_resets=sum(
            "association" in e["pipeline"]["costs"]["reset_slots"] for e in events
        ),
        verification_resets=sum(
            "report_verification" in e["pipeline"]["costs"]["reset_slots"] for e in events
        ),
    )
    outcome = evaluate(contract, events, result)
    for name, value in (
        ("events.json", events),
        ("results.json", result),
        ("outcome.json", outcome),
    ):
        save(destination / name, value)
    return dict(
        item,
        path=str(destination.relative_to(root)),
        calls=len(events),
        full_success=outcome["mission_success"],
        phases_succeeded=sum(p["success"] for p in outcome["phases"]),
        report_recall=outcome["reporting"]["recall_by_deadline"],
        failed_constraints=outcome["failed_constraints"],
        report_count=result["report_quality"]["total_reports"],
        false_reports=result["report_quality"]["false_reports"],
        duplicate_reports=result["report_quality"]["duplicate_correct_reports"],
        final_battery=result["final_battery_frac"],
        pipeline_energy_j=pipeline.cumulative_energy_j,
        association_resets=result["association_resets"],
        verification_resets=result["verification_resets"],
        selections=dict(
            Counter(
                e["selection"]["association_id"] + "/" + e["selection"]["verification_id"]
                for e in events
            )
        ),
        artifacts_sha256={
            n: sha(destination / n) for n in ("events.json", "results.json", "outcome.json")
        },
    )


def execute(root):
    root = Path(root)
    plan = verify(root)
    require(not (root / "run_index.json").exists(), "Preserve previous executions")
    terrain, _ = load_public_terrain(Path(plan["assets"]))
    rows = []
    for item in plan["grid"]:
        verify(root)
        rows.append(execute_case(root, plan, item, terrain))
        save(root / "progress.json", dict(completed=len(rows), expected=plan["expected_runs"]))
        print(f"Completed {len(rows)}/{plan['expected_runs']}", flush=True)
    save(
        root / "run_index.json",
        dict(complete_grid=True, plan_sha256=sha(root / "plan.json"), cases=rows),
    )
    return rows


def finalize(root):
    root = Path(root)
    plan = verify(root)
    index = read(root / "run_index.json")
    validation = read(root / "validation.json")
    require(
        validation["complete"]
        and validation["plan_sha256"] == sha(root / "plan.json")
        and validation["run_index_sha256"] == sha(root / "run_index.json"),
        "Complete bound independent audit required",
    )
    auditor_path = str(Path(__file__).with_name("audit.py").resolve())
    require(
        validation["auditor_sha256"] == plan["source_sha256"][auditor_path]
        and validation["runs"] == plan["expected_runs"] == len(index["cases"])
        and validation["decisions"]
        == validation["independent_cost_checks"]
        == sum(row["calls"] for row in index["cases"]),
        "Audit coverage/source mismatch",
    )
    require(
        all(
            validation[key] is True
            for key in (
                "all_public_decisions_reselected",
                "raw_selected_masks_and_reports_reconstructed",
                "clock_resources_and_terminal_flags_reconstructed",
                "original_canonical_endpoint_verified",
                "no_mission_runner_called",
            )
        ),
        "Incomplete independent reconstruction",
    )
    require(not (root / "completion.json").exists(), "Preserve final results")
    for relative, expected in validation["output_sha256"].items():
        require(sha(root / relative) == expected, "Changed audited output")
    rows = index["cases"]
    summary = dict(
        complete=True,
        runs=len(rows),
        decisions=sum(r["calls"] for r in rows),
        policies={
            p: dict(
                runs=len([r for r in rows if r["policy"] == p]),
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
            dict(
                flight=r["flight"],
                mission=r["mission"],
                policy=r["policy"],
                failed_constraints=r["failed_constraints"],
            )
            for r in rows
            if not r["full_success"]
        ],
        fresh_evaluation=False,
        learned_downstream_models=False,
        scope=plan["scope"],
    )
    save(root / "summary.json", summary)
    lines = [
        "# Classical downstream task-routing demonstration",
        "",
        plan["scope"],
        "",
        "Original mission contracts and physical recordings are unchanged. "
        "The fixed learned upstream is YOLO11s; "
        "three association algorithms and two selected evidence verifiers are classical. "
        "Downstream costs are authored assumptions. Strict intent can make the rules coincide.",
        "",
        "| Policy | Full success / 4 | Mean report recall | False reports | Duplicate reports |",
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
            "Four conditions combine two development recordings and two intents. "
            "They are not four independent environments.",
            "All condition outcomes and failures remain in run_index.json and summary.json. "
            "Neither learned heterogeneous routing nor adaptive superiority is established.",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    save(
        root / "completion.json",
        dict(
            complete=True,
            expected_runs=plan["expected_runs"],
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "verify", "finalize"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = globals()[args.action](args.output)
    print(json.dumps({"action": args.action, "complete": True}))
