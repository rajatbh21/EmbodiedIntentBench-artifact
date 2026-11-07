"""Integrated task-deadline, network, battery and intent development replay.

The input is an existing Gazebo RGB flight and real cached neural predictions.
This is a new discrete-event evaluation, not a new flight, wireless measurement
or electrical-power measurement. Existing studies and their outcomes are read
only. The design and implementation hashes are frozen before any replay output.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import math
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_tracking_evaluation import RunnerReportEvaluator
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_joint_constraints import cell_name
from experiments.mission_joint_constraints import verify as verify_parent
from experiments.mission_joint_constraints_scenario import (
    actions_for,
    load_profiles,
    policy_costs,
    scenario_for,
    trace_for,
)
from experiments.mission_pipeline_routing.runtime import CompletedPrediction, PublicPose, Selection
from experiments.mission_pipeline_routing.study import public_context
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_runtime_profile import REPO, read, save, sha
from experiments.mission_task_deadlines.contract import DeadlineProfile, TaskKind
from experiments.mission_task_deadlines.pipeline import DeadlinePipeline
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.runner import evaluation_frame
from experiments.v4_gazebo_pilot.timing import seconds_to_ns

PARENT = REPO / "results/mission_joint_constraints_v1_20261027"
OUTPUT = REPO / "results/mission_task_deadlines_v1_20261028"
POLICIES = ("fixed_local_fast", "fixed_local_heavy", "fixed_remote_union", "deadline_aware")
DEFAULT_PROFILES = (
    DeadlineProfile("tight", 2_000_000_000, 80_000_000, 10_000_000),
    DeadlineProfile("relaxed", 8_000_000_000, 150_000_000, 30_000_000),
)
DOWNSTREAM = Selection("ASSOC_TERRAIN_WORLD", "VERIFY_REPEAT2")


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
    config = config or StudyConfig()
    grid = []
    for flight, mission, battery, link, profile, policy in itertools.product(
        config.flights,
        config.missions,
        config.batteries,
        config.links,
        config.deadline_profiles,
        config.policies,
    ):
        grid.append(
            dict(
                run_id=f"run_{len(grid):04d}",
                cell=cell_name(flight, mission, battery, link, 1.0),
                flight=flight,
                mission=mission,
                initial_battery_frac=battery,
                link=link,
                deadline_profile=profile.name,
                policy=policy,
            )
        )
    return dict(
        schema_version="task_deadline_integrated_spec_v1",
        factors=config.to_dict(),
        grid=grid,
        expected_runs=len(grid),
        downstream_selection=asdict(DOWNSTREAM),
        downstream_routing_latency_ns=50_000_000,
        downstream_routing_energy_j=0.75,
        comparison="Deadline profile is crossed independently with intent, battery and link.",
        phase_availability="Canonical phase evidence uses final selected pipeline finish; "
        "perception admission and downstream task deadlines are separately recorded.",
        task_failure_endpoint="Paid task deadline misses, remote timeouts and explicit "
        "abstentions permit later recovery. Mission evidence, report and resource requirements "
        "are unchanged. All-task-deadlines compliance is a separate diagnostic.",
        scope="Known development flights, cached actual vision outputs, fixed trajectories; "
        "authored virtual network and downstream/energy costs, existing host vision p95. "
        "No fresh flight, radio/power measurement, generalization or superiority claim.",
        strict_intent_feasibility="verified_inspection is a known sensor-infeasible control "
        "on these reused recordings; rapid feasibility is not assumed under new task budgets.",
        new_neural_inference=False,
        future_link_visible_to_policy=False,
        policy_receives_ground_truth=False,
        known_previous_results=True,
    )


def prepare(root, config=None):
    root = Path(root).resolve()
    require(not root.exists(), "Preserve existing study output")
    parent = verify_parent(PARENT)
    require(read(PARENT / "completion.json")["complete"] is True, "Incomplete parent study")
    spec = specification(config)
    require(all(row["cell"] in parent["cases"] for row in spec["grid"]), "Missing parent input")
    source_paths = dict(parent["implementation_sha256"])
    for path in [
        *Path(__file__).parent.glob("*.py"),
        Path(__file__).with_name("defaults.json"),
        *REPO.glob("experiments/mission_pipeline_routing/*.py"),
        *REPO.glob("tests/test_mission_task_deadline*.py"),
    ]:
        source_paths[str(path.resolve())] = sha(path)
    inputs = dict(parent["input_sha256"])
    for relative, expected in parent["copied_input_sha256"].items():
        inputs[str((PARENT / relative).resolve())] = expected
    for name in ("plan.json", "completion.json", "validation.json"):
        inputs[str((PARENT / name).resolve())] = sha(PARENT / name)
    root.mkdir(parents=True)
    snapshots = {}
    for path, expected in source_paths.items():
        require(sha(path) == expected, "Changed source before freeze")
        relative = Path("source_snapshot") / Path(path).relative_to(REPO)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        snapshots[str(relative)] = expected
    save(root / "specification.json", spec)
    plan = dict(
        spec,
        schema_version="task_deadline_integrated_plan_v1",
        parent_root=str(PARENT),
        parent_plan_sha256=sha(PARENT / "plan.json"),
        prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_input_sha256=inputs,
        implementation_sha256=source_paths,
        snapshot_sha256=snapshots,
        outcome_files_exist_at_freeze=False,
    )
    save(root / "plan.json", plan)
    return verify(root)


def verify(root):
    root = Path(root)
    plan = read(root / "plan.json")
    require(plan["schema_version"] == "task_deadline_integrated_plan_v1", "Unknown study schema")
    spec = specification(StudyConfig.from_dict(plan["factors"]))
    for key, expected in spec.items():
        if key != "schema_version":
            require(plan[key] == expected, "Changed frozen design: " + key)
    require(read(root / "specification.json") == spec, "Changed specification")
    for field in ("source_input_sha256", "implementation_sha256"):
        for path, expected in plan[field].items():
            require(sha(path) == expected, "Changed frozen input/source: " + path)
    for relative, expected in plan["snapshot_sha256"].items():
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root.resolve()) and sha(path) == expected,
            "Changed or escaped source snapshot",
        )
    require(
        sha(Path(plan["parent_root"]) / "plan.json") == plan["parent_plan_sha256"],
        "Changed parent plan",
    )
    return plan


def task_metrics(events):
    metrics = {
        task.value: dict(
            attempts=0, on_time=0, deadline_misses=0, discarded_outputs=0, lateness_ns=0
        )
        for task in TaskKind
    }
    for event in events:
        rows = []
        if event["selected_action_id"] is not None:
            p = event["perception"]
            rows.append(
                dict(
                    p,
                    committed=p["admitted"],
                    metric_on_time=p["task_deadline_met"],
                    metric_missed=p["deadline_missed"],
                )
            )
        if event["pipeline"] is not None:
            rows.extend(s for s in event["pipeline"]["stages"] if s["dispatched"])
        for row in rows:
            value = metrics[row["task"]]
            value["attempts"] += 1
            value["on_time"] += int(row.get("metric_on_time", row["on_time"] and row["committed"]))
            value["deadline_misses"] += int(row.get("metric_missed", not row["on_time"]))
            value["discarded_outputs"] += int(not row["committed"])
            value["lateness_ns"] += row["lateness_ns"]
    return metrics


def evaluate_recoverable(contract, events, result):
    """Replace only execution validity; all intent/resource checks remain canonical."""
    original = evaluate(contract, events, result)
    outcome = copy.deepcopy(original)
    allowed = {
        "completed",
        "task_deadline_miss",
        "transport_timeout",
        "mission_deadline_exceeded",
        "abstained",
        "timeout",
        "deadline",
    }
    expected = all(
        e["perception"]["status"] in allowed
        and (e["execution"]["success"] is e["perception"]["admitted"])
        and (
            e["perception"]["admitted"]
            or (
                e["emitted_reports"] == []
                and e["pipeline"] is None
                and not e["score"]["matched_target_ids"]
                and e["score"]["false_positive_components"] == 0
            )
        )
        and (
            e["pipeline"] is None
            or e["pipeline"]["status"] in ("completed", "task_deadline_miss", "battery_depleted")
        )
        for e in events
    )
    valid = (
        result["status"] == "completed"
        and result["error"] is None
        and result["mission_completed"] is True
        and result["termination_reason"] == "mission_window_complete"
        and result["final_time_s"] >= contract.document()["duration_s"]
        and result["processed_observations"] == len(events)
        and result["router_errors"] == result["invalid_actions"] == 0
        and result["inference_failures"] == sum(not e["execution"]["success"] for e in events)
        and expected
    )
    outcome["constraints"]["execution"] = valid
    metrics = task_metrics(events)
    outcome.update(
        schema_version="task_deadline_recoverable_intent_result_v1",
        mission_success=all(outcome["constraints"].values()),
        failed_constraints=[k for k, value in outcome["constraints"].items() if not value],
        strict_original_endpoint_success=original["mission_success"],
        task_metrics=metrics,
        all_dispatched_task_deadlines_met=all(
            v["on_time"] == v["attempts"] for v in metrics.values()
        ),
        abstentions=sum(e["selected_action_id"] is None for e in events),
        endpoint_deviation="Recoverable paid task misses/transport failures/abstentions do not "
        "by themselves fail the mission. Original intent evidence/report/resource checks retained.",
    )
    return outcome


def frame_bank(scenario, directory):
    """Verify every recording hash once; share immutable captures across paired runs."""
    from experiments.mission_network_study_v1.executors import prepare_payload

    backend = ReplayBackend(scenario, directory)
    frames, payloads = {}, {}
    try:
        for index, stamp in enumerate(backend.frames):
            backend.advance_to(stamp)
            frame = backend.capture()
            frame.rgb.setflags(write=False)
            frame.object_ids.setflags(write=False)
            frames[index] = frame
            payloads[index] = prepare_payload(frame.rgb)
    finally:
        backend.close()
    return frames, payloads, backend.provenance


def execute_case(root, plan, item, terrain, bank, manifest):
    from experiments.mission_task_deadlines.perception import execute_perception
    from experiments.mission_task_deadlines.policy import DeadlineAwarePolicy

    root, parent_root = Path(root), Path(plan["parent_root"])
    parent = read(parent_root / "plan.json")
    scenario = scenario_for(parent_root, parent, item["cell"])
    contract = IntentContract(
        read(parent_root / "missions" / item["cell"] / item["mission"] / "intent.json")
    )
    scenario = bind_scenario(scenario, contract)
    require(
        manifest["recording_identity"] == scenario.recording_identity, "Wrong prediction flight"
    )
    profile = next(
        DeadlineProfile.from_dict(p)
        for p in plan["factors"]["deadline_profiles"]
        if p["name"] == item["deadline_profile"]
    )
    actions = actions_for(parent, scenario)
    descriptors = {a.action_id: a for a in actions}
    costs, trace = policy_costs(parent, scenario), trace_for(parent, item["cell"])
    profiles = load_profiles(parent_root, parent)
    policy = DeadlineAwarePolicy(
        item["policy"], actions, costs, profile, profiles["recall"], profiles["risk"]
    )
    pipeline = DeadlinePipeline(
        profile=profile,
        terrain=terrain,
        step_ns=scenario.schedule.step_ns,
        battery_capacity_j=scenario.battery_capacity_wh * 3600,
        flight_power_w=scenario.flight_power_w,
    )
    scorer = MissionEvaluator(
        scenario.matching_iou_threshold, len(scenario.targets), scenario.observation_interval_s
    )
    reporter = RunnerReportEvaluator(scenario.matching_iou_threshold, scenario.targets.values(), 2)
    masks = {
        a.action_id: VerifiedMasks(
            Path(parent["cohort"]) / item["flight"] / "predictions",
            manifest["models"][a.source_config_id],
        )
        for a in actions
    }
    frames, payloads, provenance = bank
    require(
        provenance["recording_identity"] == scenario.recording_identity, "Wrong shared RGB bank"
    )
    destination = root / "runs" / item["run_id"]
    destination.mkdir(parents=True, exist_ok=False)
    save(
        destination / "protocol.json",
        dict(
            case=item,
            plan_sha256=sha(root / "plan.json"),
            contract_sha256=contract.sha256,
            public_contract_sha256=contract.public_sha256,
            recording=provenance,
            prediction_manifest_sha256=sha(
                Path(parent["cohort"]) / item["flight"] / "predictions/manifest.json"
            ),
            deadline_profile=profile.to_dict(),
            downstream_selection=asdict(DOWNSTREAM),
        ),
    )
    save(destination / "intent.json", contract.document())
    schedule = scenario.schedule
    capacity = scenario.battery_capacity_wh * 3600
    nonflight = communication = peak_memory = 0.0
    current_action = None
    now_ns = index = 0
    events, skipped_ids = [], []
    termination = "mission_window_complete"

    def battery(at_ns, extra_j=0.0):
        energy = at_ns / 1e9 * scenario.flight_power_w + nonflight + extra_j
        return max(0.0, scenario.initial_battery_frac - energy / capacity)

    while schedule.at(index) < schedule.end_ns:
        if battery(now_ns) <= 0:
            termination = "battery_depleted"
            break
        capture_ns = schedule.at(index) - schedule.epoch_ns
        require(capture_ns >= now_ns, "Capture occurred while preceding work was busy")
        if battery(capture_ns) <= 0:
            now_ns, termination = capture_ns, "battery_depleted"
            break
        if capture_ns > seconds_to_ns(scenario.contract.deadline_s):
            now_ns, termination = capture_ns, "deadline_exceeded"
            break
        frame, prepared = frames[index], payloads[index]
        public_intent = contract.public_document()
        decision_ns = capture_ns + seconds_to_ns(costs.encode_s, allow_zero=True)
        observed_battery = battery(decision_ns, costs.encode_s * costs.codec_power_w)
        selection = policy.select(
            release_ns=capture_ns,
            current_action_id=current_action,
            battery_frac=observed_battery,
            cumulative_communication_mb=communication,
            link=trace.snapshot(decision_ns),
            payload_bytes=len(prepared.payload),
            response_bytes=prepared.response_bytes,
            intent=public_intent,
            rgb=frame.rgb,
        )
        action_id = selection["action_id"]
        action = descriptors[action_id] if action_id is not None else None
        perception = execute_perception(
            action=action,
            rgb=frame.rgb,
            prepared=prepared,
            predictions=masks[action_id] if action_id is not None else None,
            release_ns=capture_ns,
            profile=profile,
            mission_deadline_ns=seconds_to_ns(scenario.contract.deadline_s),
            current_action_id=current_action,
            trace=trace,
            costs=costs,
            router_latency_ns=seconds_to_ns(scenario.router_latency_s, allow_zero=True),
            router_energy_j=scenario.router_energy_j,
            step_ns=schedule.step_ns,
        )
        nonflight += perception.nonflight_energy_j
        communication += perception.communication_mb
        current_action = perception.resident_action_id
        now_ns = perception.completed_ns
        output = None
        reports = []
        downstream_skip_reason = None
        if perception.mask is not None and battery(now_ns) > 0:
            resource = scenario.resources.for_config(action_id)
            context = public_context(
                contract, capture_ns, now_ns, battery(now_ns), resource.memory_mb
            )
            output = pipeline.execute(
                DOWNSTREAM,
                CompletedPrediction(
                    action.source_config_id,
                    capture_ns,
                    now_ns,
                    perception.mask,
                    PublicPose(
                        tuple(frame.body_pose["position_m"]),
                        tuple(frame.body_pose["quaternion_xyzw"]),
                        scenario.world.get("camera_pitch_rad", math.pi / 2),
                    ),
                ),
                context,
            )
            nonflight += output["charged_energy_j"]
            now_ns = output["completed_ns"]
            reports = output["emitted_reports"]
            peak_memory = max(
                peak_memory,
                *(s["costs"]["resident_memory_mb"] for s in output["stages"] if s["dispatched"]),
            )
        elif perception.mask is not None:
            downstream_skip_reason = "battery_depleted_before_association_release"
            peak_memory = max(peak_memory, action.memory_mb + 68)
        elif action is not None:
            downstream_skip_reason = "perception_output_not_admitted"
            peak_memory = max(peak_memory, action.memory_mb + 68)
        completion, next_index, skipped = schedule.complete(index, now_ns - capture_ns)
        require(completion - schedule.epoch_ns == now_ns, "Inconsistent usable completion ticks")
        truth = evaluation_frame(frame, index, capture_ns / 1e9, scenario)
        prediction = (
            perception.mask
            if perception.mask is not None
            else np.zeros(frame.rgb.shape[:2], dtype=np.bool_)
        )
        score = scorer.update(truth, prediction)
        reporter.update(truth, prediction, reports)
        for missed in skipped:
            missed_truth = evaluation_frame(
                frames[missed], missed, (schedule.at(missed) - schedule.epoch_ns) / 1e9, scenario
            )
            scorer.note_skipped_visibility(missed_truth.visible_target_ids)
            reporter.note_visibility(missed_truth)
        events.append(
            dict(
                observation_id=index,
                capture_time_s=capture_ns / 1e9,
                capture_timestamp_ns=frame.timestamp_ns,
                completion_time_s=now_ns / 1e9,
                perception_completed_ns=perception.completed_ns,
                selected_action_id=action_id,
                source_config_id=action.source_config_id if action else None,
                source_rgb_key=prepared.rgb_sha256,
                selected_mask_sha256=hashlib.sha256(perception.mask.tobytes()).hexdigest()
                if perception.mask is not None
                else None,
                capture_body_pose=frame.body_pose,
                public_link=trace.snapshot(decision_ns).to_dict(),
                battery_at_decision=observed_battery,
                router_decision=selection,
                perception=perception.event,
                pipeline=output,
                downstream_skip_reason=downstream_skip_reason,
                execution=dict(
                    success=perception.mask is not None,
                    config_id=action_id,
                    predicted_pixels=int(prediction.sum()),
                ),
                score=score.to_dict(),
                emitted_reports=reports,
                cumulative_nonflight_energy_j=nonflight,
                cumulative_energy_j=now_ns / 1e9 * scenario.flight_power_w + nonflight,
                cumulative_communication_mb=communication,
                battery_after=battery(now_ns),
                skipped_observation_ids=list(skipped),
            )
        )
        skipped_ids.extend(skipped)
        index = next_index
    if termination == "mission_window_complete":
        now_ns = max(now_ns, seconds_to_ns(scenario.duration_s))
    if battery(now_ns) <= 0:
        termination = "battery_depleted"
    quality, report_quality = scorer.quality_details(now_ns / 1e9), reporter.summary(now_ns / 1e9)
    flags = dict(
        memory_constraint_success=peak_memory <= 4096,
        deadline_success=now_ns / 1e9 <= scenario.contract.deadline_s,
        battery_constraint_success=battery(now_ns) >= scenario.contract.min_final_battery_frac,
        communication_constraint_success=communication <= scenario.contract.communication_budget_mb,
        privacy_constraint_success=True,
    )
    result = dict(
        schema_version="task_deadline_execution_v1",
        contract=asdict(scenario.contract),
        status="completed",
        error=None,
        termination_reason=termination,
        mission_completed=termination == "mission_window_complete",
        final_time_s=now_ns / 1e9,
        final_battery_frac=battery(now_ns),
        cumulative_energy_j=now_ns / 1e9 * scenario.flight_power_w + nonflight,
        cumulative_nonflight_energy_j=nonflight,
        flight_energy_j=now_ns / 1e9 * scenario.flight_power_w,
        cumulative_communication_mb=communication,
        peak_configured_memory_mb=peak_memory,
        processed_observations=len(events),
        skipped_observation_ids=skipped_ids,
        router_errors=0,
        invalid_actions=0,
        privacy_violations=0,
        inference_failures=sum(not e["execution"]["success"] for e in events),
        constraints=flags,
        quality=quality,
        report_quality=report_quality,
        recording_identity=scenario.recording_identity,
    )
    outcome = evaluate_recoverable(contract, events, result)
    for name, value in (("events", events), ("results", result), ("outcome", outcome)):
        save(destination / (name + ".json"), value)
    return dict(
        **item,
        path=str(destination.relative_to(root)),
        mission_success=outcome["mission_success"],
        successful_phases=sum(p["success"] for p in outcome["phases"]),
        report_recall=outcome["reporting"]["recall_by_deadline"],
        final_battery_frac=result["final_battery_frac"],
        communication_mb=communication,
        task_metrics=outcome["task_metrics"],
        abstentions=outcome["abstentions"],
        failed_constraints=outcome["failed_constraints"],
        artifacts_sha256={
            name + ".json": sha(destination / (name + ".json"))
            for name in ("protocol", "intent", "events", "results", "outcome")
        },
    )


def execute(root):
    root = Path(root)
    plan = verify(root)
    require(
        not (root / "run_index.json").exists() and not (root / "runs").exists(),
        "Preserve every previous execution attempt",
    )
    parent_root = Path(plan["parent_root"])
    parent = read(parent_root / "plan.json")
    terrain, _ = load_public_terrain(Path(parent["assets"]))
    banks, manifests, rows = {}, {}, []
    for item in plan["grid"]:
        flight = item["flight"]
        if flight not in banks:
            scenario = scenario_for(parent_root, parent, item["cell"])
            banks[flight] = frame_bank(scenario, Path(parent["cohort"]) / flight / "recording")
            manifests[flight] = checked_manifest(
                Path(parent["cohort"]) / flight / "predictions",
                Path(parent["cohort"]) / "protocol.json",
            )
        rows.append(execute_case(root, plan, item, terrain, banks[flight], manifests[flight]))
        save(root / "progress.json", dict(completed=len(rows), expected=plan["expected_runs"]))
        print(f"{len(rows)}/{plan['expected_runs']} {item['run_id']}", flush=True)
    verify(root)
    save(
        root / "run_index.json",
        dict(plan_sha256=sha(root / "plan.json"), cases=rows, complete_grid=True),
    )
    return rows


def summarize(root):
    root = Path(root)
    plan = verify(root)
    require(not (root / "completion.json").exists(), "Preserve final study certificate")
    index = read(root / "run_index.json")
    require(
        index["plan_sha256"] == sha(root / "plan.json") and index["complete_grid"] is True,
        "Invalid execution index",
    )
    require(
        [{k: r[k] for k in plan["grid"][0]} for r in index["cases"]] == plan["grid"],
        "Incomplete, duplicated or reordered factorial grid",
    )
    counts, bindings = Counter(), {}
    for row in index["cases"]:
        destination = root / row["path"]
        for name, expected in row["artifacts_sha256"].items():
            require(sha(destination / name) == expected, "Changed execution artifact")
            bindings[str((destination / name).relative_to(root))] = expected
        events, result, outcome = (
            read(destination / (n + ".json")) for n in ("events", "results", "outcome")
        )
        contract = IntentContract(read(destination / "intent.json"))
        require(
            evaluate_recoverable(contract, events, result) == outcome, "Changed outcome scoring"
        )
        require(outcome["task_metrics"] == row["task_metrics"], "Changed task counts")
        counts["decisions"] += len(events)
        counts["reports"] += len(result["report_quality"]["reports"])
        for task, value in outcome["task_metrics"].items():
            counts[task + "_attempts"] += value["attempts"]
            counts[task + "_deadline_misses"] += value["deadline_misses"]
    policies = {}
    for policy in plan["factors"]["policies"]:
        rows = [r for r in index["cases"] if r["policy"] == policy]
        policies[policy] = dict(
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
    summary = dict(
        runs=len(index["cases"]),
        policies=policies,
        totals=dict(counts),
        scope=plan["scope"],
        strict_intent_feasibility=plan["strict_intent_feasibility"],
    )
    save(root / "summary.json", summary)
    save(
        root / "validation.json",
        dict(
            complete=True,
            validation_scope="Artifact hash bindings and canonical outcome/metric "
            "reconstruction; not an independent implementation of the full event scheduler.",
            output_sha256=bindings,
            complete_grid=True,
            counts=dict(counts),
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
                for name in ("plan.json", "run_index.json", "summary.json", "validation.json")
            },
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "summarize", "verify"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.output, StudyConfig.from_dict(read(args.config)) if args.config else None)
    elif args.action == "execute":
        execute(args.output)
    elif args.action == "summarize":
        summarize(args.output)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
