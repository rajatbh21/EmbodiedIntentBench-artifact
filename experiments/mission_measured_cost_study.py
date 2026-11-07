"""Development-only adaptive sensitivity to host latency and assumed active energy.

The original prediction banks, learned predictors, policies and studies are immutable.
This new study changes declared simulator costs, then reruns every routing decision.
It is neither a new held-out evaluation nor a measurement of electrical energy.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import shutil
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from experiments.mission_runtime_profile import (
    DEVELOPMENT,
    FLIGHTS,
    REPO,
    THREAD_ENV,
    digest,
    implementation_inventory,
    imports,
    read,
    require_development,
    save,
    sha,
)

OUTPUT = REPO / "results/mission_measured_cost_study_v1_20261027"
CALIBRATION = REPO / "results/mission_execution_profile_v1_20261027"
REFERENCE = REPO / "results/mission_runtime_profile_v1_20261027"
MISSION = "rapid_survey"
BATTERIES = (0.34, 0.65)
ARMS = ("declared", "measured_warm_p95")
OBJECTIVES = (
    "coverage_tail",
    "phase_survival",
    "mission_survival",
    "temporal_mission",
    "round_robin",
    "greedy_quality",
    "approxdet_eq5_adapter",
)
CONFIGS = (
    "CFG_MASKRCNN_1280",
    "CFG_MASKRCNN_1280_ROT90",
    "CFG_MASKRCNN_1280_ROT0_90",
    "CFG_YOLO11S_960_ROT0",
    "CFG_YOLO11N_640_ROT0",
    "CFG_YOLO11S_TILED_640",
    "CFG_YOLO11S_TILED_640_ROT90",
    "CFG_YOLO11S_TILED_640_GAMMA",
    "CFG_CABINET_UAVID_1280",
)
POLICIES = (*OBJECTIVES, *("static:" + c for c in CONFIGS))
REFERENCE_POLICIES = (*OBJECTIVES, "static:CFG_YOLO11S_960_ROT0")


def cell_name(flight, battery):
    return f"{flight}_b{round(battery * 1000):03d}"


def specification():
    return dict(
        schema_version="development_measured_cost_specification_v1",
        purpose="joint_host_latency_and_assumed_energy_sensitivity",
        flights=list(FLIGHTS),
        missions=[MISSION],
        batteries=list(BATTERIES),
        arms=list(ARMS),
        policies=list(POLICIES),
        expected_replays=128,
        grid=[
            dict(
                arm=arm,
                cell=cell_name(flight, battery),
                flight=flight,
                mission=MISSION,
                policy=policy,
            )
            for arm in ARMS
            for flight in FLIGHTS
            for battery in BATTERIES
            for policy in POLICIES
        ],
        selection="Two existing development day layouts; one survey contract; two fixed reserves; "
        "seven distinct routing objectives and every fixed configuration. No cell filtering.",
        latency_intervention="Per-model measured warm wall p95; no fitted latency predictor",
        tracking_rule="Use measured tracking p95 iff calibration verifies eligibility and parity; "
        "otherwise retain 0.1 s. Its schedule-dependent sample is not a worst-case guarantee.",
        unchanged_costs=dict(
            router_latency_s=0.05,
            router_energy_j=0.75,
            model_load_latency_s=0.3,
            model_load_energy_j=2.0,
            tracking_energy_j=1.0,
            tracking_memory_mb=64,
        ),
        inference_energy="15 W times the assigned inference latency; assumed, not measured power",
        actual_adaptive_replays=True,
        fixed_schedule_rescore=False,
        new_neural_inference=False,
        fresh_evaluation=False,
        confidence_intervals=False,
        trained_on_reused_development=True,
        preexisting_test_diagnostics_known=True,
        scope="Engineering sensitivity on reused development data, not generalization or "
        "algorithm-superiority evidence. Prior Lake/Depot experiments are unchanged.",
    )


def freeze_specification(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    save(root / "specification.json", specification())
    save(
        root / "specification_chronology.json",
        dict(
            written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            specification_sha256=sha(root / "specification.json"),
            new_policy_outcomes_exist=False,
            measured_profile_exists=(CALIBRATION / "profile.json").exists(),
        ),
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def positive(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def calibrated_arm(profile, signatures):
    require(
        profile.get("schema_version") == "mission_execution_host_profile_v1"
        and profile.get("complete") is True,
        "Incomplete or wrong calibration",
    )
    require(
        set(profile["latency_s"]) == set(CONFIGS) and set(profile["profiles"]) == set(CONFIGS),
        "Calibration must cover all nine models",
    )
    for config in CONFIGS:
        row = profile["profiles"][config]
        require(
            positive(profile["latency_s"][config])
            and profile["latency_s"][config] == row["warm_p95_s"],
            "Invalid warm p95",
        )
        require(
            row["signature"] == signatures[config]
            and row["prediction_parity"] is True
            and row["warm_samples"] == 24,
            "Calibration model/parity/sample mismatch",
        )
    tracking = profile["tracking"]
    eligible = tracking.get("eligible_for_sensitivity") is True
    if eligible:
        require(
            tracking.get("parity") is True and positive(tracking["warm_p95_s"]),
            "Advertised tracking calibration failed parity",
        )
    return dict(
        latency_s=copy.deepcopy(profile["latency_s"]),
        tracking_cost=dict(
            latency_s=tracking["warm_p95_s"] if eligible else 0.1, energy_j=1.0, memory_mb=64
        ),
        tracking_calibration_used=eligible,
    )


def costed_documents(original, intent, battery, arm):
    require(battery in BATTERIES, "Undeclared battery condition")
    latency = arm["latency_s"]
    require(
        set(latency) == set(CONFIGS) and all(positive(v) for v in latency.values()),
        "Every configuration requires a finite positive latency",
    )
    scenario, contract = copy.deepcopy(original), copy.deepcopy(intent)
    require(
        scenario["mission"]["router_latency_s"] == 0.05
        and scenario["mission"]["router_energy_j"] == 0.75
        and scenario["mission"]["battery_capacity_wh"] == 30
        and scenario["mission"]["flight_power_w"] == 120,
        "Unexpected declared routing/flight resources",
    )
    require(
        set(scenario["resources"]["models"]) == set(CONFIGS)
        and scenario["resources"]["memory_capacity_mb"] == 4096
        and all(
            value == dict(load_energy_j=2, load_latency_s=0.3, memory_mb=2400)
            for value in scenario["resources"]["models"].values()
        ),
        "Unexpected declared loading/memory resources",
    )
    require(
        {s["config_id"] for s in scenario["executor_configs"]} == set(CONFIGS),
        "Changed original model pool",
    )
    scenario["mission"]["initial_battery_frac"] = battery
    contract["mission"]["initial_battery_frac"] = battery
    for spec in scenario["executor_configs"]:
        spec["mission_latency_s"] = latency[spec["config_id"]]
        spec["energy_j_per_call"] = 15 * spec["mission_latency_s"]
    require(
        {p["config_id"] for p in scenario["public_profiles"]} == set(CONFIGS),
        "Public catalog differs from executor pool",
    )
    for public in scenario["public_profiles"]:
        public["expected_latency_ms"] = 1000 * latency[public["config_id"]]
    return scenario, contract


def tracking_cost(plan, arm):
    from experiments.lake_tracking_runtime import TrackingCost

    return TrackingCost(**plan["arms"][arm]["tracking_cost"])


def scenario_for(root, plan, cell, arm):
    from experiments.mission_routing_v2.flights import load_scenario

    base = load_scenario(
        Path(root) / "missions" / arm / cell / MISSION / "scenario.json",
        plan["cases"][cell]["environment"],
        Path(plan["assets"]),
    )
    # Match the prior declared 15 W scenario representation exactly. No predictor is retagged.
    return replace(
        base,
        source={
            **base.source,
            "executor_configs": [asdict(s) for s in base.executors],
            "sensitivity_active_cpu_power_w": 15,
        },
    )


def load_profiles(root, plan):
    return {key: read(Path(root) / value) for key, value in plan["profile_paths"].items()}


def reference_path(flight, policy):
    return REFERENCE / "runs" / flight / MISSION / policy.replace(":", "_") / "control"


def semantic_run(folder):
    data = {
        name: read(Path(folder) / path)
        for name, path in dict(
            events="execution/events.json",
            result="execution/results.json",
            outcome="outcome.json",
            protocol="protocol.json",
        ).items()
    }
    for event in data["events"]:
        event.pop("measured_router_wall_clock_s")
        event["execution"].pop("measured_wall_clock_s")
    data["outcome"].pop("events_sha256")
    data["outcome"].pop("execution_sha256")
    return data


def baseline_parity(current, previous):
    left, right = semantic_run(current), semantic_run(previous)
    require(left == right, "Declared 65% arm differs from the existing causal control")
    return dict(
        passed=True,
        reference=str(previous),
        semantic_sha256=digest(left),
        excluded_fields=[
            "events.measured_router_wall_clock_s",
            "events.execution.measured_wall_clock_s",
            "outcome.events_sha256",
            "outcome.execution_sha256",
        ],
    )


def prepare(root, calibration=CALIBRATION):
    from experiments import mission_measured_cost_audit
    from experiments.mission_execution_profile import verify as verify_calibration

    root, calibration = Path(root), Path(calibration)
    require(
        read(root / "specification.json") == specification(), "Changed predeclared specification"
    )
    require(
        not (root / "plan.json").exists() and not (root / "runs").exists(),
        "Preserve existing sensitivity plan/executions",
    )
    profile = verify_calibration(calibration)
    api = imports()
    p = api["development_verify"](DEVELOPMENT)
    base = REPO / "results"
    profile_paths = dict(
        recall=base / "mission_policy_joint_training_v1_20261018/all_development/recall.json",
        risk=base / "mission_routing_v4_risk_training_20261017/risk_profile.json",
        temporal=base / "mission_temporal_training_v2_20261025/profile.json",
        coverage=base / "mission_coverage_training_v1_20261025/profile.json",
    )
    profiles = {k: read(v) for k, v in profile_paths.items()}
    api["coverage_verify"](profile_paths["coverage"].parent)
    api["risk_verify"](profile_paths["risk"].parent)
    api["temporal_verify"](profile_paths["temporal"].parent)
    require_development(p, profiles["coverage"])
    scenario = api["scenario_for"](DEVELOPMENT, p, FLIGHTS[0], MISSION, 15)
    signatures = {s.config_id: api["model_signature"](s) for s in scenario.executors}
    arms = dict(
        declared=dict(
            latency_s={s.config_id: s.mission_latency_s for s in scenario.executors},
            tracking_cost=dict(latency_s=0.1, energy_j=1.0, memory_mb=64),
            tracking_calibration_used=False,
        ),
        measured_warm_p95=calibrated_arm(profile, signatures),
    )
    frozen, copied = {}, {}

    def pin(path):
        frozen[str(Path(path).resolve())] = sha(path)

    def copy_file(path, relative):
        pin(path)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(sha(destination) == sha(path), "Input copy differs")
        copied[str(relative)] = sha(destination)

    for name in ("plan.json", "profile.json", "completion.json"):
        pin(calibration / name)
    for name in ("specification.json", "specification_chronology.json"):
        pin(root / name)
    pin(DEVELOPMENT / "protocol.json")
    source = Path(p["cohort"])
    pin(source / "protocol.json")
    for key, path in profile_paths.items():
        copy_file(path, Path("profiles") / (key + ".json"))
    cases = {}
    for flight in FLIGHTS:
        folder = source / flight / "predictions"
        manifest = api["checked_manifest"](folder, source / "protocol.json")
        for name in (
            "recording/recording.json",
            "predictions/manifest.json",
            "predictions/completion.json",
            "predictions/inference.json",
        ):
            pin(source / flight / name)
        original_root = DEVELOPMENT / flight / MISSION
        originals = {
            name: read(original_root / (name + ".json"))
            for name in ("scenario", "intent", "contract")
        }
        for name in originals:
            copy_file(
                original_root / (name + ".json"), Path("originals") / flight / (name + ".json")
            )
        require(
            {
                s["config_id"]: s["mission_latency_s"]
                for s in originals["scenario"]["executor_configs"]
            }
            == arms["declared"]["latency_s"],
            "Historical costs differ between recordings",
        )
        for battery in BATTERIES:
            cell = cell_name(flight, battery)
            cases[cell] = dict(
                source_flight=flight,
                initial_battery_frac=battery,
                environment=p["cases"][flight]["environment"],
                lineage=p["cases"][flight]["lineage"],
                layout_group=p["cases"][flight]["layout_group"],
                role="development",
            )
            for arm in ARMS:
                transformed, intent = costed_documents(
                    originals["scenario"], originals["intent"], battery, arms[arm]
                )
                dest = root / "missions" / arm / cell / MISSION
                for name, value in dict(
                    scenario=transformed, intent=intent, contract=originals["contract"]
                ).items():
                    save(dest / (name + ".json"), value)
                    copied[str((dest / (name + ".json")).relative_to(root))] = sha(
                        dest / (name + ".json")
                    )
                draft = dict(cases=cases, assets=p["assets"])
                loaded = scenario_for(root, draft, cell, arm)
                require(
                    loaded.recording_identity == manifest["recording_identity"]
                    and sha(source / flight / "recording/recording.json")
                    == manifest["recording_sha256"],
                    "Cost arm changed the physical recording",
                )
                require(
                    {s.config_id: api["model_signature"](s) for s in loaded.executors}
                    == signatures
                    == {c: manifest["models"][c]["signature"] for c in CONFIGS},
                    "Cost arm changed model/cache/predictor semantics",
                )
        for policy in REFERENCE_POLICIES:
            for name in (
                "protocol.json",
                "execution/events.json",
                "execution/results.json",
                "outcome.json",
            ):
                pin(reference_path(flight, policy) / name)
    _, metadata = api["load_public_terrain"](Path(p["assets"]))
    pin(metadata["raster_path"])
    pin(Path(p["assets"]) / "harmonic_terrain/model.sdf")
    implementations = implementation_inventory()
    for path in (
        Path(__file__),
        Path(mission_measured_cost_audit.__file__),
        REPO / "tests/test_mission_measured_cost_study.py",
        REPO / "tests/test_mission_measured_cost_audit.py",
    ):
        implementations[str(path.resolve())] = sha(path)
    snapshots = {}
    for path, expected in implementations.items():
        relative = Path("source_snapshot") / Path(path).relative_to(REPO)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        require(sha(destination) == expected, "Source snapshot differs")
        copied[str(relative)] = expected
        snapshots[path] = str(relative)
    plan = specification()
    plan.update(
        cohort=str(source),
        assets=p["assets"],
        cases=cases,
        arms=arms,
        calibration=str(calibration.resolve()),
        calibration_sha256=sha(calibration / "profile.json"),
        profile_paths={k: "profiles/" + k + ".json" for k in profile_paths},
        model_signatures=signatures,
        costs_excluded_from_semantic_model_signature=True,
        profile_coefficients_unchanged=True,
        input_sha256=frozen,
        copied_input_sha256=copied,
        implementation_sha256=implementations,
        source_snapshot_paths=snapshots,
        baseline_parity_expected=16,
        baseline_reference=str(REFERENCE),
        prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    # specification carries a distinct schema; retain the study schema on the final plan.
    plan["schema_version"] = "mission_measured_cost_study_v1"
    save(root / "plan.json", plan)
    verify(root)
    return plan


def verify(root):
    root = Path(root)
    plan = read(root / "plan.json")
    require(plan["schema_version"] == "mission_measured_cost_study_v1", "Unknown study schema")
    for key, value in specification().items():
        if key not in ("schema_version", "arms"):
            require(plan[key] == value, "Changed predeclared grid/scope: " + key)
    require(list(plan["arms"]) == sorted(ARMS), "Changed cost arms")
    for field in ("input_sha256", "implementation_sha256"):
        for path, expected in plan[field].items():
            require(sha(path) == expected, "Changed source/input: " + path)
    for relative, expected in plan["copied_input_sha256"].items():
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root.resolve()) and sha(path) == expected,
            "Changed copied input/snapshot: " + relative,
        )
    require(
        sha(Path(plan["calibration"]) / "profile.json") == plan["calibration_sha256"],
        "Changed calibration",
    )
    expected_measured = calibrated_arm(
        read(Path(plan["calibration"]) / "profile.json"), plan["model_signatures"]
    )
    require(
        plan["arms"]["measured_warm_p95"] == expected_measured, "Assigned measured costs differ"
    )
    expected_cases = {cell_name(f, b) for f in FLIGHTS for b in BATTERIES}
    require(set(plan["cases"]) == expected_cases, "Changed development cells")
    source_protocol = read(DEVELOPMENT / "protocol.json")
    require(
        plan["cohort"] == source_protocol["cohort"] and plan["assets"] == source_protocol["assets"],
        "Changed development source",
    )
    for cell, case in plan["cases"].items():
        flight, battery = case["source_flight"], case["initial_battery_frac"]
        require(
            cell == cell_name(flight, battery) and flight in FLIGHTS and battery in BATTERIES,
            "Changed cell label or battery",
        )
        source_case = source_protocol["cases"][flight]
        require(
            case
            == dict(
                source_flight=flight,
                initial_battery_frac=battery,
                environment=source_case["environment"],
                lineage=source_case["lineage"],
                layout_group=source_case["layout_group"],
                role="development",
            ),
            "Changed development environment/lineage",
        )
        original = read(root / "originals" / flight / "scenario.json")
        original_intent = read(root / "originals" / flight / "intent.json")
        require(
            plan["arms"]["declared"]
            == dict(
                latency_s={
                    s["config_id"]: s["mission_latency_s"] for s in original["executor_configs"]
                },
                tracking_cost=dict(latency_s=0.1, energy_j=1.0, memory_mb=64),
                tracking_calibration_used=False,
            ),
            "Changed historical cost reference",
        )
        for arm in ARMS:
            transformed, intent = costed_documents(
                original, original_intent, battery, plan["arms"][arm]
            )
            dest = root / "missions" / arm / cell / MISSION
            require(
                read(dest / "scenario.json") == transformed
                and read(dest / "intent.json") == intent
                and read(dest / "contract.json")
                == read(root / "originals" / flight / "contract.json"),
                "Undeclared scenario/intent intervention",
            )
    return plan


def execute(root):
    root = Path(root)
    plan = verify(root)
    require(
        not (root / "runs").exists() and not (root / "run_index.json").exists(),
        "Preserve prior execution attempt",
    )
    api = imports()
    profiles = load_profiles(root, plan)
    terrain, metadata = api["load_public_terrain"](Path(plan["assets"]))
    source, rows, parity = Path(plan["cohort"]), [], []
    try:
        for number, cell in enumerate(plan["grid"]):
            verify(root)
            arm, label, flight, policy = (cell[k] for k in ("arm", "cell", "flight", "policy"))
            scenario = scenario_for(root, plan, label, arm)
            folder = source / flight / "predictions"
            manifest = api["checked_manifest"](folder, source / "protocol.json")
            require(
                manifest["recording_identity"] == scenario.recording_identity,
                "Prediction recording differs",
            )
            executors = {}
            for spec in scenario.executors:
                model = manifest["models"][spec.config_id]
                require(
                    model["signature"] == api["model_signature"](spec), "Model identity differs"
                )
                executors[spec.config_id] = api["CachedExecutor"](
                    spec, api["VerifiedMasks"](folder, model), sha(folder / "manifest.json")
                )
            intent = root / "missions" / arm / label / MISSION / "intent.json"
            contract = api["IntentContract"](read(intent))
            destination = root / "runs" / arm / label / policy.replace(":", "_")
            backend = api["ReplayBackend"](scenario, source / flight / "recording")
            try:
                outcome = api["run_tracking_mission"](
                    scenario,
                    backend,
                    api["make_policy"](policy, profiles, scenario),
                    executors,
                    contract,
                    destination,
                    config=api["TrackingConfig"](remember_reports=True),
                    terrain=terrain,
                    map_metadata=metadata,
                    router_name=policy,
                    gt_aware=False,
                    cost=tracking_cost(plan, arm),
                )
            finally:
                backend.close()
            result, events = (
                read(destination / ("execution/" + n + ".json")) for n in ("results", "events")
            )
            require(
                result["status"] == "completed"
                and not any(
                    result[k] for k in ("router_errors", "inference_failures", "invalid_actions")
                ),
                "Runtime failure; outputs retained, completion blocked",
            )
            row = dict(
                **cell,
                path=str(destination.relative_to(root)),
                full_success=outcome["mission_success"],
                phases_succeeded=sum(v["success"] for v in outcome["phases"]),
                report_recall=outcome["reporting"]["recall_by_deadline"],
                false_reports=outcome["reporting"]["false_reports"],
                duplicate_reports=outcome["reporting"]["duplicate_reports"],
                final_battery=result["final_battery_frac"],
                calls=len(events),
                selections=dict(Counter(e["execution"]["config_id"] for e in events)),
                failed_constraints=outcome["failed_constraints"],
                schedule=[[e["observation_id"], e["execution"]["config_id"]] for e in events],
            )
            for name, filename in dict(
                outcome="outcome.json",
                events="execution/events.json",
                results="execution/results.json",
                protocol="protocol.json",
            ).items():
                row[name + "_sha256"] = sha(destination / filename)
            if (
                arm == "declared"
                and scenario.initial_battery_frac == 0.65
                and policy in REFERENCE_POLICIES
            ):
                parity.append(
                    dict(
                        cell=label,
                        policy=policy,
                        **baseline_parity(destination, reference_path(flight, policy)),
                    )
                )
            rows.append(row)
            save(root / "progress.json", dict(completed=len(rows), expected=128, last=cell))
            print(f"{number + 1}/128 {arm} {label} {policy}: {len(events)} decisions", flush=True)
    except BaseException as exc:
        save(
            root / "failure.json",
            dict(
                error=type(exc).__name__,
                message=str(exc),
                completed=len(rows),
                cases=rows,
                plan_sha256=sha(root / "plan.json"),
            ),
        )
        raise
    require(len(parity) == plan["baseline_parity_expected"], "Missing historical-control parity")
    save(root / "baseline_parity.json", dict(complete=True, cases=parity, count=len(parity)))
    save(
        root / "run_index.json",
        dict(
            complete_execution_grid=True,
            cases=rows,
            expected_replays=128,
            plan_sha256=sha(root / "plan.json"),
            baseline_parity_sha256=sha(root / "baseline_parity.json"),
        ),
    )
    verify(root)
    return rows


def finalize(root):
    from experiments.mission_measured_cost_audit import complete_grid

    root = Path(root)
    require(not (root / "completion.json").exists(), "Preserve prior finalization")
    plan = verify(root)
    index, validation = read(root / "run_index.json"), read(root / "validation.json")
    require(
        index["complete_execution_grid"]
        and index["plan_sha256"] == sha(root / "plan.json")
        and complete_grid(plan, index["cases"]) == 128,
        "Incomplete execution grid",
    )
    require(
        validation["complete"]
        and validation["executions"] == 128
        and validation["plan_sha256"] == sha(root / "plan.json")
        and validation["run_index_sha256"] == sha(root / "run_index.json"),
        "Independent audit missing",
    )
    validator_path = str((REPO / "experiments/mission_measured_cost_audit.py").resolve())
    require(
        validation["validator_sha256"] == plan["implementation_sha256"][validator_path],
        "Audit did not use the frozen independent validator",
    )
    require(
        index["baseline_parity_sha256"] == sha(root / "baseline_parity.json"),
        "Changed historical-control parity",
    )
    parity = read(root / "baseline_parity.json")
    require(
        parity["complete"] and parity["count"] == len(parity["cases"]) == 16,
        "Missing historical controls",
    )
    for relative, expected in validation["source_output_sha256"].items():
        require(sha(root / relative) == expected, "Changed audited execution")
    policies, contrasts = {}, []
    for policy in POLICIES:
        policies[policy] = {}
        for arm in ARMS:
            rows = [r for r in index["cases"] if r["policy"] == policy and r["arm"] == arm]
            policies[policy][arm] = dict(
                cells=len(rows),
                successes=sum(r["full_success"] for r in rows),
                phases=sum(r["phases_succeeded"] for r in rows),
                phase_denominator=16,
                calls=sum(r["calls"] for r in rows),
            )
        for cell in plan["cases"]:
            pair = {
                r["arm"]: r for r in index["cases"] if r["policy"] == policy and r["cell"] == cell
            }
            left, right = pair["declared"], pair["measured_warm_p95"]
            contrasts.append(
                dict(
                    cell=cell,
                    policy=policy,
                    schedule_changed=left["schedule"] != right["schedule"],
                    full_success_before=left["full_success"],
                    full_success_after=right["full_success"],
                    phases_before=left["phases_succeeded"],
                    phases_after=right["phases_succeeded"],
                    calls_before=left["calls"],
                    calls_after=right["calls"],
                )
            )
    summary = dict(
        complete=True,
        plan_sha256=sha(root / "plan.json"),
        executions=128,
        decisions=sum(r["calls"] for r in index["cases"]),
        policies=policies,
        paired_cells=contrasts,
        changed_schedules=sum(r["schedule_changed"] for r in contrasts),
        scope=plan["scope"],
        inference_energy_measured=False,
    )
    save(root / "summary.json", summary)
    lines = [
        "# Adaptive host-cost sensitivity on development recordings",
        "",
        "All 128 causal replays and their independent reconstruction are complete. "
        "Each arm uses the same two physical RGB/mask banks and authored survey contract. "
        "Four cells combine two layouts and two initial charge levels; "
        "they are dependent development conditions.",
        "",
        "| Policy | Declared success /4 | Measured-cost success /4 | Declared phases /16 "
        "| Measured-cost phases /16 | Calls, declared / measured |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, arms in policies.items():
        a, b = (arms[k] for k in ARMS)
        lines.append(
            f"| {policy} | {a['successes']} | {b['successes']} | "
            f"{a['phases']} | {b['phases']} | {a['calls']} / {b['calls']} |"
        )
    lines += [
        "",
        f"Selected capture/configuration schedules changed in "
        f"{summary['changed_schedules']}/64 paired cells. "
        "Adaptive policies reselect under the assigned costs; fixed selectors retain their model. "
        "These are causal executions, not scores assigned to old schedules. "
        "All unsuccessful cells are retained.",
        "",
        "The measured arm substitutes host warm p95 inference latency and, only if eligible, "
        "tracking latency. Inference energy remains the assumption 15 W times assigned latency. "
        "Tracking energy is fixed at 1 J, routing at 50 ms/0.75 J, and loading at 0.3 s/2 J. "
        "Warm calibration does not measure loading, power, contention tails or onboard "
        "performance. Tracking calibration is schedule dependent. "
        "This joint timing/assumed-energy intervention does not isolate latency alone.",
        "",
        "Predictors were trained on reused development recordings; no model coefficients, "
        "semantic signatures, thresholds, masks or original benchmark results changed. "
        "The seven objectives and nine fixed choices are reported without a selected winner, "
        "confidence interval or generalization claim. Sixteen matching 65% declared-cost "
        "controls reproduce earlier semantic execution artifacts exactly, excluding only "
        "measured host time and hashes derived from it. "
        "Every new execution is independently reconstructed.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    save(
        root / "completion.json",
        dict(
            complete=True,
            executions=128,
            decisions=summary["decisions"],
            **{
                key + "_sha256": sha(root / name)
                for key, name in dict(
                    plan="plan.json",
                    run_index="run_index.json",
                    validation="validation.json",
                    summary="summary.json",
                    report="REPORT.md",
                    baseline_parity="baseline_parity.json",
                ).items()
            },
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("freeze-spec", "prepare", "execute", "finalize", "verify")
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    args = parser.parse_args()
    for key in THREAD_ENV:
        os.environ[key] = "1"
    if args.action == "freeze-spec":
        freeze_specification(args.output)
    elif args.action == "prepare":
        prepare(args.output, args.calibration)
    elif args.action == "execute":
        execute(args.output)
    elif args.action == "finalize":
        finalize(args.output)
    else:
        plan = verify(args.output)
        print(dict(verified=True, expected_replays=plan["expected_replays"]))


if __name__ == "__main__":
    main()
