"""Predeclared development-only causal replay under authored virtual links."""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from experiments.mission_measured_cost_study_v2 import CONFIGS, calibrated_arm, require
from experiments.mission_network_policies import POLICIES, ActionDescriptor
from experiments.mission_network_study_v1.scenario import (
    MISSION,
    actions_for,
    base_scenario_for,
    derivative_documents,
    policy_costs,
    policy_for,
    scenario_for,
    trace_for,
    tracking_cost,
)
from experiments.mission_network_study_v1.scenario import (
    load_profiles as load_profiles,
)
from experiments.mission_runtime_profile import (
    DEVELOPMENT,
    FLIGHTS,
    REPO,
    implementation_inventory,
    imports,
    read,
    save,
    sha,
)

OUTPUT = REPO / "results/mission_network_study_v1_20261027"
CALIBRATION = REPO / "results/mission_execution_profile_v1_20261027"
LINKS = ("good", "constrained", "intermittent")


def authored_traces():
    def row(start, end, up=50_000_000, down=100_000_000, rtt=30_000_000, outage=False):
        return dict(
            start_ns=start * 10**9,
            end_ns=end * 10**9,
            upload_bps=up,
            download_bps=down,
            rtt_ns=rtt,
            outage=outage,
        )

    return dict(
        good=[row(0, 140)],
        constrained=[row(0, 140, 1_000_000, 5_000_000, 300_000_000)],
        intermittent=[
            row(0, 45),
            row(45, 65, outage=True),
            row(65, 95),
            row(95, 105, outage=True),
            row(105, 140),
        ],
    )


def cell_name(flight, link):
    return flight + "__" + link


def specification():
    return dict(
        schema_version="virtual_network_development_spec_v1",
        flights=list(FLIGHTS),
        missions=[MISSION],
        links=list(LINKS),
        policies=list(POLICIES),
        expected_replays=84,
        initial_battery_frac=0.65,
        grid=[
            dict(cell=cell_name(f, link), flight=f, link=link, mission=MISSION, policy=p)
            for f in FLIGHTS
            for link in LINKS
            for p in POLICIES
        ],
        traces=authored_traces(),
        remote_service_factor=0.25,
        communication_budget_mb=100.0,
        privacy_level="remote_allowed",
        local_power_w=15.0,
        remote_client_memory_mb=64.0,
        remote_load_latency_s=0.0,
        remote_load_energy_j=0.0,
        policy_costs=dict(
            timeout_s=8.0,
            encode_s=0.02,
            decode_s=0.01,
            codec_power_w=15.0,
            tx_power_w=2.0,
            rx_power_w=2.0,
            wait_power_w=0.5,
            quality_budget_s=2.0,
            horizon_s=8.0,
            blind_upload_bps=50_000_000,
            blind_download_bps=100_000_000,
            blind_rtt_ns=30_000_000,
        ),
        preparation="Every action pays PNG encode20ms/0.3J before state/link/policy observation; "
        "lossless decoded RGB equality; actual bytes known only after this common preparation",
        timeout="8s transport/service from post-router/post-load dispatch; successful unpack10ms "
        "and tracking follow outside timeout; outer130s deadline covers complete operation",
        payload="PNG RGB compress_level6 optimizeFalse; fixed response ceil(H*W/8) bytes",
        network_clock="Mission-origin integer ns; physical ns minus recording epoch",
        endpoint="network_recoverable_intent_result_v1: paid expected timeout/deadline "
        "allows later recovery; no output/tracker/report/same-frame fallback on failure",
        local_baseline_expansion="Initial6-policy proposal expanded before any network execution "
        "to all9 fixedlocal references plus fixedremoteunion and4 adaptive policies. "
        "Known development host-cost controls motivated complete local-reference coverage.",
        scope="Authored network/server and assumed onboard energy on reused development data; "
        "no empirical cellular representativeness, real offloading, field or generalization claim",
        future_trace_visible_to_policy=False,
        new_neural_inference=False,
        source_contract_phase_and_reporting_conditions_unchanged=True,
        positive_control="All9 fixed local references invariant to all3links within each recording",
        execution_sharding="Optional disjoint per-flight shards; exact complete84 grid required",
        fresh_evaluation=False,
        remote_server_electrical_energy_included=False,
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
            network_outcomes_exist=False,
            earlier_development_diagnostics_known=True,
        ),
    )


def prepare(root, calibration=CALIBRATION):
    from experiments import mission_network_audit
    from experiments.mission_execution_profile import verify as verify_calibration

    root, calibration = Path(root), Path(calibration)
    require(read(root / "specification.json") == specification(), "Changed preregistration")
    require(
        not (root / "plan.json").exists() and not (root / "runs").exists(), "Preserve prior plan"
    )
    api = imports()
    protocol = api["development_verify"](DEVELOPMENT)
    profile = verify_calibration(calibration)
    base = api["scenario_for"](DEVELOPMENT, protocol, FLIGHTS[0], MISSION, 15)
    signatures = {s.config_id: api["model_signature"](s) for s in base.executors}
    arm = calibrated_arm(profile, signatures)
    require(arm["tracking_calibration_used"], "Verified tracking calibration required")
    actions = []
    for spec in base.executors:
        require(spec.config_id in CONFIGS, "Unexpected neural pool")
        resource = base.resources.for_config(spec.config_id)
        latency = arm["latency_s"][spec.config_id]
        actions.extend(
            [
                ActionDescriptor(
                    spec.config_id,
                    spec.config_id,
                    "local",
                    signatures[spec.config_id],
                    latency,
                    latency * 15,
                    resource.load_latency_s,
                    resource.load_energy_j,
                    resource.memory_mb,
                ),
                ActionDescriptor(
                    "REMOTE__" + spec.config_id,
                    spec.config_id,
                    "remote",
                    signatures[spec.config_id],
                    latency * 0.25,
                    0,
                    0,
                    0,
                    64.0,
                ),
            ]
        )
    frozen, copied = {}, {}

    def pin(path):
        frozen[str(Path(path).resolve())] = sha(path)

    def copy_file(path, relative):
        pin(path)
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        require(sha(dest) == sha(path), "Copied input differs")
        copied[str(relative)] = sha(dest)

    for name in ("plan.json", "profile.json", "completion.json"):
        pin(calibration / name)
    for name in ("specification.json", "specification_chronology.json"):
        pin(root / name)
    pin(DEVELOPMENT / "protocol.json")
    # Known controls justify the broad fixed-local reference set, never tuning link outcomes.
    for name in ("summary.json", "completion.json"):
        pin(REPO / "results/mission_measured_cost_study_v2_20261027" / name)
    source = Path(protocol["cohort"])
    pin(source / "protocol.json")
    profile_sources = dict(
        recall=REPO
        / "results/mission_policy_joint_training_v1_20261018/all_development/recall.json",
        risk=REPO / "results/mission_routing_v4_risk_training_20261017/risk_profile.json",
    )
    for key, path in profile_sources.items():
        copy_file(path, Path("profiles") / (key + ".json"))
    cases = {}
    for flight in FLIGHTS:
        source_case = protocol["cases"][flight]
        require(source_case["role"] == "development", "Evaluation recording forbidden")
        manifest = api["checked_manifest"](
            source / flight / "predictions", source / "protocol.json"
        )
        for name in (
            "recording/recording.json",
            "predictions/manifest.json",
            "predictions/completion.json",
            "predictions/inference.json",
        ):
            pin(source / flight / name)
        original = {}
        for name in ("scenario", "intent", "contract"):
            path = DEVELOPMENT / flight / MISSION / (name + ".json")
            original[name] = read(path)
            copy_file(path, Path("originals") / flight / (name + ".json"))
        for link in LINKS:
            cell = cell_name(flight, link)
            cases[cell] = dict(
                source_flight=flight,
                link=link,
                role="development",
                environment=source_case["environment"],
                lineage=source_case["lineage"],
                layout_group=source_case["layout_group"],
            )
            values = derivative_documents(
                original["scenario"], original["intent"], original["contract"], arm["latency_s"]
            )
            for name, value in zip(("scenario", "intent", "contract"), values, strict=True):
                dest = root / "missions" / cell / MISSION / (name + ".json")
                save(dest, value)
                copied[str(dest.relative_to(root))] = sha(dest)
            draft = dict(
                cases=cases,
                assets=protocol["assets"],
                traces=authored_traces(),
                actions=[asdict(a) for a in actions],
                model_signatures=signatures,
            )
            loaded = scenario_for(root, draft, cell)
            require(loaded.recording_identity == manifest["recording_identity"], "Changed flight")
            require(
                {c: manifest["models"][c]["signature"] for c in CONFIGS} == signatures,
                "Changed neural prediction identity",
            )
    _, terrain_metadata = api["load_public_terrain"](Path(protocol["assets"]))
    pin(terrain_metadata["raster_path"])
    pin(Path(protocol["assets"]) / "harmonic_terrain/model.sdf")
    implementations = implementation_inventory()
    for path in [
        *Path(__file__).parent.glob("*.py"),
        Path(mission_network_audit.__file__),
        REPO / "tests/test_mission_network_study.py",
        REPO / "tests/test_mission_network_audit.py",
        REPO / "tests/test_mission_network_policies.py",
        REPO / "tests/test_mission_network_emulation.py",
    ]:
        implementations[str(path.resolve())] = sha(path)
    snapshots = {}
    for path, expected in implementations.items():
        relative = Path("source_snapshot") / Path(path).relative_to(REPO)
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        require(sha(dest) == expected, "Source snapshot differs")
        copied[str(relative)] = expected
        snapshots[path] = str(relative)
    plan = specification()
    plan.update(
        schema_version="mission_network_study_v1",
        cohort=str(source),
        assets=protocol["assets"],
        cases=cases,
        actions=[asdict(a) for a in actions],
        tracking_cost=arm["tracking_cost"],
        local_latency_s=arm["latency_s"],
        model_signatures=signatures,
        calibration=str(calibration.resolve()),
        calibration_sha256=sha(calibration / "profile.json"),
        profile_paths={k: "profiles/" + k + ".json" for k in profile_sources},
        input_sha256=frozen,
        copied_input_sha256=copied,
        implementation_sha256=implementations,
        source_snapshot_paths=snapshots,
        runtime_versions={
            name: importlib.metadata.version(name) for name in ("numpy", "Pillow", "scipy")
        },
        prepared_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    # Instantiate every policy before capturing any new selection/outcome.
    scenario = scenario_for(root, plan, next(iter(cases)))
    for name in POLICIES:
        policy_for(root, plan, name, scenario)
    save(root / "plan.json", plan)
    verify(root)
    return plan


def verify(root):
    root = Path(root)
    plan = read(root / "plan.json")
    require(plan["schema_version"] == "mission_network_study_v1", "Unknown network study")
    for key, value in specification().items():
        if key != "schema_version":
            require(plan[key] == value, "Changed declared specification: " + key)
    for field in ("input_sha256", "implementation_sha256"):
        for path, expected in plan[field].items():
            require(sha(path) == expected, "Changed frozen input/source: " + path)
    for relative, expected in plan["copied_input_sha256"].items():
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root.resolve()) and sha(path) == expected,
            "Changed copy: " + relative,
        )
    arm = calibrated_arm(read(Path(plan["calibration"]) / "profile.json"), plan["model_signatures"])
    require(
        plan["local_latency_s"] == arm["latency_s"]
        and plan["tracking_cost"] == arm["tracking_cost"],
        "Changed measured assignment",
    )
    require(
        set(plan["cases"]) == {cell_name(f, link) for f in FLIGHTS for link in LINKS},
        "Changed cells",
    )
    source_protocol = read(DEVELOPMENT / "protocol.json")
    require(
        plan["cohort"] == source_protocol["cohort"] and plan["assets"] == source_protocol["assets"],
        "Changed source cohort",
    )
    for cell, case in plan["cases"].items():
        flight, link = case["source_flight"], case["link"]
        original_case = source_protocol["cases"][flight]
        expected_case = dict(
            source_flight=flight,
            link=link,
            role="development",
            environment=original_case["environment"],
            lineage=original_case["lineage"],
            layout_group=original_case["layout_group"],
        )
        require(case == expected_case and cell == cell_name(flight, link), "Changed lineage")
        values = derivative_documents(
            *(
                read(root / "originals" / flight / (n + ".json"))
                for n in ("scenario", "intent", "contract")
            ),
            plan["local_latency_s"],
        )
        for name, expected in zip(("scenario", "intent", "contract"), values, strict=True):
            require(
                read(root / "missions" / cell / MISSION / (name + ".json")) == expected,
                "Undeclared contract/scenario intervention",
            )
    base = base_scenario_for(root, plan, next(iter(plan["cases"])))
    expected = []
    for spec in base.executors:
        resource = base.resources.for_config(spec.config_id)
        latency = plan["local_latency_s"][spec.config_id]
        expected.extend(
            [
                asdict(
                    ActionDescriptor(
                        spec.config_id,
                        spec.config_id,
                        "local",
                        plan["model_signatures"][spec.config_id],
                        latency,
                        latency * 15,
                        resource.load_latency_s,
                        resource.load_energy_j,
                        resource.memory_mb,
                    )
                ),
                asdict(
                    ActionDescriptor(
                        "REMOTE__" + spec.config_id,
                        spec.config_id,
                        "remote",
                        plan["model_signatures"][spec.config_id],
                        latency * 0.25,
                        0,
                        0,
                        0,
                        64.0,
                    )
                ),
            ]
        )
    require(plan["actions"] == expected, "Changed placement costs or identities")
    return plan


def execute(root, flight=None):
    from experiments.mission_network_study_v1.executors import PlacementExecutor
    from experiments.mission_network_study_v1.runtime import run_tracking_mission

    root = Path(root)
    plan = verify(root)
    require(flight is None or flight in FLIGHTS, "Unknown flight shard")
    tag = "" if flight is None else "_" + flight
    grid = [r for r in plan["grid"] if flight is None or r["flight"] == flight]
    require(not (root / ("run_index" + tag + ".json")).exists(), "Preserve existing index")
    require(
        all(not (root / "runs" / r["cell"] / r["policy"]).exists() for r in grid),
        "Preserve every existing execution attempt",
    )
    api = imports()
    source, rows = Path(plan["cohort"]), []
    terrain, metadata = api["load_public_terrain"](Path(plan["assets"]))
    try:
        for number, item in enumerate(grid):
            verify(root)
            cell, source_flight, policy = item["cell"], item["flight"], item["policy"]
            scenario = scenario_for(root, plan, cell)
            folder = source / source_flight / "predictions"
            manifest = api["checked_manifest"](folder, source / "protocol.json")
            require(manifest["recording_identity"] == scenario.recording_identity, "Changed flight")
            trace, costs = trace_for(plan, cell, scenario), policy_costs(plan, scenario)
            executors = {
                a.action_id: PlacementExecutor(
                    a,
                    api["VerifiedMasks"](folder, manifest["models"][a.source_config_id]),
                    sha(folder / "manifest.json"),
                    trace,
                    costs,
                )
                for a in actions_for(plan, scenario)
            }
            destination = root / "runs" / cell / policy
            contract = api["IntentContract"](
                read(root / "missions" / cell / MISSION / "intent.json")
            )
            backend = api["ReplayBackend"](scenario, source / source_flight / "recording")
            try:
                outcome = run_tracking_mission(
                    scenario,
                    backend,
                    policy_for(root, plan, policy, scenario),
                    executors,
                    contract,
                    destination,
                    config=api["TrackingConfig"](remember_reports=True),
                    terrain=terrain,
                    map_metadata=metadata,
                    router_name=policy,
                    cost=tracking_cost(plan),
                    trace=trace,
                    costs=costs,
                    study_plan_sha256=sha(root / "plan.json"),
                )
            finally:
                backend.close()
            result, events = (
                read(destination / ("execution/" + n + ".json")) for n in ("results", "events")
            )
            row = dict(
                **item,
                path=str(destination.relative_to(root)),
                full_success=outcome["mission_success"],
                calls=len(events),
                phases_succeeded=sum(p["success"] for p in outcome["phases"]),
                report_recall=outcome["reporting"]["recall_by_deadline"],
                failed_constraints=outcome["failed_constraints"],
                expected_transport_timeouts=result["expected_transport_timeouts"],
                communication_mb=result["cumulative_communication_mb"],
                final_battery=result["final_battery_frac"],
                selections=dict(Counter(e["execution"]["config_id"] for e in events)),
            )
            for name, path in dict(
                outcome="outcome.json",
                events="execution/events.json",
                results="execution/results.json",
                protocol="protocol.json",
            ).items():
                row[name + "_sha256"] = sha(destination / path)
            rows.append(row)
            save(
                root / ("progress" + tag + ".json"),
                dict(completed=len(rows), expected=len(grid), cases=rows),
            )
            require(
                result["status"] == "completed"
                and result["router_errors"] == 0
                and result["invalid_actions"] == 0
                and result["inference_failures"] == result["expected_transport_timeouts"],
                "Unexpected internal failure; retain complete attempt, no rerun",
            )
            print(f"{number + 1}/{len(grid)} {cell} {policy}: {len(events)} decisions", flush=True)
    except BaseException as exc:
        save(
            root / ("failure" + tag + ".json"),
            dict(
                error=type(exc).__name__,
                message=str(exc),
                completed=len(rows),
                cases=rows,
                plan_sha256=sha(root / "plan.json"),
            ),
        )
        raise
    save(
        root / ("run_index" + tag + ".json"),
        dict(
            complete_execution_grid=flight is None,
            complete_shard=flight is not None,
            flight=flight,
            cases=rows,
            expected_replays=len(grid),
            plan_sha256=sha(root / "plan.json"),
        ),
    )
    verify(root)
    return rows


def combine_shards(plan, shards, plan_sha256):
    require(len(shards) == len(FLIGHTS), "Every flight shard required")
    require({s["flight"] for s in shards} == set(FLIGHTS), "Duplicate or missing flight shard")
    rows = []
    for shard in shards:
        require(
            shard["complete_shard"] is True
            and shard["complete_execution_grid"] is False
            and shard["plan_sha256"] == plan_sha256
            and shard["expected_replays"] == len(shard["cases"]) == 42,
            "Incomplete or foreign flight shard",
        )
        require(all(r["flight"] == shard["flight"] for r in shard["cases"]), "Mixed flight shard")
        rows.extend(shard["cases"])
    expected = [(r["cell"], r["policy"]) for r in plan["grid"]]
    lookup = {(r["cell"], r["policy"]): r for r in rows}
    require(
        len(rows) == len(lookup) == len(expected) == 84 and set(lookup) == set(expected),
        "Shard collection requires exactly the declared84 cells and policies",
    )
    for item in plan["grid"]:
        row = lookup[(item["cell"], item["policy"])]
        require({key: row[key] for key in item} == item, "Changed shard grid metadata")
    return [lookup[key] for key in expected]


def collect(root):
    root = Path(root)
    plan = verify(root)
    require(not (root / "run_index.json").exists(), "Preserve complete index")
    paths = [root / ("run_index_" + flight + ".json") for flight in FLIGHTS]
    rows = combine_shards(plan, [read(p) for p in paths], sha(root / "plan.json"))
    for row in rows:
        for name, filename in dict(
            outcome="outcome.json",
            events="execution/events.json",
            results="execution/results.json",
            protocol="protocol.json",
        ).items():
            require(
                sha(root / row["path"] / filename) == row[name + "_sha256"],
                "Changed shard execution",
            )
    save(
        root / "run_index.json",
        dict(
            complete_execution_grid=True,
            expected_replays=84,
            cases=rows,
            plan_sha256=sha(root / "plan.json"),
            shard_sha256={p.name: sha(p) for p in paths},
        ),
    )
    return rows


def finalize(root):
    root = Path(root)
    plan = verify(root)
    index, validation = read(root / "run_index.json"), read(root / "validation.json")
    require(
        index["complete_execution_grid"] is True and len(index["cases"]) == 84,
        "The complete84-run grid is required",
    )
    require(
        validation["schema_version"] == "mission_network_audit_v1"
        and validation["complete"] is True
        and validation["executions"] == 84
        and validation["fixed_local_invariance_groups"] == 18,
        "Independent audit or all18local invariance groups incomplete",
    )
    for field in (
        "canonical_outcomes_verified",
        "public_states_and_resource_flags_reconstructed",
        "raw_selected_mask_scores_reconstructed",
        "report_identity_support_reconstructed",
        "per_event_cost_clock_partial_bytes_verified",
    ):
        require(validation[field] is True, "Independent reconstruction incomplete: " + field)
    require(
        validation["plan_sha256"] == sha(root / "plan.json") == index["plan_sha256"]
        and validation["run_index_sha256"] == sha(root / "run_index.json"),
        "Changed audited inputs",
    )
    auditor = REPO / "experiments/mission_network_audit.py"
    require(
        validation["validator_sha256"]
        == sha(auditor)
        == plan["implementation_sha256"][str(auditor)],
        "Auditor was not frozen before execution",
    )
    for path, expected in validation["source_output_sha256"].items():
        target = Path(path) if Path(path).is_absolute() else root / path
        require(sha(target) == expected, "Changed audited artifact: " + path)
    rows = index["cases"]
    require(
        validation["events"]
        == validation["public_decisions_reproduced"]
        == sum(r["calls"] for r in rows),
        "Missing public decisions",
    )
    events = {}
    for row in rows:
        for name, filename in dict(
            outcome="outcome.json",
            events="execution/events.json",
            results="execution/results.json",
            protocol="protocol.json",
        ).items():
            require(
                sha(root / row["path"] / filename) == row[name + "_sha256"],
                "Changed indexed artifact",
            )
        key = (row["flight"], row["link"], row["policy"])
        events[key] = read(root / row["path"] / "execution/events.json")
    policy_summary = []
    for policy in POLICIES:
        selected = [r for r in rows if r["policy"] == policy]
        groups = {}
        for link in LINKS:
            members = [r for r in selected if r["link"] == link]
            require(len(members) == 2, "Changed policy/link denominator")
            calls = [e for r in members for e in events[(r["flight"], link, policy)]]
            groups[link] = dict(
                conditions=2,
                full_successes=sum(r["full_success"] for r in members),
                decisions=len(calls),
                remote_decisions=sum(e["placement"] == "remote" for e in calls),
                expected_timeouts=sum(r["expected_transport_timeouts"] for r in members),
                communication_mb=sum(r["communication_mb"] for r in members),
                mean_report_recall=sum(r["report_recall"] for r in members) / 2,
                minimum_final_battery=min(r["final_battery"] for r in members),
            )
        policy_summary.append(
            dict(
                policy=policy,
                conditions=6,
                full_successes=sum(r["full_success"] for r in selected),
                links=groups,
            )
        )
    summary = dict(
        schema_version="mission_network_summary_v1",
        complete=True,
        executions=84,
        decisions=validation["events"],
        policies=policy_summary,
        all_local_invariance_groups=18,
        expected_transport_timeouts=sum(r["expected_transport_timeouts"] for r in rows),
        scope=plan["scope"],
        plan_sha256=sha(root / "plan.json"),
        run_index_sha256=sha(root / "run_index.json"),
        validation_sha256=sha(root / "validation.json"),
    )
    save(root / "summary.json", summary)
    lines = [
        "# Virtual network sensitivity on reused development recordings",
        "",
        f"All 84 causal replays and {validation['events']:,} policy decisions were independently "
        "reconstructed. All 18 fixed-local flight/reference groups were invariant across links.",
        "",
        "Full success is the conjunction of original phase, report and resource requirements, "
        "with a versioned execution condition permitting paid expected transport cancellation. "
        "Canceled attempts deliver no prediction, tracker update, report or fallback.",
        "",
        "| Policy | Good (of 2) | Constrained (of 2) | Intermittent (of 2) | All (of 6) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in policy_summary:
        lines.append(
            "| "
            + row["policy"]
            + " | "
            + " | ".join(str(row["links"][link]["full_successes"]) for link in LINKS)
            + " | "
            + str(row["full_successes"])
            + " |"
        )
    lines += [
        "",
        plan["scope"],
        "",
        "The two physical banks, source masks and authored mission are reused across links. "
        "The public API reveals only the current link after common paid PNG preparation. "
        "The private transport integrates future trace segments only during execution. "
        "Remote service is a hypothetical quarter of measured local warm p95. Onboard flight, "
        "codec, compute and radio energy are assumptions; remote-server electricity is excluded.",
        "",
        "No confidence intervals, population superiority, empirical network representativeness "
        "or real deployment are inferred. All conditions, including failures, remain.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    save(
        root / "completion.json",
        dict(
            complete=True,
            executions=84,
            decisions=summary["decisions"],
            plan_sha256=sha(root / "plan.json"),
            run_index_sha256=sha(root / "run_index.json"),
            validation_sha256=sha(root / "validation.json"),
            summary_sha256=sha(root / "summary.json"),
            report_sha256=sha(root / "REPORT.md"),
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("specify", "prepare", "execute", "verify", "collect", "finalize")
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--flight", choices=FLIGHTS)
    args = parser.parse_args()
    if args.command == "execute":
        execute(args.output, args.flight)
    else:
        require(args.flight is None, "Flight shard applies only toexecute")
        {
            "specify": freeze_specification,
            "prepare": prepare,
            "verify": verify,
            "collect": collect,
            "finalize": finalize,
        }[args.command](args.output)


if __name__ == "__main__":
    main()
