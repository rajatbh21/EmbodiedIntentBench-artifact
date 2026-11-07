"""Predeclared causal-policy development comparison on verified physical recordings.

Both lighting conditions are development data: training uses day only, but dusk
shares scenery, object layout and flight. No held-out or deployment claim is made.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments.lake_causal_router import (
    MODES,
    CausalRouter,
    MissionCosts,
    check_split,
    fit_profile,
    split_fingerprint,
)
from experiments.lake_fast_evaluation import PeriodicViews
from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import TrackingConfig
from experiments.lake_tracking_runtime import load_public_terrain, run_tracking_mission
from experiments.lake_tracking_study import scenario_and_cache
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import context_key
from experiments.v4_benchmark.probe_cache import CachedExecutor
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.scenario import digest

MISSIONS = ("sector_rescue", "perimeter_inspection", "reserve_revisit")
ENVIRONMENTS = ("day", "dusk_haze")
POLICIES = (*MODES, "static_training_best", "periodic_training_top_two")


def train(prior: Path, source: Path, output: Path, scenery: str) -> dict:
    scenario, _, _ = scenario_and_cache(prior, source, "day", "sector_rescue")
    backend = ReplayBackend(scenario, source / "day/recording")
    contexts = {}
    try:
        for index, timestamp in enumerate(backend.frames):
            backend.advance_to(timestamp)
            contexts[index] = context_key(backend.capture().rgb)
    finally:
        backend.close()
    profile = fit_profile(
        read(prior / "day/outcomes.json"),
        contexts,
        scenario.executors,
        split_fingerprint(scenario, scenery),
    )
    save(output / "calibration.json", profile)
    return profile


def verify_case(destination: Path, scenario, poses: dict, contract, profile: dict) -> dict:
    events = read(destination / "execution/events.json")
    result = read(destination / "execution/results.json")
    outcome = read(destination / "outcome.json")
    canonical = evaluate(contract, events, result)
    for key in ("mission_success", "failed_constraints", "reporting"):
        if canonical[key] != outcome[key]:
            raise ValueError("Canonical intent outcome mismatch")
    if not events or any(
        e["capture_body_pose"] != poses[e["capture_timestamp_ns"]] for e in events
    ):
        raise ValueError("Physical capture poses differ")
    if result["router_errors"] or result["inference_failures"] or result["invalid_actions"]:
        raise ValueError("Policy replay did not execute cleanly")
    components = (
        "flight_energy_j",
        "inference_energy_j",
        "communication_energy_j",
        "router_energy_j",
        "model_load_energy_j",
        "tracking_energy_j",
    )
    if abs(result["cumulative_energy_j"] - sum(result[k] for k in components)) > 1e-6:
        raise ValueError("Energy accounting mismatch")
    for index, event in enumerate(events):
        decision = event["router_decision"]
        if decision and decision.get("schema_version") == "causal_decision_v1":
            if decision["profile_sha256"] != digest(profile) or decision["uses_ground_truth"]:
                raise ValueError("Decision calibration binding mismatch")
            selected = next(
                o
                for o in decision["candidates"]
                if o["config_id"] == event["execution"]["config_id"]
            )
            if abs(selected["first_completion_s"] - event["completion_time_s"]) > 1e-7:
                raise ValueError("Forecast differs from actual inference clock")
            if (
                index + 1 < len(events)
                and selected["next_capture_s"] != events[index + 1]["capture_time_s"]
            ):
                raise ValueError("Forecast differs from next physical capture")
    return dict(
        schema_version="causal_case_verification_v1",
        exact_capture_poses=True,
        energy_accounting_verified=True,
        canonical_intent_verified=True,
        decision_clock_verified=True,
        mission_success=outcome["mission_success"],
        failed_constraints=outcome["failed_constraints"],
        report_recall=outcome["reporting"]["recall_by_deadline"],
        false_reports=result["report_quality"]["false_reports"],
        duplicate_reports=result["report_quality"]["duplicate_correct_reports"],
        final_battery_frac=result["final_battery_frac"],
        calls=len(events),
        selections=dict(Counter(e["execution"]["config_id"] for e in events)),
        outcome_sha256=sha(destination / "outcome.json"),
        events_sha256=sha(destination / "execution/events.json"),
        results_sha256=sha(destination / "execution/results.json"),
    )


def report(output: Path, rows: list[dict]) -> None:
    lines = [
        "# Causal routing development comparison",
        "",
        "All 36 executions replay real Gazebo captures and verified neural predictions. "
        "No new neural inference or live Gazebo flight is claimed. Each policy pays configured "
        "inference, flight, switching, routing and terrain-tracking costs.",
        "",
        "Calibration uses day only. Day is resubstitution; dusk shares the map, object layout "
        "and flight and is a development lighting shift. Neither is an independent test. "
        "Costs retain prior laptop latency measurements and assumed energy/memory values. "
        "Tracking assumes a perfect known terrain map and ideal odometry.",
        "",
        "| Environment | Mission | Policy | Full success | Report recall | Battery | Calls |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['environment']} | {r['mission']} | {r['policy']} | "
            f"{r['mission_success']} | {r['report_recall']:.3f} | "
            f"{r['final_battery_frac']:.3f} | {r['calls']} |"
        )
    lines += [
        "",
        "The success criterion is the unchanged report-required intent contract: all phases, "
        "correct confirmed reports, deadlines, resource limits and valid execution must pass. "
        "A per-frame calibration proxy does not establish report or mission success.",
        "",
        "## Next evidence required",
        "",
        "Freeze policies and collect independent physical flights with new scenery, layout and "
        "route; calibrate on development scenes only. Measure deployment latency/energy and "
        "test pose/map uncertainty. Mission feasibility remains separate from policy quality: "
        "previous bounds certify the inspection latency and reserve-energy cases infeasible "
        "under this fixed pool and cost declaration.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def run(prior: Path, intents: Path, output: Path) -> None:
    from experiments.lake_missions.run import freeze

    source = Path(read(prior / "protocol.json")["source"])
    freeze(source)
    terrain, metadata = load_public_terrain(Path(read(source / "suite.json")["assets"]))
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        schema_version="lake_causal_study_v1",
        scope="development",
        independent_test=False,
        training_environment="day",
        environments=ENVIRONMENTS,
        missions=MISSIONS,
        policies=POLICIES,
        source=str(source.resolve()),
        prior=str(prior.resolve()),
        intents=str(intents.resolve()),
        source_lock_sha256=sha(source / "lock.json"),
        calibration_table_sha256=sha(prior / "day/outcomes.json"),
        static_selection_rule="Highest training recall times precision; config ID breaks ties",
        periodic_selection_rule="Alternate top two under the same global training ranking",
        quality_context_min_frames=8,
        protocol_tuned_on_test_outcomes=False,
        limitation="Previously explored flight; this is not preregistration",
        implementation_sha256={
            name: sha(Path(__file__).with_name(name))
            for name in ("lake_causal_router.py", "lake_causal_study.py")
        },
        scenery_sha256=metadata["raster_sha256"],
    )
    save(output / "protocol.json", protocol)
    profile = train(prior, source, output, metadata["raster_sha256"])
    ranking = sorted(
        profile["models"],
        key=lambda c: (
            -profile["models"][c]["global_stats"]["recall"]
            * profile["models"][c]["global_stats"]["precision"],
            c,
        ),
    )
    save(
        output / "baseline_selection.json",
        dict(ranking=ranking, calibration_sha256=sha(output / "calibration.json")),
    )
    rows = []
    for env in ENVIRONMENTS:
        for mission in MISSIONS:
            contract = IntentContract(
                read(intents / "contracts" / mission / "report_required/intent.json")
            )
            scenario, available, hashes = scenario_and_cache(prior, source, env, mission)
            split = check_split(
                profile["training_lineage"],
                split_fingerprint(scenario, metadata["raster_sha256"]),
                scope="development",
            )
            try:
                check_split(
                    profile["training_lineage"],
                    split_fingerprint(scenario, metadata["raster_sha256"]),
                    scope="held_out",
                )
            except ValueError as exc:
                split["held_out_rejected"] = str(exc)
            else:
                raise ValueError("Expected shared flight to be rejected as held out")
            for policy in POLICIES:
                router = (
                    CausalRouter(
                        profile,
                        scenario.executors,
                        MissionCosts.from_scenario(scenario),
                        mode=policy,
                    )
                    if policy in MODES
                    else (
                        build_router(
                            "static:" + ranking[0],
                            scenario.profiles,
                            scenario.observation_interval_s,
                        )
                        if policy == "static_training_best"
                        else PeriodicViews(ranking[:2])
                    )
                )
                executors = {
                    s.config_id: CachedExecutor(
                        s,
                        VerifiedMasks(*available[s.config_id]),
                        sha(available[s.config_id][0] / "manifest.json"),
                    )
                    for s in scenario.executors
                }
                backend = ReplayBackend(scenario, source / env / "recording")
                poses = {t: f["body_pose"] for t, f in backend.frames.items()}
                destination = output / env / mission / policy
                try:
                    run_tracking_mission(
                        scenario,
                        backend,
                        router,
                        executors,
                        contract,
                        destination,
                        config=TrackingConfig(remember_reports=True),
                        terrain=terrain,
                        map_metadata=metadata,
                        router_name=policy,
                        gt_aware=False,
                    )
                finally:
                    backend.close()
                verification = verify_case(
                    destination, bind_scenario(scenario, contract), poses, contract, profile
                )
                verification.update(
                    environment=env,
                    mission=mission,
                    policy=policy,
                    split=split,
                    manifests_sha256=hashes,
                    output=str(destination.resolve()),
                )
                save(destination / "verification.json", verification)
                rows.append(verification)
                save(output / "progress.json", dict(completed=len(rows), expected=36, cases=rows))
                print(
                    f"{len(rows)}/36 {env}/{mission}/{policy}: "
                    f"success={verification['mission_success']}, "
                    f"reports={verification['report_recall']:.3f}",
                    flush=True,
                )
    for name, expected in protocol["implementation_sha256"].items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError("Study implementation changed during execution")
    save(
        output / "summary.json",
        dict(
            schema_version="lake_causal_summary_v1",
            scope="development",
            independent_test=False,
            cases=rows,
            complete=len(rows) == 36,
            protocol_sha256=sha(output / "protocol.json"),
            calibration_sha256=sha(output / "calibration.json"),
        ),
    )
    report(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--intents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.prior, args.intents, args.output)


if __name__ == "__main__":
    main()
