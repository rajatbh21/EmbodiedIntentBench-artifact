"""Independently replay causal decisions from public RGB/state and audit saved outcomes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from experiments.lake_causal_router import (
    MODES,
    CausalRouter,
    MissionCosts,
    check_split,
    fit_profile,
    split_fingerprint,
)
from experiments.lake_causal_study import ENVIRONMENTS, MISSIONS, POLICIES
from experiments.lake_fast_evaluation import PeriodicViews
from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, evaluate, immutable
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking_study import scenario_and_cache
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import context_key
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.scenario import digest
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(output: Path) -> dict:
    protocol, summary, profile = (
        read(output / name) for name in ("protocol.json", "summary.json", "calibration.json")
    )
    require(
        summary["complete"] and summary["protocol_sha256"] == sha(output / "protocol.json"),
        "Incomplete or changed study",
    )
    require(
        summary["calibration_sha256"] == sha(output / "calibration.json"), "Changed calibration"
    )
    require(
        summary["scope"] == "development" and not summary["independent_test"],
        "False evaluation scope",
    )
    for name, expected in protocol["implementation_sha256"].items():
        require(sha(Path(__file__).with_name(name)) == expected, "Changed implementation")
    source, prior = Path(protocol["source"]), Path(protocol["prior"])
    require(sha(source / "lock.json") == protocol["source_lock_sha256"], "Changed source lock")
    require(
        sha(prior / "day/outcomes.json") == protocol["calibration_table_sha256"],
        "Changed training table",
    )
    rows = summary["cases"]
    expected_cases = {(e, m, p) for e in ENVIRONMENTS for m in MISSIONS for p in POLICIES}
    require(
        len(rows) == len(expected_cases)
        and {(r["environment"], r["mission"], r["policy"]) for r in rows} == expected_cases,
        "Incomplete policy grid",
    )
    ranking = sorted(
        profile["models"],
        key=lambda c: (
            -profile["models"][c]["global_stats"]["recall"]
            * profile["models"][c]["global_stats"]["precision"],
            c,
        ),
    )
    require(
        read(output / "baseline_selection.json")["ranking"] == ranking,
        "Changed training baseline selection",
    )
    decisions, checked_events = 0, 0
    training_contexts = {}
    for environment in ENVIRONMENTS:
        by_capture = defaultdict(list)
        for row in (r for r in rows if r["environment"] == environment):
            destination = output / environment / row["mission"] / row["policy"]
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                require(sha(destination / name) == row[key], "Changed saved execution")
            for name, expected in row["manifests_sha256"].items():
                require(sha(Path(name)) == expected, "Changed prediction manifest")
            scenario, _, _ = scenario_and_cache(prior, source, environment, row["mission"])
            contract = IntentContract(read(destination / "intent.json"))
            scenario = bind_scenario(scenario, contract)
            lineage = split_fingerprint(scenario, protocol["scenery_sha256"])
            split = check_split(profile["training_lineage"], lineage, scope="development")
            require(split["overlaps"] == row["split"]["overlaps"], "Changed split declaration")
            events, result = (
                read(destination / "execution/events.json"),
                read(destination / "execution/results.json"),
            )
            canonical = evaluate(contract, events, result)
            outcome = read(destination / "outcome.json")
            for key in ("mission_success", "failed_constraints", "reporting"):
                require(canonical[key] == outcome[key], "Canonical outcome mismatch")
            policy = row["policy"]
            router = (
                CausalRouter(
                    profile, scenario.executors, MissionCosts.from_scenario(scenario), mode=policy
                )
                if policy in MODES
                else (
                    build_router(
                        "static:" + ranking[0], scenario.profiles, scenario.observation_interval_s
                    )
                    if policy == "static_training_best"
                    else PeriodicViews(ranking[:2])
                )
            )
            tracking = read(destination / "protocol.json")["public_tracking"]
            require(
                sha(destination / "public_intent.json")
                == sha(
                    Path(protocol["intents"])
                    / "contracts"
                    / row["mission"]
                    / "report_required/public_intent.json"
                ),
                "Public intent differs from declaration",
            )
            energy, reports, previous, expected_index = 0.0, 0, None, 0
            specs = {s.config_id: s for s in scenario.executors}
            for event in events:
                index, time = event["observation_id"], event["capture_time_s"]
                require(index == expected_index, "Capture skipped outside the inference clock")
                state = event["runtime_at_capture"]
                capture_energy = energy + time * scenario.flight_power_w
                require(
                    abs(state["cumulative_energy_j"] - capture_energy) < 1e-6,
                    "Pre-decision energy mismatch",
                )
                require(
                    abs(
                        state["battery_frac"]
                        - (
                            scenario.initial_battery_frac
                            - capture_energy / (scenario.battery_capacity_wh * 3600)
                        )
                    )
                    < 1e-9,
                    "Pre-decision battery mismatch",
                )
                require(
                    state["current_config_id"] == previous
                    and state["frame_id"] == index
                    and state["current_time_s"] == time,
                    "Invalid public runtime state",
                )
                config = event["execution"]["config_id"]
                spec, resources = specs[config], scenario.resources.for_config(config)
                load_s = resources.load_latency_s if previous != config else 0
                load_j = resources.load_energy_j if previous != config else 0
                latency = (
                    spec.mission_latency_s
                    + load_s
                    + scenario.router_latency_s
                    + tracking["cost"]["latency_s"]
                )
                completion, expected_index, skipped = scenario.schedule.complete(
                    index, seconds_to_ns(latency, allow_zero=True)
                )
                require(
                    event["capture_timestamp_ns"] == scenario.schedule.at(index),
                    "Capture timestamp mismatch",
                )
                require(
                    abs(
                        event["completion_time_s"] - (completion - scenario.schedule.epoch_ns) / 1e9
                    )
                    < 1e-9
                    and event["skipped_observation_ids"] == list(skipped),
                    "Completion/skip mismatch",
                )
                energy += (
                    spec.energy_j_per_call
                    + load_j
                    + scenario.router_energy_j
                    + tracking["cost"]["energy_j"]
                )
                require(
                    abs(
                        event["cumulative_energy_j"]
                        - energy
                        - event["completion_time_s"] * scenario.flight_power_w
                    )
                    < 1e-6,
                    "Per-event energy mismatch",
                )
                require(
                    event["resident_memory_mb"]
                    == resources.memory_mb + tracking["cost"]["memory_mb"],
                    "Resident memory mismatch",
                )
                by_capture[index].append((router, event, scenario, contract, tracking, reports))
                reports += len(event["emitted_reports"])
                previous = config
                checked_events += 1
            require(
                abs(
                    result["cumulative_energy_j"]
                    - energy
                    - result["final_time_s"] * scenario.flight_power_w
                )
                < 1e-6,
                "Final energy mismatch",
            )
        backend = ReplayBackend(scenario, source / environment / "recording")
        try:
            for index, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                if environment == "day":
                    training_contexts[index] = context_key(frame.rgb)
                for router, event, run_scenario, contract, tracking, reports in by_capture[index]:
                    require(event["capture_body_pose"] == frame.body_pose, "Changed physical pose")
                    public = SimpleNamespace(
                        contract=run_scenario.contract,
                        intent_contract=immutable(contract.public_document()),
                        state=SimpleNamespace(**event["runtime_at_capture"]),
                        configs=run_scenario.catalog,
                        resources=run_scenario.resources,
                        tracking=immutable(tracking),
                        confirmed_reports_so_far=reports,
                        rgb=frame.rgb,
                    )
                    choice = router.select(public)
                    require(
                        choice == event["execution"]["config_id"], "Public-only selection differs"
                    )
                    if isinstance(router, CausalRouter):
                        require(
                            router.last_decision == event["router_decision"],
                            "Public-only decision trace differs",
                        )
                    decisions += 1
        finally:
            backend.close()
    training_scenario, _, _ = scenario_and_cache(prior, source, "day", "sector_rescue")
    rebuilt = fit_profile(
        read(prior / "day/outcomes.json"),
        training_contexts,
        training_scenario.executors,
        split_fingerprint(training_scenario, protocol["scenery_sha256"]),
    )
    require(rebuilt == profile, "Calibration differs from declared development data")
    result = dict(
        schema_version="lake_causal_validation_v1",
        training_aggregation_reproduced=True,
        complete=True,
        cases=len(rows),
        events=checked_events,
        public_only_decisions_reproduced=decisions,
        physical_poses_verified=True,
        per_event_clock_energy_memory_verified=True,
        canonical_outcomes_verified=True,
        calibration_sha256=digest(profile),
        summary_sha256=sha(output / "summary.json"),
        validator_sha256=sha(Path(__file__)),
        independent_test=False,
    )
    save(output / "validation.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(validate(parser.parse_args().output))
