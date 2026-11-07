"""Rebuild prediction history and routing decisions, audit clocks and mission scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract, evaluate, immutable
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import SurfaceProjector, TrackingConfig, TrackingPipeline
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_routing_v2.feedback import snapshot
from experiments.mission_routing_v2.flights import split_check
from experiments.mission_routing_v2.policy import Policy as PreviousPolicy
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_routing_v3.flights import verify
from experiments.mission_routing_v3.policy import Policy
from experiments.mission_routing_v3.study import POLICIES, make_policy, scenario_for
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_gazebo_pilot.timing import seconds_to_ns


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run(root):
    p = verify(root)
    profile = read(Path(p["screen"]) / "calibration.json")
    terrain, _ = load_public_terrain(Path(p["assets"]))
    counts = dict(cases=0, events=0, public_decisions_reproduced=0)
    source_hashes = {}
    for flight, case in p["cases"].items():
        split_check(profile["training_lineage"], case["lineage"])
        folder = root / flight / "predictions"
        manifest = checked_manifest(folder, root / "protocol.json")
        masks = {c: VerifiedMasks(folder, m) for c, m in manifest["models"].items()}
        by_capture = defaultdict(list)
        for power in p["active_cpu_power_sensitivity_w"]:
            summary_path = root / f"summary_{flight}_{power}.json"
            summary = read(summary_path)
            source_hashes[summary_path.name] = sha(summary_path)
            require(
                summary["complete"] and summary["protocol_sha256"] == sha(root / "protocol.json"),
                "Wrong summary",
            )
            expected = {(mission, policy) for mission in p["missions"] for policy in POLICIES}
            actual = [(r["mission"], r["policy"]) for r in summary["cases"]]
            require(
                len(actual) == len(expected) and set(actual) == expected, "Incomplete policy grid"
            )
            for row in summary["cases"]:
                destination = root / row["path"]
                for filename, key in (
                    ("outcome.json", "outcome_sha256"),
                    ("execution/events.json", "events_sha256"),
                    ("execution/results.json", "results_sha256"),
                ):
                    require(sha(destination / filename) == row[key], "Changed result artifact")
                scenario = scenario_for(root, p, flight, row["mission"], power)
                contract = IntentContract(read(root / flight / row["mission"] / "intent.json"))
                scenario = bind_scenario(scenario, contract)
                events = read(destination / "execution/events.json")
                result = read(destination / "execution/results.json")
                outcome = read(destination / "outcome.json")
                canonical = evaluate(contract, events, result)
                for key in ("mission_success", "failed_constraints", "reporting", "phases"):
                    require(canonical[key] == outcome[key], "Canonical outcome mismatch")
                require(
                    row["full_success"] == canonical["mission_success"], "False success summary"
                )
                policy = make_policy(row["policy"], profile, scenario)
                tracking = read(destination / "protocol.json")["public_tracking"]
                tracker = TrackingPipeline(
                    TrackingConfig(remember_reports=True), SurfaceProjector(terrain)
                )
                specs = {s.config_id: s for s in scenario.executors}
                previous, index, energy = None, 0, 0.0
                for e in events:
                    t = e["capture_time_s"]
                    state = e["runtime_at_capture"]
                    require(
                        e["observation_id"] == index and state["current_config_id"] == previous,
                        "Illegal decision schedule",
                    )
                    before = energy + t * scenario.flight_power_w
                    require(
                        abs(state["cumulative_energy_j"] - before) < 1e-6, "Capture energy mismatch"
                    )
                    require(
                        abs(
                            state["battery_frac"]
                            - (
                                scenario.initial_battery_frac
                                - before / (scenario.battery_capacity_wh * 3600)
                            )
                        )
                        < 1e-8,
                        "Battery mismatch",
                    )
                    config = e["execution"]["config_id"]
                    spec, resource = specs[config], scenario.resources.for_config(config)
                    load_s = resource.load_latency_s if config != previous else 0
                    load_j = resource.load_energy_j if config != previous else 0
                    completion, next_index, skipped = scenario.schedule.complete(
                        index,
                        seconds_to_ns(
                            spec.mission_latency_s
                            + load_s
                            + scenario.router_latency_s
                            + tracking["cost"]["latency_s"]
                        ),
                    )
                    require(
                        e["capture_timestamp_ns"] == scenario.schedule.at(index),
                        "Capture clock mismatch",
                    )
                    require(
                        abs(
                            e["completion_time_s"] - (completion - scenario.schedule.epoch_ns) / 1e9
                        )
                        < 1e-8
                        and e["skipped_observation_ids"] == list(skipped),
                        "Completion clock mismatch",
                    )
                    energy += (
                        spec.energy_j_per_call
                        + load_j
                        + scenario.router_energy_j
                        + tracking["cost"]["energy_j"]
                    )
                    require(
                        abs(
                            e["cumulative_energy_j"]
                            - energy
                            - e["completion_time_s"] * scenario.flight_power_w
                        )
                        < 1e-6,
                        "Event energy mismatch",
                    )
                    require(
                        e["resident_memory_mb"]
                        == resource.memory_mb + tracking["cost"]["memory_mb"],
                        "Memory mismatch",
                    )
                    by_capture[index].append((policy, tracker, e, scenario, contract, tracking))
                    previous, index = config, next_index
                    counts["events"] += 1
                require(
                    abs(
                        result["cumulative_energy_j"]
                        - energy
                        - result["final_time_s"] * scenario.flight_power_w
                    )
                    < 1e-6,
                    "Final energy mismatch",
                )
                counts["cases"] += 1
        backend = ReplayBackend(scenario, root / flight / "recording")
        try:
            for index, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                for policy, tracker, event, scenario, contract, tracking in by_capture[index]:
                    require(
                        frame.body_pose == event["capture_body_pose"],
                        "Physical trajectory mismatch",
                    )
                    history = snapshot(tracker, event["capture_time_s"])
                    observation = SimpleNamespace(
                        contract=scenario.contract,
                        intent_contract=immutable(contract.public_document()),
                        state=SimpleNamespace(**event["runtime_at_capture"]),
                        configs=scenario.catalog,
                        resources=scenario.resources,
                        tracking=immutable(tracking),
                        evidence_history=immutable(history),
                        confirmed_reports_so_far=tracker.report_count,
                        rgb=frame.rgb,
                    )
                    choice = policy.select(observation)
                    require(
                        choice == event["execution"]["config_id"], "Public-only routing differs"
                    )
                    if isinstance(policy, (Policy, PreviousPolicy)):
                        require(
                            policy.last_decision == event["router_decision"],
                            "Causal history or decision trace differs",
                        )
                    reports = tracker.update(
                        masks[choice][rgb_key(frame.rgb)],
                        event["capture_time_s"],
                        event["completion_time_s"],
                        pose=frame.body_pose,
                        camera_pitch_rad=scenario.world["camera_pitch_rad"],
                    )
                    require(
                        reports == event["emitted_reports"], "Prediction history does not reproduce"
                    )
                    counts["public_decisions_reproduced"] += 1
        finally:
            backend.close()
    result = dict(
        complete=True,
        scope=p["scope"],
        **counts,
        source_summaries_sha256=source_hashes,
        physical_poses_verified=True,
        per_event_clock_energy_memory_verified=True,
        prediction_history_rebuilt_without_ground_truth=True,
        canonical_outcomes_verified=True,
        validator_sha256=sha(Path(__file__)),
    )
    save(root / "validation.json", result)
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
