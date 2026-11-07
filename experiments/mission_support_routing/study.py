"""Factorial support policies with the same physical sensor bank and paid costs."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from experiments.lake_causal_router import MissionCosts
from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import TrackingConfig
from experiments.lake_tracking_runtime import load_public_terrain
from experiments.mission_routing_v2.flights import load_scenario
from experiments.mission_routing_v2.runtime import run_tracking_mission
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_routing_v5.policy import Policy as Baseline
from experiments.mission_support_routing.policy import Policy
from experiments.mission_support_routing.protocol import verify
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import CachedExecutor

POLICIES = ("baseline", "weakest_support", "provenance_mean", "provenance_weakest")


def load_profiles(protocol, flight=None):
    return dict(
        recall=read(Path(protocol["training"]) / "all_development/recall.json"),
        risk=read(Path(protocol["risk_training"]) / "risk_profile.json"),
    )


def make_policy(name, profiles, scenario, risk_profile=None):
    if name not in POLICIES:
        raise ValueError("Undeclared support policy")
    args = (
        dict(profiles["recall"], routing_variant="base_recall"),
        scenario.executors,
        MissionCosts.from_scenario(scenario),
        profiles["risk"],
    )
    if name == "baseline":
        return Baseline(*args)
    return Policy(
        *args, provenance=name.startswith("provenance"), weakest=name != "provenance_mean"
    )


def scenario_for(root, protocol, flight, mission, power):
    base = load_scenario(
        root / flight / mission / "scenario.json",
        protocol["cases"][flight]["environment"],
        Path(protocol["assets"]),
    )
    specs = tuple(replace(s, energy_j_per_call=s.mission_latency_s * power) for s in base.executors)
    return replace(
        base,
        executors=specs,
        source={
            **base.source,
            "executor_configs": [asdict(spec) for spec in specs],
            "sensitivity_active_cpu_power_w": power,
        },
    )


def run(root, flight, power):
    p = verify(root)
    if power not in p["active_cpu_power_sensitivity_w"]:
        raise ValueError("Undeclared power condition")
    profile = load_profiles(p, flight)
    source = Path(p["cohort"])
    source_flight = p["cases"][flight]["source_flight"]
    risk_profile = None
    terrain, metadata = load_public_terrain(Path(p["assets"]))
    folder = source / source_flight / "predictions"
    manifest = checked_manifest(folder, source / "protocol.json")
    rows = []
    for mission in p["missions"]:
        scenario = scenario_for(root, p, flight, mission, power)
        if manifest["recording_identity"] != scenario.recording_identity or manifest[
            "recording_sha256"
        ] != sha(source / source_flight / "recording/recording.json"):
            raise ValueError("Predictions belong to a different flight")
        executors = {}
        for s in scenario.executors:
            model = manifest["models"][s.config_id]
            if model["signature"] != model_signature(s):
                raise ValueError("Model signature mismatch")
            executors[s.config_id] = CachedExecutor(
                s, VerifiedMasks(folder, model), sha(folder / "manifest.json")
            )
        contract = IntentContract(read(root / flight / mission / "intent.json"))
        for policy in POLICIES:
            destination = (
                root / "runs" / flight / f"power_{power}" / mission / policy.replace(":", "_")
            )
            if destination.exists():
                raise FileExistsError("Preserve prior execution: " + str(destination))
            backend = ReplayBackend(scenario, source / source_flight / "recording")
            try:
                outcome = run_tracking_mission(
                    scenario,
                    backend,
                    make_policy(policy, profile, scenario, risk_profile),
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
            result = read(destination / "execution/results.json")
            events = read(destination / "execution/events.json")
            if result["router_errors"] or result["inference_failures"] or result["invalid_actions"]:
                raise ValueError("Execution error: " + str(destination))
            phases = outcome["phases"]
            row = dict(
                flight=flight,
                mission=mission,
                policy=policy,
                power_w=power,
                path=str(destination.relative_to(root)),
                full_success=outcome["mission_success"],
                phases_succeeded=sum(v["success"] for v in phases),
                report_recall=outcome["reporting"]["recall_by_deadline"],
                false_reports=outcome["reporting"]["false_reports"],
                duplicate_reports=outcome["reporting"]["duplicate_reports"],
                final_battery=result["final_battery_frac"],
                calls=len(events),
                selections=dict(Counter(e["execution"]["config_id"] for e in events)),
                failed_constraints=outcome["failed_constraints"],
                outcome_sha256=sha(destination / "outcome.json"),
                events_sha256=sha(destination / "execution/events.json"),
                results_sha256=sha(destination / "execution/results.json"),
            )
            rows.append(row)
            print(
                f"{flight} {power}W {mission} {policy}: success={row['full_success']} "
                f"phases={row['phases_succeeded']}/4 reports={row['report_recall']:.2f}",
                flush=True,
            )
    save(
        root / f"summary_{flight}_{power}.json",
        dict(complete=True, cases=rows, protocol_sha256=sha(root / "protocol.json")),
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight", required=True)
    parser.add_argument("--power", type=int, default=15)
    args = parser.parse_args()
    run(args.output, args.flight, args.power)
