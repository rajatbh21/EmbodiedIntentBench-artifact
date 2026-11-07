"""Search evaluator-only schedules and verify them through the paid reporting runner.

Never part of the deployable policy comparison. Search uses GT phase evidence;
only a canonical full-contract replay establishes a witness. Beam failure is
inconclusive. Frozen models, costs, contracts and all predefined results stay intact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_search import HindsightSchedule, search
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import TrackingConfig
from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST
from experiments.mission_environment_transfer_v3.scenario import load_public_terrain
from experiments.mission_routing_v2.runtime import run_tracking_mission
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_transfer_comparison_v4.integrity import check_execution
from experiments.mission_transfer_comparison_v4.protocol import check_prediction_admission, verify
from experiments.mission_transfer_comparison_v4.study import scenario_for
from experiments.mission_transfer_within_recording.admission import verify as verify_admission
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.probe_cache import CachedExecutor


def search_scenario(scenario):
    """Account for identical tracking costs in search; leave replay input unchanged."""
    cost = DEFAULT_TRACKING_COST
    capacity = scenario.resources.memory_capacity_mb
    if capacity is not None and any(
        scenario.resources.for_config(s.config_id).memory_mb + cost.memory_mb > capacity
        for s in scenario.executors
    ):
        raise ValueError("This witness adapter requires all candidates to fit with tracking")
    return replace(
        scenario,
        router_latency_s=scenario.router_latency_s + cost.latency_s,
        router_energy_j=scenario.router_energy_j + cost.energy_j,
    )


def run(root, flight):
    p = verify(root)
    source = Path(p["cohort"])
    verify_admission(Path(p["admission"]))
    if flight not in p["cases"] or p["active_cpu_power_sensitivity_w"] != [15]:
        raise ValueError("Expected a declared condition at the fixed 15 W cost profile")
    source_flight = p["cases"][flight]["source_flight"]
    output = root / "witnesses" / flight
    output.mkdir(parents=True, exist_ok=False)
    terrain, metadata = load_public_terrain(Path(p["assets"]))
    folder = source / source_flight / "predictions"
    manifest = checked_manifest(folder, source / "protocol.json")
    check_prediction_admission(manifest, p)
    evidence = source / source_flight / "diagnostic_outcomes.json"
    diagnostic = read(source / "feasibility_diagnostics.json")
    d = diagnostic["flights"][source_flight]
    if (
        diagnostic["schema_version"] != "independent_map_posthoc_feasibility_v1"
        or not diagnostic["gt_aware"]
        or diagnostic["deployable_policy"]
        or diagnostic["diagnostic_implementation_sha256"]
        != sha(Path(__file__).with_name("diagnostics.py"))
        or diagnostic["protocol_sha256"] != sha(source / "protocol.json")
        or diagnostic["comparison_protocol_sha256"] != sha(root / "protocol.json")
        or diagnostic["routing_admission_sha256"] != p["routing_admission_sha256"]
        or d["source_manifest_sha256"] != sha(folder / "manifest.json")
        or d["outcomes_sha256"] != sha(evidence)
    ):
        raise ValueError("Changed diagnostic evidence")
    save(
        output / "plan.json",
        dict(
            gt_aware=True,
            deployable_policy=False,
            posthoc=True,
            beam_width=500,
            source_sha256=sha(Path(__file__)),
            helper_sha256={
                str(path.resolve()): sha(path)
                for path in [
                    *Path(__file__).parent.glob("*.py"),
                    Path(__file__).parents[1] / "lake_hindsight_search.py",
                    Path(__file__).parents[1] / "mission_transfer_comparison_v4/integrity.py",
                ]
            },
            evidence_sha256=sha(evidence),
            protocol_sha256=sha(root / "protocol.json"),
        ),
    )
    rows = []
    for mission in p["missions"]:
        scenario = scenario_for(root, p, flight, mission, 15)
        contract = IntentContract(read(root / flight / mission / "intent.json"))
        found = search(
            contract.document()["mission"],
            search_scenario(bind_scenario(scenario, contract)),
            read(evidence),
            beam_width=500,
        )
        destination = output / mission
        destination.mkdir(parents=True)
        save(destination / "search.json", found)
        row = dict(
            flight=flight,
            mission=mission,
            search_conclusion=found["conclusion"],
            full_contract_witness=False,
            search_sha256=sha(destination / "search.json"),
        )
        if found["schedule"] is not None:
            executors = {}
            for spec in scenario.executors:
                model = manifest["models"][spec.config_id]
                if model["signature"] != model_signature(spec):
                    raise ValueError("Wrong witness model signature")
                executors[spec.config_id] = CachedExecutor(
                    spec, VerifiedMasks(folder, model), sha(folder / "manifest.json")
                )
            backend = ReplayBackend(scenario, source / source_flight / "recording")
            try:
                outcome = run_tracking_mission(
                    scenario,
                    backend,
                    HindsightSchedule(found["schedule"]),
                    executors,
                    contract,
                    destination / "replay",
                    config=TrackingConfig(remember_reports=True),
                    terrain=terrain,
                    map_metadata=metadata,
                    router_name="gt_aware_witness",
                    gt_aware=True,
                )
            finally:
                backend.close()
            replay = destination / "replay"
            events = read(replay / "execution/events.json")
            actual = [
                dict(capture_id=e["observation_id"], config_id=e["execution"]["config_id"])
                for e in events
            ]
            if actual != found["schedule"]:
                raise ValueError("Paid replay differs from witness schedule")
            result = read(replay / "execution/results.json")
            expected_energy = (
                scenario.duration_s * scenario.flight_power_w
                + found["planned_nonflight_and_tail_energy_j"]
            )
            if abs(result["cumulative_energy_j"] - expected_energy) > 1e-6:
                raise ValueError("Witness search/replay energy differs")
            row.update(
                policy="gt_aware_witness",
                power_w=15,
                path=str(replay.relative_to(root)),
                full_success=outcome["mission_success"],
                phases_succeeded=sum(v["success"] for v in outcome["phases"]),
                report_recall=outcome["reporting"]["recall_by_deadline"],
                false_reports=outcome["reporting"]["false_reports"],
                duplicate_reports=outcome["reporting"]["duplicate_reports"],
                selections=dict(Counter(e["execution"]["config_id"] for e in events)),
                final_battery=result["final_battery_frac"],
                calls=len(events),
                full_contract_witness=outcome["mission_success"],
                failed_constraints=outcome["failed_constraints"],
                outcome_sha256=sha(replay / "outcome.json"),
                events_sha256=sha(replay / "execution/events.json"),
                results_sha256=sha(replay / "execution/results.json"),
            )
            check_execution(
                bind_scenario(scenario, contract),
                events,
                result,
                row,
                evaluate(contract, events, result),
            )
        rows.append(row)
        print(f"{flight} {mission}: {row}", flush=True)
    save(
        output / "summary.json",
        dict(
            complete=True,
            gt_aware=True,
            deployable_policy=False,
            protocol_sha256=sha(root / "protocol.json"),
            cases=rows,
        ),
    )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flight", required=True)
    args = parser.parse_args()
    run(args.output, args.flight)
