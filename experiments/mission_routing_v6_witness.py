"""Search evaluator-only schedules and verify them through the paid reporting runner.

Never part of the deployable policy comparison. Search uses GT phase evidence;
only a canonical full-contract replay establishes a witness. Beam failure is
inconclusive. Frozen models, costs, contracts and all predefined results stay intact.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_hindsight_search import HindsightSchedule, search
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_intent_contract import IntentContract
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import TrackingConfig
from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST, load_public_terrain
from experiments.mission_routing_v2.runtime import run_tracking_mission
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_routing_v2_checked import require_route
from experiments.mission_routing_v6.flights import verify
from experiments.mission_routing_v6.study import scenario_for
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


def run(root):
    require_route(root)
    p = verify(root)
    diagnostic = read(root / "feasibility_diagnostics.json")
    if diagnostic["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Changed diagnostic protocol")
    output = root / "witnesses"
    output.mkdir(exist_ok=False)
    plan = dict(
        schema_version="mission_routing_v6_witness_v1",
        gt_aware=True,
        deployable_policy=False,
        posthoc=True,
        beam_width=500,
        inference_power_w=15,
        protocol_sha256=sha(root / "protocol.json"),
        source_sha256=sha(Path(__file__)),
        search_sha256=sha(Path(__file__).with_name("lake_hindsight_search.py")),
        purpose="Feasibility only; excluded from the 90-case policy grid",
    )
    save(output / "protocol.json", plan)
    terrain, metadata = load_public_terrain(Path(p["assets"]))
    rows = []
    for flight in p["cases"]:
        folder = root / flight / "predictions"
        manifest = checked_manifest(folder, root / "protocol.json")
        evidence = root / flight / "diagnostic_outcomes.json"
        d = diagnostic["flights"][flight]
        if d["outcomes_sha256"] != sha(evidence) or d["source_manifest_sha256"] != sha(
            folder / "manifest.json"
        ):
            raise ValueError("Changed witness evidence")
        for mission in p["missions"]:
            scenario = scenario_for(root, p, flight, mission, 15)
            contract = IntentContract(read(root / flight / mission / "intent.json"))
            if manifest["recording_identity"] != scenario.recording_identity or manifest[
                "recording_sha256"
            ] != sha(root / flight / "recording/recording.json"):
                raise ValueError("Wrong witness recording")
            found = search(
                contract.document()["mission"],
                search_scenario(bind_scenario(scenario, contract)),
                read(evidence),
                beam_width=plan["beam_width"],
            )
            destination = output / flight / mission
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
                backend = ReplayBackend(scenario, root / flight / "recording")
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
                    full_contract_witness=outcome["mission_success"],
                    failed_constraints=outcome["failed_constraints"],
                    outcome_sha256=sha(replay / "outcome.json"),
                    events_sha256=sha(replay / "execution/events.json"),
                    results_sha256=sha(replay / "execution/results.json"),
                )
            rows.append(row)
            print(f"{flight} {mission}: {row}", flush=True)
    save(
        output / "summary.json",
        dict(
            complete=True,
            gt_aware=True,
            deployable_policy=False,
            protocol_sha256=sha(output / "protocol.json"),
            cases=rows,
        ),
    )
    lines = [
        "# Evaluator-only feasibility witnesses",
        "",
        "All schedules use hidden test evidence and are excluded from policy rankings. "
        "A successful paid replay establishes feasibility only. Beam failure or a "
        "failed reporting replay does not prove impossibility. No contracts were relaxed.",
        "",
        "| Flight | Intent | Search conclusion | Full-contract witness |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r['flight']} | {r['mission']} | {r['search_conclusion']} | "
        f"{r['full_contract_witness']} |"
        for r in rows
    ]
    lines += [
        "",
        "Search charges tracking, inference, routing, switching and tail-flight energy. "
        "Each candidate schedule is replayed with the unchanged predictive tracker and "
        "canonical report scoring. Capture/model schedules and energy totals must match.",
        "",
    ]
    (root / "WITNESSES.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
