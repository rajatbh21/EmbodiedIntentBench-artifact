"""Post-validation local controller timing; never changes frozen charged latency.

Times select() including RGB features, DP construction/reuse and decision trace
construction. Excludes frame decoding, tracking and JSON I/O. This is not an
onboard end-to-end deployment measurement or an energy measurement.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.lake_causal_router import MissionCosts
from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, immutable
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v3.flights import verify
from experiments.mission_routing_v3.policy import MODES, Policy
from experiments.mission_routing_v3.study import scenario_for
from experiments.mission_routing_v3_report import run as report


def describe(values):
    return dict(
        samples=len(values),
        median_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        max_ms=max(values),
        above_50ms=sum(x > 50 for x in values),
    )


def run(root):
    p, audited = verify(root), report(root)
    profile = read(Path(p["screen"]) / "calibration.json")
    cold, warm, rows, counterfactuals, resource_audit = [], [], [], [], []
    images = {}
    for row in audited["cases"]:
        if row["policy"] != "confirmation":
            continue
        folder = root / row["path"]
        flight, mission = row["flight"], row["mission"]
        contract = IntentContract(read(root / flight / mission / "intent.json"))
        scenario = bind_scenario(scenario_for(root, p, flight, mission, row["power_w"]), contract)
        tracker = read(folder / "protocol.json")["public_tracking"]
        policy = Policy(profile, scenario.executors, MissionCosts.from_scenario(scenario))
        alternatives = {
            mode: Policy(profile, scenario.executors, MissionCosts.from_scenario(scenario), mode)
            for mode in MODES
            if mode != "confirmation"
        }
        records = read(root / flight / "recording/recording.json")["frames"]
        events = read(folder / "execution/events.json")
        for index, event in enumerate(events):
            capture = event["observation_id"]
            if (flight, capture) not in images:
                frame = records[capture]
                path = root / flight / "recording" / frame["file"]
                if sha(path) != frame["sha256"]:
                    raise ValueError("Changed physical frame")
                with np.load(path, allow_pickle=False) as data:
                    images[flight, capture] = data["rgb"]
            o = SimpleNamespace(
                state=SimpleNamespace(**event["runtime_at_capture"]),
                rgb=images[flight, capture],
                contract=scenario.contract,
                intent_contract=immutable(contract.public_document()),
                configs=scenario.catalog,
                resources=scenario.resources,
                tracking=immutable(tracker),
                evidence_history=immutable(event["router_decision"]["history"]),
            )
            # Repeat identical public inputs. The policy holds only the resource
            # table cache; its causal evidence state comes entirely from o.
            for repeat in range(3):
                started = time.perf_counter_ns()
                choice = policy.select(o)
                ms = (time.perf_counter_ns() - started) / 1e6
                if (
                    choice != event["execution"]["config_id"]
                    or policy.last_decision != event["router_decision"]
                ):
                    raise ValueError("Timed public decision differs from validated run")
                first = index == 0 and repeat == 0
                (cold if first else warm).append(ms)
                rows.append(
                    dict(
                        flight=flight,
                        mission=mission,
                        capture_id=capture,
                        first_decision=first,
                        repeat=repeat,
                        measured_ms=ms,
                    )
                )
            for mode, alternative in alternatives.items():
                alternate_choice = alternative.select(o)
                counterfactuals.append(
                    dict(
                        flight=flight,
                        mission=mission,
                        capture_id=capture,
                        ablation=mode,
                        actual_choice=choice,
                        counterfactual_choice=alternate_choice,
                        changed=choice != alternate_choice,
                    )
                )
            trace = event["router_decision"]
            admitted = [
                c["config_id"]
                for c in trace["candidates"]
                if c["feasible"] and c["constant_energy_j"] > trace["available_energy_j"] + 1e-8
            ]
            resource_audit.append(
                dict(
                    flight=flight,
                    mission=mission,
                    capture_id=capture,
                    newly_admitted_configurations=admitted,
                    selected_requires_mixed_continuation=choice in admitted,
                    resource_feasible=trace["resource_feasible"],
                )
            )
    result = dict(
        schema_version="v3_local_router_timing_v1",
        cold=describe(cold),
        warm=describe(warm),
        configured_router_latency_ms=50,
        configured_costs_changed=False,
        scope="Local select() only; no inference processes running; other host load uncontrolled",
        excludes=["frame decoding", "tracking", "event JSON serialization", "model inference"],
        measured_energy=False,
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        helper_sha256=sha(Path(__file__)),
        samples=rows,
        same_state_counterfactuals=counterfactuals,
        same_state_interpretation=(
            "Each ablation observes the full controller's identical validated public state. "
            "Changed actions demonstrate mechanism activation, not mission benefit. "
            "These extra diagnostic evaluations are outside the benchmark mission clock."
        ),
        same_state_changes={
            mode: sum(r["changed"] for r in counterfactuals if r["ablation"] == mode)
            for mode in alternatives
        },
        resource_admission_audit=resource_audit,
        mixed_budget_newly_admits_at_decisions=sum(
            bool(r["newly_admitted_configurations"]) for r in resource_audit
        ),
        selected_requires_mixed_continuation=sum(
            r["selected_requires_mixed_continuation"] for r in resource_audit
        ),
        resource_infeasible_decisions=sum(not r["resource_feasible"] for r in resource_audit),
    )
    save(root / "router_timing.json", result)
    print({k: result[k] for k in ("cold", "warm", "same_state_changes")}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
