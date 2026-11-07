"""Time stateful public-only routing and audit same-state mechanism changes.

Three independent controller instances replay each complete validated trajectory.
No idempotent retries enter timings. Counterfactuals borrow the full policy's
pre-decision expected ledger and public history, not evaluator labels.
"""

from __future__ import annotations

import argparse
import copy
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
from experiments.mission_routing_v3_timing import describe
from experiments.mission_routing_v4.flights import verify
from experiments.mission_routing_v4.policy import MODES, Policy
from experiments.mission_routing_v4.study import scenario_for
from experiments.mission_routing_v4_report import run as report


def run(root):
    p, audited = verify(root), report(root)
    quality = read(Path(p["screen"]) / "calibration.json")
    risk = read(Path(p["risk_training"]) / "risk_profile.json")
    cold, warm, timings, counterfactuals = [], [], [], []
    images = {}
    for row in audited["cases"]:
        if row["policy"] != "risk_aware":
            continue
        flight, mission = row["flight"], row["mission"]
        folder = root / row["path"]
        contract = IntentContract(read(root / flight / mission / "intent.json"))
        scenario = bind_scenario(scenario_for(root, p, flight, mission, row["power_w"]), contract)
        costs = MissionCosts.from_scenario(scenario)
        tracking = read(folder / "protocol.json")["public_tracking"]
        records = read(root / flight / "recording/recording.json")["frames"]
        policies = [Policy(quality, scenario.executors, costs, risk) for _ in range(3)]
        for index, event in enumerate(read(folder / "execution/events.json")):
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
                tracking=immutable(tracking),
                evidence_history=immutable(event["router_decision"]["history"]),
            )
            for repeat, policy in enumerate(policies):
                started = time.perf_counter_ns()
                choice = policy.select(o)
                elapsed = (time.perf_counter_ns() - started) / 1e6
                if (
                    choice != event["execution"]["config_id"]
                    or policy.last_decision != event["router_decision"]
                ):
                    raise ValueError(
                        "Timed stateful decision differs from independently validated run"
                    )
                (cold if index == 0 else warm).append(elapsed)
                timings.append(
                    dict(
                        flight=flight,
                        mission=mission,
                        capture_id=capture,
                        repeat=repeat,
                        ms=elapsed,
                    )
                )
            for mode in MODES:
                if mode == "risk_aware":
                    continue
                alternate = Policy(quality, scenario.executors, costs, risk, mode)
                alternate.ledger = copy.deepcopy(event["router_decision"]["risk_ledger_before"])
                other = alternate.select(o)
                counterfactuals.append(
                    dict(
                        flight=flight,
                        mission=mission,
                        capture_id=capture,
                        ablation=mode,
                        actual_choice=choice,
                        counterfactual_choice=other,
                        changed=choice != other,
                    )
                )
    result = dict(
        schema_version="v4_local_router_timing_v1",
        cold=describe(cold),
        warm=describe(warm),
        configured_router_latency_ms=50,
        configured_costs_changed=False,
        measured_energy=False,
        scope="Local select only; no model inference running; other host load uncontrolled",
        excludes=["frame decoding", "tracking", "event I/O", "model inference"],
        repeated_idempotent_calls_timed=False,
        platform=platform.platform(),
        python=platform.python_version(),
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        source_sha256=sha(Path(__file__)),
        timings=timings,
        same_state_counterfactuals=counterfactuals,
        same_state_changes={
            mode: sum(c["changed"] for c in counterfactuals if c["ablation"] == mode)
            for mode in MODES
            if mode != "risk_aware"
        },
        counterfactual_interpretation=(
            "Same public input and full-policy ledger; mechanism activation, "
            "not benefit; outside mission clock"
        ),
    )
    save(root / "router_timing.json", result)
    print({k: result[k] for k in ("cold", "warm", "same_state_changes")}, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
