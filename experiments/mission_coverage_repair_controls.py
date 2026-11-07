"""Prediction-independent input masking controls using a frozen public observation."""

import argparse
import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import TrackingConfig
from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST
from experiments.mission_coverage_repair.protocol import verify
from experiments.mission_coverage_repair.study import load_profiles, make_policy, scenario_for
from experiments.v4_benchmark.backends import ReplayBackend


def run(root):
    p = verify(root)
    flight = next(iter(p["cases"]))
    case = p["cases"][flight]
    mission = "verified_inspection"
    scenario = scenario_for(root, p, flight, mission, 15)
    contract = IntentContract(read(root / flight / mission / "intent.json"))
    scenario = bind_scenario(scenario, contract)
    backend = ReplayBackend(scenario, Path(p["cohort"]) / case["source_flight"] / "recording")
    try:
        backend.advance_to(next(iter(backend.frames)))
        frame = backend.capture()
    finally:
        backend.close()
    obs = SimpleNamespace(
        contract=scenario.contract,
        intent_contract=contract.public_document(),
        state=SimpleNamespace(
            current_time_s=0,
            frame_id=0,
            current_config_id=None,
            battery_frac=0.34,
            cumulative_energy_j=0,
        ),
        configs=scenario.catalog,
        resources=scenario.resources,
        rgb=frame.rgb,
        tracking=dict(
            configuration=asdict(TrackingConfig(remember_reports=True)),
            cost=asdict(DEFAULT_TRACKING_COST),
        ),
        evidence_history=dict(
            available_at_s=0, last_completion_s=None, counts_are_ground_truth=False, tracks=[]
        ),
        confirmed_reports_so_far=0,
    )
    profiles = load_profiles(p)
    checks = []

    def selected(name, o):
        policy = make_policy(name, profiles, scenario)
        choice = policy.select(o)
        score = [
            (c["config_id"], c["coverage_log_score"]) for c in policy.last_decision["candidates"]
        ]
        return choice, score

    for name, changed in (
        ("coverage_no_context", "image"),
        ("coverage_no_intent", "intent"),
        ("coverage_no_battery", "battery"),
    ):
        other = copy.copy(obs)
        if changed == "image":
            other.rgb = np.full_like(obs.rgb, 255)
        elif changed == "intent":
            other.intent_contract = copy.deepcopy(obs.intent_contract)
            for ph in other.intent_contract["mission"]["phases"]:
                ph.update(min_hits=2, min_evidence_span_s=2, required_recall=0.5)
        else:
            other.state = SimpleNamespace(**dict(vars(obs.state), battery_frac=0.65))
            other.intent_contract = copy.deepcopy(obs.intent_contract)
            other.intent_contract["mission"]["initial_battery_frac"] = 0.65
        checks.append(
            dict(
                policy=name, changed=changed, invariant=selected(name, obs) == selected(name, other)
            )
        )
    for invalid in ("future", "ground_truth"):
        other = copy.copy(obs)
        other.evidence_history = copy.deepcopy(obs.evidence_history)
        if invalid == "future":
            other.evidence_history["last_completion_s"] = 1
        else:
            other.evidence_history["counts_are_ground_truth"] = True
        rejected = False
        try:
            selected("coverage_tail", other)
        except ValueError:
            rejected = True
        checks.append(dict(policy="coverage_tail", changed=invalid, invariant=rejected))
    if not all(c["invariant"] for c in checks):
        raise ValueError("Failed input boundary control")
    result = dict(
        complete=True,
        controls=checks,
        protocol_sha256=sha(root / "protocol.json"),
        source_sha256=sha(Path(__file__)),
        prediction_outputs_used=False,
        scope="First captured frame, matched input interventions; not an outcome comparison",
    )
    save(root / "input_controls.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
