"""Freeze battery interventions without changing trajectories or evidence contracts."""

import copy
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.mission_policy_selection.train import verify as verify_training
from experiments.mission_routing_v2.flights import load_scenario, split_check
from experiments.mission_routing_v4.train import verify as verify_risk
from experiments.mission_routing_v6.flights import verify as verify_cohort

BATTERIES = (0.335, 0.34, 0.35)
MISSIONS = ("rapid_survey", "verified_inspection")


def prepare(cohort, training, risk_training, selection, output):
    from experiments.resource_routing_study.study import POLICIES

    if output.exists():
        raise FileExistsError("Preserve previous resource intervention")
    old = verify_cohort(cohort)
    from experiments.mission_routing_v6.sensor import verify_gate

    verify_gate(cohort)
    verify_training(training)
    risk = verify_risk(risk_training)
    chosen = read(selection / "analysis.json")
    completion = read(selection / "completion.json")
    if (
        sha(selection / "analysis.json") != completion["analysis_sha256"]
        or chosen["selection_result"]["best_adaptive_candidate"] != "base_original_risk"
    ):
        raise ValueError("Wrong development-selected adaptive comparator")
    output.mkdir(parents=True)
    cases, frozen = {}, {}
    for flight, case in old["cases"].items():
        for battery in BATTERIES:
            name = f"{flight}_b{round(battery * 1000)}"
            cases[name] = dict(case, source_flight=flight, initial_battery=battery)
            for mission in MISSIONS:
                folder = output / name / mission
                folder.mkdir(parents=True)
                doc = copy.deepcopy(read(cohort / flight / mission / "scenario.json"))
                doc["mission"]["initial_battery_frac"] = battery
                intent = read(cohort / flight / mission / "intent.json")
                intent["mission"]["initial_battery_frac"] = battery
                save(folder / "scenario.json", doc)
                save(folder / "contract.json", read(cohort / flight / mission / "contract.json"))
                save(folder / "intent.json", intent)
                scenario = bind_scenario(
                    load_scenario(
                        folder / "scenario.json", case["environment"], Path(old["assets"])
                    ),
                    IntentContract(intent),
                )
                if (
                    scenario.recording_identity
                    != read(cohort / flight / "recording/recording.json")["recording_identity"]
                ):
                    raise ValueError("Battery intervention changed physical lineage")
                for path in folder.glob("*.json"):
                    frozen[str(path.relative_to(output))] = sha(path)
            recall = read(training / "all_development/recall.json")
            split_check(recall["training_lineage"], case["lineage"])
            split_check(risk["training_lineage"], case["lineage"])
    repo = Path(__file__).resolve().parents[2]
    sources = [
        p for tree in ("experiments", "missionbench") for p in (repo / tree).rglob("*.py")
    ]
    inputs = [
        cohort / "protocol.json",
        cohort / "sensor_gate.json",
        cohort / "validation.json",
        cohort / "feasibility_diagnostics.json",
        training / "completion.json",
        risk_training / "completion.json",
        selection / "analysis.json",
        selection / "completion.json",
    ]
    for flight in old["cases"]:
        inputs += [
            cohort / flight / "diagnostic_outcomes.json",
            cohort / flight / "predictions/manifest.json",
            cohort / flight / "recording/recording.json",
        ]
    save(
        output / "protocol.json",
        dict(
            schema_version="resource_routing_study_v1",
            scope="exploratory_reuse_of_inspected_v6",
            cohort=str(cohort.resolve()),
            training=str(training.resolve()),
            risk_training=str(risk_training.resolve()),
            selection=str(selection.resolve()),
            assets=old["assets"],
            cases=cases,
            missions=list(MISSIONS),
            policies=list(POLICIES),
            active_cpu_power_sensitivity_w=[15],
            batteries=list(BATTERIES),
            primary_battery=0.34,
            frozen_files=frozen,
            source_sha256={str(p): sha(p) for p in sources},
            inputs_sha256={str(p.resolve()): sha(p) for p in inputs},
            fresh_test=False,
            new_physical_flights=0,
            new_neural_calls=0,
            profile_refitting=False,
            witnesses=dict(gt_aware=True, beam_width=500, excluded_from_policy_grid=True),
            budget_blind_assumed_initial_battery=0.65,
            primary_comparison="Paired full-contract success: adaptive versus battery_blind at 34%",
            reserve_patrol_omitted_reason=(
                "Same evidence contract as rapid_survey once initial battery is controlled"
            ),
            caveats=[
                "Previously inspected known-map cohort; not prospective generalization",
                "Battery and 15-W inference energy are configured, not measured",
                "A changed schedule or preserved reserve is not necessarily a mission benefit",
                "Witness search failure is inconclusive",
                "All cells retained, including possible quality/resource infeasibility",
            ],
        ),
    )
    verify(output)


def verify(output):
    from experiments.resource_routing_study.study import POLICIES

    p = read(output / "protocol.json")
    if (
        p["schema_version"] != "resource_routing_study_v1"
        or p["policies"] != list(POLICIES)
        or p["missions"] != list(MISSIONS)
        or p["batteries"] != list(BATTERIES)
    ):
        raise ValueError("Wrong resource intervention grid")
    verify_cohort(Path(p["cohort"]))
    verify_training(Path(p["training"]))
    risk = verify_risk(Path(p["risk_training"]))
    for field in ("inputs_sha256", "source_sha256"):
        for path, expected in p[field].items():
            if sha(Path(path)) != expected:
                raise ValueError("Changed study input or source: " + path)
    for path, expected in p["frozen_files"].items():
        if sha(output / path) != expected:
            raise ValueError("Changed intervention contract")
    recall = read(Path(p["training"]) / "all_development/recall.json")
    expected = {(f, b) for f in verify_cohort(Path(p["cohort"]))["cases"] for b in BATTERIES}
    actual = [(c["source_flight"], c["initial_battery"]) for c in p["cases"].values()]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete battery intervention grid")
    for case in p["cases"].values():
        split_check(recall["training_lineage"], case["lineage"])
        split_check(risk["training_lineage"], case["lineage"])
    return p
