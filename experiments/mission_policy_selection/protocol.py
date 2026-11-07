"""Freeze a development-only mission selection grid before running any candidates."""

import copy
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract
from experiments.lake_missions.assets import save, sha
from experiments.mission_policy_selection.train import verify as verify_training
from experiments.mission_routing_v2.flights import contract_document, load_scenario, split_check
from experiments.mission_routing_v4.train import verify as verify_risk
from experiments.routing_development_matrix.cohort import verify as verify_cohort

MISSIONS = ("rapid_survey", "verified_inspection", "reserve_patrol")


def prepare(cohort, training, original_risk, output):
    from experiments.mission_policy_selection.study import POLICIES

    if output.exists():
        raise FileExistsError("Preserve previous mission selection")
    p = verify_cohort(cohort)
    train = verify_training(training)
    verify_risk(original_risk)
    if Path(train["cohort"]).resolve() != cohort.resolve():
        raise ValueError("Wrong training cohort")
    output.mkdir(parents=True)
    frozen, sources = {}, {}
    # Pin implementation dependencies as well as the new package. These are not
    # executable instructions from artifacts; only files in this repository.
    repo = Path(__file__).resolve().parents[2]
    for tree in ("experiments", "missionbench"):
        for path in sorted((repo / tree).rglob("*.py")):
            sources[str(path)] = sha(path)
    for flight, case in p["cases"].items():
        template = cohort / flight / "rapid_survey"
        for mission in MISSIONS:
            folder = output / flight / mission
            folder.mkdir(parents=True)
            doc = copy.deepcopy(read(template / "scenario.json"))
            targets = [o["object_id"] for o in doc["world"]["objects"] if o["is_target"]]
            intent = IntentContract(contract_document(targets, mission))
            doc["mission"]["initial_battery_frac"] = intent.document()["mission"][
                "initial_battery_frac"
            ]
            save(folder / "scenario.json", doc)
            save(folder / "contract.json", read(template / "contract.json"))
            save(folder / "intent.json", intent.document())
            scenario = load_scenario(
                folder / "scenario.json", case["environment"], Path(p["assets"])
            )
            recording = read(cohort / flight / "recording/recording.json")
            if scenario.recording_identity != recording["recording_identity"]:
                raise ValueError("Mission variant changed physical observations")
            for path in folder.glob("*.json"):
                frozen[str(path.relative_to(output))] = sha(path)
        for filename in ("recall.json", "risk.json"):
            profile = read(training / case["layout_group"] / filename)
            split_check(profile["training_lineage"], case["lineage"])
        split_check(verify_risk(original_risk)["training_lineage"], case["lineage"])
    save(
        output / "protocol.json",
        dict(
            schema_version="mission_policy_selection_v1",
            scope="development_selection_known_map",
            cohort=str(cohort.resolve()),
            training=str(training.resolve()),
            original_risk=str(original_risk.resolve()),
            cases=p["cases"],
            assets=p["assets"],
            missions=list(MISSIONS),
            policies=list(POLICIES),
            active_cpu_power_sensitivity_w=[15],
            source_sha256=sources,
            frozen_files=frozen,
            inputs_sha256={
                str(path.resolve()): sha(path)
                for path in (
                    cohort / "protocol.json",
                    cohort / "observability.json",
                    training / "completion.json",
                    original_risk / "completion.json",
                )
            },
            selection_order=[
                "macro_layout_full_success",
                "macro_layout_phase_success",
                "macro_layout_report_recall",
                "ascending_policy_name",
            ],
            selection_all_cases=True,
            sensor_infeasible_cases_removed=False,
            replay_layout_blocks=2,
            training_layout_blocks=3,
            original_development_role=(
                "Fit and risk diagnostics only; original flight does not implement "
                "the 120-second mission protocol"
            ),
            fresh_test=False,
            inference_rerun=False,
            new_physical_flights=0,
            caveats=[
                "Mission selection on inspected development data is not an unbiased test",
                "All strict inspection development flights are sensor-infeasible; retain them",
                "Same known map, two authored layout/route blocks",
                "Power and latency are configured, not onboard measurements",
                "Selection cannot establish adaptive superiority or resource adaptation",
            ],
        ),
    )
    verify(output)


def verify(output):
    from experiments.mission_policy_selection.study import POLICIES

    p = read(output / "protocol.json")
    if (
        p["schema_version"] != "mission_policy_selection_v1"
        or p["policies"] != list(POLICIES)
        or p["missions"] != list(MISSIONS)
        or p["active_cpu_power_sensitivity_w"] != [15]
    ):
        raise ValueError("Changed declared selection grid")
    verify_cohort(Path(p["cohort"]))
    verify_training(Path(p["training"]))
    original_risk = verify_risk(Path(p["original_risk"]))
    for field in ("inputs_sha256", "source_sha256"):
        for path, expected in p[field].items():
            if sha(Path(path)) != expected:
                raise ValueError("Changed selection input or implementation: " + path)
    for path, expected in p["frozen_files"].items():
        if sha(output / path) != expected:
            raise ValueError("Changed mission file")
    for case in p["cases"].values():
        for filename in ("recall.json", "risk.json"):
            profile = read(Path(p["training"]) / case["layout_group"] / filename)
            split_check(profile["training_lineage"], case["lineage"])
        split_check(original_risk["training_lineage"], case["lineage"])
    return p
