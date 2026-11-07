"""Pre-acquisition policy/intent/resource factorial; no post-outcome selection."""

import copy
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.cohort import verify as verify_cohort
from experiments.mission_routing_v2.flights import split_check

BATTERIES = (0.34, 0.65)


def prepare(cohort, output):
    from experiments.mission_coverage_routing.study import POLICIES, load_profiles

    old = verify_cohort(cohort)
    output.mkdir(parents=True, exist_ok=False)
    cases, frozen = {}, {}
    for flight, c in old["cases"].items():
        for battery in BATTERIES:
            name = f"{flight}_b{round(battery * 1000)}"
            cases[name] = dict(c, source_flight=flight, initial_battery=battery)
            for mission in old["missions"]:
                folder = output / name / mission
                folder.mkdir(parents=True)
                for n in ("scenario.json", "contract.json", "intent.json"):
                    doc = copy.deepcopy(read(cohort / flight / mission / n))
                    if n != "contract.json":
                        doc["mission"]["initial_battery_frac"] = battery
                    save(folder / n, doc)
                    frozen[str((folder / n).relative_to(output))] = sha(folder / n)
    base = cohort.resolve().parent
    p = dict(
        schema_version="mission_coverage_study_v1",
        cohort=str(cohort.resolve()),
        training=str(base / "mission_policy_joint_training_v1_20261018"),
        risk_training=str(base / "mission_routing_v4_risk_training_20261017"),
        temporal_training=str(base / "mission_temporal_training_v2_20261025"),
        coverage_training=old["coverage_training"],
        assets=old["assets"],
        cases=cases,
        missions=old["missions"],
        batteries=list(BATTERIES),
        policies=list(POLICIES),
        frozen_files=frozen,
        cohort_protocol_sha256=sha(cohort / "protocol.json"),
        active_cpu_power_sensitivity_w=[15],
        scope="prospective_known_map_layout_transfer",
        primary_policy="coverage_tail",
        primary_endpoint="Full mission contract success in all 24 paired cells",
        primary_comparisons=[
            "coverage_no_battery",
            "coverage_global",
            "coverage_short",
            "phase_survival",
            "temporal_mission",
            "mission_survival",
        ],
        secondary_endpoints=[
            "phases_succeeded",
            "report_recall",
            "false_components",
            "reserve_margin_j",
        ],
        selection_after_evaluation=False,
        hyperparameter_search=False,
        fresh_test=True,
        new_physical_flights=6,
        new_neural_calls=6120,
        independent_evaluation_layouts=3,
        statistical_unit=(
            "Spatial layout; lighting, intents and batteries are paired repeated measures"
        ),
        claims=(
            "Descriptive three-layout transfer on one known map; "
            "no population significance or unseen-map claim"
        ),
    )
    profiles = load_profiles(p)
    for c in cases.values():
        for profile in profiles.values():
            split_check(profile["training_lineage"], c["lineage"])
    p["profile_sha256"] = {k: sha(v) for k, v in profile_paths(p).items()}
    save(output / "protocol.json", p)
    return verify(output)


def profile_paths(p):
    return dict(
        recall=Path(p["training"]) / "all_development/recall.json",
        risk=Path(p["risk_training"]) / "risk_profile.json",
        temporal=Path(p["temporal_training"]) / "profile.json",
        coverage=Path(p["coverage_training"]) / "profile.json",
    )


def verify(root):
    from experiments.mission_coverage_routing.study import POLICIES

    p = read(root / "protocol.json")
    if (
        p["schema_version"] != "mission_coverage_study_v1"
        or p["policies"] != list(POLICIES)
        or p["batteries"] != list(BATTERIES)
    ):
        raise ValueError("Changed evaluation grid")
    cohort = Path(p["cohort"])
    old = verify_cohort(cohort)
    if sha(cohort / "protocol.json") != p["cohort_protocol_sha256"]:
        raise ValueError("Changed cohort")
    for name, path in profile_paths(p).items():
        if sha(path) != p["profile_sha256"][name]:
            raise ValueError("Changed profile")
    for path, expected in p["frozen_files"].items():
        if sha(root / path) != expected:
            raise ValueError("Changed mission")
    for c in p["cases"].values():
        if c["lineage"] != old["cases"][c["source_flight"]]["lineage"]:
            raise ValueError("Changed pairing")
    return p
