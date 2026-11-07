"""Freeze policies and paired interventions before acquiring a new map's RGB."""

import argparse
import copy
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.protocol import profile_paths
from experiments.mission_environment_transfer_v3.cohort import verify as verify_cohort
from experiments.mission_routing_v2.flights import split_check
from experiments.mission_transfer_comparison.protocol import verify as verify_rejected_study
from experiments.mission_transfer_comparison_v3.prehistory import (
    method_parity,
    rejected_acquisition,
    unchanged_design,
    unchanged_interventions,
)

BATTERIES = (0.34, 0.65)
PROVENANCE = dict(
    fresh_test=False,
    prospective_policy_evaluation=True,
    new_capture_run=True,
    strict_unmodified_prospective_test=False,
    fully_sensor_unseen_environment=False,
    map_unseen_by_development_training=True,
    geometry_revised_after_rejected_acquisition=True,
    raw_rgb_and_geometry_gt_seen_before_revision=True,
    model_outputs_seen_before_freeze=False,
    policy_outcomes_seen_before_freeze=False,
    outcome_driven_method_change=False,
)


def source_paths(cohort_protocol, rejected_protocol):
    repo = Path(__file__).resolve().parents[2]
    inherited = {
        **rejected_protocol["implementation_sha256"],
        **cohort_protocol["implementation_sha256"],
    }
    own = [
        p for p in Path(__file__).parent.glob("*.py") if p.name not in ("report.py", "pipeline.py")
    ]
    own += [
        repo / "tests/test_mission_transfer_comparison_v3.py",
        repo / "experiments/mission_transfer_acquire.py",
        repo / "tests/test_transfer_acquisition_admission.py",
        repo / "docs/mission_transfer_comparison_v3.md",
    ]
    if any(not p.is_file() for p in own):
        raise ValueError("Complete v3 source, tests and method documentation before freezing")
    return inherited, own


def prepare(cohort, previous, output, *, failed_cohort):
    from experiments.mission_coverage_repair.protocol import verify as verify_previous
    from experiments.mission_transfer_comparison_v3.study import POLICIES, load_profiles

    old = verify_cohort(cohort)
    prior = verify_previous(previous)
    history = rejected_acquisition(failed_cohort)
    rejected = verify_rejected_study(failed_cohort / "study")
    parity = method_parity()
    interventions = unchanged_interventions(cohort, failed_cohort)
    if Path(old["engineering_revision"]["rejected_cohort"]).resolve() != failed_cohort.resolve():
        raise ValueError("Comparison and environment must reference the same rejected acquisition")
    if Path(rejected["previous"]).resolve() != previous.resolve():
        raise ValueError("v3 must retain the original lake-study reference")
    if any((cohort / f / "recording").exists() for f in old["cases"]):
        raise ValueError("Freeze comparison before the first evaluation recording")
    if any((cohort / f / "predictions").exists() for f in old["cases"]):
        raise ValueError("Evaluation predictions must not exist before method freeze")
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
    sources, own = source_paths(old, rejected)
    for path in own:
        sources[str(path.resolve())] = sha(path)
    p = {
        k: prior[k] for k in ("training", "risk_training", "temporal_training", "coverage_training")
    }
    p.update(
        schema_version="mission_map_transfer_comparison_v3",
        cohort=str(cohort.resolve()),
        previous=str(previous.resolve()),
        previous_protocol_sha256=sha(previous / "protocol.json"),
        assets=old["assets"],
        cases=cases,
        missions=old["missions"],
        batteries=list(BATTERIES),
        policies=list(POLICIES),
        frozen_files=frozen,
        cohort_protocol_sha256=sha(cohort / "protocol.json"),
        implementation_sha256=sources,
        active_cpu_power_sensitivity_w=[15],
        scope="prospective_policy_transfer_after_disclosed_sensor_geometry_revision",
        primary_policy="coverage_tail",
        primary_comparisons=["approxdet_eq5_adapter", "coverage_global", "coverage_no_battery"],
        primary_endpoint="Full mission contract success across the entire paired factorial",
        methods_frozen_before_capture=True,
        hyperparameter_search=False,
        selection_after_evaluation=False,
        inherited_numerical_repair_fixed_before_acquisition=True,
        independent_evaluation_layouts=len({c["layout_group"] for c in cases.values()}),
        physical_recordings=len(old["cases"]),
        statistical_unit=(
            "Map is the domain-shift unit; layouts and paired conditions are repeated measures"
        ),
        claims=(
            "Descriptive policy transfer to one map unseen by development training; "
            "geometry was engineered after rejected raw-sensor inspection, "
            "not a sensor-unseen or unmodified prospective environment test"
        ),
        rejected_acquisition=history,
        inherited_method_parity=parity,
        inherited_interventions=interventions,
        **PROVENANCE,
        published_baseline_scope=(
            "ApproxDet paper Eq.5 scheduler adaptation; not original-system reproduction"
        ),
    )
    profiles = load_profiles(p)
    for c in cases.values():
        for profile in profiles.values():
            split_check(profile["training_lineage"], c["lineage"])
    p["profile_sha256"] = {k: sha(v) for k, v in profile_paths(p).items()}
    unchanged_design(p, rejected)
    save(output / "protocol.json", p)
    return verify(output)


def verify(root):
    from experiments.mission_transfer_comparison_v3.study import POLICIES

    p = read(root / "protocol.json")
    if (
        p["schema_version"] != "mission_map_transfer_comparison_v3"
        or p["policies"] != list(POLICIES)
        or p["batteries"] != list(BATTERIES)
    ):
        raise ValueError("Changed comparison grid")
    cohort = Path(p["cohort"])
    old = verify_cohort(cohort)
    failed_cohort = Path(p["rejected_acquisition"]["cohort"])
    history = rejected_acquisition(failed_cohort)
    if p["rejected_acquisition"] != history or p["inherited_method_parity"] != method_parity():
        raise ValueError("Changed engineering prehistory or inherited method binding")
    if (
        p["inherited_interventions"] != unchanged_interventions(cohort, failed_cohort)
        or Path(old["engineering_revision"]["rejected_cohort"]).resolve() != failed_cohort.resolve()
    ):
        raise ValueError("Changed inherited non-geometric interventions or environment prehistory")
    rejected = verify_rejected_study(failed_cohort / "study")
    unchanged_design(p, rejected)
    if any(p[k] != v for k, v in PROVENANCE.items()):
        raise ValueError("Changed prospective-policy/observed-sensor provenance")
    inherited_sources, own = source_paths(old, rejected)
    if set(p["implementation_sha256"]) != {
        *inherited_sources,
        *(str(path.resolve()) for path in own),
    }:
        raise ValueError("Incomplete or changed v3 source pins")
    if any(
        p["implementation_sha256"][name] != expected for name, expected in inherited_sources.items()
    ):
        raise ValueError("Changed inherited source pin")
    if sha(cohort / "protocol.json") != p["cohort_protocol_sha256"]:
        raise ValueError("Changed cohort")
    if sha(Path(p["previous"]) / "protocol.json") != p["previous_protocol_sha256"]:
        raise ValueError("Changed previous study reference")
    prior = read(Path(p["previous"]) / "protocol.json")
    expected_cases = {
        f"{flight}_b{round(battery * 1000)}": dict(c, source_flight=flight, initial_battery=battery)
        for flight, c in old["cases"].items()
        for battery in BATTERIES
    }
    if (
        p["cases"] != expected_cases
        or p["missions"] != old["missions"]
        or p["assets"] != old["assets"]
        or p["active_cpu_power_sensitivity_w"] != [15]
        or p["primary_policy"] != "coverage_tail"
        or p["primary_comparisons"]
        != ["approxdet_eq5_adapter", "coverage_global", "coverage_no_battery"]
        or p["physical_recordings"] != len(old["cases"])
        or p["independent_evaluation_layouts"]
        != len({c["layout_group"] for c in old["cases"].values()})
    ):
        raise ValueError("Changed pairing, population or declared evaluation condition")
    for key in ("training", "risk_training", "temporal_training", "coverage_training"):
        if p[key] != prior[key]:
            raise ValueError("Changed inherited development source")
    expected_files = {
        f"{flight}/{mission}/{name}"
        for flight in expected_cases
        for mission in old["missions"]
        for name in ("scenario.json", "contract.json", "intent.json")
    }
    if set(p["frozen_files"]) != expected_files:
        raise ValueError("Incomplete intervention file pins")
    for group in ("implementation_sha256",):
        for name, expected in p[group].items():
            if sha(Path(name)) != expected:
                raise ValueError("Changed frozen source: " + name)
    for name, path in profile_paths(p).items():
        if sha(path) != p["profile_sha256"][name]:
            raise ValueError("Changed development profile")
    for name, expected in p["frozen_files"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed mission intervention")
    for c in p["cases"].values():
        if c["lineage"] != old["cases"][c["source_flight"]]["lineage"]:
            raise ValueError("Changed physical pairing")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failed-cohort", type=Path, required=True)
    a = parser.parse_args()
    prepare(a.cohort, a.previous, a.output, failed_cohort=a.failed_cohort)
