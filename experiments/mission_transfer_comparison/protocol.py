"""Freeze policies and paired interventions before acquiring a new map's RGB."""

import argparse
import copy
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.protocol import profile_paths
from experiments.mission_environment_transfer.cohort import verify as verify_cohort
from experiments.mission_routing_v2.flights import split_check

BATTERIES = (0.34, 0.65)


def prepare(cohort, previous, output):
    from experiments.mission_coverage_repair.protocol import verify as verify_previous
    from experiments.mission_transfer_comparison.study import POLICIES, load_profiles

    old = verify_cohort(cohort)
    prior = verify_previous(previous)
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
    repo = Path(__file__).resolve().parents[2]
    inherited = read(Path(prior["cohort"]) / "protocol.json")["implementation_sha256"]
    sources = {**inherited, **prior["repair_source_sha256"]}
    for package in (
        "mission_transfer_comparison",
        "literature_routing",
        "mission_environment_transfer",
    ):
        for path in (repo / "experiments" / package).glob("*.py"):
            # Post-hoc reporting does not select actions or change scored outcomes.
            if path.name not in ("report.py", "pipeline.py"):
                sources[str(path)] = sha(path)
    for path in [
        repo / "experiments/mission_transfer_identity.py",
        *repo.glob("tests/test_*transfer*.py"),
        repo / "tests/test_literature_routing.py",
        repo / "tests/test_mission_protocol_invariants_v2.py",
        repo / "docs/literature_routing.md",
        repo / "docs/mission_transfer_comparison.md",
    ]:
        if path.exists():
            sources[str(path)] = sha(path)
    p = {
        k: prior[k] for k in ("training", "risk_training", "temporal_training", "coverage_training")
    }
    p.update(
        schema_version="mission_map_transfer_comparison_v1",
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
        scope="prospective_independently_sourced_map_transfer",
        primary_policy="coverage_tail",
        primary_comparisons=["approxdet_eq5_adapter", "coverage_global", "coverage_no_battery"],
        primary_endpoint="Full mission contract success across the entire paired factorial",
        fresh_test=True,
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
            "Descriptive transfer to one independently sourced map, not population generalization"
        ),
        published_baseline_scope=(
            "ApproxDet paper Eq.5 scheduler adaptation; not original-system reproduction"
        ),
    )
    profiles = load_profiles(p)
    for c in cases.values():
        for profile in profiles.values():
            split_check(profile["training_lineage"], c["lineage"])
    p["profile_sha256"] = {k: sha(v) for k, v in profile_paths(p).items()}
    save(output / "protocol.json", p)
    return verify(output)


def verify(root):
    from experiments.mission_transfer_comparison.study import POLICIES

    p = read(root / "protocol.json")
    if (
        p["schema_version"] != "mission_map_transfer_comparison_v1"
        or p["policies"] != list(POLICIES)
        or p["batteries"] != list(BATTERIES)
    ):
        raise ValueError("Changed comparison grid")
    cohort = Path(p["cohort"])
    old = verify_cohort(cohort)
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
    a = parser.parse_args()
    prepare(a.cohort, a.previous, a.output)
