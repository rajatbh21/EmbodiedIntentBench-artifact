"""Frozen report-routing experiment: development fit, validation, held-out scoring.

All geometry, thresholds, profiles and policies are fixed before scoring. A held-out
run consumes that split: later tuning requires new evaluation data.
"""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.mission_requirements import MissionRequirements
from experiments.v4_benchmark.probe_cache import build_cached
from experiments.v4_benchmark.probe_cache import freeze as freeze_cache
from experiments.v4_benchmark.reporting import ReportingConfig
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import exported_source, load_scenario
from experiments.v4_benchmark.splits import checksum, freeze, verify
from experiments.v4_benchmark.validation_study import verify_router_lock

BASE = Path(__file__).parent


def read(path):
    return json.loads(path.read_text())


def cached(scenario, scene, partition, original, expanded, screen):
    old = [s for s in scenario.executors if s.kind != "torch_cabinet_segmentation"]
    new = [s for s in scenario.executors if s.kind == "torch_cabinet_segmentation"]
    executors = build_cached(old, original / scene / "cache", scenario.recording_identity)
    executors.update(
        build_cached(
            new,
            (screen if partition == "development" else expanded) / scene / "cache",
            scenario.recording_identity,
        )
    )
    return executors


def execute(scenario, recording, executors, router, artifact, destination):
    backend = ReplayBackend(scenario, recording)
    try:
        return run_mission(
            scenario,
            backend,
            build_router(
                router,
                scenario.profiles,
                scenario.observation_interval_s,
                dict(artifact_path=str(artifact))
                if router in ("calibrated", "report_calibrated")
                else {},
            ),
            executors,
            router_name=router,
            output=destination,
        )
    finally:
        backend.close()


def implementation_hashes():
    # Include the pipeline and scoring/policy dependencies, not just the fitted weights.
    paths = sorted(BASE.glob("*.py"))
    paths += sorted((BASE.parents[1] / "missionbench/v2").glob("*.py"))
    paths += sorted((BASE.parent / "v4_gazebo_pilot").glob("*.py"))
    return {str(p.resolve()): checksum(p) for p in paths}


def fit(original, expanded, screen, output):
    verify(expanded / "missions_split.json")
    verify_router_lock(expanded)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profiles = read(BASE / "report_mission_profiles.json")
    # Predeclared metric-space gate, unchanged after observing validation or evaluation.
    reporting = ReportingConfig(association="ground_plane", max_distance_m=1.0, max_gap_s=2.0)
    save_json(
        output / "protocol.json",
        dict(
            reporting=reporting.to_dict(),
            profiles=profiles,
            costs="frozen configured costs; preloaded default residency",
            original=str(original.resolve()),
            expanded=str(expanded.resolve()),
            screen=str(screen.resolve()),
            development_only=True,
            scope="synthetic shared-appearance diagnostic, not real-world validation",
        ),
    )
    rows = []
    for entry in read(expanded / "missions_split.json")["entries"]:
        profile = entry["case_id"].split("__")[1]
        scenario = replace(
            load_scenario(Path(entry["scenario_path"])),
            reporting=reporting,
            requirements=MissionRequirements.parse(profiles[profile]),
        )
        folder = output / "missions" / entry["case_id"]
        folder.mkdir(parents=True)
        save_json(folder / "scenario.json", exported_source(scenario))
        save_json(folder / "contract.json", {"schema_version": "1.0", **asdict(scenario.contract)})
        rows.append(
            dict(
                case_id=entry["case_id"],
                partition=entry["partition"],
                family_id=entry["family_id"],
                scenario=str(folder / "scenario.json"),
                recording=entry["recording_path"],
            )
        )
    save_json(output / "split_source.json", dict(split_schema_version="0.2", entries=rows))
    freeze(output / "split_source.json", output / "missions_split.json")
    artifact = read(expanded / "development_profile.json")
    outcomes = {}
    for entry in read(output / "missions_split.json")["entries"]:
        if entry["partition"] != "development" or not entry["case_id"].endswith("__rapid_search"):
            continue
        scene = entry["case_id"].split("__")[0]
        scenario = load_scenario(Path(entry["scenario_path"]))
        context = "below_6m" if "low_oblique" in scene else "at_least_6m"
        outcomes[context] = {}
        for spec in scenario.executors:
            result = execute(
                scenario,
                Path(entry["recording_path"]),
                cached(scenario, scene, "development", original, expanded, screen),
                "static:" + spec.config_id,
                None,
                output / "development" / scene / spec.config_id,
            )
            if result["status"] != "completed":
                raise RuntimeError(result["error"])
            outcomes[context][spec.config_id] = {
                k: v for k, v in result["report_quality"].items() if k != "reports"
            }
    artifact["report_outcomes"] = outcomes
    artifact["report_protocol"] = dict(
        reporting=reporting.to_dict(),
        duration_s=scenario.duration_s,
        interval_s=scenario.observation_interval_s,
        resources=scenario.resources.to_dict(),
        execution_costs={
            spec.config_id: [spec.mission_latency_s, spec.energy_j_per_call]
            for spec in scenario.executors
        },
    )
    save_json(output / "development_profile.json", artifact)
    save_json(
        output / "lock.json",
        dict(
            artifact=checksum(output / "development_profile.json"),
            protocol=checksum(output / "protocol.json"),
            split=checksum(output / "missions_split.json"),
            code=implementation_hashes(),
        ),
    )


def check_lock(study):
    lock = read(study / "lock.json")
    for name, file in [
        ("artifact", "development_profile.json"),
        ("protocol", "protocol.json"),
        ("split", "missions_split.json"),
    ]:
        if checksum(study / file) != lock[name]:
            raise ValueError("frozen report study changed: " + name)
    if implementation_hashes() != lock["code"]:
        raise ValueError("frozen report study implementation changed")
    verify(study / "missions_split.json")


def score(study, partition):
    if partition not in ("validation", "evaluation"):
        raise ValueError("only validation or evaluation may be scored")
    if partition == "evaluation" and not (study / "validation" / "report.json").is_file():
        raise ValueError("complete frozen validation before held-out evaluation")
    check_lock(study)
    protocol = read(study / "protocol.json")
    destination = study / partition
    destination.mkdir(exist_ok=False)
    save_json(
        destination / "started.json",
        dict(
            partition=partition,
            lock_sha256=checksum(study / "lock.json"),
            note="Evaluation is consumed when started; never tune on these results.",
        ),
    )
    original, expanded, screen = (Path(protocol[k]) for k in ("original", "expanded", "screen"))
    rows = []
    for entry in read(study / "missions_split.json")["entries"]:
        if entry["partition"] != partition:
            continue
        scene = entry["case_id"].split("__")[0]
        scenario = load_scenario(Path(entry["scenario_path"]))
        if partition == "evaluation":
            cache = destination / scene / "cache"
            if not cache.exists():
                from experiments.v4_benchmark.model_probe import probe

                folder = destination / scene
                # Dense execution is shared across mission intents, never used to fit the router.
                probe(
                    Path(entry["scenario_path"]),
                    Path(entry["recording_path"]),
                    folder / "probe",
                    max_frames=24,
                )
                freeze_cache(
                    Path(entry["scenario_path"]),
                    Path(entry["recording_path"]),
                    folder / "probe",
                    cache,
                )
            executors = build_cached(scenario.executors, cache, scenario.recording_identity)
        else:
            executors = cached(scenario, scene, partition, original, expanded, screen)
        routers = ["static:" + s.config_id for s in scenario.executors] + [
            "calibrated",
            "report_calibrated",
        ]
        for router in routers:
            result = execute(
                scenario,
                Path(entry["recording_path"]),
                executors,
                router,
                study / "development_profile.json",
                destination / entry["case_id"] / router.replace(":", "_"),
            )
            rows.append(
                dict(
                    case_id=entry["case_id"],
                    router=router,
                    status=result["status"],
                    error=result["error"],
                    mission_success=result["mission_success"],
                    constraints=result["constraints"],
                    energy_j=result["cumulative_energy_j"],
                    report_quality=result["report_quality"],
                    choices=result["config_selection_history"],
                )
            )
            save_json(destination / "rows.json", rows)
    check_lock(study)
    if any(r["status"] != "completed" for r in rows):
        raise RuntimeError("component errors invalidate the study")
    save_json(
        destination / "report.json",
        dict(
            partition=partition,
            rows=rows,
            lock_sha256=checksum(study / "lock.json"),
            scope="two correlated views of one synthetic family; no population success estimate",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "validation", "evaluation"])
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument(
        "--original", type=Path, default=Path("results/v4_validation_study_20261014")
    )
    parser.add_argument(
        "--expanded", type=Path, default=Path("results/v4_cabinet_validation_20261014")
    )
    parser.add_argument("--screen", type=Path, default=Path("results/v4_cabinet_screen_20261014"))
    args = parser.parse_args()
    if args.stage == "fit":
        fit(args.original, args.expanded, args.screen, args.study)
    else:
        score(args.study, args.stage)


if __name__ == "__main__":
    main()
