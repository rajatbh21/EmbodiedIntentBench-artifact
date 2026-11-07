"""Fit only development recordings and compare preregistered routers on validation.

Evaluation inputs are integrity-checked and frozen, never probed or scored here.
Selection is a small synthetic diagnostic, not evidence of population superiority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v4_benchmark.backends import ReplayBackend, save_json
from experiments.v4_benchmark.calibrated_router import fit, merge_profiles
from experiments.v4_benchmark.generalization_study import mission_diagnostics
from experiments.v4_benchmark.model_pool import candidates, prepare
from experiments.v4_benchmark.model_probe import probe
from experiments.v4_benchmark.probe_cache import build_cached
from experiments.v4_benchmark.probe_cache import freeze as freeze_cache
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import load_scenario
from experiments.v4_benchmark.splits import checksum, freeze, verify
from experiments.v4_benchmark.suite import merge


def partition_entries(manifest: dict, partition: str) -> list[dict]:
    """This workflow deliberately has no evaluation execution path."""
    if partition not in {"development", "validation"}:
        raise ValueError("only development/validation execution is allowed")
    return [row for row in manifest["entries"] if row["partition"] == partition]


def policies(artifact: Path) -> list[dict]:
    ids = [c["config_id"] for c in candidates()]
    result = [dict(label=f"static:{key}", name=f"static:{key}", parameters={}) for key in ids]
    result.extend(
        [
            dict(label="intent_visual", name="intent_visual", parameters={}),
            dict(
                label="calibrated_context",
                name="calibrated",
                parameters=dict(artifact_path=str(artifact), use_context=True),
            ),
            dict(
                label="calibrated_global",
                name="calibrated",
                parameters=dict(artifact_path=str(artifact), use_context=False),
            ),
            dict(label="periodic", name="periodic", parameters=dict(config_ids=ids, period_s=2)),
        ]
    )
    result.extend(
        dict(
            label=f"random_{seed}", name="seeded_random", parameters=dict(config_ids=ids, seed=seed)
        )
        for seed in range(3)
    )
    return result


def select_per_profile(rows: list[dict]) -> list[dict]:
    """Lexicographic preregistered selection; failures never earn energy savings."""
    selections = []
    for profile in sorted({r["profile"] for r in rows}):
        group = [r for r in rows if r["profile"] == profile]
        ranking = []
        for label in sorted({r["router"] for r in group}):
            runs = [r for r in group if r["router"] == label]
            success = [r for r in runs if r["mission_success"]]
            ranking.append(
                dict(
                    router=label,
                    successes=len(success),
                    missions=len(runs),
                    successful_energy_j=sum(r["energy_j"] for r in success),
                )
            )
        ranking.sort(key=lambda r: (-r["successes"], r["successful_energy_j"], r["router"]))
        best = ranking[0]
        selections.append(
            dict(
                profile=profile,
                selected_router=best["router"] if best["successes"] else None,
                all_validation_missions_passed=best["successes"] == best["missions"],
                ranking=ranking,
            )
        )
    return selections


def verify_router_lock(output: Path) -> None:
    """Reject changes to the development-fitted router or preregistered protocol."""
    lock = json.loads((output / "frozen_router.json").read_text())
    for name, key in (
        ("development_profile.json", "artifact_sha256"),
        ("protocol.json", "protocol_sha256"),
    ):
        if checksum(output / name) != lock[key]:
            raise ValueError(f"frozen validation input changed: {name}")


def run_study(bundle: Path, output: Path, cost_profile: Path, profiles_path: Path) -> dict:
    verify(bundle)
    source = json.loads(bundle.read_text())
    if source["split_schema_version"] != "0.2":
        raise ValueError("three-way frozen split required")
    profiles = json.loads(profiles_path.read_text())["profiles"]
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact = output / "development_profile.json"
    routers = policies(artifact)
    save_json(
        output / "protocol.json",
        dict(
            validation_study_version="0.1",
            source_split_sha256=checksum(bundle),
            cost_profile_sha256=checksum(cost_profile),
            mission_profiles=profiles,
            routers=routers,
            selection_rule=(
                "success count descending, successful-run energy ascending, "
                "label ascending; zero successes => no selection"
            ),
            fitting_partition="development",
            scoring_partition="validation",
            evaluation_scored=False,
            independence_verified=False,
            costs="Previously measured diagnostic latency; assumed energy; no loading/switch costs",
        ),
    )
    entries, mission_entries = [], []
    for row in source["entries"]:
        case = row["case_id"]
        scenario_path = prepare(
            Path(row["scenario_path"]), output / case / "pool", cost_profile=cost_profile
        )
        entry = dict(
            case_id=case,
            partition=row["partition"],
            family_id=row["family_id"],
            scenario=str(scenario_path),
            recording=row["recording_path"],
        )
        entries.append(entry)
        base = json.loads(scenario_path.read_text())
        contract = json.loads((scenario_path.parent / "contract.json").read_text())
        for profile in profiles:
            directory = output / case / profile["case_id"]
            directory.mkdir()
            save_json(directory / "contract.json", merge(contract, profile["contract_overrides"]))
            save_json(directory / "scenario.json", merge(base, profile["scenario_overrides"]))
            mission_entries.append(
                dict(
                    entry,
                    case_id=f"{case}__{profile['case_id']}",
                    scenario=str(directory / "scenario.json"),
                )
            )
    for name, rows in (("pool", entries), ("missions", mission_entries)):
        path = output / f"{name}_source.json"
        save_json(path, dict(split_schema_version="0.2", entries=rows))
        freeze(path, output / f"{name}_split.json")
    # Only after every input, cost, mission and candidate is fixed do models run.
    manifest = json.loads((output / "pool_split.json").read_text())
    training = []
    for partition in ("development", "validation"):
        if partition == "validation":
            merge_profiles(training, artifact)
            save_json(
                output / "frozen_router.json",
                dict(
                    artifact_sha256=checksum(artifact),
                    protocol_sha256=checksum(output / "protocol.json"),
                ),
            )
        for row in partition_entries(manifest, partition):
            case = row["case_id"]
            folder = output / case
            scenario_path, recording = Path(row["scenario_path"]), Path(row["recording_path"])
            scenario = load_scenario(scenario_path)
            slots = len(
                range(
                    scenario.schedule.epoch_ns,
                    scenario.schedule.end_ns,
                    scenario.schedule.interval_ns,
                )
            )
            probe(scenario_path, recording, folder / "probe", max_frames=slots, threads=2)
            freeze_cache(scenario_path, recording, folder / "probe", folder / "cache")
            if partition == "development":
                profile_path = folder / "profile.json"
                fit(scenario_path, recording, folder / "probe", profile_path, use_altitude=True)
                training.append(profile_path)
            print(f"Measured {partition}: {case}", flush=True)
    verify(output / "missions_split.json")
    verify_router_lock(output)
    rows = []
    for entry in partition_entries(manifest, "validation"):
        folder = output / entry["case_id"]
        for profile in profiles:
            scenario = load_scenario(folder / profile["case_id"] / "scenario.json")
            for index, router in enumerate(routers):
                destination = folder / profile["case_id"] / f"router_{index:02d}"
                backend = ReplayBackend(scenario, Path(entry["recording_path"]))
                try:
                    result = run_mission(
                        scenario,
                        backend,
                        build_router(
                            router["name"],
                            scenario.profiles,
                            scenario.observation_interval_s,
                            router["parameters"],
                        ),
                        build_cached(
                            scenario.executors, folder / "cache", scenario.recording_identity
                        ),
                        router_name=router["label"],
                        output=destination,
                    )
                finally:
                    backend.close()
                events = json.loads((destination / "events.json").read_text())
                rows.append(
                    dict(
                        case_id=entry["case_id"],
                        profile=profile["case_id"],
                        router=router["label"],
                        mission_success=result["mission_success"],
                        status=result["status"],
                        error=result["error"],
                        recall=result["quality"]["target_recall"],
                        precision=result["quality"]["detection_precision"],
                        energy_j=result["cumulative_energy_j"],
                        failed_constraints=[k for k, v in result["constraints"].items() if not v],
                        choices=result["config_selection_history"],
                        **mission_diagnostics(result, events),
                    )
                )
                save_json(output / "rows.json", rows)
    verify_router_lock(output)
    if any(r["status"] == "error" for r in rows):
        raise RuntimeError("component errors invalidate validation selection; inspect rows.json")
    report = dict(
        validation_study_version="0.1",
        evaluation_scored=False,
        scope="one validation family, two views; synthetic geometry diagnostic",
        training_recordings=len(training),
        mission_count=len(rows),
        selection=select_per_profile(rows),
        rows=rows,
    )
    save_json(output / "report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("bundle", "output", "cost-profile", "profiles"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    result = run_study(args.bundle, args.output, args.cost_profile, args.profiles)
    print(json.dumps(result["selection"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
