"""Record, audit, infer and compare multi-stage missions on identical observations."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from experiments.lake_missions.assets import save, sha
from experiments.lake_missions.objectives import enough_evidence, public_phases, score, validate
from experiments.lake_missions.router import PhaseRouter
from experiments.lake_missions.scenario import load
from experiments.routing_mission.experiment import router_for
from experiments.routing_mission.inference import compute
from experiments.v4_benchmark.backends import ReplayBackend, record_world
from experiments.v4_benchmark.probe_cache import build_cached
from experiments.v4_benchmark.runner import run_mission

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def inputs(root):
    suite = read(root / "suite.json")
    missions = read(root / "missions.json")["missions"]
    data = read(root / "scenario.json")
    for mission in missions:
        validate(
            mission,
            data["mission"]["duration_s"],
            [o["object_id"] for o in data["world"]["objects"]],
        )
    return suite, missions


def freeze(root):
    files = [
        root / p
        for p in ["suite.json", "missions.json", "scenario.json", "contract.json", "evidence.json"]
    ]
    for pattern in [
        "missionbench/**/*.py",
        "experiments/lake_missions/*.py",
        "experiments/routing_mission/*.py",
        "experiments/v4_benchmark/*.py",
        "experiments/v4_gazebo_pilot/*.py",
    ]:
        files.extend(ROOT.glob(pattern))
    lock = {str(p.resolve().relative_to(ROOT)): sha(p) for p in sorted(files)}
    path = root / "lock.json"
    if path.exists() and read(path) != lock:
        raise ValueError("Frozen lake experiment changed; create a new run directory")
    if not path.exists():
        save(path, lock)


def record(root):
    from experiments.lake_missions.backend import LiveBackend

    suite, _ = inputs(root)
    for name, env in suite["environments"].items():
        scenario = load(root / "scenario.json", env, Path(suite["assets"]))
        folder = root / name / "recording"
        if (folder / "recording.json").exists():
            ReplayBackend(scenario, folder).close()
            continue
        backend = LiveBackend(scenario, folder)
        try:
            record_world(scenario, backend, folder)
        finally:
            backend.close()


def audit(root):
    from experiments.lake_missions.placement import collision_bounds, obstacles_at, terrain_height

    suite, missions = inputs(root)
    bounds = collision_bounds(Path(suite["assets"]))
    for obj in read(root / "scenario.json")["world"]["objects"]:
        x, y, z = obj["position_m"]
        floor = terrain_height(Path(suite["assets"]), x, y)
        if floor < 1.2 or abs(z - floor - 0.04) > 0.01 or obstacles_at(x, y, floor, bounds):
            raise ValueError(
                "Target intersects water/scenery or floats above terrain: " + obj["object_id"]
            )
    environments = {}
    for name, env in suite["environments"].items():
        scenario = load(root / "scenario.json", env, Path(suite["assets"]))
        backend = ReplayBackend(scenario, root / name / "recording")
        visibility = {target: [] for target in scenario.targets.values()}
        pixels = {target: 0 for target in scenario.targets.values()}
        positions = []
        for timestamp in backend.frames:
            backend.advance_to(timestamp)
            frame = backend.capture()
            positions.append(frame.body_pose["position_m"])
            for label, target in scenario.targets.items():
                count = int(np.count_nonzero(frame.object_ids == label))
                pixels[target] = max(pixels[target], count)
                if count >= 4:
                    visibility[target].append((timestamp - scenario.schedule.epoch_ns) / 1e9)
        backend.close()
        rows = []
        for mission in missions:
            for p in mission["phases"]:
                possible = [
                    target
                    for target in p["evaluator_target_ids"]
                    if enough_evidence(
                        [t for t in visibility[target] if p["start_s"] <= t < p["end_s"]],
                        p,
                    )
                ]
                bound = len(possible) / len(p["evaluator_target_ids"])
                rows.append(
                    dict(
                        mission_id=mission["mission_id"],
                        phase_id=p["phase_id"],
                        visible_repeated_evidence_upper_bound=bound,
                        observable=bound >= p["required_recall"],
                        visible_targets=possible,
                    )
                )
        environments[name] = dict(
            captures=len(positions),
            position_min=np.min(positions, axis=0).tolist(),
            position_max=np.max(positions, axis=0).tolist(),
            target_visible_capture_counts={k: len(v) for k, v in visibility.items()},
            target_max_pixels=pixels,
            stages=rows,
            all_stages_observable=all(r["observable"] for r in rows),
            note="Visibility-only upper bound; does not guarantee model feasibility or success.",
        )
    report = dict(
        environments=environments,
        all_stages_observable=all(e["all_stages_observable"] for e in environments.values()),
    )
    save(root / "physical_audit.json", report)
    return report


def infer(root):
    suite, _ = inputs(root)
    audit_report = audit(root)
    if not audit_report["all_stages_observable"]:
        raise ValueError(
            "Stage has insufficient physical evidence opportunities; fix scenario before inference"
        )
    freeze(root)
    for name, env in suite["environments"].items():
        scenario = load(root / "scenario.json", env, Path(suite["assets"]))
        output = root / name / "predictions"
        if (output / "manifest.json").exists():
            build_cached(scenario.executors, output, scenario.recording_identity)
        else:
            compute(scenario, root / name / "recording", output, threads=2)


def evaluate(root, environment=None, on_demand=False):
    suite, missions = inputs(root)
    freeze(root)
    if not audit(root)["all_stages_observable"]:
        raise ValueError("Unobservable mission stage")
    rows = []
    selected = (
        suite["environments"]
        if environment is None
        else {environment: suite["environments"][environment]}
    )
    for name, env in selected.items():
        rows = []
        base = load(root / "scenario.json", env, Path(suite["assets"]))
        if on_demand:
            from experiments.lake_missions.predictions import build

            cached = build(base, root / name / "on_demand_predictions", root / name / "recording")
        else:
            cached = build_cached(
                base.executors, root / name / "predictions", base.recording_identity
            )
        canonical = {
            r["timestamp_ns"]: r["body_pose"]
            for r in read(root / name / "recording/recording.json")["frames"]
        }
        for mission in missions:
            folder = root / name / "comparison" / mission["mission_id"]
            folder.mkdir(parents=True, exist_ok=True)
            data, contract = read(root / "scenario.json"), read(root / "contract.json")
            data["mission"]["initial_battery_frac"] = mission["initial_battery_frac"]
            contract["min_final_battery_frac"] = mission["min_final_battery_frac"]
            save(folder / "scenario.json", data)
            save(folder / "contract.json", contract)
            scenario = load(folder / "scenario.json", env, Path(suite["assets"]))
            assert scenario.recording_identity == base.recording_identity
            for i, policy in enumerate(suite["policies"]):
                output = folder / f"policy_{i:02d}"
                backend = ReplayBackend(scenario, root / name / "recording")
                router = router_for(
                    "context" if policy == "phase_context" else policy, scenario, suite["seed"]
                )
                if policy == "phase_context":
                    router = PhaseRouter(router, public_phases(mission))
                try:
                    result = run_mission(
                        scenario,
                        backend,
                        router,
                        {k: copy.copy(v) for k, v in cached.items()},
                        router_name=policy,
                        output=output,
                    )
                finally:
                    backend.close()
                events = read(output / "events.json")
                for event in events:
                    if event["capture_body_pose"] != canonical[event["capture_timestamp_ns"]]:
                        raise ValueError("Policy changed the physical trajectory")
                objective = score(mission, events, result)
                objective["routing_only_audit"] = dict(
                    identical_recording=True,
                    exact_capture_poses=True,
                    router_receives_gt=False,
                    scene_control_access=False,
                    public_phase_schema=sorted(public_phases(mission)[0]),
                )
                save(output / "objectives.json", objective)
                rows.append(
                    dict(
                        environment=name,
                        mission_id=mission["mission_id"],
                        policy=policy,
                        mission_success=objective["mission_success"],
                        phase_success_rate=objective["phase_success_rate"],
                        weighted_evidence_recall=objective["weighted_evidence_recall"],
                        resource_success=objective["resource_success"],
                        execution_success=objective["execution_success"],
                        final_battery_frac=result["final_battery_frac"],
                        switches=result["config_switches"],
                        processed=result["processed_observations"],
                        recording_sha256=backend.provenance["recording_sha256"],
                        output=str(output.relative_to(root)),
                    )
                )
                save(root / name / "comparison.partial.json", rows)
                print(
                    f"{name}/{mission['mission_id']}/{policy}: "
                    f"{objective['phase_success_rate']:.0%} stages",
                    flush=True,
                )
        save(root / name / "comparison.json", dict(rows=rows))
    if all((root / key / "comparison.json").exists() for key in suite["environments"]):
        merged = [
            r for key in suite["environments"] for r in read(root / key / "comparison.json")["rows"]
        ]
        save(
            root / "comparison.json",
            dict(
                rows=merged,
                complete=all(r["execution_success"] for r in merged),
                purpose="development scenarios; no generalization or adaptive-superiority claim",
            ),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["record", "audit", "infer", "evaluate", "run", "selected"]
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--environment", choices=["day", "dusk_haze"])
    args = parser.parse_args()
    if args.command == "run":
        record(args.root)
        infer(args.root)
        evaluate(args.root)
    elif args.command == "selected":
        evaluate(args.root, args.environment, on_demand=True)
    else:
        globals()[args.command](args.root)
