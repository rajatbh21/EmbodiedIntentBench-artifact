"""One-command physical recording, actual model inference and paired policy scoring."""

import argparse
import copy
import json
import re
import tarfile
from pathlib import Path

from experiments.routing_mission.environment import load
from experiments.routing_mission.inference import checksum, compute
from experiments.routing_mission.router import ContextRouter
from experiments.v4_benchmark.backends import ReplayBackend, record_world, save_json
from experiments.v4_benchmark.probe_cache import build_cached
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.runner import run_mission

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text())


def freeze(root):
    paths = [
        root / name
        for name in (
            "experiment.json",
            "scenario.json",
            "contract.json",
            "intent.json",
            "evidence.json",
        )
    ]
    source_paths = sorted(
        [
            *ROOT.glob("missionbench/**/*.py"),
            *ROOT.glob("experiments/routing_mission/*.py"),
            *ROOT.glob("experiments/v4_benchmark/*.py"),
            *ROOT.glob("experiments/v4_gazebo_pilot/*.py"),
            *ROOT.glob("experiments/evidence_environment/*.py"),
        ]
    )
    lock = dict(
        inputs={p.name: checksum(p) for p in paths},
        implementation={str(p.relative_to(ROOT)): checksum(p) for p in source_paths},
    )
    if (root / "lock.json").exists():
        if read(root / "lock.json") != lock:
            raise ValueError("locked experiment inputs/code changed")
    else:
        save_json(root / "lock.json", lock)
        with tarfile.open(root / "implementation_snapshot.tar.gz", "w:gz") as archive:
            for p in source_paths:
                archive.add(p, arcname=str(p.relative_to(ROOT)))
    return lock


def specification(root):
    spec = read(root / "experiment.json")
    if set(spec) != {
        "schema_version",
        "scenario",
        "environments",
        "cases",
        "routers",
        "seed",
        "threads",
    }:
        raise ValueError("unexpected experiment fields")
    if spec["schema_version"] != "1.0" or spec["scenario"] != "scenario.json":
        raise ValueError("unsupported experiment version/scenario")
    if type(spec["threads"]) is not int or spec["threads"] < 1 or type(spec["seed"]) is not int:
        raise ValueError("invalid runtime settings")
    if not spec["environments"] or not spec["cases"] or not spec["routers"]:
        raise ValueError("empty experiment")
    ids = [c["case_id"] for c in spec["cases"]]
    if len(ids) != len(set(ids)) or len(spec["routers"]) != len(set(spec["routers"])):
        raise ValueError("duplicate case/router")
    for key in [*spec["environments"], *ids]:
        if not re.fullmatch("[A-Za-z0-9_-]+", key):
            raise ValueError("invalid case/environment name")
    for case in spec["cases"]:
        if set(case) != {"case_id", "contract", "requirements", "mission"}:
            raise ValueError("invalid case fields")
        if set(case["mission"]) - {"initial_battery_frac"}:
            raise ValueError("paired cases may only change initial battery, not path/sensors/costs")
    return spec


def scenario_case(root, case, environment, folder):
    data = read(root / "scenario.json")
    contract = read(root / "contract.json")
    contract.update(case["contract"])
    data["mission"].update(case["mission"])
    data["mission_requirements"].update(case["requirements"])
    folder.mkdir(parents=True, exist_ok=True)
    save_json(folder / "contract.json", contract)
    save_json(folder / "scenario.json", data)
    scenario = load(folder / "scenario.json", environment)
    base = load(root / "scenario.json", environment)
    if scenario.recording_identity != base.recording_identity:
        raise ValueError("paired case changed physical trajectory or capture protocol")
    return scenario


def router_for(name, scenario, seed):
    if name.startswith("context"):
        if name not in (
            "context",
            "context_no_intent",
            "context_no_visual",
            "context_no_resources",
        ):
            raise ValueError("unknown context ablation")
        ranks = {
            "CFG_LRASPP_320": 0,
            "CFG_DEEPLAB_MOBILE_320": 1,
            "CFG_CABINET_640": 1,
            "CFG_MASKRCNN_640": 2,
            "CFG_MASKRCNN_1280": 3,
        }
        return ContextRouter(
            dict(
                duration_s=scenario.duration_s,
                interval_s=scenario.observation_interval_s,
                capacity_j=scenario.battery_capacity_wh * 3600,
                flight_power_w=scenario.flight_power_w,
                costs={
                    s.config_id: dict(
                        latency_s=s.mission_latency_s,
                        energy_j=s.energy_j_per_call,
                        rank=ranks.get(s.config_id, 1),
                    )
                    for s in scenario.executors
                },
            ),
            mode=name,
        )
    params = {}
    if name in ("periodic", "seeded_random"):
        params = dict(config_ids=list(scenario.catalog.ids()), seed=seed, period_s=30)
    return build_router(name, scenario.profiles, scenario.observation_interval_s, params)


def record(root):
    from experiments.routing_mission.backend import LiveBackend

    spec = specification(root)
    freeze(root)
    for name, environment in spec["environments"].items():
        scenario = load(root / "scenario.json", environment)
        folder = root / name / "recording"
        if (folder / "recording.json").exists():
            ReplayBackend(scenario, folder).close()
            continue
        backend = LiveBackend(scenario, folder)
        try:
            record_world(scenario, backend, folder)
        finally:
            backend.close()
        print("recorded " + name, flush=True)


def infer(root):
    spec = specification(root)
    freeze(root)
    for name, environment in spec["environments"].items():
        scenario = load(root / "scenario.json", environment)
        folder = root / name
        if (folder / "cache/manifest.json").exists():
            checked_cache(scenario, folder)
            continue
        compute(scenario, folder / "recording", folder / "cache", spec["threads"])


def checked_cache(scenario, folder):
    manifest = read(folder / "cache/manifest.json")
    if manifest["recording_sha256"] != checksum(folder / "recording/recording.json"):
        raise ValueError("cached predictions belong to different captured bytes")
    if manifest["source_inference_sha256"] != checksum(folder / "cache/inference.json"):
        raise ValueError("actual inference provenance changed")
    captures = len(
        range(scenario.schedule.epoch_ns, scenario.schedule.end_ns, scenario.schedule.interval_ns)
    )
    rows = read(folder / "cache/inference.json")
    expected = {(i, s.config_id) for i in range(captures) for s in scenario.executors}
    keys = [(r["capture_id"], r["config_id"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("incomplete or duplicate actual inference calls")
    if manifest["capture_count"] != captures or not all(r["execution"]["success"] for r in rows):
        raise ValueError("failed or incomplete inference cache")
    return build_cached(scenario.executors, folder / "cache", scenario.recording_identity)


def evaluate(root):
    spec = specification(root)
    freeze(root)
    # Prepare and validate every policy/case before consuming any evaluation result.
    prepared = []
    for name, environment in spec["environments"].items():
        for case in spec["cases"]:
            folder = root / name / "comparison" / case["case_id"]
            scenario = scenario_case(root, case, environment, folder)
            ReplayBackend(scenario, root / name / "recording").close()
            for router in spec["routers"]:
                router_for(router, scenario, spec["seed"])
            prepared.append((name, case["case_id"], folder, scenario))
    rows = []
    for name, case_id, folder, scenario in prepared:
        # Immutable prediction arrays shared, executor objects copied per mission.
        cached = checked_cache(scenario, root / name)
        for index, policy in enumerate(spec["routers"]):
            output = folder / f"policy_{index:02d}"
            if output.exists():
                raise ValueError(
                    "evaluation output exists; reuse summary or create a new experiment"
                )
            backend = ReplayBackend(scenario, root / name / "recording")
            try:
                result = run_mission(
                    scenario,
                    backend,
                    router_for(policy, scenario, spec["seed"]),
                    {k: copy.copy(v) for k, v in cached.items()},
                    router_name=policy,
                    output=output,
                )
            finally:
                backend.close()
            events = read(output / "events.json")
            canonical = {
                r["timestamp_ns"]: r
                for r in read(root / name / "recording/recording.json")["frames"]
            }
            for event in events:
                if (
                    event["capture_body_pose"]
                    != canonical[event["capture_timestamp_ns"]]["body_pose"]
                ):
                    raise ValueError("policy-dependent trajectory detected")
            result["physical_environment"] = vars(scenario.environment)
            result["user_intent_provenance"] = read(root / "intent.json")
            result["routing_only_audit"] = dict(
                same_recording=True,
                canonical_capture_poses_verified=True,
                flight_control_access=False,
                gt_in_router_input=False,
                note="trusted Python plugin boundary, not a hostile-code sandbox",
            )
            save_json(output / "results.json", result)
            rows.append(
                dict(
                    environment=name,
                    case_id=case_id,
                    policy=policy,
                    status=result["status"],
                    mission_success=result["mission_success"],
                    report_recall=result["report_quality"]["report_recall"],
                    false_reports=result["report_quality"]["false_reports"],
                    first_confirmed_s=result["report_quality"]["first_confirmed_correct_report_s"],
                    final_battery_frac=result["final_battery_frac"],
                    energy_j=result["cumulative_energy_j"],
                    switches=result["config_switches"],
                    processed=result["processed_observations"],
                    skipped=len(result["skipped_observation_ids"]),
                    failed_constraints=[k for k, v in result["constraints"].items() if not v],
                    router_errors=result["router_errors"],
                    invalid_actions=result["invalid_actions"],
                    recording_sha256=backend.provenance["recording_sha256"],
                    result=str(output.relative_to(root) / "results.json"),
                )
            )
            save_json(root / "comparison.partial.json", rows)
            print(f"evaluated {name}/{case_id}/{policy}: {result['mission_success']}", flush=True)
    # Same environment/case must use the exact recording for every policy.
    for name in spec["environments"]:
        if len({r["recording_sha256"] for r in rows if r["environment"] == name}) != 1:
            raise ValueError("policy recording mismatch")
    report = dict(
        scope="paired routing-policy development evaluation; no population success rate",
        physical_environments=len(spec["environments"]),
        cases=len(spec["cases"]),
        policies=len(spec["routers"]),
        rows=rows,
        complete=all(
            r["status"] == "completed" and not r["router_errors"] and not r["invalid_actions"]
            for r in rows
        ),
    )
    save_json(root / "comparison.json", report)
    lines = [
        "# Fixed-flight routing-policy comparison",
        "",
        report["scope"],
        "",
        "| Environment | Case | Policy | Mission | Report recall | False reports | Reserve |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['environment']} | {r['case_id']} | {r['policy']} | "
            f"{r['mission_success']} | {r['report_recall']:.3f} | {r['false_reports']} | "
            f"{r['final_battery_frac']:.3f} |"
        )
    (root / "comparison.md").write_text("\n".join(lines) + "\n")
    if not report["complete"]:
        raise RuntimeError("component/policy errors remain; inspect comparison")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    create = sub.add_parser("prepare")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument(
        "--source", type=Path, default=Path("results/v4_external_evidence_20261014")
    )
    create.add_argument(
        "--asset", type=Path, default=Path("results/v4_assets/standing_person/manifest.json")
    )
    create.add_argument("--request", required=True)
    create.add_argument("--source-run", default="D-4D-141520")
    create.add_argument("--duration", type=float)
    create.add_argument("--interval", type=float, default=5)
    for key in ("record", "infer", "evaluate", "run", "audit"):
        s = sub.add_parser(key)
        s.add_argument("--experiment", type=Path, required=True)
    a = p.parse_args()
    if a.command == "prepare":
        from experiments.routing_mission.prepare import prepare

        prepare(a.output, a.source, a.asset, a.request, a.source_run, a.duration, a.interval)
    elif a.command == "record":
        record(a.experiment)
    elif a.command == "infer":
        infer(a.experiment)
    elif a.command == "evaluate":
        evaluate(a.experiment)
    elif a.command == "audit":
        from experiments.routing_mission.audit import audit

        audit(a.experiment)
    else:
        record(a.experiment)
        infer(a.experiment)
        evaluate(a.experiment)
        from experiments.routing_mission.audit import audit

        audit(a.experiment)


if __name__ == "__main__":
    main()
