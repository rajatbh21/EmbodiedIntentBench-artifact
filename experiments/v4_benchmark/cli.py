"""CLI for configurable V4 router evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v4_benchmark.scenario import load_scenario


def main() -> int:
    parser = argparse.ArgumentParser(prog="v4-benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scenario", type=Path, required=True)
    common.add_argument(
        "--router",
        required=True,
        help="static:ID, rule_based, budget_planner, intent_visual, intent_only, "
        "visual_only, resource_only, periodic, seeded_random, calibrated, or python:module:factory",
    )
    common.add_argument("--output", type=Path, required=True)
    common.add_argument("--recording", type=Path)
    run = sub.add_parser("run", parents=[common])
    run.add_argument("--router-parameters", type=Path)
    run.add_argument("--decision-stride", type=int, default=1)
    run.add_argument("--split-manifest", type=Path)
    run.add_argument("--partition", choices=("development", "validation", "evaluation"))
    record = sub.add_parser("record")
    record.add_argument("--scenario", type=Path, required=True)
    record.add_argument("--live-python", default="/usr/bin/python3")
    # Accept the old unused arguments for existing command histories.
    record.add_argument("--router", help=argparse.SUPPRESS)
    record.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    record.add_argument("--recording-output", type=Path, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--scenario", type=Path, required=True)
    args = parser.parse_args()
    scenario = load_scenario(args.scenario)
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "scenario_id": scenario.scenario_id,
                    "recording_identity": scenario.recording_identity,
                    "config_ids": scenario.catalog.ids(),
                    "capture_slots": (
                        scenario.schedule.end_ns
                        - scenario.schedule.epoch_ns
                        + scenario.schedule.interval_ns
                        - 1
                    )
                    // scenario.schedule.interval_ns,
                },
                indent=2,
            )
        )
        return 0
    if args.command == "record":
        from experiments.v4_benchmark.backends import LiveBackend, record_world

        backend = LiveBackend(scenario, args.recording_output, python=args.live_python)
        try:
            manifest = record_world(scenario, backend, args.recording_output)
        finally:
            backend.close()
        print(
            json.dumps(
                {"recording": str(args.recording_output), "frames": len(manifest["frames"])},
                indent=2,
            )
        )
        return 0
    if args.recording is None:
        raise SystemExit("run requires --recording; use record first")
    split_info = None
    if bool(args.split_manifest) != bool(args.partition):
        raise SystemExit("--split-manifest and --partition must be supplied together")
    if args.split_manifest:
        from experiments.v4_benchmark.splits import verify_membership

        split_info = verify_membership(
            args.split_manifest, args.scenario, args.recording, args.partition
        )
    from missionbench.v2.executors import build_executors
    from experiments.v4_benchmark.backends import ReplayBackend
    from experiments.v4_benchmark.routers import build_router
    from experiments.v4_benchmark.runner import run_mission

    backend = ReplayBackend(scenario, args.recording)
    try:
        parameters = {}
        if args.router_parameters:
            parameters = json.loads(args.router_parameters.read_text())
        router = build_router(
            args.router,
            scenario.profiles,
            scenario.observation_interval_s * args.decision_stride,
            parameters,
        )
        result = run_mission(
            scenario,
            backend,
            router,
            build_executors(scenario.executors),
            router_name=args.router,
            decision_stride=args.decision_stride,
            output=args.output,
        )
    finally:
        backend.close()
    if split_info:
        from experiments.v4_benchmark.backends import save_json

        result["split_protocol"] = split_info
        save_json(args.output / "results.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
