"""Benchmark CLI.

    python -m missionbench.run_benchmark \\
      --episode data/episodes/episode_001.json \\
      --contract data/contracts/contract_001.json \\
      --policy rule_based \\
      --output results/episode_001_rule_based.json

    # every shipped episode, aggregated
    python -m missionbench.run_benchmark --suite \\
      --contract data/contracts/contract_001.json \\
      --policy rule_based --output results/suite_rule_based.json

A thin shell over :mod:`missionbench.benchmark`: parse arguments, run, write JSON, print
a summary. Keeping it thin is deliberate -- everything worth testing lives in the library, so
the tests exercise the same path a user does rather than a parallel one.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from missionbench.benchmark import DEFAULT_EXECUTOR, BenchmarkData, SuiteResult, run_suite
from missionbench.executor.registry import executor_registry
from missionbench.policies.registry import policy_registry
from missionbench.schemas.contract import load_contract
from missionbench.schemas.episode import load_episode
from missionbench.schemas.loading import SchemaValidationError

__all__ = ["build_parser", "main"]

_DEFAULT_DATA_ROOT = Path("data")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="missionbench",
        description="Run EmbodiedIntentBench synthetic core fixtures and write JSON metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "All shipped fixture values are synthetic placeholders, not measurements.\n"
            "For the bundled paper experiments, use python tools/artifact.py --help.\n"
            f"Policies:  {', '.join(policy_registry.names())}\n"
            f"Executors: {', '.join(executor_registry.names())}"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help="Directory holding the benchmark fixtures (default: %(default)s).",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--episode",
        type=Path,
        action="append",
        dest="episodes",
        metavar="PATH",
        help="Episode file to run. Repeat for several.",
    )
    selection.add_argument(
        "--suite",
        action="store_true",
        help="Run every episode in the data root and aggregate.",
    )
    parser.add_argument("--contract", type=Path, required=True, help="Contract file.")
    parser.add_argument(
        "--policy",
        default="rule_based",
        choices=policy_registry.names(),
        help="Policy to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--executor",
        default=DEFAULT_EXECUTOR,
        choices=executor_registry.names(),
        help=(
            "Execution backend (default: %(default)s). "
            "'profile' synthesises costs from a per-platform profile (all values synthetic); "
            "'replay' serves precomputed frame x config records, the bridge to real "
            "predictions; 'real_segmentation' is a V1 stub and fails immediately."
        ),
    )
    parser.add_argument(
        "--replay-strict",
        action="store_true",
        help=(
            "With --executor replay, treat a missing record as an error rather than a "
            "recorded failed inference."
        ),
    )
    parser.add_argument(
        "--fallback-config-id",
        default=None,
        metavar="CONFIG_ID",
        help=(
            "Safe fallback configuration for invalid policy actions, overriding the episode's "
            "own fallback. Must be in the catalog and the episode's allowed pool. When omitted, "
            "the fallback is resolved from the episode (fallback_config_id, then "
            "initial_config_id, then the legacy CFG_LOCAL_LIGHT if present)."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run each episode under N seeds and pool the results (default: %(default)s). "
            "Three episodes can only produce a success rate of 0, 1/3, 2/3 or 1; raise this "
            "to get an interval narrow enough to compare policies."
        ),
    )
    parser.add_argument("--output", type=Path, help="Where to write the result JSON.")
    parser.add_argument(
        "--include-detail",
        action="store_true",
        help=(
            "Include the step log and raw evidence. Large, and the raw evidence contains "
            "ground-truth-derived fields -- do not publish a detailed result file."
        ),
    )
    parser.add_argument(
        "--hide-profiles",
        action="store_true",
        help="Withhold public configuration profiles from the policy.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress the printed summary.")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Log per-step policy and validation detail."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        data = BenchmarkData(args.data_root)
        contract = load_contract(args.contract)
        episodes = (
            data.episodes() if args.suite else tuple(load_episode(path) for path in args.episodes)
        )
        result = run_suite(
            data=data,
            episodes=episodes,
            contract=contract,
            policy_name=args.policy,
            executor_name=args.executor,
            disclose_profiles=not args.hide_profiles,
            replay_strict=args.replay_strict,
            repeats=args.repeats,
            fallback_config_id=args.fallback_config_id,
        )
    except NotImplementedError as error:
        # A selected-but-unbuilt backend (real_segmentation) fails at construction, before
        # any episode runs. Report it as a clean usage error, not a traceback.
        print(f"missionbench: {error}", file=sys.stderr)
        return 2
    except SchemaValidationError as error:
        print(f"missionbench: {error}", file=sys.stderr)
        return 2

    payload = result.to_dict(include_detail=args.include_detail)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # sort_keys so that re-running an unchanged benchmark produces a byte-identical file,
        # which makes results diffable and determinism regressions visible in review.
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    elif args.quiet:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")

    if not args.quiet:
        _print_summary(result, output=args.output)
    return 0


def _print_summary(result: SuiteResult, *, output: Path | None) -> None:
    aggregate = result.aggregate
    print(f"policy: {result.policy_name}   executor: {result.executor_id}")
    if result.repeats == 1:
        print(
            f"{'episode':<16}{'success':>9}{'quality':>9}{'MB':>9}{'batt':>8}{'time s':>9}"
            f"{'switch':>8}"
        )
        for episode in result.episodes:
            metrics = episode.metrics
            print(
                f"{metrics.episode_id:<16}"
                f"{'PASS' if metrics.mission_success else 'fail':>9}"
                f"{metrics.quality_value:>9.3f}"
                f"{metrics.total_communication_mb:>9.0f}"
                f"{metrics.final_battery_fraction:>8.3f}"
                f"{metrics.mission_completion_time_s:>9.0f}"
                f"{metrics.configuration_switch_count:>8}"
            )
    else:
        # One line per episode would be `repeats` lines each; summarise per episode instead.
        print(f"{'episode':<16}{'runs':>6}{'passed':>8}{'rate':>8}{'mean quality':>14}")
        for episode_id in dict.fromkeys(e.metrics.episode_id for e in result.episodes):
            runs = [e.metrics for e in result.episodes if e.metrics.episode_id == episode_id]
            passed = sum(m.mission_success for m in runs)
            mean_quality = sum(m.quality_value for m in runs) / len(runs)
            print(
                f"{episode_id:<16}{len(runs):>6}{passed:>8}"
                f"{passed / len(runs):>8.0%}{mean_quality:>14.3f}"
            )

    low, high = aggregate.mission_success_ci
    print(
        f"\nMission Success Rate: {aggregate.mission_success_rate:.1%} "
        f"({aggregate.mission_success_count}/{aggregate.episode_count})   "
        f"95% CI [{low:.1%}, {high:.1%}]"
    )
    if aggregate.episode_count < 30:
        print(
            "  note: that interval is wide. Use --repeats to pool more seeds before "
            "comparing policies."
        )
    print(
        f"  quality {aggregate.quality_success_rate:.0%}   "
        f"deadline {aggregate.deadline_success_rate:.0%}   "
        f"battery {aggregate.battery_constraint_success_rate:.0%}   "
        f"communication {aggregate.communication_constraint_success_rate:.0%}   "
        f"privacy {aggregate.privacy_constraint_success_rate:.0%}"
    )
    if output is not None:
        print(f"\nwrote {output}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
