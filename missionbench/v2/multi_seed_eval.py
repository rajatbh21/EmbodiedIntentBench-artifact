"""V2.4 multi-seed hard-scenario evaluation: family generation + batch runs + statistics.

Answers, with controlled evidence rather than one hand-designed episode, whether an
adaptive configuration-selection policy succeeds more reliably than static baselines::

    python -m missionbench.v2.multi_seed_eval \
      --base-scenario data/v2_scenarios/demo_img1_hard_tradeoff.json \
      --seeds 0:29 --output results/v2_hard_multiseed

Method: every seed's scenario variant (see :mod:`scenario_family`) is evaluated under
**all** policies — paired by construction, so scenario-level comparisons are valid.
Per-run records restate the mission evaluator's own outputs (never a second
mission-success computation); aggregates use the same Wilson interval as the V1
benchmark; seeds that contradict the hypothesis are preserved and surfaced, and the
fragility section exists to catch an over-tuned family, not to hide one.

Outputs (all strict JSON, deterministic for identical inputs — measured wall-clock is
deliberately excluded from records):

    output/
    ├── experiment_manifest.json   # provenance: family version, base sha256, commit
    ├── scenarios/seed_NNNN.json   # the generated variants (regenerated idempotently)
    ├── runs.jsonl                 # one record per (seed, policy), sorted
    ├── aggregate.json             # per-policy statistics + Wilson 95% CIs
    ├── paired_comparison.json     # pattern counts, counterexample seed lists
    ├── report.md                  # human-readable summary + fragility flags
    └── representative_replays/    # replay bundles for auto-selected seeds

Reruns are resumable: existing (seed, policy) records are kept unless ``--force``.
Real-model execution requires the ``[v2-real-models]`` extra and the local-only
assets; generation, validation, and all statistics are importable without Torch.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

from missionbench.metrics.aggregate_metrics import wilson_interval
from missionbench.schemas.loading import SchemaValidationError, SchemaVersionError
from missionbench.v2.scenario_family import FAMILY_VERSION, write_variant

__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "aggregate_records",
    "fragility_report",
    "main",
    "paired_comparison",
    "run_record",
    "select_representatives",
]

EXPERIMENT_SCHEMA_VERSION: Final = "1.0"
DEFAULT_POLICIES: Final = ("always_light_real", "always_strong_real", "rule_based")
#: The default policy set for bases that declare a ``simulated_remote`` executor
#: (V3 P2): the remote static baseline is part of the designed 4-way outcome pattern.
REMOTE_DEFAULT_POLICIES: Final = (
    "always_light_real",
    "always_strong_real",
    "always_remote_strong",
    "rule_based",
)
_FRAGILE_BATTERY_MARGIN: Final = 0.01


# --- per-run records (restating the evaluator's outputs) -------------------------------------


def _privacy_status(result: Any, scenario: Any) -> str:
    """Privacy standing for a run: NOT_APPLICABLE where no remote path exists (V3 P1)."""
    has_remote = any(
        spec.kind == "simulated_remote" for spec in getattr(scenario, "executor_configs", ())
    )
    if not has_remote:
        return "NOT_APPLICABLE"
    satisfied = result.constraints.get("privacy_constraint_success", True)
    return "satisfied" if satisfied else "violated"


def run_record(result: Any, scenario: Any, seed: int) -> dict[str, Any]:
    """One structured record for a (scenario, policy) run.

    Every value restates the mission result / evaluator output; nothing is recomputed.
    Measured wall-clock diagnostics are excluded so records are deterministic.
    """
    quality = result.quality
    compute_energy = sum(log.execution["energy_j"] for log in result.observations)
    found = sorted({t for log in result.observations for t in log.score["matched_target_ids"]})
    small_total = sum(1 for o in scenario.targets if o.object_id.startswith("TGT_S"))
    late_total = sum(1 for o in scenario.targets if o.object_id.startswith("TGT_E"))
    history = list(result.config_selection_history)
    switches = sum(1 for a, b in itertools.pairwise(history) if a != b)
    invalid = sum(1 for log in result.observations if not log.action_valid)
    rle: list[list[Any]] = []
    for config in history:
        if rle and rle[-1][0] == config:
            rle[-1][1] += 1
        else:
            rle.append([config, 1])

    return {
        "scenario_id": result.scenario_id,
        "seed": seed,
        "policy": result.policy_name,
        "mission_success": result.mission_success,
        "constraints": dict(result.constraints),
        "privacy_status": _privacy_status(result, scenario),
        "termination_reason": result.termination_reason,
        "target_recall": quality["target_recall"],
        "detection_precision": quality["detection_precision"],
        "false_positive_detections": quality["false_positive_detections"],
        "false_positives_per_processed_minute": quality["false_positives_per_processed_minute"],
        "unique_targets_found": quality["unique_targets_found"],
        "total_unique_targets": quality["total_unique_targets"],
        "small_targets_found": sum(1 for t in found if t.startswith("TGT_S")),
        "small_targets_total": small_total,
        "late_targets_found": sum(1 for t in found if t.startswith("TGT_E")),
        "late_targets_total": late_total,
        "found_target_ids": found,
        "final_battery_frac": result.final_battery_frac,
        "battery_margin_frac": result.final_battery_frac - scenario.contract.min_final_battery_frac,
        "total_energy_j": result.cumulative_energy_j,
        "compute_energy_j": compute_energy,
        "flight_energy_j": result.cumulative_energy_j - compute_energy,
        "communication_mb": result.cumulative_communication_mb,
        "communication_margin_mb": (
            scenario.contract.communication_budget_mb - result.cumulative_communication_mb
        ),
        # Simulated radio energy (V3 P1); 0.0 for local-only missions, where no radio
        # is modelled and nothing can be spent.
        "communication_energy_j": result.communication_energy_j,
        "network_behaviour": dict(result.network_behaviour),
        "mean_executor_latency_s": result.mean_executor_latency_s,
        "processed_observations": result.processed_observation_count,
        "skipped_observations": result.skipped_observation_count,
        "config_switch_count": switches,
        "invalid_action_count": invalid,
        "fallback_count": invalid,  # in V2 an invalid action always executes the fallback
        "config_history_rle": rle,
    }


# --- aggregation -----------------------------------------------------------------------------

_AGGREGATED_METRICS: Final = (
    "target_recall",
    "detection_precision",
    "false_positive_detections",
    "final_battery_frac",
    "battery_margin_frac",
    "total_energy_j",
    "compute_energy_j",
    "communication_mb",
    "communication_margin_mb",
    "communication_energy_j",
    "skipped_observations",
    "config_switch_count",
)


def _metric_summary(values: list[float]) -> dict[str, Any] | None:
    """Summary statistics, or ``None`` when no record carries the metric."""
    if not values:
        return None
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-policy aggregate statistics with the benchmark's Wilson 95% intervals."""
    policies = sorted({r["policy"] for r in records})
    out: dict[str, Any] = {}
    for policy in policies:
        rows = [r for r in records if r["policy"] == policy]
        n = len(rows)
        successes = sum(1 for r in rows if r["mission_success"])
        low, high = wilson_interval(successes, n)
        out[policy] = {
            "evaluated_scenarios": n,
            "mission_success_count": successes,
            "mission_success_rate": successes / n,
            "mission_success_ci_95": [low, high],
            "failure_rates": {
                "quality": sum(1 for r in rows if not r["constraints"]["quality_success"]) / n,
                "deadline": sum(1 for r in rows if not r["constraints"]["deadline_success"]) / n,
                "battery": sum(
                    1 for r in rows if not r["constraints"]["battery_constraint_success"]
                )
                / n,
                "communication": sum(
                    1 for r in rows if not r["constraints"]["communication_constraint_success"]
                )
                / n,
                # V2 records predating the remote path carry no privacy key; absence
                # means the constraint could not fail, not that it was unmeasured.
                "privacy": sum(
                    1 for r in rows if not r["constraints"].get("privacy_constraint_success", True)
                )
                / n,
            },
            "small_target_detection_rate": statistics.fmean(
                r["small_targets_found"] / r["small_targets_total"]
                for r in rows
                if r["small_targets_total"]
            ),
            "late_target_detection_rate": statistics.fmean(
                r["late_targets_found"] / r["late_targets_total"]
                for r in rows
                if r["late_targets_total"]
            ),
            "metrics": {
                name: _metric_summary([float(r[name]) for r in rows if r.get(name) is not None])
                for name in _AGGREGATED_METRICS
            },
        }
    return out


# --- paired comparison -----------------------------------------------------------------------


def paired_comparison(
    records: list[dict[str, Any]],
    *,
    adaptive_policy: str = "rule_based",
    static_policies: tuple[str, ...] = ("always_light_real", "always_strong_real"),
) -> dict[str, Any]:
    """Scenario-level paired outcomes (valid because every policy ran the same seeds)."""
    by_seed: dict[int, dict[str, bool]] = {}
    for r in records:
        by_seed.setdefault(r["seed"], {})[r["policy"]] = r["mission_success"]

    complete = {
        seed: outcome
        for seed, outcome in sorted(by_seed.items())
        if adaptive_policy in outcome and all(p in outcome for p in static_policies)
    }

    def seeds_where(predicate) -> list[int]:
        return [seed for seed, o in complete.items() if predicate(o)]

    counts = {
        **{
            f"adaptive_succeeds_{static}_fails": seeds_where(
                lambda o, s=static: o[adaptive_policy] and not o[s]
            )
            for static in static_policies
        },
        "adaptive_succeeds_all_static_fail": seeds_where(
            lambda o: o[adaptive_policy] and not any(o[s] for s in static_policies)
        ),
        "adaptive_fails_any_static_succeeds": seeds_where(
            lambda o: not o[adaptive_policy] and any(o[s] for s in static_policies)
        ),
        "all_succeed": seeds_where(lambda o: all(o.values())),
        "all_fail": seeds_where(lambda o: not any(o.values())),
    }
    #: The design's reference outcome: every static baseline fails, the adaptive
    #: policy succeeds. Seeds that deviate are the interesting ones and are listed.
    reference = {**{static: False for static in static_policies}, adaptive_policy: True}
    return {
        "adaptive_policy": adaptive_policy,
        "static_policies": list(static_policies),
        "paired_seed_count": len(complete),
        "patterns": {name: {"count": len(seeds), "seeds": seeds} for name, seeds in counts.items()},
        "counterexamples": {
            **{
                f"adaptive_underperforms_{static}": seeds_where(
                    lambda o, s=static: o[s] and not o[adaptive_policy]
                )
                for static in static_policies
            },
            "pattern_differs_from_reference": seeds_where(
                lambda o: {p: o[p] for p in reference} != reference
            ),
        },
        "outcomes_by_seed": {str(seed): outcome for seed, outcome in complete.items()},
    }


# --- fragility / sensitivity -----------------------------------------------------------------


def fragility_report(
    records: list[dict[str, Any]],
    paired: dict[str, Any],
    *,
    adaptive_policy: str = "rule_based",
) -> dict[str, Any]:
    """Explicit-rule fragility analysis: is the family over-tuned or knife-edged?"""
    adaptive_rows = [r for r in records if r["policy"] == adaptive_policy]
    successes = [r for r in adaptive_rows if r["mission_success"]]
    margins = [r["battery_margin_frac"] for r in successes]
    fragile_margin = [
        r["seed"] for r in successes if r["battery_margin_frac"] < _FRAGILE_BATTERY_MARGIN
    ]
    single_switch = [r for r in successes if r["config_switch_count"] == 1]

    flags: list[str] = []
    if fragile_margin:
        flags.append(
            f"{len(fragile_margin)}/{len(successes)} adaptive successes clear the battery "
            f"floor by < {_FRAGILE_BATTERY_MARGIN} (seeds {fragile_margin}): the family is "
            "knife-edged on battery; a small energy perturbation flips these missions"
        )
    if successes and len(single_switch) == len(successes):
        flags.append(
            "every adaptive success uses exactly one configuration switch: the result "
            "pattern depends on a single strong-to-light transition (battery-pressure "
            "rule), not on richer adaptation"
        )
    if not paired["counterexamples"]["pattern_differs_from_reference"]:
        flags.append(
            "no seed deviates from the reference outcome pattern: the family may be "
            "over-controlled; widen variation before claiming robustness"
        )
    if successes and len(successes) == len(adaptive_rows):
        flags.append(
            "the adaptive policy never fails on the evaluated seeds; that supports the "
            "claim on this family but says nothing outside it"
        )

    switch_distribution: dict[str, int] = {}
    for row in successes:
        key = str(row["config_switch_count"])
        switch_distribution[key] = switch_distribution.get(key, 0) + 1

    return {
        "adaptive_battery_margin": {
            "n_successes": len(successes),
            "min": min(margins) if margins else None,
            "mean": statistics.fmean(margins) if margins else None,
            "below_threshold_seeds": fragile_margin,
            "threshold": _FRAGILE_BATTERY_MARGIN,
        },
        "adaptive_single_switch_successes": {
            "count": len(single_switch),
            "of_successes": len(successes),
        },
        "adaptive_switch_count_distribution": dict(sorted(switch_distribution.items())),
        "flags": flags,
    }


# --- representative replay selection ---------------------------------------------------------


def select_representatives(
    paired: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    adaptive_policy: str = "rule_based",
    limit: int = 5,
) -> list[dict[str, Any]]:
    """A small, reasoned set of seeds worth watching in the replay viewer."""
    patterns = paired["patterns"]
    margin_by_seed = {
        r["seed"]: r["battery_margin_frac"]
        for r in records
        if r["policy"] == adaptive_policy and r["mission_success"]
    }
    fragile = sorted(s for s, m in margin_by_seed.items() if m < _FRAGILE_BATTERY_MARGIN)
    adaptive_failures = sorted(
        {r["seed"] for r in records if r["policy"] == adaptive_policy and not r["mission_success"]}
    )
    static_success = sorted(
        set(patterns["all_succeed"]["seeds"])
        | set(patterns["adaptive_fails_any_static_succeeds"]["seeds"])
    )
    candidates = [
        ("only_adaptive_succeeds", patterns["adaptive_succeeds_all_static_fail"]["seeds"]),
        ("all_policies_fail", patterns["all_fail"]["seeds"]),
        ("a_static_policy_succeeds", static_success),
        ("fragile_adaptive_success_battery_margin", fragile),
        ("adaptive_failure_counterexample", adaptive_failures),
    ]
    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for reason, seeds in candidates:
        for seed in seeds:
            if seed not in used:
                chosen.append({"seed": seed, "reason": reason})
                used.add(seed)
                break
        if len(chosen) >= limit:
            break
    return chosen


# --- report ----------------------------------------------------------------------------------


def render_report(
    manifest: dict[str, Any],
    aggregate: dict[str, Any],
    paired: dict[str, Any],
    fragility: dict[str, Any],
    representatives: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    add = lines.append
    add("# Multi-seed hard-scenario evaluation")
    add("")
    add(
        f"Family `{manifest['scenario_family_version']}` of base "
        f"`{manifest['base_scenario_id']}` — {len(manifest['seeds'])} seeds x "
        f"{len(manifest['policies'])} policies, paired by construction. "
        f"Base commit `{manifest['base_commit']}`."
    )
    add("")
    add(
        "> All numbers are synthetic-human controlled-observability diagnostics with "
        "configured (simulated) latency/energy — never real aerial-human perception "
        "or real hardware measurements."
    )
    add("")
    add("## Per-policy outcomes")
    add("")
    add(
        "| policy | n | success | rate | Wilson 95% CI | quality fail | battery fail "
        "| comm fail | privacy fail |"
    )
    add("|---|---|---|---|---|---|---|---|---|")
    for policy, stats in aggregate.items():
        low, high = stats["mission_success_ci_95"]
        add(
            f"| {policy} | {stats['evaluated_scenarios']} | "
            f"{stats['mission_success_count']} | {stats['mission_success_rate']:.3f} | "
            f"[{low:.3f}, {high:.3f}] | {stats['failure_rates']['quality']:.3f} | "
            f"{stats['failure_rates']['battery']:.3f} | "
            f"{stats['failure_rates']['communication']:.3f} | "
            f"{stats['failure_rates']['privacy']:.3f} |"
        )
    add("")
    add(
        "| policy | recall mean | precision mean | final battery mean "
        "| comm MB mean | switches mean | skipped mean |"
    )
    add("|---|---|---|---|---|---|---|")
    for policy, stats in aggregate.items():
        m = stats["metrics"]
        comm = m.get("communication_mb")
        comm_cell = f"{comm['mean']:.1f}" if comm else "n/a"
        add(
            f"| {policy} | {m['target_recall']['mean']:.3f} | "
            f"{m['detection_precision']['mean']:.3f} | "
            f"{m['final_battery_frac']['mean']:.3f} | "
            f"{comm_cell} | "
            f"{m['config_switch_count']['mean']:.2f} | "
            f"{m['skipped_observations']['mean']:.1f} |"
        )
    add("")
    add("## Target-class detection (why outcomes differ)")
    add("")
    add("| policy | small-target detection rate | late-target detection rate |")
    add("|---|---|---|")
    for policy, stats in aggregate.items():
        add(
            f"| {policy} | {stats['small_target_detection_rate']:.3f} | "
            f"{stats['late_target_detection_rate']:.3f} |"
        )
    add("")
    add("## Paired scenario-level comparison")
    add("")
    for name, entry in paired["patterns"].items():
        add(f"- **{name}**: {entry['count']}  (seeds: {entry['seeds']})")
    add("")
    add("### Counterexamples (preserved, not hidden)")
    add("")
    for name, seeds in paired["counterexamples"].items():
        add(f"- {name}: {seeds if seeds else 'none'}")
    add("")
    add("## Fragility / sensitivity")
    add("")
    margin = fragility["adaptive_battery_margin"]
    if margin["n_successes"]:
        add(
            f"- adaptive battery margin over successes: min {margin['min']:.4f}, "
            f"mean {margin['mean']:.4f} (fragile if < {margin['threshold']})"
        )
    single = fragility["adaptive_single_switch_successes"]
    add(f"- adaptive successes with exactly one switch: {single['count']}/{single['of_successes']}")
    distribution = fragility.get("adaptive_switch_count_distribution")
    if distribution:
        cells = ", ".join(f"{k} switch(es): {v}" for k, v in distribution.items())
        add(f"- adaptive switch-count distribution over successes: {cells}")
    for flag in fragility["flags"]:
        add(f"- **FLAG**: {flag}")
    if not fragility["flags"]:
        add("- no fragility flags raised by the explicit rules")
    add("")
    add("## Representative replays")
    add("")
    if not representatives:
        add("(none selected)")
    by_key = {(r["seed"], r["policy"]): r for r in records}
    for entry in representatives:
        seed = entry["seed"]
        add(f"### seed {seed:04d} — {entry['reason']}")
        add("")
        add(
            "| policy | mission | recall | final battery | skips | switches "
            "| termination | replay |"
        )
        add("|---|---|---|---|---|---|---|---|")
        for policy in manifest["policies"]:
            r = by_key.get((seed, policy))
            if r is None:
                continue
            link = f"representative_replays/seed_{seed:04d}/{policy}/index.html"
            add(
                f"| {policy} | {'SUCCESS' if r['mission_success'] else 'FAIL'} | "
                f"{r['target_recall']:.3f} | {r['final_battery_frac']:.3f} | "
                f"{r['skipped_observations']} | {r['config_switch_count']} | "
                f"{r['termination_reason']} | [{link}]({link}) |"
            )
        add("")
    return "\n".join(lines) + "\n"


# --- experiment driver -----------------------------------------------------------------------


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def _dump(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        start_s, _, end_s = args.seeds.partition(":")
        start, end = int(start_s), int(end_s)
        if end < start:
            raise SystemExit(f"--seeds {args.seeds}: end must be >= start")
        return list(range(start, end + 1))
    return list(range(args.seed_start, args.seed_start + args.num_seeds))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="missionbench.v2.multi_seed_eval",
        description="Generate a hard-scenario family and evaluate policies on every seed.",
    )
    parser.add_argument(
        "--base-scenario",
        type=Path,
        default=Path("data/v2_scenarios/demo_img1_hard_tradeoff.json"),
    )
    parser.add_argument("--seeds", help="Inclusive range, e.g. 0:29.")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument(
        "--policies",
        default=None,
        help="Comma-separated policy names; defaults to the base scenario's designed "
        "set (remote bases add always_remote_strong).",
    )
    parser.add_argument("--adaptive-policy", default="rule_based")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--force", action="store_true", help="Re-run pairs already present in runs.jsonl."
    )
    parser.add_argument("--no-replays", action="store_true")
    parser.add_argument("--max-replays", type=int, default=5)
    args = parser.parse_args(argv)

    seeds = _parse_seeds(args)

    from missionbench.v2.scenario import load_scenario

    try:
        base_scenario = load_scenario(args.base_scenario)
    except (SchemaValidationError, SchemaVersionError) as error:
        print(f"base scenario failed to load: {error}", file=sys.stderr)
        return 2
    base_id = base_scenario.scenario_id
    if args.policies is not None:
        policies = tuple(p for p in args.policies.split(",") if p)
    elif any(spec.kind == "simulated_remote" for spec in base_scenario.executor_configs):
        policies = REMOTE_DEFAULT_POLICIES
    else:
        policies = DEFAULT_POLICIES
    statics = tuple(p for p in policies if p != args.adaptive_policy)

    out = args.output
    scenarios_dir = out / "scenarios"
    out.mkdir(parents=True, exist_ok=True)

    base_sha = hashlib.sha256(args.base_scenario.read_bytes()).hexdigest()
    manifest_path = out / "experiment_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if (
            previous.get("scenario_family_version") != FAMILY_VERSION
            or previous.get("base_scenario_sha256") != base_sha
        ):
            print(
                f"{out} holds results for a different family/base scenario; use a fresh "
                "--output directory instead of mixing experiments",
                file=sys.stderr,
            )
            return 2

    try:
        scenario_paths: dict[int, Path] = {}
        for seed in seeds:
            scenario_paths[seed] = write_variant(
                args.base_scenario, seed, scenarios_dir / f"seed_{seed:04d}.json"
            )
    except (SchemaValidationError, SchemaVersionError) as error:
        print(f"scenario generation failed: {error}", file=sys.stderr)
        return 2

    runs_path = out / "runs.jsonl"
    existing: dict[tuple[int, str], dict[str, Any]] = {}
    if runs_path.exists() and not args.force:
        for line in runs_path.read_text().splitlines():
            record = json.loads(line)
            existing[(record["seed"], record["policy"])] = record

    from missionbench.v2.runner import run_mission

    records: list[dict[str, Any]] = []
    for seed in seeds:
        scenario = load_scenario(scenario_paths[seed])
        for policy in policies:
            key = (seed, policy)
            if key in existing:
                records.append(existing[key])
                continue
            result = run_mission(scenario, policy)
            record = run_record(result, scenario, seed)
            records.append(record)
            print(
                f"seed {seed:04d} {policy:<20} success={record['mission_success']} "
                f"recall={record['target_recall']:.3f} "
                f"battery={record['final_battery_frac']:.3f} "
                f"switches={record['config_switch_count']}"
            )
    records.sort(key=lambda r: (r["seed"], r["policy"]))
    runs_path.write_text(
        "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in records),
        encoding="utf-8",
    )

    aggregate = aggregate_records(records)
    paired = paired_comparison(
        records, adaptive_policy=args.adaptive_policy, static_policies=statics
    )
    fragility = fragility_report(records, paired, adaptive_policy=args.adaptive_policy)
    representatives = select_representatives(
        paired, records, adaptive_policy=args.adaptive_policy, limit=args.max_replays
    )

    manifest = {
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "scenario_family_version": FAMILY_VERSION,
        "base_scenario": args.base_scenario.name,
        "base_scenario_id": base_id,
        "base_scenario_sha256": base_sha,
        "base_commit": _git_commit(),
        "seeds": seeds,
        "policies": list(policies),
        "adaptive_policy": args.adaptive_policy,
        "honesty": {
            "targets": "synthetic generated humans (owner-supplied); never real people",
            "measurements": "latency/energy configured (simulated); deterministic missions",
            "claim_scope": (
                "results characterise this scenario family only; they are not real "
                "aerial-human perception performance"
            ),
        },
    }
    _dump(manifest_path, manifest)
    _dump(out / "aggregate.json", aggregate)
    _dump(out / "paired_comparison.json", paired)
    (out / "report.md").write_text(
        render_report(manifest, aggregate, paired, fragility, representatives, records),
        encoding="utf-8",
    )
    _dump(out / "fragility.json", fragility)

    if not args.no_replays:
        from missionbench.v2.replay_export import export_replay_bundle

        for entry in representatives:
            seed = entry["seed"]
            for policy in policies:
                bundle_dir = out / "representative_replays" / f"seed_{seed:04d}" / policy
                if not (bundle_dir / "index.html").exists() or args.force:
                    export_replay_bundle(scenario_paths[seed], policy, bundle_dir)
            print(f"replays exported for seed {seed:04d} ({entry['reason']})")

    adaptive = aggregate.get(args.adaptive_policy)
    if adaptive:
        low, high = adaptive["mission_success_ci_95"]
        print(
            f"\n{args.adaptive_policy}: {adaptive['mission_success_count']}/"
            f"{adaptive['evaluated_scenarios']} success "
            f"(rate {adaptive['mission_success_rate']:.3f}, Wilson 95% [{low:.3f}, {high:.3f}])"
        )
    for name, entry in paired["patterns"].items():
        print(f"{name}: {entry['count']}")
    print(f"wrote {out / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
