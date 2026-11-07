"""Energy robustness of observed schedules, not a rerun of adaptive policies."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.protocol import verify


def breakpoint(available, total_energy, inference_seconds, reference_power=15):
    if inference_seconds <= 0:
        raise ValueError("Positive inference time required")
    fixed = total_energy - reference_power * inference_seconds
    return (available - fixed) / inference_seconds


def run(root):
    p = verify(root)
    completion = read(root / "completion.json")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed completed study")
    analysis = read(root / "analysis.json")
    rows = []
    for r in analysis["policy_cases"]:
        scenario = read(root / r["flight"] / r["mission"] / "scenario.json")
        specs = {s["config_id"]: s for s in scenario["executor_configs"]}
        events = read(root / r["path"] / "execution/events.json")
        if sha(root / r["path"] / "execution/events.json") != r["events_sha256"]:
            raise ValueError("Changed schedule")
        seconds = sum(specs[e["execution"]["config_id"]]["mission_latency_s"] for e in events)
        limit = breakpoint((r["battery"] - 0.2) * 108000, r["energy_j"], seconds)
        rows.append(
            dict(
                flight=r["flight"],
                mission=r["mission"],
                policy=r["policy"],
                battery=r["battery"],
                full_success_at_reference=r["full_success"],
                inference_seconds=seconds,
                break_even_active_cpu_power_w=limit,
                extra_energy_tolerance_j=r["reserve_margin_j"],
                reserve_at_power={str(w): bool(w <= limit + 1e-9) for w in (15, 16, 18, 20, 25)},
            )
        )
    summary = []
    for policy in p["policies"]:
        selected = [r for r in rows if r["policy"] == policy and r["battery"] == 0.34]
        successful = [r for r in selected if r["full_success_at_reference"]]
        summary.append(
            dict(
                policy=policy,
                conditions=len(selected),
                reference_successes=len(successful),
                min_power_tolerance_of_successes_w=min(
                    (r["break_even_active_cpu_power_w"] for r in successful), default=None
                ),
                recorded_successes_retaining_reserve={
                    str(w): sum(r["reserve_at_power"][str(w)] for r in successful)
                    for w in (15, 16, 18, 20, 25)
                },
            )
        )
    result = dict(
        complete=True,
        cases=rows,
        low_battery_summary=summary,
        source_sha256=sha(Path(__file__)),
        analysis_sha256=sha(root / "analysis.json"),
        scope="Post-hoc energy sensitivity of fixed observed schedules",
        adaptive_policy_rerun=False,
        mission_outcomes_under_changed_power_claimed=False,
        assumptions=(
            "All timings, non-inference costs and selected actions held fixed; "
            "only active inference power varies"
        ),
    )
    save(root / "energy_robustness.json", result)
    print(summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
