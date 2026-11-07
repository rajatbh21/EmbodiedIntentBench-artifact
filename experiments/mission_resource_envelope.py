"""Cost-only resource screening for future protocols; no perception or mission feasibility claim."""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict
from pathlib import Path

from experiments.lake_causal_router import MissionCosts
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST
from experiments.mission_policy_selection.protocol import verify
from experiments.mission_policy_selection.study import scenario_for
from experiments.mission_routing_v3.budget import Action, Continuation
from experiments.v4_gazebo_pilot.timing import seconds_to_ns

POWERS = (15, 30, 45, 60)
BATTERIES = (0.33, 0.335, 0.34, 0.35, 0.40, 0.65)


def audit_path(costs, actions, path, deadline):
    """Recompute DP witness timing and total costs without Continuation.transition."""
    schedule = costs.schedule()
    by_id = {a.config_id: a for a in actions}
    index, previous, total, end = 0, None, 0.0, 0.0
    for i, config, completion, _ in path:
        if i != index or schedule.at(i) >= schedule.end_ns:
            raise ValueError("Illegal cost witness capture")
        action = by_id[config]
        switch = config != previous
        delay = action.latency_s + (action.load_latency_s if switch else 0)
        finish, next_index, _ = schedule.complete(i, seconds_to_ns(delay, allow_zero=True))
        end = finish / 1e9
        if abs(end - completion) > 1e-8 or end > deadline:
            raise ValueError("Cost witness timing mismatch")
        total += action.energy_j + (action.load_energy_j if switch else 0)
        previous, index = config, next_index
    if not path or schedule.at(index) < schedule.end_ns:
        raise ValueError("Incomplete cost witness")
    total += max(costs.duration_s, end) * costs.flight_power_w
    return total


def solve(costs, actions, deadline, reserve):
    if not 0 <= reserve <= 1:
        raise ValueError("Invalid reserve fraction")
    dp = Continuation(costs, actions, deadline)
    value = dp.values[0, None]
    if not math.isfinite(value):
        return dict(resource_feasible=False, minimum_initial_battery=None)
    path = dp.witness(0, None, dp.choices[0, None])
    actual = audit_path(costs, actions, path, deadline)
    if abs(value - actual) > 1e-6:
        raise ValueError("Dynamic-programming cost differs from independent audit")
    return dict(
        resource_feasible=reserve + value / costs.capacity_j <= 1,
        minimum_energy_j=value,
        minimum_initial_battery=reserve + value / costs.capacity_j,
        actions=len(path),
        configs=sorted({r[1] for r in path}),
        witness=path,
    )


def run(selection, output):
    if output.exists():
        raise FileExistsError("Preserve previous resource analysis")
    p = verify(selection)
    flight = sorted(p["cases"])[0]
    tracking = DEFAULT_TRACKING_COST
    rows, grid = [], []
    for power in POWERS:
        scenario = scenario_for(selection, p, flight, "rapid_survey", power)
        costs = MissionCosts.from_scenario(scenario)
        actions = []
        for spec in scenario.executors:
            resource = scenario.resources.for_config(spec.config_id)
            if resource.memory_mb + tracking.memory_mb > scenario.resources.memory_capacity_mb:
                continue
            actions.append(
                Action(
                    spec.config_id,
                    spec.mission_latency_s + costs.router_latency_s + tracking.latency_s,
                    spec.energy_j_per_call + costs.router_energy_j + tracking.energy_j,
                    resource.load_latency_s,
                    resource.load_energy_j,
                )
            )
        deadline, reserve = scenario.contract.deadline_s, scenario.contract.min_final_battery_frac
        mixed = solve(costs, actions, deadline, reserve)
        fixed = {a.config_id: solve(costs, [a], deadline, reserve) for a in actions}
        first = {
            a.config_id: reserve
            + Continuation(costs, actions, deadline).required(0, None, a.config_id)
            / costs.capacity_j
            for a in actions
        }
        rows.append(
            dict(
                power_w=power,
                mixed=mixed,
                fixed=fixed,
                minimum_battery_by_first_action=first,
                costs=asdict(costs),
                tracking=asdict(tracking),
                actions=[asdict(a) for a in actions],
            )
        )
        for battery in BATTERIES:
            feasible = [
                c
                for c, r in fixed.items()
                if r["minimum_initial_battery"] is not None
                and r["minimum_initial_battery"] <= battery + 1e-12
            ]
            first_feasible = [c for c, b in first.items() if b <= battery + 1e-12]
            grid.append(
                dict(
                    power_w=power,
                    initial_battery=battery,
                    fixed_resource_feasible=feasible,
                    first_action_with_mixed_continuation_feasible=first_feasible,
                    mixed_resource_feasible=mixed["minimum_initial_battery"] <= battery + 1e-12,
                )
            )
    output.mkdir(parents=True)
    sources = [
        Path(__file__),
        Path(__file__).parent / "mission_routing_v3/budget.py",
        Path(__file__).parent / "v4_gazebo_pilot/timing.py",
        Path(__file__).parent / "lake_causal_router.py",
        Path(__file__).parent / "lake_tracking_runtime.py",
    ]
    result = dict(
        complete=True,
        scope="configured_resource_design_only",
        prediction_labels_used=False,
        new_policy_runs=0,
        mission_feasibility_proven=False,
        energy_measured=False,
        selection_protocol_sha256=sha(selection / "protocol.json"),
        source_sha256={str(s.resolve()): sha(s) for s in sources},
        powers=list(POWERS),
        batteries=list(BATTERIES),
        rows=rows,
        grid=grid,
        caveats=[
            "Timing is held fixed as configured power changes",
            "No model accuracy, report correctness or phase evidence constraint is solved",
            "Same 120-second flight and 30-Wh capacity",
            "Always executing a model at the next available capture; no idle or abstain action",
        ],
    )
    save(output / "envelope.json", result)
    lines = [
        "# Configured resource envelope",
        "",
        (
            "Cost-only analytical screening for a future protocol. No new physical flight, "
            "inference or policy mission is run. All finite minimum-energy schedules "
            "are independently reconstructed."
        ),
        "",
        (
            "| Active model power | Initial battery | Feasible fixed configurations / 9 "
            "| Feasible first actions / 9 |"
        ),
        "|---|---:|---:|---:|",
    ]
    lines += [
        (
            f"| {r['power_w']} W | {r['initial_battery']:.3f} | "
            f"{len(r['fixed_resource_feasible'])} | "
            f"{len(r['first_action_with_mixed_continuation_feasible'])} |"
        )
        for r in grid
    ]
    lines += [
        "",
        (
            "The first-action column permits an optimal mixed-model continuation; "
            "it is not a guarantee that the router will follow it or meet perception requirements. "
            "A cost-infeasible case can be identified, but a cost-feasible case is not "
            "certified mission-feasible."
        ),
        "",
        *["- " + s for s in result["caveats"]],
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    save(
        output / "completion.json",
        dict(
            complete=True,
            envelope_sha256=sha(output / "envelope.json"),
            report_sha256=sha(output / "REPORT.md"),
            source_sha256=sha(Path(__file__)),
        ),
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.selection, args.output)
