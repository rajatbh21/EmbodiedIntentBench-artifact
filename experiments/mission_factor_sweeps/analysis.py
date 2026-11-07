"""Predeclared baseline-controlled slices and finite differences of four factors.

Every slice is a view of one complete paired factorial execution bank. Overlapping
memberships are never counted as independent runs. Inclusion-exclusion contrasts
identify descriptive interactions, without confidence intervals or population
claims from the two deterministic recordings. Intent changes the task contract;
intent-involving contrasts are distinguished from resource-only comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

FACTORS = ("deadline", "network", "battery", "intent")
LEVELS = {
    "deadline": ("perception_2s", "perception_4s", "perception_8s"),
    "network": ("good", "constrained", "intermittent"),
    "battery": (0.34, 0.65),
    "intent": ("rapid_survey", "verified_inspection"),
}
BASELINE = {
    "deadline": "perception_8s",
    "network": "good",
    "battery": 0.65,
    "intent": "rapid_survey",
}
FLIGHTS = ("east_shore_day", "west_rocks_day")
POLICIES = ("fixed_local_fast", "fixed_local_heavy", "fixed_remote_union", "deadline_aware")
METRICS = (
    "mission_success",
    "phase_fraction",
    "report_recall",
    "final_battery_frac",
    "energy_j",
    "communication_mb",
    "perception_admission_rate",
    "perception_deadline_miss_rate",
    "perception_attempts",
    "perception_deadline_misses",
    "association_deadline_misses",
    "verification_deadline_misses",
    "capture_coverage",
    "admitted_capture_fraction",
    "discarded_task_energy_j",
    "abstentions",
)
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/mission_factor_sweeps_v1_20261028"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def specification():
    slices = []
    blocks = len(FLIGHTS) * len(POLICIES)
    for order in range(1, 5):
        for selected in itertools.combinations(FACTORS, order):
            cells = math.prod(len(LEVELS[k]) for k in selected)
            slices.append(
                dict(
                    slice_id="+".join(selected),
                    order=order,
                    varied_factors=list(selected),
                    held_fixed={k: BASELINE[k] for k in FACTORS if k not in selected},
                    condition_count=cells,
                    execution_memberships=cells * blocks,
                )
            )
    partition = [blocks]
    for order in range(1, 5):
        partition.append(
            blocks
            * sum(
                math.prod(len(LEVELS[k]) - 1 for k in selected)
                for selected in itertools.combinations(FACTORS, order)
            )
        )
    return dict(
        schema_version="mission_factor_sweep_analysis_spec_v1",
        factors=list(FACTORS),
        levels={k: list(v) for k, v in LEVELS.items()},
        baseline=dict(BASELINE),
        flights=list(FLIGHTS),
        policies=list(POLICIES),
        expected_runs=math.prod(map(len, LEVELS.values())) * blocks,
        slices=slices,
        slice_memberships=sum(s["execution_memberships"] for s in slices),
        exact_change_counts=partition,
        anchored_contrast_counts=partition[1:],
        metric_names=list(METRICS),
        controlled_deadline_scope="Only perception invocation budget varies. Association "
        "150 ms and verification 30 ms remain fixed, along with all other execution costs.",
        intent_scope="An intent-profile intervention changes reference requirements; "
        "verified_inspection remains a known sensor-infeasible control on these recordings.",
        network_scope="Categorical public/private link profiles, not an ordinal severity scale.",
        comparison_scope="All 15 baseline-controlled slices share the same 288 fresh replays. "
        "Conditions outside each selected factor set remain at the declared baseline.",
        interaction_definition="For selected nonbaseline levels S, sum over T subset S of "
        "(-1)^(|S|-|T|) times the outcome with T at selected levels and every other factor "
        "at baseline. An order-one contrast is a main effect; higher orders are interactions.",
        units="Matched recording/policy blocks; overlapping deterministic contrasts are "
        "descriptive, not independent samples. No run-level p-values or confidence intervals.",
        timing_rate_scope="Report per-run admission rates together with attempt counts and "
        "potential-capture coverage. Undefined zero-attempt rates remain null, never zero.",
        discarded_energy_scope="Nonflight energy charged to selected discarded tasks, "
        "including retained loading cost; excludes baseline flight and explicit abstentions. "
        "This is not a claim that all these costs could have been avoided.",
        comparison_with_previous="Earlier outcomes are known. New 4s level and fixed "
        "downstream budgets are declared before new executions; policies are not tuned.",
    )


def factors_for(item):
    return dict(
        deadline=item["deadline_profile"],
        network=item["link"],
        battery=item["initial_battery_frac"],
        intent=item["mission"],
    )


def cell_key(cell):
    return (cell["flight"], cell["policy"], *(cell["factors"][k] for k in FACTORS))


def validate_cells(cells):
    expected = set(itertools.product(FLIGHTS, POLICIES, *(LEVELS[k] for k in FACTORS)))
    keys = [cell_key(c) for c in cells]
    require(
        len(keys) == len(set(keys)) and set(keys) == expected, "Incomplete/duplicate factor grid"
    )
    require(len({c["run_id"] for c in cells}) == len(cells), "Duplicate run identity")
    for cell in cells:
        require(set(cell["metrics"]) == set(METRICS), "Wrong per-run metric schema")
        require(set(cell["factors"]) == set(FACTORS), "Unknown factor field")
        changed = [k for k in FACTORS if cell["factors"][k] != BASELINE[k]]
        require(
            cell["changed_factors"] == changed and cell["change_order"] == len(changed),
            "Wrong exact change-order partition",
        )
        require(
            cell["known_infeasible_intent"] == (cell["factors"]["intent"] == "verified_inspection"),
            "Wrong known-infeasible intent label",
        )
        for value in cell["metrics"].values():
            require(
                value is None or (type(value) in (int, float) and math.isfinite(value)),
                "Nonfinite/invalid metric",
            )


def aggregate(cells):
    means, counts = {}, {}
    for metric in METRICS:
        values = [c["metrics"][metric] for c in cells if c["metrics"][metric] is not None]
        counts[metric] = len(values)
        means[metric] = math.fsum(values) / len(values) if values else None
    return dict(
        runs=len(cells),
        mission_successes=sum(c["metrics"]["mission_success"] for c in cells),
        mean_metrics=means,
        metric_counts=counts,
    )


def policy_summaries(cells):
    result = []
    for policy in POLICIES:
        selected = [c for c in cells if c["policy"] == policy]
        result.append(
            dict(
                policy=policy,
                **aggregate(selected),
                intent_breakdown={
                    level: aggregate([c for c in selected if c["factors"]["intent"] == level])
                    for level in LEVELS["intent"]
                },
            )
        )
    return result


def controlled_slices(cells):
    output = []
    for description in specification()["slices"]:
        selected = [
            c
            for c in cells
            if all(c["factors"][k] == v for k, v in description["held_fixed"].items())
        ]
        require(len(selected) == description["execution_memberships"], "Wrong controlled slice")
        output.append(
            dict(
                **description,
                run_ids=[c["run_id"] for c in selected],
                policy_summaries=policy_summaries(selected),
            )
        )
    return output


def interactions(cells):
    lookup = {cell_key(c): c for c in cells}
    result = []
    for order in range(1, 5):
        for selected in itertools.combinations(FACTORS, order):
            for values in itertools.product(
                *(tuple(v for v in LEVELS[k] if v != BASELINE[k]) for k in selected)
            ):
                levels = dict(zip(selected, values, strict=True))
                for flight, policy in itertools.product(FLIGHTS, POLICIES):
                    terms, corners = [], []
                    for active in itertools.product((False, True), repeat=order):
                        condition = dict(BASELINE)
                        condition.update(
                            {k: levels[k] for k, on in zip(selected, active, strict=True) if on}
                        )
                        corner = lookup[(flight, policy, *(condition[k] for k in FACTORS))]
                        sign = (-1) ** (order - sum(active))
                        terms.append(dict(run_id=corner["run_id"], coefficient=sign))
                        corners.append((corner, sign))
                    effects = {}
                    for metric in METRICS:
                        values = [c["metrics"][metric] for c, _ in corners]
                        effects[metric] = (
                            None
                            if any(v is None for v in values)
                            else math.fsum(c["metrics"][metric] * sign for c, sign in corners)
                        )
                    result.append(
                        dict(
                            contrast_id=f"contrast_{len(result):04d}",
                            order=order,
                            varied_factors=list(selected),
                            levels=levels,
                            flight=flight,
                            policy=policy,
                            involves_intent="intent" in selected,
                            corner_terms=terms,
                            effects=effects,
                        )
                    )
    return result


def interaction_summary(rows):
    result = []
    for order in range(1, 5):
        selected = [r for r in rows if r["order"] == order]
        resources = [r for r in selected if not r["involves_intent"]]

        def count(group):
            return {
                m: sum(r["effects"][m] is not None and abs(r["effects"][m]) > 1e-10 for r in group)
                for m in METRICS
            }

        result.append(
            dict(
                order=order,
                contrasts=len(selected),
                nonzero_by_metric=count(selected),
                resource_only_contrasts=len(resources),
                resource_only_nonzero_by_metric=count(resources),
            )
        )
    return result


def build_analysis(cells, sources):
    validate_cells(cells)
    contrasts = interactions(cells)
    return dict(
        schema_version="mission_factor_sweep_analysis_v1",
        complete=True,
        runs=len(cells),
        design=specification(),
        source_sha256=sources,
        cells=cells,
        slices=controlled_slices(cells),
        exact_change_groups=[
            dict(
                order=order,
                runs=sum(c["change_order"] == order for c in cells),
                policy_summaries=policy_summaries([c for c in cells if c["change_order"] == order]),
            )
            for order in range(5)
        ],
        interactions=contrasts,
        interaction_summary=interaction_summary(contrasts),
    )


def extract_cell(item, events, outcome, result, potential_captures):
    require(
        type(potential_captures) is int and potential_captures > 0, "Positive capture bank required"
    )
    task = outcome["task_metrics"]
    attempts = task["perception"]["attempts"]
    discarded = []
    for event in events:
        if event["selected_action_id"] is not None and not event["perception"]["admitted"]:
            discarded.append(event["perception"]["costs"]["charged_energy_j"])
        if event["pipeline"] is not None:
            discarded.extend(
                s["costs"]["charged_energy_j"]
                for s in event["pipeline"]["stages"]
                if s["dispatched"] and not s["committed"]
            )
    require(len(events) == result["processed_observations"], "Changed processed capture count")
    metrics = dict(
        mission_success=int(outcome["mission_success"]),
        phase_fraction=sum(p["success"] for p in outcome["phases"]) / len(outcome["phases"]),
        report_recall=outcome["reporting"]["recall_by_deadline"],
        final_battery_frac=result["final_battery_frac"],
        energy_j=result["cumulative_energy_j"],
        communication_mb=result["cumulative_communication_mb"],
        perception_admission_rate=task["perception"]["on_time"] / attempts if attempts else None,
        perception_deadline_miss_rate=task["perception"]["deadline_misses"] / attempts
        if attempts
        else None,
        perception_attempts=attempts,
        perception_deadline_misses=task["perception"]["deadline_misses"],
        association_deadline_misses=task["association"]["deadline_misses"],
        verification_deadline_misses=task["verification"]["deadline_misses"],
        capture_coverage=len(events) / potential_captures,
        admitted_capture_fraction=task["perception"]["on_time"] / potential_captures,
        discarded_task_energy_j=math.fsum(discarded),
        abstentions=outcome["abstentions"],
    )
    factors = factors_for(item)
    changed = [k for k in FACTORS if factors[k] != BASELINE[k]]
    return dict(
        run_id=item["run_id"],
        flight=item["flight"],
        policy=item["policy"],
        factors=factors,
        changed_factors=changed,
        change_order=len(changed),
        known_infeasible_intent=factors["intent"] == "verified_inspection",
        metrics=metrics,
    )


def verify_inputs(root):
    # Imported only during real analysis; specification/math are standard-library only.
    from experiments.mission_factor_sweeps.execution import verify

    root = Path(root).resolve()
    plan, _ = verify(root)
    require(read(root / "analysis_design.json") == specification(), "Changed predeclared analysis")
    index, completion, audit = (
        read(root / n)
        for n in ("run_index.json", "completion.json", "independent_deadline_audit.json")
    )
    require(
        completion["complete"] is True and audit["passed"] is True,
        "Completed audited grid required",
    )
    require(index["complete_grid"] is True, "Incomplete execution index")
    require(completion["runs"] == audit["runs"] == len(index["cases"]) == 288, "Wrong run count")
    require(
        index["plan_sha256"] == audit["plan_sha256"] == sha(root / "plan.json"),
        "Plan binding mismatch",
    )
    require(
        audit["completion_sha256"] == sha(root / "completion.json"), "Completion binding mismatch"
    )
    require(
        sha(audit["audit_source_path"]) == audit["audit_source_sha256"], "Changed receipt auditor"
    )
    for name, expected in completion["artifacts_sha256"].items():
        require(sha(root / name) == expected, "Changed completed artifact")
    require(len(plan["grid"]) == 288, "Wrong frozen design size")
    bindings, cells = {}, []
    for item, row in zip(plan["grid"], index["cases"], strict=True):
        require(all(row[k] == v for k, v in item.items()), "Grid identity changed")
        directory = (root / row["path"]).resolve()
        require(directory.is_relative_to(root), "Run path escaped output")
        for name, expected in row["artifacts_sha256"].items():
            path = (directory / name).resolve()
            require(
                path.is_relative_to(directory) and sha(path) == expected, "Changed run artifact"
            )
            bindings[str(path.relative_to(root))] = expected
        events, result, outcome = (
            read(directory / (name + ".json")) for name in ("events", "results", "outcome")
        )
        protocol = read(directory / "protocol.json")
        require(protocol["plan_sha256"] == sha(root / "plan.json"), "Per-run plan mismatch")
        scenario = read(
            Path(plan["parent_root"])
            / "missions"
            / item["cell"]
            / item["mission"]
            / "scenario.json"
        )
        potential = math.ceil(
            scenario["mission"]["duration_s"] / scenario["mission"]["observation_interval_s"]
        )
        cells.append(extract_cell(item, events, outcome, result, potential))
    require(bindings == audit["audited_artifacts_sha256"], "Independent audit inventory mismatch")
    sources = {
        name: sha(root / name)
        for name in (
            "plan.json",
            "orchestration_manifest.json",
            "analysis_design.json",
            "run_index.json",
            "summary.json",
            "validation.json",
            "completion.json",
            "independent_deadline_audit.json",
        )
    }
    return cells, sources


def analyze(root=OUTPUT):
    root = Path(root).resolve()
    require(not (root / "analysis.json").exists(), "Preserve previous analysis output")
    cells, sources = verify_inputs(root)
    output = build_analysis(cells, sources)
    with (root / "analysis.json").open("x") as stream:
        json.dump(output, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = analyze(args.output)
    print(
        json.dumps(
            dict(
                runs=result["runs"],
                slices=len(result["slices"]),
                interactions=len(result["interactions"]),
            )
        )
    )


if __name__ == "__main__":
    main()
