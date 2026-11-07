"""Report every battery cell, paired ablations and paid evaluator feasibility witnesses."""

from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.resource_routing_study.protocol import BATTERIES, verify
from experiments.resource_routing_study.study import POLICIES


def classify(certified_resource_infeasible, policy_success, witness_success):
    if certified_resource_infeasible and (policy_success or witness_success):
        raise ValueError("Resource certificate contradicts a successful execution")
    if certified_resource_infeasible:
        return "certified_resource_infeasible"
    if policy_success or witness_success:
        return "full_contract_witnessed_feasible"
    return "unresolved"


def require_grid(rows, protocol):
    expected = {
        (f, m, n) for f in protocol["cases"] for m in protocol["missions"] for n in POLICIES
    }
    actual = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicated policy grid")


def run(root):
    p = verify(root)
    validation = read(root / "validation.json")
    if (
        not validation["complete"]
        or validation["protocol_sha256"] != sha(root / "protocol.json")
        or validation["validator_sha256"] != sha(Path(__file__).with_name("validate.py"))
    ):
        raise ValueError("Independent reconstruction required")
    for name, expected in validation["source_summaries_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed validated summary")
    rows, witnesses, schedules = [], [], {}
    for case in p["cases"]:
        current = read(root / f"summary_{case}_15.json")["cases"]
        rows.extend(current)
        witness_rows = read(root / "witnesses" / case / "summary.json")["cases"]
        witnesses.extend(witness_rows)
        for r in [*current, *(v for v in witness_rows if "path" in v)]:
            for name, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                if sha(root / r["path"] / name) != r[key]:
                    raise ValueError("Changed replay artifact")
            events = read(root / r["path"] / "execution/events.json")
            schedules[case, r["mission"], r["policy"]] = [
                (e["observation_id"], e["execution"]["config_id"], e["completion_time_s"])
                for e in events
            ]
        for r in witness_rows:
            path = root / "witnesses" / case / r["mission"] / "search.json"
            if sha(path) != r["search_sha256"]:
                raise ValueError("Changed witness search")
    require_grid(rows, p)
    if validation["policy_cases"] != len(rows):
        raise ValueError("Wrong validated policy count")
    outcomes = []
    pairs = []
    for name, case in p["cases"].items():
        for mission in p["missions"]:
            selected = [r for r in rows if r["flight"] == name and r["mission"] == mission]
            witness = next(r for r in witnesses if r["flight"] == name and r["mission"] == mission)
            feasible = classify(
                witness["search_conclusion"] == "certified_resource_infeasible",
                any(r["full_success"] for r in selected),
                witness["full_contract_witness"],
            )
            outcomes.append(
                dict(
                    case=name,
                    source_flight=case["source_flight"],
                    initial_battery=case["initial_battery"],
                    mission=mission,
                    classification=feasible,
                    witness_success=witness["full_contract_witness"],
                    successful_policies=[r["policy"] for r in selected if r["full_success"]],
                )
            )
            aware = next(r for r in selected if r["policy"] == "adaptive")
            blind = next(r for r in selected if r["policy"] == "battery_blind")
            pairs.append(
                dict(
                    case=name,
                    initial_battery=case["initial_battery"],
                    mission=mission,
                    aware_success=aware["full_success"],
                    blind_success=blind["full_success"],
                    aware_reserve=aware["final_battery"] >= 0.2,
                    blind_reserve=blind["final_battery"] >= 0.2,
                    aware_report_recall=aware["report_recall"],
                    blind_report_recall=blind["report_recall"],
                    schedule_changed=schedules[name, mission, "adaptive"]
                    != schedules[name, mission, "battery_blind"],
                )
            )
    invariance = []
    for flight in {c["source_flight"] for c in p["cases"].values()}:
        cases = [name for name, c in p["cases"].items() if c["source_flight"] == flight]
        for mission in p["missions"]:
            for policy in POLICIES:
                values = [schedules[name, mission, policy] for name in cases]
                same = all(v == values[0] for v in values)
                if policy != "adaptive" and not same:
                    raise ValueError("Battery-blind or fixed control changed its schedule")
                invariance.append(
                    dict(
                        source_flight=flight,
                        mission=mission,
                        policy=policy,
                        identical_across_battery=same,
                    )
                )
    table = []
    for battery in BATTERIES:
        for policy in POLICIES:
            rs = [
                r
                for r in rows
                if r["policy"] == policy and p["cases"][r["flight"]]["initial_battery"] == battery
            ]
            table.append(
                dict(
                    initial_battery=battery,
                    policy=policy,
                    cases=len(rs),
                    successes=sum(r["full_success"] for r in rs),
                    reserve_satisfied=sum(r["final_battery"] >= 0.2 for r in rs),
                    phases=sum(r["phases_succeeded"] for r in rs),
                )
            )
    primary = [r for r in pairs if r["initial_battery"] == p["primary_battery"]]
    analysis = dict(
        complete=True,
        schema_version="resource_intervention_analysis_v1",
        scope=p["scope"],
        fresh_test=False,
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        policy_cases=rows,
        policy_summary=table,
        feasibility=outcomes,
        witnesses=witnesses,
        battery_pairs=pairs,
        control_schedule_invariance=invariance,
        primary_comparison=dict(
            battery=p["primary_battery"],
            cases=len(primary),
            aware_wins=sum(r["aware_success"] and not r["blind_success"] for r in primary),
            aware_losses=sum(r["blind_success"] and not r["aware_success"] for r in primary),
            schedule_changes=sum(r["schedule_changed"] for r in primary),
        ),
        policy_replays=len(rows),
        witness_replays=validation["witness_cases"],
        new_physical_flights=0,
        new_neural_calls=0,
        source_sha256=sha(Path(__file__)),
    )
    save(root / "analysis.json", analysis)
    lines = [
        "# Exploratory battery intervention",
        "",
        (
            "The inspected v6 physical recordings are reused. These are configured-cost "
            "interventions, not fresh tests or measured battery experiments."
        ),
        "",
        "| Initial battery | Policy | Full success / 4 | Reserve satisfied / 4 | Phases / 16 |",
        "|---|---|---:|---:|---:|",
    ]
    lines += [
        (
            f"| {100 * r['initial_battery']:g}% | {r['policy']} | {r['successes']} | "
            f"{r['reserve_satisfied']} | {r['phases']} |"
        )
        for r in table
    ]
    lines += [
        "",
        "## Evaluator feasibility",
        "",
        "| Condition | Intent | Status | Paid GT witness |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {r['case']} | {r['mission']} | {r['classification']} | {r['witness_success']} |"
        for r in outcomes
    ]
    q = analysis["primary_comparison"]
    lines += [
        "",
        (
            f"At the predeclared 34% condition, adaptive routing has {q['aware_wins']} "
            f"full-success wins and {q['aware_losses']} losses versus its battery-blind "
            f"control across four correlated conditions. "
            f"Schedules differ in {q['schedule_changes']}/4."
        ),
        "",
        (
            "All battery-blind and fixed-control schedules are identical across battery levels. "
            "Only the real runner's budget differs for those controls. Successful GT-aware "
            "schedules establish feasibility and are excluded from policy comparisons; "
            "unsuccessful bounded search remains unresolved."
        ),
        "",
        *["- " + v for v in p["caveats"]],
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    save(
        root / "completion.json",
        dict(
            complete=True,
            protocol_sha256=sha(root / "protocol.json"),
            analysis_sha256=sha(root / "analysis.json"),
            validation_sha256=sha(root / "validation.json"),
            report_sha256=sha(root / "REPORT.md"),
            policy_replays=len(rows),
            witness_replays=validation["witness_cases"],
            public_decisions=validation["public_decisions_reproduced"],
            witness_decisions=validation["witness_decisions_reproduced"],
            fresh_test=False,
            research_readiness_certified=False,
        ),
    )
    print(analysis["primary_comparison"], flush=True)
    return analysis
