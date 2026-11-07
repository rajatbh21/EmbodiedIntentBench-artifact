"""Complete factorial comparison with immutable fixed controls and baseline parity."""

from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_support_routing.protocol import verify
from experiments.mission_support_routing.study import POLICIES


def schedule(events):
    return [
        (e["observation_id"], e["execution"]["config_id"], e["completion_time_s"]) for e in events
    ]


def checked_events(root, row):
    for name, key in (
        ("outcome.json", "outcome_sha256"),
        ("execution/events.json", "events_sha256"),
        ("execution/results.json", "results_sha256"),
    ):
        if sha(root / row["path"] / name) != row[key]:
            raise ValueError("Changed execution artifact")
    return read(root / row["path"] / "execution/events.json")


def run(root):
    p = verify(root)
    v = read(root / "validation.json")
    if (
        not v["complete"]
        or v["policy_cases"] != 48
        or v["witness_cases"] != 0
        or v["protocol_sha256"] != sha(root / "protocol.json")
        or v["validator_sha256"] != sha(Path(__file__).with_name("validate.py"))
    ):
        raise ValueError("Complete independent reconstruction required")
    for name, expected in v["source_summaries_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed validated summary")
    rows = [r for case in p["cases"] for r in read(root / f"summary_{case}_15.json")["cases"]]
    keys = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    expected = {(f, m, n) for f in p["cases"] for m in p["missions"] for n in POLICIES}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate factorial grid")
    parent = Path(p["parent"])
    old = read(parent / "analysis.json")
    events = {(r["flight"], r["mission"], r["policy"]): checked_events(root, r) for r in rows}
    fixed = [r for r in old["policy_cases"] if r["policy"].startswith("static:")]
    for r in fixed:
        checked_events(parent, r)
    parity = []
    for r in rows:
        if r["policy"] != "baseline":
            continue
        before = next(
            x
            for x in old["policy_cases"]
            if x["flight"] == r["flight"]
            and x["mission"] == r["mission"]
            and x["policy"] == "adaptive"
        )
        prior_events = checked_events(parent, before)
        current = events[r["flight"], r["mission"], "baseline"]
        if schedule(current) != schedule(prior_events) or any(
            a["router_decision"] != b["router_decision"]
            for a, b in zip(current, prior_events, strict=True)
        ):
            raise ValueError("Baseline decision parity failed")
        for key in (
            "full_success",
            "phases_succeeded",
            "report_recall",
            "failed_constraints",
            "final_battery",
            "selections",
        ):
            if r[key] != before[key]:
                raise ValueError("Baseline outcome parity failed")
        parity.append(dict(case=r["flight"], mission=r["mission"], identical=True))
    paired = []
    for case in p["cases"]:
        for mission in p["missions"]:
            subset = {
                r["policy"]: r for r in rows if r["flight"] == case and r["mission"] == mission
            }
            for left, right in (
                ("provenance_weakest", "baseline"),
                ("provenance_weakest", "weakest_support"),
                ("provenance_weakest", "provenance_mean"),
                ("provenance_mean", "baseline"),
                ("weakest_support", "baseline"),
            ):
                a, b = subset[left], subset[right]
                paired.append(
                    dict(
                        case=case,
                        mission=mission,
                        left=left,
                        right=right,
                        success_delta=int(a["full_success"]) - int(b["full_success"]),
                        phase_delta=a["phases_succeeded"] - b["phases_succeeded"],
                        report_recall_delta=a["report_recall"] - b["report_recall"],
                        schedule_changed=schedule(events[case, mission, left])
                        != schedule(events[case, mission, right]),
                    )
                )
    summaries = []
    for battery in p["batteries"]:
        for name in (*POLICIES, *dict.fromkeys(r["policy"] for r in fixed)):
            subset = [
                r
                for r in [*rows, *fixed]
                if r["policy"] == name and p["cases"][r["flight"]]["initial_battery"] == battery
            ]
            summaries.append(
                dict(
                    battery=battery,
                    policy=name,
                    cases=len(subset),
                    successes=sum(r["full_success"] for r in subset),
                    phases=sum(r["phases_succeeded"] for r in subset),
                    reserve_satisfied=sum(r["final_battery"] >= 0.2 for r in subset),
                    report_recall=sum(r["report_recall"] for r in subset) / len(subset),
                    reused=name.startswith("static:"),
                )
            )
    timing = []
    for name in POLICIES:
        es = [e for key, stream in events.items() if key[2] == name for e in stream]
        samples = [e["measured_router_wall_clock_s"] for e in es]
        timing.append(
            dict(
                policy=name,
                decisions=len(es),
                p95_s=float(np.percentile(samples, 95)),
                max_s=max(samples),
                configured_s=0.05,
                measurements_above_configured=sum(t > 0.05 for t in samples),
                target_device_measurement=False,
            )
        )
    primary = [x for x in paired if x["left"] == "provenance_weakest" and x["right"] == "baseline"]
    a = dict(
        complete=True,
        scope=p["scope"],
        fresh_test=False,
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        policy_cases=rows,
        reused_fixed_controls=fixed,
        fixed_controls_source=str(parent),
        policy_summary=summaries,
        pairs=paired,
        baseline_parity=parity,
        feasibility=old["feasibility"],
        feasibility_source=str(parent),
        router_timing=timing,
        primary_comparison=dict(
            cases=12,
            wins=sum(x["success_delta"] > 0 for x in primary),
            losses=sum(x["success_delta"] < 0 for x in primary),
            changed_schedules=sum(x["schedule_changed"] for x in primary),
        ),
        new_policy_replays=len(rows),
        reused_fixed_replays=len(fixed),
        new_physical_flights=0,
        new_neural_calls=0,
        promote_as_validated_improvement=False,
        source_sha256=sha(Path(__file__)),
    )
    save(root / "analysis.json", a)
    lines = [
        "# Exploratory prediction-support routing",
        "",
        "48 new policy replays; 108 immutable fixed-control references. "
        "Inspected physical recordings, configured costs, no fresh test.",
        "",
        "| Battery | Policy | Success / 4 | Phases / 16 | Reserve / 4 | Reused |",
        "|---|---|---:|---:|---:|---|",
    ]
    lines += [
        f"| {100 * r['battery']:g}% | {r['policy']} | {r['successes']} | {r['phases']} | "
        f"{r['reserve_satisfied']} | {r['reused']} |"
        for r in summaries
    ]
    q = a["primary_comparison"]
    lines += [
        "",
        f"Combined policy versus baseline: {q['wins']} wins, {q['losses']} losses; "
        f"{q['changed_schedules']}/12 schedules differ.",
        "",
        "All 12 baseline decision traces and endpoints match the previous study. "
        "Reference witnesses are evaluator-only. Unresolved conditions remain included.",
        "",
        "Public prediction history cannot reveal never-detected targets. "
        "Reassociation frequency is not correctness, visibility or calibrated recall. "
        "No generalization or hardware robustness claim is supported.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    save(
        root / "completion.json",
        dict(
            complete=True,
            fresh_test=False,
            artifacts_sha256={
                name: sha(root / name)
                for name in (
                    "protocol.json",
                    "analysis.json",
                    "validation.json",
                    "REPORT.md",
                )
            },
            source_sha256=sha(Path(__file__)),
        ),
    )
    print(a["primary_comparison"], flush=True)
