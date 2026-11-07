"""Complete factorial comparison with immutable fixed controls and baseline parity."""

from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_frustum_routing.protocol import verify
from experiments.mission_frustum_routing.study import POLICIES


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
        or v["cases"] != 32
        or v["frustum_audit_sha256"] != sha(root / "frustum_audit.json")
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
    fixed = [
        r
        for r in old["cases"]
        if r["policy"].startswith("static:") and r["mission"] in p["missions"]
    ]
    for r in fixed:
        checked_events(parent, r)
    parity = []
    for r in rows:
        if r["policy"] != "baseline":
            continue
        before = next(
            x
            for x in old["cases"]
            if x["flight"] == r["flight"]
            and x["mission"] == r["mission"]
            and x["policy"] == "base_original_risk"
        )
        prior_events = checked_events(parent, before)
        current = events[r["flight"], r["mission"], "baseline"]
        if schedule(current) != schedule(prior_events) or any(
            {k: v for k, v in a["router_decision"].items() if k != "frustum_feedback"}
            != b["router_decision"]
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
                ("frustum_full", "provenance_mean"),
                ("frustum_trials", "provenance_mean"),
                ("frustum_full", "frustum_trials"),
                ("provenance_mean", "baseline"),
                ("frustum_full", "baseline"),
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
    for name in (*POLICIES, *dict.fromkeys(r["policy"] for r in fixed)):
        subset = [r for r in [*rows, *fixed] if r["policy"] == name]
        summaries.append(
            dict(
                policy=name,
                cases=len(subset),
                successes=sum(r["full_success"] for r in subset),
                phases=sum(r["phases_succeeded"] for r in subset),
                report_recall=sum(r["report_recall"] for r in subset) / len(subset),
                reused=name.startswith("static:"),
            )
        )
    audit = read(root / "frustum_audit.json")["cases"]
    geometry_counts = []
    for name in POLICIES:
        for status in (
            "inside_frustum",
            "outside_frustum",
            "boundary_uncertain",
            "unknown_geometry",
        ):
            subset = [r for r in audit if r["policy"] == name and r["status"] == status]
            geometry_counts.append(
                dict(
                    policy=name,
                    status=status,
                    opportunities=len(subset),
                    reassociated=sum(r["prediction_reassociated"] for r in subset),
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
    primary = [x for x in paired if x["left"] == "frustum_full" and x["right"] == "provenance_mean"]
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
        sensor_bounds=old["sensor_bounds"],
        frustum_counts=geometry_counts,
        occlusion_measured=False,
        feasibility_source=str(parent),
        router_timing=timing,
        primary_comparison=dict(
            cases=8,
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
        "# Development frustum-feedback routing",
        "",
        "32 new policy replays; 72 immutable fixed-control references. "
        "Two inspected development layouts, configured costs, no fresh test.",
        "",
        "| Policy | Success / 8 | Phases / 32 | Mean report recall | Reused |",
        "|---|---:|---:|---:|---|",
    ]
    lines += [
        f"| {r['policy']} | {r['successes']} | {r['phases']} | "
        f"{r['report_recall']:.3f} | {r['reused']} |"
        for r in summaries
    ]
    q = a["primary_comparison"]
    lines += [
        "",
        f"Frustum full versus provenance mean: {q['wins']} wins, "
        f"{q['losses']} losses; {q['changed_schedules']}/8 schedules differ.",
        "",
        "All eight baseline schedules, underlying decision traces and endpoints reproduce "
        "the development selection. All four strict cases are sensor-infeasible and retained.",
        "",
        "Frustum estimates use completed predicted positions, ideal public pose and terrain. "
        "The configured uncertainty box is not calibrated. Inclusion is not visibility; "
        "reassociation is not true-target correctness. No occlusion measurement, independent "
        "generalization, hardware robustness or validated improvement is claimed.",
        "",
        "| Policy | Geometric status | Pre-action tracks | Reassociated predictions |",
        "|---|---|---:|---:|",
    ]
    lines += [
        f"| {r['policy']} | {r['status']} | {r['opportunities']} | {r['reassociated']} |"
        for r in geometry_counts
    ]
    lines.append("")
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
                    "frustum_audit.json",
                    "REPORT.md",
                )
            },
            source_sha256=sha(Path(__file__)),
        ),
    )
    print(a["primary_comparison"], flush=True)
