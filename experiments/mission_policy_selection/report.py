"""Select on layout-macro mission outcomes; never label selection scores as test results."""

from collections import Counter
from pathlib import Path
from statistics import mean

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_policy_selection.protocol import verify
from experiments.mission_policy_selection.study import POLICIES
from experiments.mission_routing_v4_report import risk_audit


def rank(rows, cases, missions, policies):
    expected = {(f, m, n) for f in cases for m in missions for n in policies}
    actual = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Missing, duplicate or undeclared selection cases")
    if any(
        type(r["full_success"]) is not bool
        or not 0 <= r["phases_succeeded"] <= 4
        or not 0 <= r["report_recall"] <= 1
        for r in rows
    ):
        raise ValueError("Invalid selection endpoint")
    groups = sorted({c["layout_group"] for c in cases.values()})
    table = []
    for policy in policies:
        selected = [r for r in rows if r["policy"] == policy]
        blocks = []
        for group in groups:
            block = [r for r in selected if cases[r["flight"]]["layout_group"] == group]
            blocks.append(
                dict(
                    group=group,
                    full_success=mean(r["full_success"] for r in block),
                    phase_success=mean(r["phases_succeeded"] / 4 for r in block),
                    report_recall=mean(r["report_recall"] for r in block),
                )
            )
        table.append(
            dict(
                policy=policy,
                cases=len(selected),
                successes=sum(r["full_success"] for r in selected),
                phases=sum(r["phases_succeeded"] for r in selected),
                macro_full_success=mean(b["full_success"] for b in blocks),
                macro_phase_success=mean(b["phase_success"] for b in blocks),
                macro_report_recall=mean(b["report_recall"] for b in blocks),
                blocks=blocks,
            )
        )
    return sorted(
        table,
        key=lambda r: (
            -r["macro_full_success"],
            -r["macro_phase_success"],
            -r["macro_report_recall"],
            r["policy"],
        ),
    )


def run(root):
    p = verify(root)
    validation = read(root / "validation.json")
    if (
        not validation["complete"]
        or validation["protocol_sha256"] != sha(root / "protocol.json")
        or validation["validator_sha256"] != sha(Path(__file__).with_name("validate.py"))
    ):
        raise ValueError("Independent reconstruction required before selection")
    rows, audits, contrasts = [], [], []
    for flight in p["cases"]:
        summary = root / f"summary_{flight}_15.json"
        if sha(summary) != validation["source_summaries_sha256"][summary.name]:
            raise ValueError("Changed validated summary")
        current = read(summary)["cases"]
        rows.extend(current)
        schedules = {}
        for row in current:
            folder = root / row["path"]
            for filename, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                if sha(folder / filename) != row[key]:
                    raise ValueError("Changed validated execution")
            events = read(folder / "execution/events.json")
            schedules[row["mission"], row["policy"]] = [
                (e["observation_id"], e["execution"]["config_id"], e["completion_time_s"])
                for e in events
            ]
            if row["policy"] in ("base_recall", "base_original_risk"):
                phases = read(root / flight / row["mission"] / "intent.json")["mission"]["phases"]
                audits.append(
                    dict(
                        flight=flight,
                        mission=row["mission"],
                        policy=row["policy"],
                        audit=risk_audit(events, phases),
                    )
                )
        for policy in POLICIES:
            contrasts.append(
                dict(
                    flight=flight,
                    policy=policy,
                    identical_schedule=schedules["rapid_survey", policy]
                    == schedules["reserve_patrol", policy],
                )
            )
    if validation["cases"] != len(rows):
        raise ValueError("Validated case count differs")
    ranking = rank(rows, p["cases"], p["missions"], POLICIES)
    adaptive = [
        r
        for r in ranking
        if r["policy"] in ("base_recall", "spatial_recall", "global_recall", "base_original_risk")
    ]
    selected = ranking[0]["policy"]
    training = Path(p["training"])
    # Profiles trained on every development block are for a future frozen test.
    # These are never substituted for out-of-fold profiles in this report.
    deployment = dict(
        recall=str(training / "all_development/recall.json"),
        risk=str(training / "all_development/risk.json"),
    )
    if selected == "base_original_risk":
        deployment["risk"] = str(Path(p["original_risk"]) / "risk_profile.json")
    if selected.startswith("static:"):
        deployment = dict(config_id=selected.removeprefix("static:"))
    elif selected == "periodic_training_top_two":
        model = read(training / "all_development/recall.json")["models"]
        deployment = dict(
            config_ids=sorted(model, key=lambda c: (-model[c]["base"]["global_mean"], c))[:2]
        )
    analysis = dict(
        schema_version="mission_policy_selection_analysis_v1",
        complete=True,
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        scope=p["scope"],
        fresh_test=False,
        independent_maps=0,
        replay_layout_blocks=2,
        training_layout_blocks=3,
        selection_result=dict(
            selected_policy=selected,
            best_adaptive_candidate=adaptive[0]["policy"],
            deployment=deployment,
            deployment_evaluated_on_fresh_test=False,
        ),
        ranking=ranking,
        cases=rows,
        risk_audits=audits,
        battery_schedule_contrasts=contrasts,
        sensor_bounds=read(Path(p["cohort"]) / "observability.json")["cases"],
        all_cases_retained=True,
        number_of_policy_replays=len(rows),
        failed_constraint_counts=dict(Counter(c for r in rows for c in r["failed_constraints"])),
        limitations=p["caveats"],
    )
    save(root / "analysis.json", analysis)
    lines = [
        "# Layout-blocked mission policy selection",
        "",
        (
            "Development selection only. These scores are not an unbiased estimate of the "
            "selected policy's test performance."
        ),
        "",
        f"Selected candidate: `{selected}`. Best adaptive candidate: `{adaptive[0]['policy']}`.",
        "",
        (
            "Recall and risk are fitted without either lighting flight of each held-out layout. "
            "Two layouts support paid mission replays; a third original block contributes training "
            "and count diagnostics only. All strict-inspection cases are sensor-infeasible "
            "and remain in every denominator."
        ),
        "",
        (
            "The fixed order is macro layout full-contract success, phase success, report recall, "
            "then ascending policy name. No test result or energy tie-break is used."
        ),
        "",
        "| Policy | Full success / 12 | Phases / 48 | Macro report recall |",
        "|---|---:|---:|---:|",
    ]
    lines += [
        f"| {r['policy']} | {r['successes']} | {r['phases']} | {r['macro_report_recall']:.3f} |"
        for r in ranking
    ]
    lines += [
        "",
        (
            f"Reconstructed {validation['public_decisions_reproduced']:,} decisions in {len(rows)} "
            "paid replays. No new neural inference or physical acquisition was performed; "
            "the existing 240 captures and 4,080 actual dense model calls are reused."
        ),
        "",
        (
            f"Rapid/reserve schedules are identical in "
            f"{sum(c['identical_schedule'] for c in contrasts)}/{len(contrasts)} "
            "paired policy/flight comparisons. This is a descriptive activation check, "
            "not proof of useful battery adaptation."
        ),
        "",
        "## Limits",
        "",
        *["- " + s for s in p["caveats"]],
        "",
        (
            "The next evaluation must freeze the selected candidate before collecting a new "
            "sensor-valid cohort. Do not relabel v6 or these inspected development recordings "
            "as untouched tests."
        ),
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    save(
        root / "completion.json",
        dict(
            complete=True,
            protocol_sha256=sha(root / "protocol.json"),
            analysis_sha256=sha(root / "analysis.json"),
            report_sha256=sha(root / "REPORT.md"),
            validation_sha256=sha(root / "validation.json"),
            source_sha256=sha(Path(__file__)),
            fresh_test=False,
            research_readiness_certified=False,
        ),
    )
    print(f"Development selection complete: {selected}; {len(rows)} replays", flush=True)
    return analysis
