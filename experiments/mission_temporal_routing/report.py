"""Complete comparisons, causal ablations, negative controls, and reusable figures."""

from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_strategy_routing.report import checked_events, schedule
from experiments.mission_temporal_routing.protocol import verify
from experiments.mission_temporal_routing.study import POLICIES


def assert_grid(rows, cases, missions, policies):
    keys = [(r["flight"], r["mission"], r["policy"]) for r in rows]
    expected = {(f, m, p) for f in cases for m in missions for p in policies}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Incomplete, extra, or duplicate comparison cells")


def summarize(rows):
    return dict(
        cases=len(rows),
        successes=sum(r["full_success"] for r in rows),
        phases=sum(r["phases_succeeded"] for r in rows),
        report_recall=float(np.mean([r["report_recall"] for r in rows])),
        false_components=sum(r["false_components"] for r in rows),
        calls=sum(r["calls"] for r in rows),
        reserve_satisfied=sum(r["final_battery"] >= 0.2 for r in rows),
        min_reserve_margin_j=min(r["reserve_margin_j"] for r in rows),
    )


def run(root):
    p, v = verify(root), read(root / "validation.json")
    if (
        not v["complete"]
        or v["policy_cases"] != 84
        or v["protocol_sha256"] != sha(root / "protocol.json")
        or v["validator_sha256"] != sha(Path(__file__).with_name("validate.py"))
    ):
        raise ValueError("Complete independent reconstruction required")
    for name, expected in v["source_summaries_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed validated summary")
    rows = [r for c in p["cases"] for r in read(root / f"summary_{c}_15.json")["cases"]]
    assert_grid(rows, p["cases"], p["missions"], POLICIES)
    parent = Path(p["parent"])
    old = read(parent / "analysis.json")
    references = [(parent, r) for r in old["policy_cases"]]
    fixed_root = Path(old["fixed_controls_source"])
    references += [(fixed_root, r) for r in old["reused_fixed_controls"]]
    augmented, streams = [], {}
    for source, original in [(root, r) for r in rows] + references:
        row = dict(original, source=str(source.resolve()), reused=source != root)
        events = checked_events(source, row)
        result = read(source / row["path"] / "execution/results.json")
        row["false_components"] = sum(e["score"]["false_positive_components"] for e in events)
        row["energy_j"] = result["cumulative_energy_j"]
        row["reserve_margin_j"] = (row["final_battery"] - 0.2) * 30 * 3600
        row["battery"] = p["cases"][row["flight"]]["initial_battery"]
        augmented.append(row)
        streams[row["flight"], row["mission"], row["policy"]] = events
    reference_names = list(dict.fromkeys(r["policy"] for _, r in references))
    assert_grid([r for r in augmented if r["reused"]], p["cases"], p["missions"], reference_names)
    indexed = {(r["flight"], r["mission"], r["policy"]): r for r in augmented}
    parity, pairs = [], []
    for case in p["cases"]:
        for mission in p["missions"]:
            before = streams[case, mission, "mission_survival"]
            after = streams[case, mission, "survival_reference"]
            if schedule(before) != schedule(after) or any(
                a["router_decision"] != b["router_decision"]
                for a, b in zip(before, after, strict=True)
            ):
                raise ValueError("Survival reproduction control failed")
            for field in (
                "full_success",
                "phases_succeeded",
                "report_recall",
                "final_battery",
                "failed_constraints",
            ):
                if (
                    indexed[case, mission, "mission_survival"][field]
                    != indexed[case, mission, "survival_reference"][field]
                ):
                    raise ValueError("Reproduction endpoint mismatch")
            parity.append(dict(case=case, mission=mission, identical=True))
            for left, right in p["primary_comparisons"]:
                a, b = indexed[case, mission, left], indexed[case, mission, right]
                ae, be = streams[case, mission, left], streams[case, mission, right]
                divergence = next(
                    (
                        i
                        for i, (x, y) in enumerate(zip(ae, be, strict=False))
                        if x["execution"]["config_id"] != y["execution"]["config_id"]
                    ),
                    None,
                )
                mechanism = None
                if divergence is not None:
                    x, y = ae[divergence], be[divergence]
                    mechanism = dict(
                        decision_index=divergence,
                        capture_time_s=x["capture_time_s"],
                        left_config=x["execution"]["config_id"],
                        right_config=y["execution"]["config_id"],
                        identical_prior_schedule=schedule(ae[:divergence])
                        == schedule(be[:divergence]),
                        identical_actual_runtime=x["runtime_at_capture"] == y["runtime_at_capture"],
                        identical_prediction_history=x["router_decision"]["history"]
                        == y["router_decision"]["history"],
                        later_histories_are_matched=False,
                    )
                pairs.append(
                    dict(
                        case=case,
                        mission=mission,
                        left=left,
                        right=right,
                        success_delta=int(a["full_success"]) - int(b["full_success"]),
                        phase_delta=a["phases_succeeded"] - b["phases_succeeded"],
                        report_recall_delta=a["report_recall"] - b["report_recall"],
                        false_components_delta=a["false_components"] - b["false_components"],
                        reserve_margin_delta_j=a["reserve_margin_j"] - b["reserve_margin_j"],
                        schedule_changed=schedule(ae) != schedule(be),
                        first_divergence=mechanism,
                    )
                )
    controls = []
    for case in p["cases"]:
        schedules = [schedule(streams[case, m, "temporal_no_repetition"]) for m in p["missions"]]
        controls.append(
            dict(
                kind="repetition_blind_across_missions",
                case=case,
                invariant=all(s == schedules[0] for s in schedules),
            )
        )
    for source_flight in sorted({c["source_flight"] for c in p["cases"].values()}):
        for mission in p["missions"]:
            schedules = [
                schedule(streams[c, mission, "temporal_no_battery"])
                for c, info in p["cases"].items()
                if info["source_flight"] == source_flight
            ]
            controls.append(
                dict(
                    kind="battery_blind_across_batteries",
                    source_flight=source_flight,
                    mission=mission,
                    invariant=all(s == schedules[0] for s in schedules),
                )
            )
    if not all(c["invariant"] for c in controls):
        raise ValueError("Matched input negative control failed")
    names = [n for n in POLICIES if n != "survival_reference"] + reference_names
    summaries = [
        dict(policy=n, **summarize([r for r in augmented if r["policy"] == n])) for n in names
    ]
    stratified = [
        dict(
            policy=n,
            battery=b,
            mission=m,
            **summarize(
                [
                    r
                    for r in augmented
                    if r["policy"] == n and r["battery"] == b and r["mission"] == m
                ]
            ),
        )
        for n in names
        for b in p["batteries"]
        for m in p["missions"]
    ]
    contrasts = []
    for left, right in p["primary_comparisons"]:
        ps = [x for x in pairs if x["left"] == left and x["right"] == right]
        contrasts.append(
            dict(
                left=left,
                right=right,
                cases=len(ps),
                wins=sum(x["success_delta"] > 0 for x in ps),
                losses=sum(x["success_delta"] < 0 for x in ps),
                ties=sum(x["success_delta"] == 0 for x in ps),
                changed_schedules=sum(x["schedule_changed"] for x in ps),
                phase_delta=sum(x["phase_delta"] for x in ps),
                independent_recordings_with_wins=len(
                    {p["cases"][x["case"]]["source_flight"] for x in ps if x["success_delta"] > 0}
                ),
            )
        )
    timing = []
    for n in POLICIES:
        samples = [
            e["measured_router_wall_clock_s"]
            for key, es in streams.items()
            if key[2] == n
            for e in es
        ]
        timing.append(
            dict(
                policy=n,
                decisions=len(samples),
                p95_s=float(np.percentile(samples, 95)),
                max_s=max(samples),
                above_configured_50ms=sum(t > 0.05 for t in samples),
                target_hardware=False,
            )
        )
    analysis = dict(
        complete=True,
        fresh_test=False,
        independent_evaluation_layouts=1,
        policy_cases=augmented,
        policy_summary=summaries,
        stratified=stratified,
        paired_contrasts=contrasts,
        pairs=pairs,
        negative_controls=controls,
        reproduction_parity=parity,
        router_timing=timing,
        feasibility=old["feasibility"],
        new_policy_replays=84,
        reused_reference_replays=180,
        new_physical_flights=0,
        new_neural_calls=0,
        promote_as_validated_improvement=False,
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        source_sha256=sha(Path(__file__)),
    )
    save(root / "analysis.json", analysis)
    lines = [
        "# Learned temporal routing and benchmark discrimination",
        "",
        "84 new cost-accounted replays (including 12 reproduction controls) and 180 immutable "
        "reference runs. The 21 distinct policies share two previously inspected physical "
        "recordings, one layout, one route, two intents, and three battery levels. "
        "No new physics capture or neural execution was performed. This is an exploratory "
        "within-cohort comparison, not an independent generalization test.",
        "",
        "## Complete policy comparison",
        "",
        "| Policy | Full success / 12 | Phases / 48 | Report recall | False components | "
        "Reserve / 12 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['policy']} | {r['successes']} | {r['phases']} | {r['report_recall']:.3f} | "
        f"{r['false_components']} | {r['reserve_satisfied']} |"
        for r in summaries
    ]
    lines += [
        "",
        "## Paired, predeclared contrasts",
        "",
        "Positive wins and phase differences favor temporal_mission. Ties remain in the "
        "denominator.",
        "",
        "| Comparator | Wins | Losses | Ties | Changed schedules / 12 | Phase difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['right']} | {r['wins']} | {r['losses']} | {r['ties']} | "
        f"{r['changed_schedules']} | {r['phase_delta']:+d} |"
        for r in contrasts
    ]
    lines += [
        "",
        "## Interpretation and scope",
        "",
        "A schedule change establishes input activation; it does not establish a mission benefit. "
        "The matched repetition ablation hides only required hit counts and evidence "
        "spans, retaining "
        "phase deadlines, phase weights and safety limits. The history ablation hides track "
        "confirmation state while retaining the selected-action estimated error ledger. "
        "The battery ablation assumes 65% at takeoff but pays actual execution costs.",
        "",
        "Temporal labels use future development observations offline; runtime policies see current "
        "RGB and completed prediction history only. They predict repeated support of identities "
        "visible at window start, not mission success, new arrivals, or actual tracker "
        "association. "
        "Training normalization and coefficients are fit separately within each "
        "complete-layout fold. "
        "The final heads are refit on all development layouts, excluding evaluation lineage. "
        "Overlapping windows and two lighting recordings do not supply independent trials.",
        "",
        "All 12 survival reproduction controls match the prior study. All ten input "
        "negative controls "
        "pass. The validator independently reconstructs predictions, tracking, scores, clocks, "
        f"energy and {v['public_decisions_reproduced']} public routing decisions. "
        "Ten conditions have a paid evaluator witness; two remain unresolved. "
        "Failed and unresolved conditions are retained. No router is promoted automatically.",
        "",
        "Energy and runtime charges are configured models. Host timing is reported separately. "
        "No frame-level p-values, population confidence intervals, hardware battery guarantee, "
        "or out-of-map validity claim is made.",
        "",
        "Full per-condition results, first divergent decisions, matched actual states, energy "
        "margins, training diagnostics and timing are in the accompanying JSON artifacts.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    plot(root, p, names, stratified)
    save(
        root / "completion.json",
        dict(
            complete=True,
            fresh_test=False,
            research_readiness_certified=False,
            artifacts_sha256={
                n: sha(root / n)
                for n in (
                    "protocol.json",
                    "validation.json",
                    "analysis.json",
                    "REPORT.md",
                    "comparison.png",
                )
            },
            source_sha256=sha(Path(__file__)),
        ),
    )
    print(contrasts, flush=True)


def plot(root, protocol, names, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 11), sharey=True, layout="constrained")
    columns = [(b, m) for b in protocol["batteries"] for m in protocol["missions"]]
    for ax, field, vmax, title in zip(
        axes,
        ("successes", "phases"),
        (2, 8),
        ("Full mission success / 2 lighting conditions", "Completed phases / 8"),
        strict=True,
    ):
        values = np.asarray(
            [
                [
                    next(
                        r[field]
                        for r in rows
                        if r["policy"] == name and r["battery"] == b and r["mission"] == m
                    )
                    for b, m in columns
                ]
                for name in names
            ]
        )
        ax.imshow(values, vmin=0, vmax=vmax, cmap="Blues", aspect="auto")
        ax.set_xticks(
            range(len(columns)),
            [f"{b * 100:g}%\n{'Survey' if m == 'rapid_survey' else 'Inspect'}" for b, m in columns],
        )
        ax.set_title(title)
        ax.axhline(5.5, color="#777777", linewidth=1)
        for i in range(len(names)):
            for j in range(len(columns)):
                ax.text(
                    j,
                    i,
                    str(values[i, j]),
                    ha="center",
                    va="center",
                    color="white" if values[i, j] > vmax * 0.6 else "black",
                    fontsize=10,
                )
    axes[0].set_yticks(
        range(len(names)),
        [n.replace("static:CFG_", "Fixed ").replace("_", " ") for n in names],
        fontsize=10,
    )
    fig.suptitle(
        "Temporal routing and matched ablations\nSame physical trajectory; inspected "
        "recordings; configured execution costs",
        fontsize=14,
    )
    fig.savefig(root / "comparison.png", dpi=150)
    plt.close(fig)
