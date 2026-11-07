"""Complete-grid, layout-level reporting for the frozen prospective experiment."""

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_routing.audit import complete_grid
from experiments.mission_coverage_routing.protocol import verify


def paired(rows, primary, other):
    left = {(r["flight"], r["mission"]): r for r in rows if r["policy"] == primary}
    right = {(r["flight"], r["mission"]): r for r in rows if r["policy"] == other}
    if not left or set(left) != set(right):
        raise ValueError("Unmatched comparison")
    wins = sum(left[k]["full_success"] > right[k]["full_success"] for k in left)
    losses = sum(left[k]["full_success"] < right[k]["full_success"] for k in left)
    return dict(
        primary=primary,
        comparator=other,
        n=len(left),
        wins=wins,
        losses=losses,
        ties=len(left) - wins - losses,
        phase_difference=sum(
            left[k]["phases_succeeded"] - right[k]["phases_succeeded"] for k in left
        ),
        changed_schedules=sum(left[k]["schedule"] != right[k]["schedule"] for k in left),
    )


def summarize(rows):
    if not rows:
        raise ValueError("Nonempty denominator required")
    return dict(
        cases=len(rows),
        successes=sum(r["full_success"] for r in rows),
        phases_succeeded=sum(r["phases_succeeded"] for r in rows),
        mean_report_recall=float(np.mean([r["report_recall"] for r in rows])),
        false_components=sum(r["false_components"] for r in rows),
        false_reports=sum(r["false_reports"] for r in rows),
        reserve_passes=sum(r["reserve_margin_j"] >= -1e-8 for r in rows),
        min_reserve_margin_j=min(r["reserve_margin_j"] for r in rows),
    )


def run(root):
    if (root / "completion.json").exists():
        raise FileExistsError("Preserve complete report")
    p = verify(root)
    validation = read(root / "validation.json")
    if not validation["complete"] or validation["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Independent validation required")
    rows = []
    host = {}
    for f, c in p["cases"].items():
        audit = root / f"validation_{f}.json"
        if sha(audit) != validation["condition_audits_sha256"][audit.name]:
            raise ValueError("Changed audit")
        a = read(audit)
        for name, expected in a["source_summaries_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError("Changed validated summary")
            for r in read(root / name)["cases"]:
                dest = root / r["path"]
                for n, k in (
                    ("outcome.json", "outcome_sha256"),
                    ("execution/events.json", "events_sha256"),
                    ("execution/results.json", "results_sha256"),
                ):
                    if sha(dest / n) != r[k]:
                        raise ValueError("Changed audited execution")
                result = read(dest / "execution/results.json")
                events = read(dest / "execution/events.json")
                energy = result["cumulative_energy_j"]
                rows.append(
                    dict(
                        r,
                        layout=c["layout_group"],
                        lighting=c["lighting"],
                        battery=c["initial_battery"],
                        energy_j=energy,
                        reserve_margin_j=(c["initial_battery"] - 0.2) * 108000 - energy,
                        false_components=sum(
                            e["score"]["false_positive_components"] for e in events
                        ),
                        schedule=[
                            (e["observation_id"], e["execution"]["config_id"]) for e in events
                        ],
                    )
                )
                host.setdefault(r["policy"], []).extend(
                    e["measured_router_wall_clock_s"] for e in events
                )
    complete_grid(rows, p)
    summary = [
        dict(policy=name, **summarize([r for r in rows if r["policy"] == name]))
        for name in p["policies"]
    ]
    cells = []
    for f in p["cases"]:
        for m in p["missions"]:
            selected = [r for r in rows if r["flight"] == f and r["mission"] == m]
            cells.append(
                dict(
                    flight=f,
                    mission=m,
                    successful_policies=[r["policy"] for r in selected if r["full_success"]],
                    success_varies=len({r["full_success"] for r in selected}) > 1,
                    phases_varies=len({r["phases_succeeded"] for r in selected}) > 1,
                    report_recall_varies=len({r["report_recall"] for r in selected}) > 1,
                )
            )
    contrasts = [
        paired(rows, p["primary_policy"], c) for c in p["policies"] if c != p["primary_policy"]
    ]
    stratified = {
        key: [
            dict(
                value=value,
                policy=policy,
                **summarize([r for r in rows if r[key] == value and r["policy"] == policy]),
            )
            for value in sorted({r[key] for r in rows})
            for policy in p["policies"]
        ]
        for key in ("layout", "mission", "battery", "lighting")
    }
    analysis = dict(
        complete=True,
        fresh_test=True,
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        policy_cases=rows,
        policy_summary=summary,
        paired_contrasts=contrasts,
        stratified=stratified,
        cells=cells,
        router_timing={
            c: dict(
                count=len(v),
                p95_ms=float(np.percentile(v, 95) * 1000),
                max_ms=max(v) * 1000,
                above_charged_50ms=sum(t > 0.05 for t in v),
            )
            for c, v in host.items()
        },
        independent_layouts=3,
        evaluation_maps=1,
        unseen_evaluation_maps=0,
        statistical_inference=False,
        superiority_claim=False,
        source_sha256=sha(Path(__file__)),
    )
    save(root / "analysis.json", analysis)
    plot(root, p, rows, summary)
    primary = next(r for r in summary if r["policy"] == p["primary_policy"])
    fixed = max(r["successes"] for r in summary if r["policy"].startswith("static:"))
    lines = [
        "# Prospective mission-level routing comparison",
        "",
        f"The predeclared primary policy completes {primary['successes']}/24 conditions; "
        f"the best fixed-model result is {fixed}/24 (selected descriptively after evaluation).",
        (
            "All 504 executions, 21 policies and 24 conditions are retained. Six "
            "new physics recordings supply 360 captures, 3,240 frame/configuration "
            "pairs and 6,120 dataset neural forwards. The three spatial layout "
            "groups are held out from policy training; the scenery map is known. No"
            " model or policy was retuned on these outcomes."
        ),
        "",
        "Independent reconstruction reproduces "
        f"{validation['public_decisions_reproduced']:,} public routing decisions, "
        "raw mask scores, tracker reports, integer clocks and configured resource debits. "
        "Lighting and battery mates share the same physical trajectories.",
        "",
        "## Complete policy comparison",
        "",
        "| Policy | Full /24 | Phases /96 | Mean report recall | False components | Reserve /24 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['policy']} | {r['successes']} | {r['phases_succeeded']} | "
        f"{r['mean_report_recall']:.3f} | {r['false_components']} | {r['reserve_passes']} |"
        for r in summary
    ]
    lines += [
        "",
        "## Paired primary-policy contrasts",
        "",
        (
            "Wins/losses/ties refer to full contract success, with identical "
            "physical recording, intent and initial battery. Conditions are "
            "repeated measures, not independent samples."
        ),
        "",
        "| Comparator | Wins | Losses | Ties | Phase difference | Changed schedules |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['comparator']} | {r['wins']} | {r['losses']} | {r['ties']} | "
        f"{r['phase_difference']:+d} | "
        f"{r['changed_schedules']} |"
        for r in contrasts
    ]
    lines += [
        "",
        "## Discrimination and failure retention",
        "",
        "Full success differs between policies in "
        f"{sum(c['success_varies'] for c in cells)}/24 conditions; "
        f"phase completion differs in {sum(c['phases_varies'] for c in cells)}/24; "
        f"report recall differs in {sum(c['report_recall_varies'] for c in cells)}/24. "
        f"All policies fail {sum(not c['successful_policies'] for c in cells)}/24 conditions. "
        "These cells remain in the primary denominator.",
        "",
        "## Layout-level results",
        "",
        "| Layout | Policy | Full /8 | Phases /32 | Report recall |",
        "|---|---|---:|---:|---:|",
    ]
    show = {
        p["primary_policy"],
        "coverage_no_battery",
        "coverage_global",
        "coverage_short",
        "phase_survival",
        "mission_survival",
        "temporal_mission",
        "greedy_quality",
    }
    lines += [
        f"| {r['value']} | {r['policy']} | {r['successes']} | {r['phases_succeeded']} | "
        f"{r['mean_report_recall']:.3f} |"
        for r in stratified["layout"]
        if r["policy"] in show
    ]
    lines += [
        "",
        "## Scope and limitations",
        "",
        (
            "These are authored static-person missions in a physics simulator, not "
            "replayed real UAV missions. This study supports only descriptive "
            "transfer to three fresh layouts on one known map. It does not "
            "establish unseen-map generalization, field robustness, calibrated "
            "success prediction or statistical population superiority. Energy, "
            "memory and router cost are configured; prior CPU inference timings are"
            " used in the mission clock. Host timing under concurrent study "
            "execution is diagnostic, not target-device calibration."
        ),
        "",
        (
            "Sensor admission establishes opportunity to observe targets, not paid "
            "mission feasibility. Evaluator-only optimistic bounds and paid "
            "witnesses, when present, are separate from deployable rankings. Search"
            " failure does not prove infeasibility. A proposed router does not need"
            " to win for the benchmark to expose meaningful differences, but a new "
            "routing-algorithm superiority claim requires stronger evidence."
        ),
        "",
        (
            "The policy definition, comparison grid and sensor gate are documented "
            "in `docs/mission_coverage_routing.md`; all coefficients and source "
            "pins are in the acquisition/study protocols. `analysis.json` retains "
            "every case and all layout/intent/battery/lighting strata."
        ),
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    names = (
        "protocol.json",
        "analysis.json",
        "validation.json",
        "REPORT.md",
        "comparison.png",
        "comparison.pdf",
        "success_matrix.png",
    )
    save(
        root / "completion.json",
        dict(
            complete=True,
            fresh_test=True,
            source_sha256=sha(Path(__file__)),
            artifacts_sha256={n: sha(root / n) for n in names},
            research_readiness_certified=False,
        ),
    )
    print(
        dict(
            primary=primary,
            discriminating_cells=sum(c["success_varies"] for c in cells),
            all_policy_failure_cells=sum(not c["successful_policies"] for c in cells),
        ),
        flush=True,
    )


def plot(root, p, rows, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = p["policies"]
    labels = [n.replace("static:CFG_", "") for n in names]
    fig, axs = plt.subplots(1, 3, figsize=(15, 9), sharey=True, layout="constrained")
    for ax, key, title, denom in zip(
        axs,
        ("successes", "phases_succeeded", "mean_report_recall"),
        ("Full mission success / 24", "Completed phases / 96", "Mean report recall"),
        (24, 96, 1),
        strict=True,
    ):
        values = [r[key] for r in summary]
        ax.barh(
            range(len(names)),
            values,
            color=["#155e75" if n == p["primary_policy"] else "#94a3b8" for n in names],
        )
        ax.set(yticks=range(len(names)), yticklabels=labels, xlim=(0, denom), title=title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axs[0].invert_yaxis()
    fig.suptitle(
        (
            "Fresh layouts on a known map: identical flights, different routing "
            "policies\n3 layouts x 2 lighting conditions x 2 intents x 2 battery "
            "levels"
        ),
        fontsize=13,
    )
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    cells = [(f, m) for f in p["cases"] for m in p["missions"]]
    by = {(r["flight"], r["mission"], r["policy"]): r for r in rows}
    data = np.array([[by[f, m, n]["full_success"] for f, m in cells] for n in names], int)
    fig, ax = plt.subplots(figsize=(17, 9), layout="constrained")
    ax.imshow(
        data,
        cmap=matplotlib.colors.ListedColormap(["#e2e8f0", "#0f766e"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set(
        yticks=range(len(names)),
        yticklabels=labels,
        xticks=range(len(cells)),
        xticklabels=[
            f.replace("_", " ") + "\n" + ("survey" if m == "rapid_survey" else "inspection")
            for f, m in cells
        ],
        title=(
            "Complete paired factorial: green = full success; grey = failure (all cells retained)"
        ),
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.savefig(root / "success_matrix.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
