"""Static figures for the validation report (reads only committed CSV/JSON).

Responsibility: render the report figures from ``examples/validation`` tables.
Requires matplotlib (figure-only; not a benchmark dependency). Each figure has a
CSV table view next to it. Light-mode reference palette; categorical slots in
fixed order (validated: 3 slots all-pairs, 5 slots adjacent); single-hue
sequential ramp for magnitude; text in ink colors, never series colors.

Usage: python -m experiments.validation.figures [--report examples/validation]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = Path(__file__).resolve().parents[2]
SURFACE, INK, INK2, MUTED, GRID, AXIS = (
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
)
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQ = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
POLICY_ORDER = (
    "fixed_local_yolo",
    "fixed_local_cabinet",
    "fastest_current",
    "fixed_local_mask",
    "fixed_remote_mask",
    "quality_initial",
    "quality_current",
    "risk_initial",
    "risk_current",
)
SHORT = {
    "fixed_local_yolo": "Fixed local YOLO11s",
    "fixed_local_cabinet": "Fixed local CABiNet",
    "fastest_current": "Fastest feasible",
    "fixed_local_mask": "Fixed local Mask R-CNN",
    "fixed_remote_mask": "Fixed remote Mask R-CNN",
    "quality_initial": "Quality, urgency-blind",
    "quality_current": "Quality, urgency-aware",
    "risk_initial": "Quality+risk, urgency-blind",
    "risk_current": "Quality+risk, urgency-aware",
}
CONTRACT_LABEL = {
    "survey": "Survey\n(rescored only)",
    "confirm": "Confirm\n(reference)",
    "confirm_strict": "Strict confirm\n(3 obs / 4 s)",
    "revisit": "Revisit\n(both halves)",
    "time_critical": "Time-critical\n(report by 36 s)",
}


def style():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.size": 9,
            "text.color": INK,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "svg.hashsalt": "missionbench-validation",
        }
    )


def rows(report, name):
    with (report / name).open() as f:
        return list(csv.DictReader(f))


def save(fig, report, name):
    for ext in ("png", "svg"):
        # Fixed metadata so re-rendering is byte-stable (no timestamps or random SVG ids).
        meta = {"Date": None} if ext == "svg" else {"Software": None}
        fig.savefig(report / f"{name}.{ext}", dpi=200, bbox_inches="tight", metadata=meta)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 1
def fig_intent_heatmap(report):
    rerun = rows(report, "exp1_success_rerun.csv")
    rescore = rows(report, "exp1_success_rescore.csv")
    contracts = ("survey", "confirm", "confirm_strict", "revisit", "time_critical")
    cell = {}
    for r in rescore:
        if r["contract"] == "survey":
            cell[(r["policy"], "survey")] = r
    for r in rerun:
        cell[(r["policy"], r["contract"])] = r
    cmap = LinearSegmentedColormap.from_list("seq", ("#f0efec", *SEQ))
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for yi, p in enumerate(POLICY_ORDER):
        for xi, c in enumerate(contracts):
            r = cell[(p, c)]
            k, n = int(r["successes_feasible"]), int(r["conditions_feasible"])
            rate = k / n
            ax.add_patch(plt.Rectangle((xi + 0.03, yi + 0.04), 0.94, 0.92, color=cmap(rate), lw=0))
            txt = INK if rate < 0.55 else "#ffffff"
            ax.text(
                xi + 0.5, yi + 0.5, f"{k}/{n}", ha="center", va="center", fontsize=8.5, color=txt
            )
    for xi, c in enumerate(contracts):
        best = max(int(cell[(p, c)]["successes_feasible"]) for p in POLICY_ORDER)
        for yi, p in enumerate(POLICY_ORDER):
            if int(cell[(p, c)]["successes_feasible"]) == best and best > 0:
                ax.add_patch(
                    plt.Rectangle((xi + 0.06, yi + 0.07), 0.88, 0.86, fill=False, ec=INK, lw=1.4)
                )
    ax.set_xlim(0, len(contracts))
    ax.set_ylim(len(POLICY_ORDER), 0)
    ax.set_xticks([i + 0.5 for i in range(len(contracts))], [CONTRACT_LABEL[c] for c in contracts])
    ax.set_yticks([i + 0.5 for i in range(len(POLICY_ORDER))], [SHORT[p] for p in POLICY_ORDER])
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(
        "Mission successes per contract (feasible conditions only; outlined = best)", pad=34
    )
    fig.text(
        0.0,
        -0.02,
        "Cells: successes / conditions not certified infeasible by the model-union bound "
        "(72 conditions = 6 recordings x 12 replay conditions).\n"
        "Reruns supply each contract to policy and pipeline; survey cannot be a runtime input "
        "(2-observation report floor) and is rescored.",
        fontsize=7.5,
        color=INK2,
        ha="left",
        va="top",
    )
    save(fig, report, "fig1_intent_success_heatmap")


# ---------------------------------------------------------------- Figure 2
CLASS_ORDER = (
    ("A_physically_infeasible", "Physically infeasible (A)"),
    ("B_model_pool_infeasible", "Model pool cannot supply evidence (B, certified)"),
    ("B_unresolved", "Unresolved: no schedule found (B)"),
    ("C_hindsight_witness", "Feasible: hindsight schedule only (C)"),
    ("C_released_policy", "Feasible: a released policy succeeds (C)"),
    ("C_rescored_released_history", "Feasible: rescored released history (C, survey)"),
)


def fig_feasibility(report):
    classes = rows(report, "exp2_condition_classes.csv")
    contracts = ("survey", "confirm", "confirm_strict", "revisit", "time_critical")
    colors = {
        "A_physically_infeasible": SLOTS[7],
        "B_model_pool_infeasible": SLOTS[1],
        "B_unresolved": "#c3c2b7",
        "C_hindsight_witness": SLOTS[3],
        "C_released_policy": SLOTS[0],
        "C_rescored_released_history": SLOTS[2],
    }
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11.5, 3.6), gridspec_kw=dict(width_ratios=(1.25, 1), wspace=0.95)
    )
    used = set()
    for yi, c in enumerate(contracts):
        left = 0
        for key, _ in CLASS_ORDER:
            n = sum(1 for r in classes if r["contract"] == c and r["feasibility_class"] == key)
            if not n:
                continue
            used.add(key)
            ax.barh(yi, n - 0.4, left=left + 0.2, height=0.62, color=colors[key], lw=0)
            if n >= 5:
                ink = INK if key in ("B_unresolved", "C_hindsight_witness") else "#ffffff"
                ax.text(left + n / 2, yi, str(n), ha="center", va="center", fontsize=8, color=ink)
            left += n
    ax.set_yticks(range(len(contracts)), [CONTRACT_LABEL[c].replace("\n", " ") for c in contracts])
    ax.invert_yaxis()
    ax.set_xlim(0, 72)
    ax.set_xlabel("Conditions (of 72)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title("Where can a mission be satisfied at all?")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[k]) for k, _ in CLASS_ORDER if k in used]
    ax.legend(
        handles,
        [lbl for k, lbl in CLASS_ORDER if k in used],
        loc="upper left",
        bbox_to_anchor=(0, -0.2),
        ncol=2,
        fontsize=7.5,
        handlelength=1,
    )
    gap = [r for r in rows(report, "exp2_policy_gap.csv") if r["contract"] == "confirm"]
    order = [p for p in POLICY_ORDER]
    for yi, p in enumerate(order):
        r = next(x for x in gap if x["policy"] == p)
        n, k = int(r["demonstrated_feasible_conditions"]), int(r["successes_in_feasible"])
        bx.barh(yi, n, height=0.62, color="#f0efec", lw=0)
        bx.barh(yi, k, height=0.62, color=SLOTS[0], lw=0)
        bx.text(n + 0.8, yi, f"{k}/{n}", va="center", fontsize=8, color=INK2)
    bx.set_yticks(range(len(order)), [SHORT[p] for p in order])
    bx.invert_yaxis()
    bx.set_xlim(0, 60)
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("Demonstrated-feasible conditions (confirm)")
    bx.set_title("Policy successes vs. verified feasible schedules")
    fig.text(
        0.0,
        -0.3,
        "No condition is physically infeasible under any contract (perfect-sensor bound passes on "
        "all 6 recordings). Certified model-pool infeasibility = every model on every capture at "
        "zero latency still\ncannot confirm enough people. Hindsight schedules are GT-aware search "
        "results replayed through the unchanged runtime; search failure is inconclusive. Survey "
        "cannot be a runtime input, so it has no hindsight search.",
        fontsize=7.5,
        color=INK2,
        ha="left",
        va="top",
    )
    save(fig, report, "fig2_feasibility_classes")


# ---------------------------------------------------------------- Figure 3
def fig_timeline(report):
    timelines = json.loads((report / "exp3_timelines.json").read_text())
    cases = list(dict.fromkeys(t["run_id"] for t in timelines))
    fig, axes = plt.subplots(len(cases), 1, figsize=(9.6, 2.5 * len(cases)), sharex=True)
    axes = [axes] if len(cases) == 1 else axes
    for ax, run in zip(axes, cases, strict=True):
        tls = sorted((t for t in timelines if t["run_id"] == run), key=lambda t: t["protocol"])
        focal = tls[0]["focal"]
        for block, tl in enumerate(tls):
            y0 = block * (len(focal) + 1.8)
            ax.text(-1.0, y0, "captures", ha="right", va="center", fontsize=7, color=MUTED)
            proc = [c["t"] for c in tl["captures"] if c["processed"]]
            skip = [c["t"] for c in tl["captures"] if not c["processed"]]
            late = [c["t"] for c in tl["captures"] if c["processed"] and c["late"]]
            ax.scatter(proc, [y0] * len(proc), s=26, color=INK2, lw=0, zorder=3)
            ax.scatter(
                skip, [y0] * len(skip), s=26, facecolors="none", edgecolors=INK2, lw=0.9, zorder=3
            )
            ax.scatter(late, [y0] * len(late), s=46, marker="x", color=SLOTS[7], lw=1.2, zorder=4)
            for j, t in enumerate(focal):
                y = y0 + 1 + j
                ax.text(
                    -1.0, y, f"person {j + 1}", ha="right", va="center", fontsize=7, color=MUTED
                )
                vis = [c["t"] for c in tl["captures"] if t in c["visible"]]
                ax.scatter(vis, [y] * len(vis), s=18, color="#e1e0d9", marker="s", lw=0, zorder=1)
                hits = [c["t"] for c in tl["captures"] if t in c["matched"] and c["admitted"]]
                ax.scatter(hits, [y] * len(hits), s=42, color=SLOTS[j], lw=0, zorder=3)
                reps = [r["done"] for r in tl["reports"] if r["target"] == t]
                ax.scatter(reps, [y] * len(reps), s=60, marker="v", color=INK, lw=0, zorder=4)
            label = "Causal replay" if tl["protocol"] == "causal" else "No-skip ablation"
            ax.text(
                -7.5,
                y0 + len(focal) / 2,
                f"{label}\n{'SUCCESS' if tl['success'] else 'FAIL'}",
                ha="right",
                va="center",
                fontsize=8,
                fontweight="bold",
                color=INK,
            )
        ax.invert_yaxis()
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        for x in (36, 44):
            ax.axvline(x, color=AXIS, lw=0.8)
        policy = run.rsplit("/", 1)[1]
        note = tls[0].get("selection", "")
        ax.set_title(f"{SHORT[policy]} - {'/'.join(run.split('/')[:5])}   {note}", fontsize=9)
    axes[-1].set_xlabel(
        "Mission time (s).  Vertical lines: end of pass (36 s), report deadline (44 s)"
    )
    axes[-1].set_xlim(-0.8, 45.5)
    handles = [
        plt.Line2D([], [], ls="", marker="o", color=INK2, label="Capture processed"),
        plt.Line2D(
            [],
            [],
            ls="",
            marker="o",
            markerfacecolor="none",
            color=INK2,
            label="Capture unavailable",
        ),
        plt.Line2D([], [], ls="", marker="x", color=SLOTS[7], label="Request missed its deadline"),
        plt.Line2D([], [], ls="", marker="s", color="#e1e0d9", label="Person visible (GT)"),
        *[
            plt.Line2D([], [], ls="", marker="o", color=SLOTS[i], label=f"Person {i + 1} matched")
            for i in range(3)
        ],
        plt.Line2D([], [], ls="", marker="v", color=INK, label="Report delivered"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.12, 0.0), ncol=4, fontsize=7.5)
    save(fig, report, "fig3_causal_timeline")


# ---------------------------------------------------------------- Figure 4
def fig_latency(report):
    m = rows(report, "exp3_metrics.csv")
    policies = ("fixed_local_mask", "fixed_remote_mask", "risk_current")
    metrics = (
        ("mean_processed_observations", "Processed observations / run"),
        ("mean_valid_confirmations", "Confirmed people / run"),
        ("mean_completed_reports", "Timely reports / run"),
        ("successes", "Mission successes (of 72)"),
    )
    mults = (0.5, 1.0, 2.0, 4.0)
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), gridspec_kw=dict(wspace=0.38))
    for ax, (key, title) in zip(axes, metrics, strict=True):
        for color, p in zip(SLOTS, policies, strict=False):
            for proto, ls, alpha in (("causal", "-", 1.0), ("no_skip", (0, (4, 2)), 0.75)):
                ys = [
                    float(
                        next(
                            r[key]
                            for r in m
                            if r["policy"] == p
                            and r["protocol"] == proto
                            and float(r["multiplier"]) == x
                        )
                    )
                    for x in mults
                ]
                ax.plot(mults, ys, ls=ls, color=color, lw=2, alpha=alpha, marker="o", ms=4)
        ax.set_xscale("log", base=2)
        ax.set_xticks(mults, ["0.5x", "1x", "2x", "4x"])
        ax.minorticks_off()
        ax.grid(axis="y")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(bottom=0)
    axes[0].set_xlabel("Inference latency multiplier")
    handles = [
        plt.Line2D([], [], color=c, lw=2, label=SHORT[p])
        for c, p in zip(SLOTS, policies, strict=False)
    ]
    handles += [
        plt.Line2D([], [], color=INK2, lw=2, label="Causal replay (solid)"),
        plt.Line2D([], [], color=INK2, lw=2, ls=(0, (4, 2)), label="No-skip ablation (dashed)"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=7.5
    )
    save(fig, report, "fig4_latency_sweep")


# ---------------------------------------------------------------- Figure 5
def fig_attribution(report):
    urg = [
        r
        for r in rows(report, "exp4_urgency_comparison.csv")
        if r["split"] == "transfer" and r["schedule"] == "all"
    ]
    funnel = {r["policy"]: r for r in rows(report, "exp4_evidence_funnel.csv")}
    fig, axes = plt.subplots(
        1, 4, figsize=(12.5, 3.0), gridspec_kw=dict(wspace=0.55, width_ratios=(1, 1, 1, 2.3))
    )
    labels = []
    vals = dict(miss=[], energy=[], succ=[])
    for r in urg:
        fam = "Quality" if r["aware"].startswith("quality") else "Quality+risk"
        for side, name in (("blind", "urgency-blind"), ("aware", "urgency-aware")):
            labels.append(f"{fam}, {name}")
            vals["miss"].append(int(r[f"{side}_deadline_misses"]))
            vals["energy"].append(float(r[f"{side}_mean_nonflight_energy_j"]))
            vals["succ"].append(int(r[f"{side}_successes"]))
    colors = [SLOTS[1], SLOTS[0]] * 2
    for ax, key, title, fmt in (
        (axes[0], "miss", "Request deadline misses", "{:.0f}"),
        (axes[1], "energy", "Mean task energy (J)", "{:.0f}"),
        (axes[2], "succ", "Mission successes (of 48)", "{:.0f}"),
    ):
        ys = list(range(len(labels)))
        ax.barh(ys, vals[key], color=colors, height=0.6, lw=0)
        for y, v in zip(ys, vals[key], strict=True):
            ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8, color=INK2)
        ax.set_yticks(ys, labels if ax is axes[0] else [""] * len(ys))
        ax.invert_yaxis()
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0, max(vals[key]) * 1.3 + 1)
    stages = (
        ("visible_on_processed_capture", "Visible"),
        ("matched_at_least_once", "Matched"),
        ("confirmed_repeated_evidence", "Confirmed"),
        ("reported_on_time", "Reported"),
    )
    fx = axes[3]
    show = ("fixed_local_yolo", "fixed_remote_mask", "risk_initial", "risk_current")
    ends = []
    for color, p in zip((SLOTS[2], SLOTS[3], SLOTS[1], SLOTS[0]), show, strict=True):
        ys = [int(funnel[p][k]) / int(funnel[p]["targets"]) for k, _ in stages]
        fx.plot(range(len(stages)), ys, color=color, lw=2, marker="o", ms=5)
        ends.append([ys[-1], p])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels at least 0.06 apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06)
    for y, p in ends:
        fx.text(len(stages) - 0.85, y, SHORT[p], va="center", fontsize=7.5, color=INK2)
    fx.set_xticks(range(len(stages)), [s for _, s in stages], fontsize=8, rotation=20)
    fx.set_ylim(0, 1.05)
    fx.set_xlim(-0.2, len(stages) + 1.6)
    fx.grid(axis="y")
    fx.set_title("Per-person evidence funnel, all 72 conditions", fontsize=9)
    fx.set_ylabel("Fraction of 216 person-runs")
    fx.text(
        0,
        -0.3,
        "Visible on a processed capture > matched at least once >\n"
        "confirmed (2 obs, 2 s) > reported by 44 s",
        transform=fx.transAxes,
        fontsize=7,
        color=MUTED,
    )
    fig.suptitle(
        "Urgency awareness removes deadline misses but the remaining failures are evidence failures"
        " (transfer split, 48 conditions)",
        x=0.06,
        ha="left",
        fontsize=10,
        fontweight="bold",
        y=1.04,
    )
    save(fig, report, "fig5_failure_attribution")


# ---------------------------------------------------------------- Figure 6
def fig_sensitivity(report):
    path = report / "exp1_sensitivity_settings.csv"
    if not path.exists():
        return
    rows = [r for r in rows_of(path) if r["deadline_s"] == "44"]
    fast = ("fixed_local_yolo", "fixed_local_cabinet", "fastest_current")
    fixed = ("fixed_local_mask", "fixed_remote_mask")
    adaptive = ("quality_initial", "quality_current", "risk_initial", "risk_current")

    def best(r, group):
        return max(int(r[p]) for p in group)

    metrics = (
        (
            "Best fast minus best other strategy",
            lambda r: best(r, fast) - max(best(r, fixed), best(r, adaptive)),
            "fast_beats_all_others",
        ),
        (
            "Best adaptive minus best accurate fixed",
            lambda r: best(r, adaptive) - best(r, fixed),
            "adaptive_beats_accurate_fixed",
        ),
    )
    caps = ("0", "1", "2", "none")
    grid = [(k, h) for k in ("1", "2", "3") for h in ("1", "2", "3")]
    blue = LinearSegmentedColormap.from_list("div", ("#e34948", "#f0efec", "#2a78d6"))
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4), gridspec_kw=dict(hspace=0.45, wspace=0.15))
    for mi, (title, value, flag) in enumerate(metrics):
        vals = [value(r) for r in rows]
        lim = max(abs(v) for v in vals) or 1
        for ci, required in enumerate(("True", "False")):
            ax = axes[mi][ci]
            for yi, (k, h) in enumerate(grid):
                for xi, cap in enumerate(caps):
                    r = next(
                        x
                        for x in rows
                        if x["people_required"] == k
                        and x["sightings_per_person"] == h
                        and x["false_alarm_cap"] == cap
                        and x["report_required"] == required
                    )
                    v = value(r)
                    shade = 0.5 + 0.5 * v / lim
                    ax.add_patch(
                        plt.Rectangle((xi + 0.03, yi + 0.04), 0.94, 0.92, color=blue(shade), lw=0)
                    )
                    ink = "#ffffff" if abs(v) / lim > 0.6 else INK
                    ref = (k, h, cap, required) == ("2", "2", "2", "True")
                    ax.text(
                        xi + 0.5,
                        yi + 0.5,
                        f"{v:+d}" + ("*" if ref else ""),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=ink,
                    )
                    if r[flag] == "True":
                        ax.add_patch(
                            plt.Rectangle(
                                (xi + 0.07, yi + 0.08), 0.86, 0.84, fill=False, ec=INK, lw=1.3
                            )
                        )
            ax.set_xlim(0, len(caps))
            ax.set_ylim(len(grid), 0)
            ax.set_xticks([i + 0.5 for i in range(len(caps))], ["0", "1", "2", "no cap"])
            ax.set_yticks(
                [i + 0.5 for i in range(len(grid))],
                [f"{k} of 3 people, {h} sighting{'s' if h != '1' else ''}" for k, h in grid]
                if ci == 0
                else [],
            )
            ax.tick_params(length=0)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_xlabel("False-alarm cap (raw false components)")
            ax.set_title(
                title
                + "\n"
                + ("Reports required" if required == "True" else "Sightings only (no report)"),
                fontsize=9,
            )
    fig.text(
        0.0,
        0.0,
        "Cells: difference in mission successes (of 72 conditions), deadline 44 s, rescoring of "
        "the 648 reference histories.\nBlue = first-named class ahead, red = behind (each row has "
        "its own symmetric scale). Outlined = material win by the\npredeclared rule (>= 3 "
        "conditions and no recording in the opposite direction). * = reference contract.\nAt 36 s "
        "only fixed remote Mask R-CNN (-4, 12 cells) and the urgency-blind routers (-1, 8 cells) "
        "change;\nall 144 settings are in exp1_sensitivity_settings.csv.",
        fontsize=7.5,
        color=INK2,
        ha="left",
        va="top",
    )
    save(fig, report, "fig6_sensitivity_map")


def rows_of(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPO / "examples/validation")
    args = parser.parse_args()
    style()
    fig_intent_heatmap(args.report)
    fig_feasibility(args.report)
    fig_timeline(args.report)
    fig_latency(args.report)
    fig_attribution(args.report)
    fig_sensitivity(args.report)
    print("figures written to", args.report)


if __name__ == "__main__":
    main()
