"""Render the preregistered factorial replay analysis without changing outcomes.

The report is descriptive: conditions and known development recordings are not
independent random samples. No confidence intervals or p-values are invented.
Matplotlib is imported lazily so verification and text rendering need only the
standard library. The system runtime can render the scientific figures with
PYTHONNOUSERSITE=1 when a user-installed NumPy conflicts with system Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

POLICY_LABELS = {
    "fixed_local_fast": "Local fast",
    "fixed_local_heavy": "Local heavy",
    "fixed_remote_union": "Remote union",
    "deadline_aware": "Deadline aware",
}
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
MARKERS = ("o", "s", "^", "D")
FACTORS = ("deadline", "network", "battery", "intent")
FACTOR_LABELS = {
    "deadline": "Perception deadline",
    "network": "Network",
    "battery": "Initial battery",
    "intent": "Intent",
}
INTENT_LABELS = {
    "rapid_survey": "Rapid survey",
    "verified_inspection": "Strict inspection (infeasible control)",
}
REQUIRED_SOURCES = {
    "plan.json",
    "completion.json",
    "independent_deadline_audit.json",
    "orchestration_manifest.json",
    "run_index.json",
    "summary.json",
    "analysis_design.json",
}
PLOT_NAMES = (
    "single_factor_mission",
    "single_factor_phase",
    "pairwise_mission",
    "pairwise_phase",
    "full_grid_summary",
    "interaction_magnitude",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def value_label(factor, value, short=False):
    if factor == "deadline":
        return str(value).removeprefix("perception_").replace("s", " s")
    if factor == "battery":
        return f"{value * 100:.0f}%"
    if factor == "intent":
        return {"rapid_survey": "Rapid", "verified_inspection": "Strict*"}[value]
    if short:
        return {"good": "Good", "constrained": "Limited", "intermittent": "Intermittent"}[value]
    return str(value).capitalize()


def numeric_mean(values):
    valid = [v for v in values if v is not None]
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in valid),
        "Finite numeric metrics required",
    )
    return sum(valid) / len(valid) if valid else None


def mean_metric(cells, name):
    return numeric_mean([c["metrics"][name] for c in cells])


def select_cells(analysis, **criteria):
    return [
        c
        for c in analysis["cells"]
        if all(
            (c[k] if k in ("policy", "flight") else c["factors"][k]) == v
            for k, v in criteria.items()
        )
    ]


def anchored_cells(analysis, varied_factors, **criteria):
    held = {k: v for k, v in analysis["design"]["baseline"].items() if k not in varied_factors}
    return select_cells(analysis, **(held | criteria))


def verify_analysis(root):
    """Reject incomplete, changed, escaped or unaudited input before publication."""
    root = Path(root).resolve()
    analysis = read_json(root / "analysis.json")
    require(
        analysis.get("schema_version") == "mission_factor_sweep_analysis_v1",
        "Unknown analysis schema",
    )
    require(
        analysis.get("complete") is True and analysis.get("runs") == 288,
        "Only the complete 288-run frozen design can be reported",
    )
    sources = analysis.get("source_sha256", {})
    require(set(sources) >= REQUIRED_SOURCES, "Missing analysis source bindings")
    for relative, expected in sources.items():
        path = (root / relative).resolve()
        require(path.is_relative_to(root), "Escaped analysis source")
        require(path.is_file() and digest(path) == expected, "Changed analysis source: " + relative)
    completion = read_json(root / "completion.json")
    audit = read_json(root / "independent_deadline_audit.json")
    require(
        completion.get("complete") is True and completion.get("runs") == 288,
        "Incomplete execution manifest",
    )
    require(
        audit.get("passed") is True and audit.get("runs") == 288,
        "Independent receipt audit did not pass",
    )
    require(
        audit.get("completion_sha256") == digest(root / "completion.json")
        and audit.get("plan_sha256") == digest(root / "plan.json"),
        "Audit is not bound to this execution",
    )
    audited = audit.get("audited_artifacts_sha256", {})
    require(len(audited) == 288 * 5, "Audit must bind all five artifacts for every run")
    for relative, expected in audited.items():
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root) and path.is_file() and digest(path) == expected,
            "Changed audited artifact: " + relative,
        )
    manifest = read_json(root / "orchestration_manifest.json")
    require(
        manifest.get("implementation_sha256", {}).get(str(Path(__file__).resolve()))
        == digest(__file__),
        "Report generator differs from the frozen implementation",
    )
    for relative, expected in manifest.get("snapshot_sha256", {}).items():
        path = (root / relative).resolve()
        require(
            path.is_relative_to(root) and path.is_file() and digest(path) == expected,
            "Changed orchestration source snapshot: " + relative,
        )
    design, cells = analysis["design"], analysis["cells"]
    require(tuple(design["factors"]) == FACTORS, "Unknown factor ordering")
    require(
        len(cells) == 288 and len({c["run_id"] for c in cells}) == 288,
        "Missing or duplicate analysis cells",
    )
    expected = set(
        itertools.product(
            design["flights"], design["policies"], *(design["levels"][f] for f in FACTORS)
        )
    )
    observed = {(c["flight"], c["policy"], *(c["factors"][f] for f in FACTORS)) for c in cells}
    require(expected == observed and len(expected) == 288, "Incomplete factorial grid")
    require(len(analysis["slices"]) == 15, "All 15 anchored slices are required")
    require(
        sum(c["known_infeasible_intent"] for c in cells) == 144,
        "Known sensor-infeasible controls must remain explicit",
    )
    return analysis


def pct(value, digits=1):
    return "NA" if value is None else f"{100 * value:.{digits}f}%"


def number(value, digits=2):
    return "NA" if value is None else f"{value:.{digits}f}"


def score_row(policy, cells):
    require(cells, "Cannot summarize an empty cell group")
    successes = sum(c["metrics"]["mission_success"] for c in cells)
    return (
        f"| {POLICY_LABELS[policy]} | {successes:.0f}/{len(cells)} "
        f"({pct(successes / len(cells))}) | {pct(mean_metric(cells, 'phase_fraction'))} "
        f"| {pct(mean_metric(cells, 'report_recall'))} "
        f"| {pct(mean_metric(cells, 'perception_deadline_miss_rate'))} "
        f"| {pct(mean_metric(cells, 'capture_coverage'))} "
        f"| {pct(mean_metric(cells, 'final_battery_frac'))} "
        f"| {number(mean_metric(cells, 'energy_j'), 1)} "
        f"| {number(mean_metric(cells, 'communication_mb'))} |"
    )


def render_markdown(analysis):
    """Build explanatory prose and every prespecified comparison from raw cells."""
    design, cells = analysis["design"], analysis["cells"]
    policies = design["policies"]
    memberships = sum(len(s["run_ids"]) for s in analysis["slices"])
    lines = [
        "# Router performance under one to four varying constraints",
        "",
        "## Completed experiment",
        "",
        f"This study contains **{len(cells)} unique completed and audited replays**: "
        "2 fixed Gazebo flight recordings x 4 routing policies x 3 per-invocation "
        "perception deadlines x 3 network traces x 2 initial battery levels x 2 intents. "
        "Every condition is crossed with every other condition. The same recorded "
        "camera trajectory and cached neural predictions are reused across policies.",
        "",
        "The requested one-, two-, three- and four-variable experiments are reported "
        f"as **15 anchored slices** of this grid, containing {memberships:,} overlapping "
        "execution memberships. A run appearing in several slices was executed once; "
        "these memberships are not additional trials or statistical replicates.",
        "",
        "| Factor | Tested levels | Baseline |",
        "|---|---|---|",
    ]
    for factor in FACTORS:
        levels = ", ".join(value_label(factor, v) for v in design["levels"][factor])
        lines.append(
            f"| {FACTOR_LABELS[factor]} | {levels} | "
            f"{value_label(factor, design['baseline'][factor])} |"
        )
    lines += [
        "",
        "The association budget is held at **150 ms** and verification at **30 ms** "
        "throughout. The deadline factor changes only the **perception invocation "
        "deadline (2, 4 or 8 seconds)**. Phase/report deadlines and the mission duration "
        "remain separate. Thus the one-variable deadline experiment does not also "
        "change downstream processing budgets.",
        "",
        "Good, constrained and intermittent networks are the previously frozen "
        "virtual link traces, including bandwidth, RTT and outages. Battery is the "
        "initial charge fraction; a common reserve requirement remains in force. "
        "Intent changes the mission evidence and reporting requirements, rather "
        "than the flight path. Exact trace segments and intent contracts are preserved "
        "in the bound plan and per-run protocol files.",
        "",
        "## Policies and common execution rules",
        "",
        "| Policy | Selection rule |",
        "|---|---|",
        "| Local fast | Always selects local YOLO11s at 960 pixels. |",
        "| Local heavy | Always selects local Mask R-CNN at 1280 pixels with the 0°/90° union. |",
        "| Remote union | Always selects the remote placement of that same heavy union. |",
        "| Deadline aware | Filters actions using forecast invocation time, current link, memory, "
        "privacy, byte and battery-reserve feasibility; ranks eligible actions using "
        "development recall/risk heads and public intent requirements; abstains if none qualify. |",
        "",
        "These are three fixed controls and one authored adaptive reference, not four "
        "independently published routing protocols. All use the same terrain-based "
        "association and Repeat2 verifier. The adaptive router sees the current link "
        "snapshot and redacted intent, not future network segments or evaluator labels. "
        "The candidate pool contains 9 vision configurations with local and remote "
        "placements. This experiment evaluates perception routing; downstream "
        "algorithm choices remain fixed.",
        "",
        "Preparation, routing, loading, communication and inference costs count "
        "toward the relevant invocation. Late predictions and late downstream "
        "results do not become mission evidence. Local work is nonpreemptive: "
        "finishing and discarding still consumes time and energy. Remote transfers "
        "can stop at the task bound, and consumed bytes/energy remain charged. "
        "A paid task miss permits later mission recovery; it does not automatically "
        "erase all later work. Full mission success still requires the intent, "
        "reporting and resource constraints to pass.",
        "",
        "## Full-grid performance, with intent separated",
        "",
        "**Strict inspection is a known sensor-infeasible control on these two "
        "recordings (144 runs).** Its full-mission failure is not evidence that "
        "every router is poor. It is reported separately from rapid survey "
        "(144 runs). Rapid-survey feasibility is not assumed under every resource "
        "combination. The two recordings are paired development blocks, not a "
        "representative sample of real-world missions.",
        "",
        "Rates below are means across runs; deadline-miss rate is the mean of "
        "per-run miss/attempt rates, not a pooled ratio. NA denotes a zero-denominator "
        "metric. Capture coverage measures attempted capture slots; it is distinct "
        "from the fraction that produces an admitted prediction. Energy includes "
        "the common flight cost. Report recall and phase completion are diagnostics, "
        "not substitutes for full mission success.",
        "",
    ]
    for intent in design["levels"]["intent"]:
        lines += [
            f"### {INTENT_LABELS[intent]}",
            "",
            "| Policy | Full missions | Phases passed | Report recall | "
            "Perception deadline misses | Capture coverage | Final battery | "
            "Energy (J) | Traffic (MB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for policy in policies:
            lines.append(score_row(policy, select_cells(analysis, intent=intent, policy=policy)))
        lines.append("")
    lines += [
        "![Full-grid performance](figures/full_grid_summary.png)",
        "",
        "## One-variable sensitivity",
        "",
        "Each panel varies only its named factor; all other factors remain "
        "at the baseline (8 s, good network, 65% initial battery, rapid survey). "
        "Each point averages the two fixed recordings for the same policy. "
        "Lines connect tested categorical levels and do not imply interpolation. "
        "Strict* identifies the known infeasible intent control. Error bars "
        "are omitted because these are deterministic paired development replays.",
        "",
        "![One-factor mission success](figures/single_factor_mission.png)",
        "",
        "![One-factor phase completion](figures/single_factor_phase.png)",
        "",
        "## Two-variable slices",
        "",
        "Each panel crosses its two named factors while keeping the other two "
        "at baseline. Every heatmap cell is the mean of the two recordings. "
        "Panels containing intent explicitly include the strict infeasible control. "
        "These are conditional sensitivity maps, not estimates of statistical interaction.",
        "",
        "![Two-factor mission success](figures/pairwise_mission.png)",
        "",
        "![Two-factor phase completion](figures/pairwise_phase.png)",
        "",
        "## All requested anchored slices",
        "",
        "The slice order is the number of factors allowed to vary, including "
        "their baseline levels. A four-factor slice is the entire grid; it is "
        "not restricted to cells where all four factors differ from baseline. "
        "The table gives full success counts; per-cell and secondary metrics "
        "are available in `analysis.json` and `cells.csv`.",
        "",
        "| Varied factors | Order | Conditions | Execution memberships | "
        + " | ".join(POLICY_LABELS[p] for p in policies)
        + " |",
        "|---|---:|---:|---:|" + "---:|" * len(policies),
    ]
    for part in analysis["slices"]:
        subset = {c["run_id"]: c for c in cells if c["run_id"] in set(part["run_ids"])}
        values = []
        for policy in policies:
            group = [c for c in subset.values() if c["policy"] == policy]
            values.append(f"{sum(c['metrics']['mission_success'] for c in group):.0f}/{len(group)}")
        lines.append(
            f"| {' + '.join(FACTOR_LABELS[f] for f in part['varied_factors'])} | "
            f"{part['order']} | {part['condition_count']} | {len(part['run_ids'])} | "
            + " | ".join(values)
            + " |"
        )
    lines += [
        "",
        "### Exactly how many factors differ from baseline",
        "",
        "These disjoint groups partition the 288 executions. They answer a "
        "different question from the overlapping slices above. Their mixtures "
        "of factor levels differ; a trend across order cannot be attributed "
        "to the number of changed factors alone.",
        "",
        "| Factors different from baseline | Unique runs | "
        + " | ".join(POLICY_LABELS[p] for p in policies)
        + " |",
        "|---:|---:|" + "---:|" * len(policies),
    ]
    for order in range(5):
        group = [c for c in cells if c["change_order"] == order]
        counts = []
        for policy in policies:
            selected = [c for c in group if c["policy"] == policy]
            counts.append(
                f"{sum(c['metrics']['mission_success'] for c in selected):.0f}/{len(selected)}"
            )
        lines.append(f"| {order} | {len(group)} | " + " | ".join(counts) + " |")
    lines += [
        "",
        "## Factor interaction contrasts",
        "",
        "A changed-cell minus baseline difference contains main effects and "
        "interactions together. We therefore use anchored inclusion-exclusion "
        "contrasts. For two factors, the contrast is "
        "`Y(A,B) - Y(A,base) - Y(base,B) + Y(base,base)`. "
        "Higher orders use the analogous signed sum over all 2^k corners, "
        "within the same flight and policy. Order 1 is a simple finite difference; "
        "only orders 2-4 measure non-additivity. These descriptive contrasts "
        "are not significance tests or causal effects beyond this simulator.",
        "",
        "Intent-involving contrasts also change what counts as success. "
        "They describe objective sensitivity and must not be interpreted as "
        "resource-only interactions. Resource-only contrasts hold rapid intent "
        "fixed and are reported separately. Zero effects can arise from a "
        "binary success floor; phase completion and costs remain informative.",
        "",
        "| Order | Contrasts | Nonzero mission contrasts | Resource-only contrasts | "
        "Nonzero resource-only mission contrasts |",
        "|---:|---:|---:|---:|---:|",
    ]
    for summary in analysis["interaction_summary"]:
        lines.append(
            f"| {summary['order']} | {summary['contrasts']} | "
            f"{summary['nonzero_by_metric']['mission_success']} | "
            f"{summary['resource_only_contrasts']} | "
            f"{summary['resource_only_nonzero_by_metric']['mission_success']} |"
        )
    lines += [
        "",
        "![Interaction magnitudes](figures/interaction_magnitude.png)",
        "",
        "The plot shows the mean absolute finite-difference magnitude, in "
        "percentage points, within each policy and order. It removes sign "
        "cancellation; it does not indicate improvement or the probability "
        "of an interaction. Orders contain different level combinations. "
        "The four-factor resource-only contrast is undefined because the "
        "fourth factor is intent; it is omitted.",
        "",
        "## Evidence boundaries and next experiment",
        "",
        "This fresh execution tests an unchanged router under a prespecified "
        "factor grid. It does not train or tune a policy after seeing these "
        "outcomes, and it preserves the earlier 192-run study. Successful "
        "differentiation shows that this benchmark responds to model-selection "
        "and resource decisions within these recordings; it does not prove "
        "adaptive-policy superiority or generalization.",
        "",
        "The trajectories and RGB originate in Gazebo, while neural masks "
        "are cached outputs of actual models. This run performs no new "
        "neural inference, live remote service request, wireless measurement "
        "or electrical-power measurement. Network, downstream processing "
        "and energy costs remain authored models; invocation accounting "
        "is virtual and uses the fixed scheduling tick. Only two known "
        "development recordings and four references are included. Stronger "
        "claims require feasible held-out missions, more flight/environment "
        "blocks, measured remote and power traces, and independently designed "
        "adaptive routing baselines.",
        "",
        "The next policy revision should use these results for diagnosis, "
        "then be frozen and tested on newly held-out feasible recordings. "
        "It should not be evaluated as a new method on this same grid "
        "without labeling the comparison as development tuning.",
        "",
        "## Reproducibility",
        "",
        "`analysis_design.json` defines the levels and slices. "
        "`orchestration_manifest.json` freezes execution/analysis/report sources "
        "before outcomes. `plan.json` binds the scenario, neural prediction, "
        "cost and source inputs. `completion.json` and "
        "`independent_deadline_audit.json` bind the completed run artifacts. "
        "`analysis.json`, `cells.csv` and `interactions.csv` retain the computed "
        "metrics and contrast terms. `report_sources.json` binds this report "
        "and its figures to the checked analysis. Figure PDFs are vector "
        "exports of the PNG charts, suitable for manuscript reuse.",
        "",
    ]
    return "\n".join(lines)


def make_figures(analysis, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    policies = analysis["design"]["policies"]
    figures = []

    def save(fig, name):
        for extension in ("png", "pdf"):
            path = output / f"{name}.{extension}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            figures.append(path)
        plt.close(fig)

    for metric, suffix, label in (
        ("mission_success", "mission", "Full mission success (%)"),
        ("phase_fraction", "phase", "Phases passed (%)"),
    ):
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.3), sharey=True)
        for ax, factor in zip(axes, FACTORS, strict=True):
            levels = analysis["design"]["levels"][factor]
            for policy, color, marker in zip(policies, COLORS, MARKERS, strict=True):
                ys = [
                    100
                    * mean_metric(
                        anchored_cells(analysis, [factor], policy=policy, **{factor: value}), metric
                    )
                    for value in levels
                ]
                ax.plot(
                    range(len(levels)),
                    ys,
                    color=color,
                    marker=marker,
                    linewidth=1.6,
                    label=POLICY_LABELS[policy],
                    markersize=7,
                    alpha=0.9,
                )
            ax.set_xticks(range(len(levels)), [value_label(factor, v) for v in levels])
            ax.set_title(FACTOR_LABELS[factor])
            ax.set_ylim(-5, 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis="y", alpha=0.2)
        axes[0].set_ylabel(label)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04), frameon=False
        )
        fig.text(
            0.5,
            -0.025,
            "Other factors fixed: 8 s / good link / 65% battery / rapid intent. "
            "Each point: two recordings. Strict*: infeasible control.",
            ha="center",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.94))
        save(fig, f"single_factor_{suffix}")

        fig, axes = plt.subplots(3, 2, figsize=(18, 13))
        for ax, pair in zip(axes.flat, itertools.combinations(FACTORS, 2), strict=True):
            combinations = list(itertools.product(*(analysis["design"]["levels"][f] for f in pair)))
            data = [
                [
                    100
                    * mean_metric(
                        anchored_cells(
                            analysis, pair, policy=policy, **dict(zip(pair, levels, strict=True))
                        ),
                        metric,
                    )
                    for levels in combinations
                ]
                for policy in policies
            ]
            im = ax.imshow(data, aspect="auto", cmap="Blues", vmin=0, vmax=100)
            ax.set_yticks(range(len(policies)), [POLICY_LABELS[p] for p in policies])
            ax.set_xticks(
                range(len(combinations)),
                [
                    "\n".join(value_label(f, v, True) for f, v in zip(pair, levels, strict=True))
                    for levels in combinations
                ],
            )
            ax.set_title(" + ".join(FACTOR_LABELS[f] for f in pair), pad=12)
            for row, values in enumerate(data):
                for col, value in enumerate(values):
                    ax.text(
                        col,
                        row,
                        f"{value:.0f}",
                        ha="center",
                        va="center",
                        color="white" if value >= 60 else "#15283b",
                        fontsize=11,
                    )
        fig.suptitle(
            label + " — two varying factors; other factors held at baseline", fontsize=16, y=0.99
        )
        fig.text(
            0.5,
            0.012,
            "Cells average two fixed recordings. Strict*: known sensor-infeasible control. "
            "Conditional sensitivity is not an interaction estimate.",
            ha="center",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0.04, 0.94, 0.965), h_pad=2.5, w_pad=2)
        bar_ax = fig.add_axes([0.957, 0.21, 0.012, 0.58])
        fig.colorbar(im, cax=bar_ax, label=label)
        save(fig, f"pairwise_{suffix}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4))
    metrics = (
        ("mission_success", "Full mission success (%)"),
        ("phase_fraction", "Phases passed (%)"),
        ("perception_deadline_miss_rate", "Perception deadline misses (%)"),
    )
    for row, intent in enumerate(analysis["design"]["levels"]["intent"]):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            values = [
                100 * mean_metric(select_cells(analysis, policy=p, intent=intent), metric)
                for p in policies
            ]
            bars = ax.bar(range(len(policies)), values, color=COLORS)
            ax.bar_label(bars, [f"{v:.1f}" for v in values], padding=4, fontsize=11)
            ax.set_xticks(
                range(len(policies)), [POLICY_LABELS[p].replace(" ", "\n", 1) for p in policies]
            )
            ax.set_ylim(0, 110)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_title(label)
            ax.grid(axis="y", alpha=0.2)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(
                    "Rapid survey" if row == 0 else "Strict inspection\n(infeasible control)"
                )
    fig.suptitle("Full factorial grid: 36 runs per policy and intent", fontsize=16)
    fig.text(
        0.5,
        0.015,
        "Each condition uses two fixed development recordings. "
        "Perception miss rate is the unweighted mean of per-run rates. No statistical error bars.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=2)
    save(fig, "full_grid_summary")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8))
    for row, resource_only in enumerate((False, True)):
        for col, metric in enumerate(("mission_success", "phase_fraction")):
            ax = axes[row, col]
            orders = range(1, 4 if resource_only else 5)
            for policy, color, marker in zip(policies, COLORS, MARKERS, strict=True):
                values = []
                for order in orders:
                    terms = [
                        abs(c["effects"][metric])
                        for c in analysis["interactions"]
                        if c["policy"] == policy
                        and c["order"] == order
                        and (not resource_only or not c["involves_intent"])
                        and c["effects"][metric] is not None
                    ]
                    values.append(100 * numeric_mean(terms))
                ax.plot(
                    list(orders),
                    values,
                    color=color,
                    marker=marker,
                    label=POLICY_LABELS[policy],
                    markersize=7,
                )
            ax.set_xticks(list(orders))
            ax.set_xlabel("Contrast order (1 = main difference)")
            ax.set_ylabel("Mean absolute contrast (pp)")
            ax.set_title(
                ("Resources only" if resource_only else "All factors, including intent")
                + "\n"
                + ("Full mission success" if col == 0 else "Phase completion")
            )
            ax.grid(alpha=0.2)
            ax.set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.text(
        0.5,
        0.01,
        "Anchored inclusion-exclusion within each flight/policy. "
        "Magnitude is not direction, probability, significance, or a raw baseline difference.",
        ha="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), h_pad=2)
    save(fig, "interaction_magnitude")
    return figures


def export_csv(analysis, root):
    """Preserve raw run rows and signed contrast terms for external inspection."""
    root = Path(root)
    paths = []
    for name, rows in (
        (
            "cells.csv",
            [
                dict(
                    run_id=c["run_id"],
                    flight=c["flight"],
                    policy=c["policy"],
                    **c["factors"],
                    change_order=c["change_order"],
                    known_infeasible_intent=c["known_infeasible_intent"],
                    **c["metrics"],
                )
                for c in analysis["cells"]
            ],
        ),
        (
            "interactions.csv",
            [
                dict(
                    contrast_id=c["contrast_id"],
                    flight=c["flight"],
                    policy=c["policy"],
                    order=c["order"],
                    involves_intent=c["involves_intent"],
                    varied_factors=json.dumps(c["varied_factors"], sort_keys=True),
                    levels=json.dumps(c["levels"], sort_keys=True),
                    corner_terms=json.dumps(c["corner_terms"], sort_keys=True),
                    **c["effects"],
                )
                for c in analysis["interactions"]
            ],
        ),
    ):
        require(rows, "Cannot export empty analysis table")
        path = root / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths


def build(root):
    root = Path(root).resolve()
    analysis = verify_analysis(root)
    markdown = render_markdown(analysis)
    paths = make_figures(analysis, root / "figures")
    paths += export_csv(analysis, root)
    report = root / "REPORT.md"
    report.write_text(markdown)
    paths.append(report)
    provenance = {
        "schema_version": "mission_factor_sweep_report_sources_v1",
        "complete": True,
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": digest(__file__),
        "source_sha256": {
            "analysis.json": digest(root / "analysis.json"),
            **analysis["source_sha256"],
        },
        "generated_sha256": {str(p.relative_to(root)): digest(p) for p in paths},
        "unique_executions": len(analysis["cells"]),
        "overlapping_slice_memberships": sum(len(s["run_ids"]) for s in analysis["slices"]),
        "outcomes_changed": False,
        "statistical_inference": False,
    }
    (root / "report_sources.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = build(args.root)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "figures": len(PLOT_NAMES),
                "unique_executions": result["unique_executions"],
            }
        )
    )


if __name__ == "__main__":
    main()
