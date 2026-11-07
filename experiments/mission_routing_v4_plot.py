"""Plot primary full-contract outcomes only after the release evidence audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v4.flights import verify
from experiments.mission_routing_v4_report import run as report


def run(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    p = verify(root)
    status = report(root)
    rows = status["cases"]
    primary = {(r["flight"], r["mission"], r["policy"]): r for r in rows if r["power_w"] == 15}
    columns = [(f, m) for f in p["cases"] for m in p["missions"]]
    labels = [
        ("Day" if f == "flight_1_day" else "Dusk")
        + "\n"
        + {
            "rapid_survey": "Rapid\nsurvey",
            "verified_inspection": "Verified\ninspection",
            "reserve_patrol": "Reserve\npatrol",
        }[m]
        for f, m in columns
    ]
    readable = {
        "confirmation": "Confirmation-aware (v3)",
        "risk_aware": "Risk-aware (v4)",
        "no_risk_ledger_v4": "v4: no risk ledger",
        "no_risk_history_v4": "v4: no track history",
        "global_risk_v4": "v4: global risk estimates",
        "no_history_v3": "v3: no history",
        "no_intent_v3": "v3: no intent",
        "no_context_v3": "v3: no RGB context",
        "constant_budget_v3": "v3: constant-model budget",
        "intent_history": "Intent/history (v2)",
        "performance_predictor": "Performance predictor (v2)",
        "offline_contextual": "Offline contextual (v2)",
        "no_history": "v2: no history",
        "no_intent": "v2: no intent",
        "no_context": "v2: no RGB context",
        "resource_only": "Resource only (v2)",
        "periodic_training_top_two": "Periodic training top two",
    }
    names = [
        readable.get(
            n,
            n.replace("static:CFG_", "Fixed: ")
            .replace("_1280", "")
            .replace("_640", "")
            .replace("_960", ""),
        )
        for n in p["policies"]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), sharey=True)
    for axis, field, title, palette in zip(
        axes,
        ("full_success", "report_recall"),
        ("Full mission success", "Correct report recall by deadline"),
        (ListedColormap(["#edf1f4", "#087e76"]), "Blues"),
        strict=True,
    ):
        values = np.array(
            [[primary[f, m, policy][field] for f, m in columns] for policy in p["policies"]],
            dtype=float,
        )
        axis.imshow(values, vmin=0, vmax=1, cmap=palette, aspect="auto")
        axis.set_title(title, fontsize=13, pad=14)
        axis.set_xticks(range(len(columns)), labels, fontsize=9)
        axis.set_yticks(range(len(names)), names, fontsize=9)
        axis.tick_params(length=0, pad=8)
        for i in range(len(names)):
            for j in range(len(columns)):
                v = values[i, j]
                label = ("PASS" if v else "FAIL") if field == "full_success" else f"{v:.2f}"
                axis.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if v > 0.6 else "#243744",
                )
        for y in np.arange(0.5, len(names), 1):
            axis.axhline(y, color="white", linewidth=1)
        axis.axvline(2.5, color="#93a6b0", linewidth=2)
        for boundary in (3.5, 4.5):
            axis.axhline(boundary, color="#93a6b0", linewidth=2)
        for spine in axis.spines.values():
            spine.set_visible(False)
    fig.suptitle("Fixed flight, different routing policies", fontsize=20, y=0.98)
    fig.text(
        0.5,
        0.935,
        f"{len(p['policies'])} policies / {len(p['missions'])} intents / day and dusk | "
        f"15 W assumed compute power | {status['independent_layouts']} known-map layout(s)",
        ha="center",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.025,
        "Correlated replays; no population-level significance claim.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.27, right=0.985, top=0.87, bottom=0.10, wspace=0.10)
    path = root / "mission_outcomes.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    save(
        root / "mission_outcomes_provenance.json",
        dict(
            schema_version="mission_routing_v4_plot_v1",
            protocol_sha256=sha(root / "protocol.json"),
            status_sha256=sha(root / "analysis_v4.json"),
            figure_sha256=sha(path),
            implementation_sha256=sha(Path(__file__)),
            primary_executions=len(primary),
        ),
    )
    # Separate the two quality units: fewer component errors can coexist with
    # fewer correct reports. A failure-only success heatmap cannot show this.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(columns))
    for offset, policy, label, color in (
        (-0.19, "risk_aware", "Risk-aware v4", "#087e76"),
        (0.19, "confirmation", "Confirmation v3", "#64748b"),
    ):
        axes[0].bar(
            x + offset,
            [primary[f, m, policy]["total_false_components"] for f, m in columns],
            width=0.36,
            label=label,
            color=color,
        )
        axes[1].bar(
            x + offset,
            [primary[f, m, policy]["report_recall"] for f, m in columns],
            width=0.36,
            label=label,
            color=color,
        )
    requirements = [0.75 if m == "verified_inspection" else 0.5 for _, m in columns]
    axes[1].plot(
        x,
        requirements,
        "_",
        color="#b23a48",
        markersize=35,
        markeredgewidth=2,
        label="Required report recall",
    )
    axes[0].set_title("False component observations (lower is better)")
    axes[1].set_title("Correct report recall (higher is better)")
    axes[1].set_ylim(0, 1.02)
    for axis in axes:
        axis.set_xticks(x, labels, fontsize=9)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.legend(fontsize=9)
        for name in ("top", "right"):
            axis.spines[name].set_visible(False)
    fig.suptitle("False detections and report quality under fixed flight", fontsize=17)
    fig.text(
        0.5,
        0.025,
        "Paired fixed-flight cases; configured 15 W compute. Full success: "
        f"v4 {status['successes_by_policy']['risk_aware']}/{len(columns)}, "
        f"v3 {status['successes_by_policy']['confirmation']}/{len(columns)}.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.19, top=0.86, wspace=0.22)
    tradeoff = root / "risk_tradeoff.png"
    fig.savefig(tradeoff, dpi=170)
    plt.close(fig)
    save(
        root / "risk_tradeoff_provenance.json",
        dict(
            schema_version="v4_risk_tradeoff_figure_v1",
            status_sha256=sha(root / "analysis_v4.json"),
            figure_sha256=sha(tradeoff),
            implementation_sha256=sha(Path(__file__)),
        ),
    )
    print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
