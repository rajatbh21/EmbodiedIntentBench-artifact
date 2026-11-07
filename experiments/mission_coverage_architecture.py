"""Export a paper figure showing the tested information boundary and fixed-flight design."""

import argparse
from pathlib import Path


def run(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set(xlim=(0, 14), ylim=(0, 6))
    ax.axis("off")

    def box(x, y, w, h, title, detail, color="#e2e8f0"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.08,rounding_size=0.08",
                facecolor=color,
                edgecolor="#64748b",
                linewidth=1,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.25,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#0f172a",
        )
        ax.text(
            x + w / 2,
            y + h / 2 - 0.2,
            detail,
            ha="center",
            va="center",
            fontsize=10.5,
            linespacing=1.5,
            color="#334155",
        )

    def arrow(start, end, label=None, *, color="#475569", dashed=False, rad=0):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color=color,
                linestyle="--" if dashed else "-",
                connectionstyle=f"arc3,rad={rad}",
            )
        )
        if label:
            ax.text(
                (start[0] + end[0]) / 2,
                (start[1] + end[1]) / 2 + 0.16,
                label,
                ha="center",
                fontsize=9,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=1),
            )

    box(
        0.25,
        3.05,
        2.55,
        1.5,
        "Shared physical flight",
        "Fixed route and poses\nSynchronized RGB bank",
    )
    box(3.65, 4.4, 2.5, 1.05, "Public mission intent", "Evidence, deadlines, reserve")
    box(
        3.65,
        2.75,
        2.5,
        1.15,
        "Routing policy",
        "Current RGB + public state\nSelect one configuration",
        "#ccfbf1",
    )
    box(
        7.1,
        2.75,
        2.8,
        1.15,
        "Selected model result",
        "Frozen weights and masks\nOnly chosen output exposed",
        "#dbeafe",
    )
    box(
        10.75,
        2.75,
        2.9,
        1.15,
        "Prediction-based reports",
        "Completed outputs only\nTrack and confirm people",
    )
    box(
        4.0,
        0.75,
        5.7,
        1.1,
        "Paid simulation clock and resource ledger",
        "Inference + loading + routing + tracking\n"
        "Busy captures skipped; full-flight energy charged",
        "#f1f5f9",
    )
    box(
        10.75,
        0.65,
        2.9,
        1.25,
        "Full mission contract",
        "All phases + reports\nDeadlines + resource limits",
        "#fef3c7",
    )
    box(
        0.25,
        0.7,
        2.55,
        1.2,
        "Evaluator-only truth",
        "Target identity / visibility\nNever in policy inputs",
        "#fef3c7",
    )
    arrow((2.88, 3.55), (3.55, 3.35), "RGB")
    arrow((4.9, 4.32), (4.9, 3.98))
    arrow((6.23, 3.35), (7.0, 3.35), "Action")
    arrow((9.98, 3.35), (10.65, 3.35), "Result")
    arrow((8.5, 2.67), (8.5, 1.93), "Costs")
    arrow((4.9, 1.93), (4.9, 2.67), "Clock / battery")
    arrow((12.2, 2.67), (12.2, 1.98))
    arrow((9.78, 1.25), (10.65, 1.25))
    arrow((2.88, 1.05), (10.65, 0.65), color="#a16207", dashed=True, rad=0.15)
    arrow((11.25, 3.98), (6.22, 3.85), "Completed prediction history", rad=0.35)
    ax.text(
        0.25,
        5.65,
        "Mission Routing Benchmark: change the routing policy, hold the flight fixed",
        fontsize=16,
        fontweight="bold",
        color="#0f172a",
    )
    ax.text(
        7.0,
        5.1,
        "Dense neural inference is precomputed for reproducible replay.\n"
        "Its unselected outputs and ground truth are evaluator-only.",
        fontsize=10.5,
        color="#475569",
        linespacing=1.5,
    )
    ax.text(
        0.25,
        0.02,
        "Tested scope: authored person-survey missions, declared costs, ideal pose/terrain. "
        "Information isolation is at the API level, not an OS security sandbox.",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(pad=0.5)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / ("benchmark_architecture." + suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
