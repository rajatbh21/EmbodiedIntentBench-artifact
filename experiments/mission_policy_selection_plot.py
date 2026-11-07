"""Static development figures with explicit input and plotting provenance."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha


def resource(root):
    source = root / "envelope.json"
    data = read(source)
    matrix = np.array(
        [
            [
                len(
                    next(
                        r
                        for r in data["grid"]
                        if r["power_w"] == power and r["initial_battery"] == battery
                    )["fixed_resource_feasible"]
                )
                for battery in data["batteries"]
            ]
            for power in data["powers"]
        ]
    )
    fig, ax = plt.subplots(figsize=(10, 4.6), layout="constrained")
    im = ax.imshow(matrix, vmin=0, vmax=9, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(data["batteries"])), [f"{100 * b:g}%" for b in data["batteries"]])
    ax.set_yticks(range(len(data["powers"])), [f"{p} W" for p in data["powers"]])
    ax.set_xlabel("Initial battery (20% reserve; 30 Wh capacity)")
    ax.set_ylabel("Configured active inference power")
    ax.set_title(
        "Resource-feasible fixed configurations / 9\n"
        "Cost analysis only; mission feasibility is not established",
        pad=13,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] >= 6 else "black",
                fontsize=14,
            )
    fig.colorbar(im, ax=ax, label="Configurations", ticks=[0, 3, 6, 9])
    finish(fig, root / "resource_envelope.png", source)


def comparison(root):
    source = root / "analysis.json"
    data = read(source)
    rows = data["ranking"]
    names = [r["policy"].replace("static:CFG_", "Fixed ").replace("_", " ") for r in rows]
    y = np.arange(len(rows))
    fig, axes = plt.subplots(1, 3, figsize=(16, 8), sharey=True, layout="constrained")
    colors = ["#3675a9" if not r["policy"].startswith("static:") else "#828b95" for r in rows]
    for ax, key, limit, title in zip(
        axes,
        ["successes", "phases", "macro_report_recall"],
        [12, 48, 1],
        ["Full mission success / 12", "Successful phases / 48", "Macro report recall"],
        strict=True,
    ):
        values = [r[key] for r in rows]
        ax.barh(y, values, color=colors, height=0.65)
        ax.set_xlim(0, limit * 1.13)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.text(
                value + limit * 0.02,
                i,
                f"{value:.3f}" if key == "macro_report_recall" else str(value),
                va="center",
                fontsize=10,
            )
    axes[0].set_yticks(y, names, fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Development policy selection: 168 paid replays on two layout blocks\n"
        "All strict-inspection cases retained; selection scores are not fresh-test results",
        fontsize=14,
    )
    finish(fig, root / "policy_comparison.png", source)


def finish(fig, path, source):
    fig.savefig(path, dpi=160)
    plt.close(fig)
    save(
        path.with_suffix(".provenance.json"),
        dict(
            source=str(source.resolve()),
            source_sha256=sha(source),
            plotter_sha256=sha(Path(__file__)),
            image_sha256=sha(path),
            fresh_test=False,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("resource", "comparison"))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    (resource if a.mode == "resource" else comparison)(a.output)
