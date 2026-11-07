"""Scientific coverage and blocked-development-error figure; no mission outcome claim."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.routing_development_matrix.cohort import verify


def run(root):
    p = verify(root)
    result = read(root / "analysis.json")
    if result["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Wrong figure cohort")
    flights = list(p["cases"])
    pixels = []
    for flight in flights:
        data = read(root / flight / "development_labels/visibility.json")
        pixels.append([o["visible_pixels"] for o in data["observations"] if o["visible"]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")
    axes[0].boxplot(
        pixels,
        tick_labels=[
            f.replace("_day", "\nDay").replace("_dusk_haze", "\nDusk").replace("_", " ")
            for f in flights
        ],
        showfliers=False,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Visible GT mask area (pixels, log scale)")
    axes[0].set_title("Observed image scale; invisible targets excluded from boxes")
    axes[0].tick_params(axis="x", labelsize=8)
    groups = list(result["layout_blocked_errors"])
    modes = ("spatial", "base", "global_recall")
    for i, mode in enumerate(modes):
        values = [result["layout_blocked_errors"][g][mode]["macro_log_loss"] for g in groups]
        if any(v is None for v in values):
            raise ValueError("Do not plot unsupported folds as zero")
        axes[1].bar(
            np.arange(len(groups)) + (i - 1) * 0.24,
            values,
            width=0.24,
            label={"spatial": "Spatial RGB", "base": "Base RGB", "global_recall": "Global recall"}[
                mode
            ],
            color=("#0072b2", "#d55e00", "#009e73")[i],
        )
    axes[1].set_xticks(range(len(groups)), [g.replace("_", "\n") for g in groups])
    axes[1].set_ylabel("Macro recall log loss (lower is better)")
    axes[1].set_title("Hold out a whole layout and both lighting flights")
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Development expansion: sensor coverage and model-specific recall prediction", fontsize=14
    )
    fig.supxlabel(
        "One known map, correlated captures; no fresh policy evaluation "
        "or mission improvement claim.",
        fontsize=10,
    )
    fig.savefig(root / "coverage_and_transfer.png", dpi=160)
    plt.close(fig)
    save(
        root / "coverage_and_transfer_provenance.json",
        dict(
            schema_version="development_figure_v1",
            analysis_sha256=sha(root / "analysis.json"),
            source_sha256=sha(Path(__file__)),
            image_sha256=sha(root / "coverage_and_transfer.png"),
            fresh_policy_evaluation=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
