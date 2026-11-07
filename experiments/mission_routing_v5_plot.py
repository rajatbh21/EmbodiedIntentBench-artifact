"""Static scientific figure for the exploratory comparison, with source hashes."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v5.protocol import verify


def run(root):
    p = verify(root)
    data = read(root / "analysis_v5.json")
    old = read(Path(p["source"]) / "analysis_v4.json")
    if (
        data["protocol_sha256"] != sha(root / "protocol.json")
        or data["validation_sha256"] != sha(root / "validation.json")
        or data["source_analysis_sha256"] != sha(Path(p["source"]) / "analysis_v4.json")
    ):
        raise ValueError("Changed figure source")
    cases = [(f, m) for f in p["cases"] for m in p["missions"]]
    names = ("risk_aware", "spatial_recall", "base_recall", "global_recall")
    labels = ("v4 risk", "v5 spatial recall", "v5 base recall", "v5 global recall")
    colors = ("#6c757d", "#0072b2", "#d55e00", "#009e73")
    rows = old["cases"] + data["cases"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True, layout="constrained")
    x, width = np.arange(len(cases)), 0.19
    for i, (name, label, color) in enumerate(zip(names, labels, colors, strict=True)):
        selected = [
            next(r for r in rows if r["policy"] == name and (r["flight"], r["mission"]) == case)
            for case in cases
        ]
        axes[0].bar(
            x + (i - 1.5) * width,
            [r["report_recall"] * 6 for r in selected],
            width,
            color=color,
            label=label,
        )
        axes[1].bar(
            x + (i - 1.5) * width,
            [r["total_false_components"] for r in selected],
            width,
            color=color,
        )
    axes[0].set_ylabel("Correctly reported targets / 6")
    axes[0].set_ylim(0, 6)
    axes[0].set_yticks(range(7))
    axes[0].legend(ncol=4, fontsize=9, loc="upper center")
    axes[1].set_ylabel("False component observations")
    axes[1].set_xticks(
        x,
        [
            f"{'Day' if 'day' in f else 'Dusk'}\n"
            + {
                "rapid_survey": "Rapid",
                "verified_inspection": "Inspection*",
                "reserve_patrol": "Reserve",
            }[m]
            for f, m in cases
        ],
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Conditional recall routing: exploratory reuse of inspected flights\n"
        "All four controllers: 0/6 full mission successes",
        fontsize=14,
    )
    fig.supxlabel(
        "* Inspection is latency-infeasible. Same route, masks, costs and contracts; "
        "no fresh test.",
        fontsize=10,
    )
    fig.savefig(root / "recall_tradeoff.png", dpi=160)
    plt.close(fig)
    save(
        root / "recall_tradeoff_provenance.json",
        dict(
            schema_version="v5_figure_provenance_v1",
            source_sha256=sha(Path(__file__)),
            analysis_sha256=sha(root / "analysis_v5.json"),
            original_analysis_sha256=data["source_analysis_sha256"],
            image_sha256=sha(root / "recall_tradeoff.png"),
            fresh_test=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
