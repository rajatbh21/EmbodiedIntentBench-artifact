"""Complete fixed-grid mission comparison; no outcome-based comparator selection."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v6.flights import verify
from experiments.mission_routing_v6.study import POLICIES

LABELS = {
    "base_recall": "Expanded base recall (primary)",
    "spatial_recall": "Expanded spatial recall",
    "global_recall": "Expanded global recall",
    "original_base_recall": "Original base recall",
    "risk_aware_reference": "Historical risk-aware",
    "confirmation_reference": "Historical confirmation",
}


def run(root):
    p = verify(root)
    data = read(root / "analysis_v6.json")
    if data["protocol_sha256"] != sha(root / "protocol.json") or data["validation_sha256"] != sha(
        root / "validation.json"
    ):
        raise ValueError("Changed figure source")
    cases = [(f, m) for f in p["cases"] for m in p["missions"]]
    lookup = {(r["policy"], r["flight"], r["mission"]): r for r in data["cases"]}
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), layout="constrained")
    for ax, field, title, maximum in zip(
        axes,
        ("phases_succeeded", "correct_targets_reported"),
        ("Completed mission phases / 4", "Correctly reported targets / 6"),
        (4, 6),
        strict=True,
    ):
        values = np.array([[lookup[n, f, m][field] for f, m in cases] for n in POLICIES])
        ax.imshow(values, vmin=0, vmax=maximum, cmap="Blues", aspect="auto")
        for i, name in enumerate(POLICIES):
            for j, (flight, mission) in enumerate(cases):
                r = lookup[name, flight, mission]
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if values[i, j] > maximum * 0.55 else "#15283b",
                )
                if r["full_success"]:
                    ax.add_patch(
                        Rectangle(
                            (j - 0.46, i - 0.43),
                            0.92,
                            0.86,
                            fill=False,
                            edgecolor="#d55e00",
                            linewidth=2,
                        )
                    )
        ax.set_xticks(
            range(len(cases)),
            [
                ("Day" if "day" in f else "Dusk")
                + "\n"
                + {
                    "rapid_survey": "Rapid",
                    "verified_inspection": "Inspect",
                    "reserve_patrol": "Reserve",
                }[m]
                for f, m in cases
            ],
            fontsize=10,
        )
        ax.set_yticks(
            range(len(POLICIES)),
            [LABELS.get(n, n.replace("static:CFG_", "Fixed ")) for n in POLICIES],
            fontsize=9,
        )
        ax.set_title(title, fontsize=13, pad=12)
        ax.axhline(5.5, color="#999999", linewidth=1.5)
    fig.suptitle(
        "Fresh-layout routing evaluation: every predeclared policy\n"
        "Orange outline = full mission contract success",
        fontsize=17,
    )
    fig.supxlabel(
        "Same physical flight, nine configurations, costs and intent contracts. "
        "One known map; correlated cases, not population estimates.",
        fontsize=10,
    )
    fig.savefig(root / "policy_comparison.png", dpi=160)
    plt.close(fig)
    save(
        root / "policy_comparison_provenance.json",
        dict(
            schema_version="v6_comparison_figure_v1",
            source_sha256=sha(Path(__file__)),
            analysis_sha256=sha(root / "analysis_v6.json"),
            image_sha256=sha(root / "policy_comparison.png"),
            all_predeclared_policies=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
