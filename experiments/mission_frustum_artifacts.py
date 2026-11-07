"""Hash-check a completed frustum study and export its descriptive comparison figure."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_frustum_routing.protocol import verify
from experiments.mission_frustum_routing.report import checked_events


def run(root):
    p = verify(root)
    completion = read(root / "completion.json")
    if not completion["complete"]:
        raise ValueError("Incomplete report")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed completed artifact: " + name)
    a = read(root / "analysis.json")
    diagnostics = []
    for r in a["policy_cases"]:
        events = checked_events(root, r)
        diagnostics.append(
            dict(
                flight=r["flight"],
                mission=r["mission"],
                policy=r["policy"],
                calls=len(events),
                false_components=sum(e["score"]["false_positive_components"] for e in events),
            )
        )
    for r in a["reused_fixed_controls"]:
        checked_events(Path(p["parent"]), r)
    save(root / "diagnostics.json", dict(cases=diagnostics, source_sha256=sha(Path(__file__))))
    rows = a["policy_summary"]
    labels = [r["policy"].replace("static:CFG_", "").replace("_", " ") for r in rows]
    colors = ["#777f8b" if r["reused"] else "#246da7" for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    for ax, key, limit, title in zip(
        axes,
        ("successes", "phases"),
        (8, 32),
        ("Full mission completion / 8", "Completed phases / 32"),
        strict=True,
    ):
        ax.barh(range(len(rows)), [r[key] for r in rows], color=colors)
        ax.set_xlim(0, limit)
        ax.set_title(title)
        ax.set_yticks(range(len(rows)), labels)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        for i, r in enumerate(rows):
            ax.text(r[key] + 0.1, i, str(r[key]), va="center", fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle("Frustum feedback: two inspected development layouts", fontsize=15)
    fig.text(
        0.5,
        0.02,
        "32 new replays (blue); 72 fixed references (gray). "
        "All four strict cases are sensor-infeasible. No new physical acquisition.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(root / "frustum_comparison.png", dpi=160)
    plt.close(fig)
    save(
        root / "release_audit.json",
        dict(
            complete=True,
            fresh_test=False,
            research_readiness_certified=False,
            new_policy_replays=32,
            new_physical_flights=0,
            new_neural_calls=0,
            artifacts_sha256={
                name: sha(root / name)
                for name in (
                    "protocol.json",
                    "completion.json",
                    "analysis.json",
                    "validation.json",
                    "frustum_audit.json",
                    "REPORT.md",
                    "diagnostics.json",
                    "frustum_comparison.png",
                )
            },
            release_source_sha256=sha(Path(__file__)),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
