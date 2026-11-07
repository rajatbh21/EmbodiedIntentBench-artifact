"""Render the audited joint-resource rapid-survey grid without pooling studies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

REPO = Path(__file__).resolve().parents[1]
STUDY = REPO / "results/mission_joint_constraints_v1_20261027"
OUTPUT = REPO / "results/benchmark_scope_update_20261027"
LABELS = {
    "fixed_local_CFG_MASKRCNN_1280": "Fixed local Mask R-CNN",
    "fixed_local_CFG_MASKRCNN_1280_ROT0_90": "Fixed local Mask union",
    "fixed_local_CFG_YOLO11S_960_ROT0": "Fixed local YOLO11s",
    "fixed_remote_union": "Fixed remote Mask union",
    "snapshot_quality_2s": "Snapshot quality (2 s)",
    "mission_evidence_network": "Mission evidence / link aware",
    "mission_evidence_network_blind": "Mission evidence / link blind",
    "mission_evidence_local_only": "Mission evidence / local only",
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render():
    completion = read(STUDY / "completion.json")
    assert completion["complete"] is True
    for filename in ("summary", "validation", "plan", "run_index"):
        assert sha(STUDY / f"{filename}.json") == completion[f"{filename}_sha256"]
    summary = read(STUDY / "summary.json")
    rows = summary["executions_complete"]
    policies = [p["policy"] for p in summary["policies"]]
    assert len(rows) == 384 and set(policies) == set(LABELS)
    columns = [
        (link, b, c)
        for link in ("good", "constrained", "intermittent")
        for b in (0.34, 0.65)
        for c in (1.0, 2.0)
    ]
    grid = np.zeros((8, 12), dtype=int)
    cells = []
    for i, policy in enumerate(policies):
        for j, (link, charge, scale) in enumerate(columns):
            selected = [
                r
                for r in rows
                if r["policy"] == policy
                and r["mission"] == "rapid_survey"
                and r["link"] == link
                and r["initial_battery_frac"] == charge
                and r["onboard_latency_scale"] == scale
            ]
            assert len(selected) == 2 and len({r["flight"] for r in selected}) == 2
            grid[i, j] = sum(r["full_success"] for r in selected)
            cells.append(
                dict(
                    policy=policy,
                    link=link,
                    charge=charge,
                    compute_scale=scale,
                    successes=int(grid[i, j]),
                    denominator=2,
                    source_paths=[r["path"] for r in selected],
                )
            )
    strict = [r for r in rows if r["mission"] == "verified_inspection"]
    assert len(strict) == 192
    OUTPUT.mkdir(parents=True, exist_ok=True)
    colors = ["#f0f3f7", "#88b7d5", "#175b8d"]
    fig, ax = plt.subplots(figsize=(12.2, 5.6), layout="constrained")
    ax.imshow(
        grid,
        cmap=ListedColormap(colors),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3),
        aspect="auto",
    )
    ax.set_yticks(range(8), [LABELS[p] for p in policies], fontsize=10)
    ax.set_xticks(range(12), [f"{b:.0%}\n{c:g}x" for _, b, c in columns], fontsize=10)
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 8, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for x in (3.5, 7.5):
        ax.axvline(x, color="white", linewidth=5)
    for i in range(8):
        for j in range(12):
            ax.text(
                j,
                i,
                f"{grid[i, j]}/2",
                ha="center",
                va="center",
                fontsize=10,
                color="white" if grid[i, j] == 2 else "#182b3b",
            )
    for x, name in ((1.5, "Good link"), (5.5, "Constrained link"), (9.5, "Intermittent link")):
        ax.text(x, -0.85, name, ha="center", va="bottom", fontsize=11, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Initial battery / local inference-time multiplier", labelpad=8)
    fig.suptitle(
        "Rapid-survey mission successes under joint constraints", fontsize=15, fontweight="bold"
    )
    fig.supxlabel(
        "Each cell contains the same two reused development recordings. "
        "Authored links and power; fixed flight.\n"
        f"Separate sensor-infeasible strict controls: {sum(r['full_success'] for r in strict)}"
        "/192 successes; excluded from this rapid-only view, retained in the full study.",
        fontsize=9,
    )
    for extension in ("png", "pdf"):
        fig.savefig(OUTPUT / f"joint_constraints_rapid.{extension}", dpi=180)
    plt.close(fig)
    provenance = dict(
        study=str(STUDY),
        summary_sha256=sha(STUDY / "summary.json"),
        completion_sha256=sha(STUDY / "completion.json"),
        plot_source_sha256=sha(Path(__file__)),
        rows=192,
        independent_trials=False,
        grid=grid.tolist(),
        cells=cells,
        strict_control_successes=sum(r["full_success"] for r in strict),
    )
    (OUTPUT / "figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(OUTPUT / "joint_constraints_rapid.png")


if __name__ == "__main__":
    render()
