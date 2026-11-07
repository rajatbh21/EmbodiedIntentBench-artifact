"""Export source-geometry and actual-flight figures for the paired routing experiment."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha


def run(root):
    from experiments.mission_environment_transfer_v3.cohort import verify
    from experiments.mission_environment_transfer_v3.geometry import triangles
    from experiments.mission_transfer_comparison_v4.protocol import verify as verify_study
    from experiments.mission_transfer_within_recording.admission import verify as verify_admission

    p = verify(root)
    study = verify_study(root / "study_paired_scope_v1")
    verify_admission(Path(study["admission"]))
    output = root / "figures"
    output.mkdir(exist_ok=True)
    if (output / "scenario_layouts.png").exists():
        raise FileExistsError("Preserve the completed figure")
    xyz = triangles(p["assets"])
    low, high = xyz[:, :, 2].min(axis=1), xyz[:, :, 2].max(axis=1)
    # Show ground-level structure; the roof remains present in actual rendering/physics.
    visible = xyz[(high > 0.15) & (low < 2.2), :, :2]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), height_ratios=[1.6, 1])
    colors = ["#0072B2", "#D55E00", "#009E73"]
    inputs = {}
    for column, group in enumerate(("west_bay", "east_bay")):
        flight = group + "_day"
        scenario = read(root / flight / "rapid_survey/scenario.json")
        recording_path = root / flight / "recording/recording.json"
        recording = read(recording_path)
        inputs[flight] = sha(recording_path)
        frames = recording["frames"]
        times, poses = [], []
        for frame in frames:
            # Metadata is inspected directly; no model output or GT score enters the plot.
            times.append(frame["timestamp_ns"] / 1e9)
            poses.append(frame["body_pose"]["position_m"])
        times, poses = np.asarray(times), np.asarray(poses)
        times -= times[0]
        ax = axes[0, column]
        ax.set_facecolor("#f3efe6")
        ax.add_collection(PolyCollection(visible, facecolor="#ccc7bd", edgecolor="none"))
        ax.plot(poses[:, 0], poses[:, 1], color="#252525", linewidth=2, label="Recorded route")
        for i, obj in enumerate(scenario["world"]["objects"]):
            x, y, _ = obj["position_m"]
            ax.scatter(x, y, color=colors[i // 2], s=70, edgecolor="white", zorder=5)
            ax.annotate(str(i + 1), (x, y), xytext=(5, 6), textcoords="offset points", fontsize=9)
        ax.scatter(*poses[0, :2], marker="s", color="black", s=40, label="Start / return")
        ax.set_xlim((-15.5, 0) if column == 0 else (-1.5, 14))
        ax.set_ylim(-7.5, 7.5)
        ax.set_aspect("equal")
        ax.set_title(group.replace("_", " ").title() + " — source geometry and physical route")
        ax.set_xlabel("World x (m)")
        ax.set_ylabel("World y (m)")
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[1, column]
        for phase in range(4):
            ax.axvspan(30 * phase, 30 * (phase + 1), color=colors[[0, 1, 2, 0][phase]], alpha=0.1)
            ax.text(
                30 * phase + 15,
                0.95,
                f"Phase {phase + 1}",
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
                fontsize=9,
            )
        ax.plot(times, poses[:, 0], color="#252525", linewidth=2)
        ax.set_xlim(0, 120)
        ax.set_xlabel("Mission time (s)")
        ax.set_ylabel("Recorded x position (m)")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fixed physical flight; inference routing is the intervention", fontsize=16)
    fig.text(
        0.5,
        0.015,
        "Colors mark authored target groups: 1-2 (phases 1 and 4), 3-4 (phase 2), "
        "5-6 (phase 3). Roof omitted only in this top-down plot.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.045, 1, 0.95])
    for extension in ("png", "pdf"):
        fig.savefig(output / ("scenario_layouts." + extension), dpi=180)
    plt.close(fig)
    save(
        output / "scenario_layouts_provenance.json",
        dict(
            cohort_protocol_sha256=sha(root / "protocol.json"),
            recordings_sha256=inputs,
            source_sha256=sha(Path(__file__)),
            model_predictions_used=False,
            outputs_sha256={
                name: sha(output / name)
                for name in ("scenario_layouts.png", "scenario_layouts.pdf")
            },
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    run(parser.parse_args().cohort)
