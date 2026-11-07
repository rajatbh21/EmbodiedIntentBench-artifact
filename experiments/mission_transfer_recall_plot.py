"""Plot every frozen recall prediction against observed Depot recall without refitting."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha

LABELS = {
    "CFG_MASKRCNN_1280": "Mask R-CNN\nstandard",
    "CFG_MASKRCNN_1280_ROT90": "Mask R-CNN\nrotated",
    "CFG_MASKRCNN_1280_ROT0_90": "Mask R-CNN\nunion",
    "CFG_YOLO11S_960_ROT0": "YOLO11s\nfull frame",
    "CFG_YOLO11N_640_ROT0": "YOLO11n\nfull frame",
    "CFG_YOLO11S_TILED_640": "YOLO11s\ntiled",
    "CFG_YOLO11S_TILED_640_ROT90": "YOLO11s\ntiled, rotated",
    "CFG_YOLO11S_TILED_640_GAMMA": "YOLO11s\ntiled, gamma",
    "CFG_CABINET_UAVID_1280": "CABiNet\nUAVid",
}


def run(root):
    source = root / "recall_transfer.json"
    data = read(source)
    if data["complete"] is not True or data["evaluation_labels_fitted"] is not False:
        raise ValueError("Complete unchanged-predictor diagnostic required")
    if data["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Wrong comparison protocol")
    if data["source_sha256"] != sha(Path(__file__).with_name("mission_transfer_recall_check.py")):
        raise ValueError("Changed diagnostic implementation")
    for path, expected in data["input_sha256"].items():
        if sha(Path(path)) != expected:
            raise ValueError("Changed raw diagnostic evidence")
    summary = {r["config"]: r for r in data["summary"]}
    if set(summary) != set(LABELS) or len(summary) != len(data["summary"]):
        raise ValueError("The figure must retain all nine configurations exactly once")
    counts = {(r["frames"], r["visible_observations"]) for r in summary.values()}
    if len(counts) != 1:
        raise ValueError("Inconsistent observation denominators")
    frames, observations = next(iter(counts))
    paths = [root / ("recall_transfer." + s) for s in ("png", "pdf")]
    provenance = root / "recall_transfer_figure.json"
    if any(p.exists() for p in [*paths, provenance]):
        raise FileExistsError("Preserve completed recall figure")
    predicted = [summary[c]["mean_predicted_recall"] for c in LABELS]
    observed = [summary[c]["observed_recall"] for c in LABELS]
    if not all(np.isfinite(x) and 0 <= x <= 1 for x in [*predicted, *observed]):
        raise ValueError("Recall must be finite and within [0, 1]")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(12.8, 6.1))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.78)
    positions = np.arange(len(LABELS))
    for offset, values, color, label in (
        (-0.19, predicted, "#516979", "Frozen predictor"),
        (0.19, observed, "#15948c", "Observed recall"),
    ):
        bars = ax.bar(positions + offset, values, width=0.36, color=color, label=label)
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values], padding=4, fontsize=9)
    ax.set(
        ylim=(0, 1.02),
        ylabel="Visible-target recall",
        xticks=positions,
        xticklabels=list(LABELS.values()),
    )
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#dce4e8", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=9)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.005, 1.19), ncol=2, frameon=False)
    fig.suptitle(
        "Predicted vs observed recall on the Depot map",
        x=0.065,
        y=0.97,
        ha="left",
        fontsize=19,
        fontweight="bold",
    )
    fig.text(
        0.065,
        0.905,
        "Unchanged development predictor · All nine configurations · No evaluation refitting",
        fontsize=11,
        color="#52616a",
    )
    fig.text(
        0.065,
        0.07,
        f"Each configuration: {frames} captures and {observations} visible target-observations. "
        "One map; two spatial groups.\n"
        "Repeated observations are dependent. This descriptive figure reports no population "
        "confidence interval or mission-success probability.",
        fontsize=10,
        color="#52616a",
        linespacing=1.6,
    )
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)
    save(
        provenance,
        dict(
            complete=True,
            posthoc=True,
            evaluation_refitting=False,
            configurations=list(LABELS),
            frames_per_configuration=frames,
            visible_observations_per_configuration=observations,
            diagnostic_sha256=sha(source),
            protocol_sha256=sha(root / "protocol.json"),
            source_sha256=sha(Path(__file__)),
            artifacts_sha256={p.name: sha(p) for p in paths},
        ),
    )
    print({"figure": str(paths[0]), "provenance": str(provenance)}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
