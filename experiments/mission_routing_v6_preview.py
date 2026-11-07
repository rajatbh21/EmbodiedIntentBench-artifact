"""Fixed-time physical RGB panels; boxes are evaluator GT, never detector output."""

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


def run(root):
    p = verify(root)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.6), layout="constrained")
    sources = []
    for row, flight in enumerate(p["cases"]):
        folder = root / flight / "recording"
        record = read(folder / "recording.json")
        for col, index in enumerate((6, 30, 48)):
            ax, frame = axes[row, col], record["frames"][index]
            path = folder / frame["file"]
            if sha(path) != frame["sha256"]:
                raise ValueError("Changed physical RGB")
            with np.load(path, allow_pickle=False) as data:
                ax.imshow(data["rgb"])
                for label in range(1, 7):
                    ys, xs = np.where(data["object_ids"] == label)
                    if not len(xs):
                        continue
                    ax.add_patch(
                        Rectangle(
                            (xs.min(), ys.min()),
                            xs.max() - xs.min() + 1,
                            ys.max() - ys.min() + 1,
                            fill=False,
                            edgecolor="#00ff66",
                            linewidth=1.2,
                        )
                    )
                    ax.text(
                        xs.min(),
                        max(12, ys.min() - 3),
                        f"GT {label}",
                        color="#00ff66",
                        fontsize=7,
                        bbox=dict(facecolor="black", alpha=0.6, pad=1, edgecolor="none"),
                    )
            ax.set_title(f"{'Day' if row == 0 else 'Dusk/haze'} | {index * 2} s", fontsize=11)
            ax.axis("off")
            sources.append(dict(flight=flight, capture_id=index, frame_sha256=sha(path)))
    fig.suptitle(
        "New physical evaluation layout: fixed-time Gazebo captures\n"
        "Green boxes = evaluator ground truth; not model detections",
        fontsize=16,
    )
    fig.savefig(root / "physical_preview.png", dpi=150)
    plt.close(fig)
    save(
        root / "physical_preview_provenance.json",
        dict(
            schema_version="v6_physical_preview_v1",
            source_sha256=sha(Path(__file__)),
            protocol_sha256=sha(root / "protocol.json"),
            images=sources,
            image_sha256=sha(root / "physical_preview.png"),
            gt_aware=True,
            outcome_driven_frame_selection=False,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
