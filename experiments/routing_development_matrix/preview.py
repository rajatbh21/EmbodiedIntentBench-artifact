"""Fixed-time physical RGB panels; green boxes are evaluator GT, not detections."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.routing_development_matrix.cohort import verify


def run(root):
    p = verify(root)
    flights = [g + "_" + lighting for g in p["layout_groups"] for lighting in ("day", "dusk_haze")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), layout="constrained")
    sources = {}
    for ax, flight in zip(axes.flat, flights, strict=True):
        folder = root / flight / "recording"
        record = read(folder / "recording.json")
        frame = record["frames"][30]  # Always capture 30 (60 s), not chosen for detections.
        path = folder / frame["file"]
        if sha(path) != frame["sha256"]:
            raise ValueError("Changed physical RGB")
        with np.load(path, allow_pickle=False) as data:
            rgb, ids = data["rgb"], data["object_ids"]
            ax.imshow(rgb)
            for label in sorted(set(np.unique(ids)) - {0}):
                ys, xs = np.where(ids == label)
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
        case = p["cases"][flight]
        ax.set_title(
            f"{flight.replace('_', ' ')} | world z {case['planned_altitude_m']} m", fontsize=11
        )
        ax.axis("off")
        sources[flight] = dict(
            capture_id=30, frame_sha256=sha(path), recording_sha256=sha(folder / "recording.json")
        )
    fig.suptitle(
        "New development layouts: actual Gazebo RGB at 60 s\n"
        "Green = evaluator ground truth; no model detections shown",
        fontsize=15,
    )
    fig.savefig(root / "physical_preview.png", dpi=150)
    plt.close(fig)
    save(
        root / "physical_preview_provenance.json",
        dict(
            schema_version="development_preview_v1",
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
