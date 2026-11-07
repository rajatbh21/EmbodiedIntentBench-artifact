"""Document source limits and compare two explicit trajectory-transfer controllers."""

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.real_mission_telemetry.tracking_transfer import verify


def run(root):
    p = verify(root)
    validation = read(root / "validation.json")
    old = Path(p["original_transfer"])
    prior = read(old / "validation.json")
    if (
        not validation["complete"]
        or validation["protocol_sha256"] != sha(root / "protocol.json")
        or not prior["complete"]
        or prior["protocol_sha256"] != sha(old / "protocol.json")
    ):
        raise ValueError("Complete paired physics audits required")
    imported = read(Path(p["imported"]) / "manifest.json")
    comparisons = []
    for f, c in validation["cases"].items():
        before = prior["cases"][f]
        if (
            c["reference_positions_m"] != before["reference_positions_m"]
            or c["reference_times_s"] != before["reference_times_s"]
        ):
            raise ValueError("Changed reference motion")
        comparisons.append(
            dict(
                flight=f,
                source_identity=c["source_identity"],
                waypoint_rmse_m=before["rmse_position_m"],
                reference_controller_rmse_m=c["rmse_position_m"],
                waypoint_max_error_m=before["max_position_error_m"],
                reference_controller_max_error_m=c["max_position_error_m"],
            )
        )
    result = dict(
        complete=True,
        comparisons=comparisons,
        source_sha256=sha(Path(__file__)),
        validation_sha256=sha(root / "validation.json"),
        prior_validation_sha256=sha(old / "validation.json"),
        source_frames=imported["source_frames"],
        source_groups=imported["source_groups"],
        source_motion_only=True,
        source_scene_reconstructed=False,
        mission_success_claimed=False,
        exact_flight_replication_claimed=False,
        independent_controller_generalization_test=False,
    )
    save(root / "analysis.json", result)
    plot(root, validation, prior)
    lines = [
        "# Real UAV telemetry: verified source ingestion and physical trajectory transfer",
        "",
        (
            "Official SeaDronesSee MOT training annotations provide 27,259 frame records "
            "from 21 source-video groups. Grouping preserves platform and case-sensitive "
            "source filenames. No test split, image dataset or object-label-derived "
            "mission criterion is used for trajectory selection."
        ),
        "",
        (
            "The importer retains all records. It marks ten nonincreasing timestamp "
            "boundaries, nine missing-time intervals over one second, and 13 frames "
            "referring to a video absent from the video's metadata table (20 listed videos "
            "versus 21 observed groups). It splits trajectories at discontinuities rather "
            "than silently interpolating through them. Two continuous segments last at "
            "least 120 seconds; the first 120 seconds of each are selected "
            "deterministically."
        ),
        "",
        "## Source and transformation",
        "",
        (
            "The acquisition follows the [official dataset "
            "page](https://seadronessee.cs.uni-tuebingen.de/dataset) and the [authors' "
            "repository](https://github.com/Ben93kie/SeaDronesSee). The repository states "
            "CC0 1.0 for the dataset; the downloaded annotation JSON has an `Unknown` "
            "license field. Both statements are preserved in the source manifest instead "
            "of silently replacing the latter. Downloads, source bytes and normalized "
            "exports are hashed."
        ),
        "",
        (
            "GPS becomes local spherical east/north displacement. The vertical coordinate "
            "preserves recorded altitude differences; its datum is not established. A "
            "single spatial translation places each path over the existing lake map with "
            "geometry-derived clearance. There is no scale, rotation or time rescaling. "
            "The original water scene, actors, camera motion and mission instructions are "
            "not reconstructed. Synthetic placeholder actors remain unrelated to the "
            "source people."
        ),
        "",
        "## Actual physics results",
        "",
        (
            "Both controller variants use the same 120-second source-derived references "
            "and produce 60 synchronized physical captures per flight. The first "
            "experiment uses the inherited 3 m/s waypoint controller. The second, "
            "explicitly developed after inspecting its tracking error, adds interpolated "
            "reference velocity feedforward with the same 0.8/s position gain and an 8 m/s "
            "vector speed cap. The simulator's 2 m/s^2 acceleration limit is unchanged. "
            "This is an engineering correction on two known traces, not an independent "
            "controller comparison."
        ),
        "",
        (
            "| Source platform | Waypoint RMSE (m) | Reference RMSE (m) | Waypoint max (m) "
            "| Reference max (m) |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['source_identity']['drone']} | {r['waypoint_rmse_m']:.3f} "
        f"| {r['reference_controller_rmse_m']:.3f} | {r['waypoint_max_error_m']:.3f} "
        f"| {r['reference_controller_max_error_m']:.3f} |"
        for r in comparisons
    ]
    lines += [
        "",
        "## What this establishes",
        "",
        (
            "The system now ingests traceable real UAV telemetry, preserves gaps and "
            "source identities, transfers metric motion into a physical simulator and "
            "measures the resulting tracking error. It does not establish mission-level "
            "performance on real imagery, real search-and-rescue completion, exact GPS "
            "ground truth, deployment energy or unseen-map transfer. The 504-case routing "
            "study is a separate authored simulation experiment, and its conditions and "
            "outcomes were not changed by this extension."
        ),
        "",
        (
            "The pre-capture transfer attempt that exceeded the legacy schema's speed "
            "limit is retained in `seadronessee_physics_transfer_v1_20261025`. The bounded "
            "waypoint recordings remain in `seadronessee_physics_transfer_v2_20261025`. No "
            "failed or inferior attempt is deleted."
        ),
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    names = (
        "protocol.json",
        "validation.json",
        "analysis.json",
        "trajectory_comparison.png",
        "trajectory_comparison.pdf",
        "REPORT.md",
    )
    save(
        root / "completion.json",
        dict(
            complete=True,
            mission_benchmark_completed=False,
            artifacts_sha256={n: sha(root / n) for n in names},
            source_sha256=sha(Path(__file__)),
        ),
    )
    print(comparisons, flush=True)


def plot(root, current, prior):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for row, (flight, c) in enumerate(current["cases"].items()):
        before = prior["cases"][flight]
        reference = np.asarray(c["reference_positions_m"])
        actual = np.asarray(c["actual_positions_m"])
        legacy = np.asarray(before["actual_positions_m"])
        origin = reference[0, :2]
        ax = axs[row, 0]
        for values, label, color in (
            (reference, "Source-derived reference", "#334155"),
            (legacy, "Waypoint, 3 m/s", "#b45309"),
            (actual, "Reference controller, 8 m/s", "#0f766e"),
        ):
            points = values[:, :2] - origin
            ax.plot(points[:, 0], points[:, 1], label=label, color=color, linewidth=1.7)
        ax.set(
            title=c["source_identity"]["drone"] + " | Horizontal trajectory",
            xlabel="East displacement (m)",
            ylabel="North displacement (m)",
        )
        ax.axis("equal")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        ax = axs[row, 1]
        ax.plot(
            c["reference_times_s"],
            np.linalg.norm(legacy - reference, axis=1),
            label="Waypoint",
            color="#b45309",
        )
        ax.plot(
            c["reference_times_s"],
            np.linalg.norm(actual - reference, axis=1),
            label="Reference controller",
            color="#0f766e",
        )
        ax.set(
            xlabel="Source-relative time (s)",
            ylabel="3D position error (m)",
            title="Actual synchronized physics poses",
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle(
        (
            "Real UAV motion transferred to the same simulator | Scene and mission are not "
            "reconstructed"
        ),
        fontsize=13,
    )
    fig.savefig(root / "trajectory_comparison.png", dpi=170)
    fig.savefig(root / "trajectory_comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
