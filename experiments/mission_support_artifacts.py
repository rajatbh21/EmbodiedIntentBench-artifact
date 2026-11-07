"""Plot all support-policy results and retain evaluator-only failure diagnostics."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_support_routing.protocol import verify
from experiments.mission_support_routing.report import checked_events
from experiments.mission_support_routing.study import POLICIES


def run(root):
    p = verify(root)
    completion = read(root / "completion.json")
    if not completion["complete"] or completion["fresh_test"]:
        raise ValueError("Expected completed exploratory study")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed completed artifact")
    if completion["source_sha256"] != sha(
        Path(__file__).parent / "mission_support_routing/report.py"
    ):
        raise ValueError("Changed report implementation")
    analysis = read(root / "analysis.json")
    validation = read(root / "validation.json")
    for name, expected in validation["source_summaries_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed validated summary")
    diagnostics = []
    for row in analysis["policy_cases"]:
        events = checked_events(root, row)
        outcome = read(root / row["path"] / "outcome.json")
        active, adapted = 0, 0
        for event in events:
            trace = event["router_decision"]
            candidate = next(
                c for c in trace["candidates"] if c["config_id"] == trace["selected_config_id"]
            )
            details = [
                t for phase in candidate["phases"] for t in phase.get("predicted_track_support", [])
            ]
            active += bool(details)
            adapted += any(t["attempts"] > 0 for t in details)
        diagnostics.append(
            dict(
                case=row["flight"],
                mission=row["mission"],
                policy=row["policy"],
                decisions=len(events),
                selected_action_with_pending_track=active,
                selected_action_with_observed_support_trials=adapted,
                support_detail_available=row["policy"] != "baseline",
                failed_constraints=outcome["failed_constraints"],
                phase_diagnostics=[
                    dict(
                        phase_id=phase["phase_id"],
                        success=phase["success"],
                        insufficient_evidence=phase["evidence_recall"] < phase["required_recall"],
                        excess_false_components=phase["false_components"]
                        > phase["max_false_components"],
                        late_results=phase["late_results"],
                    )
                    for phase in outcome["phases"]
                ],
            )
        )
    for row in analysis["reused_fixed_controls"]:
        checked_events(Path(p["parent"]), row)
    save(
        root / "support_diagnostics.json",
        dict(
            scope="post_execution_diagnostics_not_router_inputs",
            analysis_sha256=sha(root / "analysis.json"),
            selected_trial_counts_are_correctness=False,
            cases=diagnostics,
            source_sha256=sha(Path(__file__)),
        ),
    )
    names = [*POLICIES, *dict.fromkeys(r["policy"] for r in analysis["reused_fixed_controls"])]
    labels = [n.replace("static:CFG_", "Fixed ").replace("_", " ") for n in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True, layout="constrained")
    for ax, key, maximum, title in zip(
        axes,
        ("successes", "phases"),
        (4, 16),
        ("Full mission success / 4", "Completed phases / 16"),
        strict=True,
    ):
        values = np.asarray(
            [
                [
                    next(
                        r[key]
                        for r in analysis["policy_summary"]
                        if r["policy"] == name and r["battery"] == battery
                    )
                    for battery in p["batteries"]
                ]
                for name in names
            ]
        )
        ax.imshow(values, vmin=0, vmax=maximum, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3), [f"{100 * b:g}%" for b in p["batteries"]])
        ax.set_xlabel("Initial battery")
        ax.set_title(title)
        ax.axhline(3.5, color="#555555", linewidth=1.5)
        for i in range(len(names)):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(values[i, j]),
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="white" if values[i, j] > maximum * 0.6 else "black",
                )
    axes[0].set_yticks(range(len(names)), labels, fontsize=10)
    fig.suptitle(
        "Prediction-support routing on inspected recordings\n"
        "48 new policy replays; 108 reused fixed-control results",
        fontsize=14,
    )
    image = root / "support_comparison.png"
    fig.savefig(image, dpi=160)
    plt.close(fig)
    save(
        root / "support_comparison.provenance.json",
        dict(
            analysis_sha256=sha(root / "analysis.json"),
            image_sha256=sha(image),
            source_sha256=sha(Path(__file__)),
            fresh_test=False,
        ),
    )
    audit = dict(
        complete=True,
        policy_replays=validation["policy_cases"],
        public_decisions=validation["public_decisions_reproduced"],
        reference_fixed_replays=len(analysis["reused_fixed_controls"]),
        baseline_parity_cases=len(analysis["baseline_parity"]),
        new_physical_flights=0,
        new_neural_calls=0,
        fresh_test=False,
        research_readiness_certified=False,
        artifacts_sha256={
            name: sha(root / name)
            for name in (
                "protocol.json",
                "analysis.json",
                "validation.json",
                "completion.json",
                "REPORT.md",
                "support_diagnostics.json",
                "support_comparison.png",
                "support_comparison.provenance.json",
            )
        },
        source_sha256=sha(Path(__file__)),
    )
    save(root / "release_audit.json", audit)
    print({k: v for k, v in audit.items() if k != "artifacts_sha256"}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
