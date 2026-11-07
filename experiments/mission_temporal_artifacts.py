"""Audit the limits of benchmark discrimination and plot both battery interventions."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_strategy_routing.report import checked_events
from experiments.mission_temporal_routing.protocol import verify
from experiments.mission_temporal_routing.report import assert_grid


def discrimination(rows, cases, missions):
    rows = [r for r in rows if r["policy"] != "survival_reference"]
    policies = sorted({r["policy"] for r in rows})
    assert_grid(rows, cases, missions, policies)
    result = []
    for case in cases:
        for mission in missions:
            subset = [r for r in rows if r["flight"] == case and r["mission"] == mission]
            result.append(
                dict(
                    case=case,
                    mission=mission,
                    policies=len(subset),
                    full_success_varies=len({r["full_success"] for r in subset}) > 1,
                    phase_success_varies=len({r["phases_succeeded"] for r in subset}) > 1,
                    report_recall_varies=len({r["report_recall"] for r in subset}) > 1,
                    successful_policies=[r["policy"] for r in subset if r["full_success"]],
                )
            )
    return result


def run(root):
    if (root / "release_audit.json").exists():
        raise FileExistsError("Preserve completed release audit")
    protocol = verify(root)
    completion = read(root / "completion.json")
    if not completion["complete"] or completion["fresh_test"]:
        raise ValueError("Completed exploratory comparison required")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed completed artifact")
    if completion["source_sha256"] != sha(
        Path(__file__).with_name("mission_temporal_routing") / "report.py"
    ):
        raise ValueError("Changed report source")
    analysis = read(root / "analysis.json")
    rows = analysis["policy_cases"]
    cells = discrimination(rows, protocol["cases"], protocol["missions"])
    controls = analysis["negative_controls"]
    if len(controls) != 10 or not all(c["invariant"] for c in controls):
        raise ValueError("Missing or failed input controls")
    # Recheck all new and referenced execution hashes, not only summary tables.
    for row in rows:
        checked_events(Path(row["source"]), row)
    by_name = {r["policy"]: r for r in analysis["policy_summary"]}
    full = by_name["temporal_mission"]
    fixed = [r for r in analysis["policy_summary"] if r["policy"].startswith("static:")]
    validity = dict(
        complete=True,
        source_sha256=sha(Path(__file__)),
        analysis_sha256=sha(root / "analysis.json"),
        cells=cells,
        distinct_policies=len(by_name),
        complete_conditions=len(cells),
        discriminating_full_success_cells=sum(c["full_success_varies"] for c in cells),
        discriminating_phase_cells=sum(c["phase_success_varies"] for c in cells),
        discriminating_report_cells=sum(c["report_recall_varies"] for c in cells),
        all_failure_cells=[
            dict(case=c["case"], mission=c["mission"])
            for c in cells
            if not c["successful_policies"]
        ],
        primary_contrasts=analysis["paired_contrasts"],
        full_policy_beats_best_fixed=full["successes"] > max(r["successes"] for r in fixed),
        independent_evaluation_layouts=1,
        independent_evaluation_maps=1,
        repeated_condition_cells_are_independent=False,
        within_cohort_discrimination_supported=any(c["full_success_varies"] for c in cells),
        novel_router_superiority_established=False,
        unseen_map_generalization_established=False,
        actual_hardware_energy_validated=False,
        fresh_test=False,
    )
    save(root / "validity.json", validity)
    plot(root, protocol, rows)
    names = (
        "protocol.json",
        "analysis.json",
        "validation.json",
        "completion.json",
        "REPORT.md",
        "comparison.png",
        "validity.json",
        "battery_mechanisms.png",
    )
    save(
        root / "release_audit.json",
        dict(
            complete=True,
            fresh_test=False,
            research_readiness_certified=False,
            source_sha256=sha(Path(__file__)),
            test_source_sha256=sha(
                Path(__file__).parents[1] / "tests/test_mission_temporal_artifacts.py"
            ),
            artifacts_sha256={n: sha(root / n) for n in names},
            new_policy_replays=84,
            reused_reference_replays=180,
            distinct_policies=len(by_name),
            public_decisions_reconstructed=read(root / "validation.json")[
                "public_decisions_reproduced"
            ],
        ),
    )
    print(
        {
            k: v
            for k, v in validity.items()
            if k not in ("cells", "primary_contrasts", "all_failure_cells")
        }
    )


def plot(root, protocol, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    chosen = [(c, info) for c, info in protocol["cases"].items() if info["initial_battery"] == 0.34]
    for col, (case, info) in enumerate(chosen):
        ax, timeline = axes[0, col], axes[1, col]
        config_ids = sorted({k for r in rows for k in r["selections"]})
        for policy, label, color, offset in (
            ("temporal_mission", "Battery aware", "#007f6e", -0.12),
            ("temporal_no_battery", "Assume 65% initial battery", "#bf5b17", 0.12),
        ):
            row = next(
                r
                for r in rows
                if r["policy"] == policy and r["flight"] == case and r["mission"] == "rapid_survey"
            )
            events = checked_events(Path(row["source"]), row)
            result = read(Path(row["source"]) / row["path"] / "execution/results.json")
            times = [0] + [e["completion_time_s"] for e in events] + [result["final_time_s"]]
            battery = (
                [34]
                + [(0.34 - e["cumulative_energy_j"] / (30 * 3600)) * 100 for e in events]
                + [row["final_battery"] * 100]
            )
            ax.plot(times, battery, color=color, label=label)
            timeline.scatter(
                [e["capture_time_s"] for e in events],
                [config_ids.index(e["execution"]["config_id"]) + offset for e in events],
                color=color,
                s=18,
                label=label,
            )
            ax.text(
                0.03,
                0.15 if offset < 0 else 0.06,
                f"{label}: {'PASS' if row['full_success'] else 'FAIL'}, "
                f"reserve margin {row['reserve_margin_j']:+.1f} J",
                transform=ax.transAxes,
                color=color,
                fontsize=9,
            )
        ax.axhline(20, color="black", linestyle="--", linewidth=1, label="20% required reserve")
        ax.set_title(info["source_flight"].replace("_", " "))
        ax.set_ylim(18, 35)
        ax.set_ylabel("Remaining battery (%)")
        ax.set_xlabel("Mission time (s)")
        timeline.set_yticks(
            range(len(config_ids)), [c.replace("CFG_", "") for c in config_ids], fontsize=8
        )
        timeline.set_ylim(-0.7, len(config_ids) - 0.3)
        timeline.set_xlabel("Selected capture time (s)")
        for line in (ax, timeline):
            for t in (30, 60, 90):
                line.axvline(t, color="#dddddd", linewidth=0.7, zorder=0)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Matched battery-information intervention at 34% initial charge\n"
        "Same physical route and rapid-survey contract; configured energy model",
        fontsize=13,
    )
    fig.savefig(root / "battery_mechanisms.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
