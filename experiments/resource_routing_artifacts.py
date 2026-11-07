"""Visualize validated battery interventions and audit numerical reserve margins."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.resource_routing_study.protocol import BATTERIES, verify
from experiments.resource_routing_study.study import POLICIES


def paired_mechanisms(root, analysis):
    """Retain the first action divergence and actual charged energy for every pair."""
    pairs = []
    for pair in analysis["battery_pairs"]:
        rows = {
            name: next(
                r
                for r in analysis["policy_cases"]
                if r["flight"] == pair["case"]
                and r["mission"] == pair["mission"]
                and r["policy"] == name
            )
            for name in ("adaptive", "battery_blind")
        }
        events = {
            name: read(root / row["path"] / "execution/events.json") for name, row in rows.items()
        }
        energy = {
            name: read(root / row["path"] / "execution/results.json")["cumulative_energy_j"]
            for name, row in rows.items()
        }
        first = None
        for aware, blind in zip(events["adaptive"], events["battery_blind"], strict=False):
            if aware["execution"]["config_id"] == blind["execution"]["config_id"]:
                continue
            # Identical earlier actions must leave the same physical/causal state.
            for key in ("observation_id", "capture_time_s", "runtime_at_capture"):
                if aware[key] != blind[key]:
                    raise ValueError("Unexpected pre-divergence state mismatch")
            first = dict(
                observation_id=aware["observation_id"],
                capture_time_s=aware["capture_time_s"],
                actual_runtime_state=aware["runtime_at_capture"],
                choices={},
            )
            selected = {e["execution"]["config_id"] for e in (aware, blind)}
            for name, event in (("adaptive", aware), ("battery_blind", blind)):
                decision = event["router_decision"]
                first["choices"][name] = dict(
                    config_id=event["execution"]["config_id"],
                    available_energy_j=decision["available_energy_j"],
                    selected_pair_candidates=[
                        c for c in decision["candidates"] if c["config_id"] in selected
                    ],
                )
            break
        if pair["schedule_changed"] != (first is not None):
            raise ValueError("Schedule difference lacks an explained first action divergence")
        pairs.append(
            dict(
                case=pair["case"],
                mission=pair["mission"],
                initial_battery=pair["initial_battery"],
                first_action_divergence=first,
                total_energy_j=energy,
                energy_saved_j=energy["battery_blind"] - energy["adaptive"],
                selections={name: row["selections"] for name, row in rows.items()},
                source_events_sha256={name: row["events_sha256"] for name, row in rows.items()},
            )
        )
    return pairs


def run(root):
    p = verify(root)
    c = read(root / "completion.json")
    for filename, key in (
        ("protocol.json", "protocol_sha256"),
        ("analysis.json", "analysis_sha256"),
        ("validation.json", "validation_sha256"),
        ("REPORT.md", "report_sha256"),
    ):
        if sha(root / filename) != c[key]:
            raise ValueError("Changed completed resource artifact")
    a = read(root / "analysis.json")
    v = read(root / "validation.json")
    if (
        a["source_sha256"] != sha(Path(__file__).parent / "resource_routing_study/report.py")
        or not c["complete"]
        or c["fresh_test"]
    ):
        raise ValueError("Wrong completed analysis")
    margins = []
    for row in a["policy_cases"]:
        folder = root / row["path"]
        for filename, key in (
            ("outcome.json", "outcome_sha256"),
            ("execution/events.json", "events_sha256"),
            ("execution/results.json", "results_sha256"),
        ):
            if sha(folder / filename) != row[key]:
                raise ValueError("Changed validated policy replay")
        scenario = read(root / row["flight"] / row["mission"] / "scenario.json")
        capacity = scenario["mission"]["battery_capacity_wh"] * 3600
        reserve = read(root / row["flight"] / row["mission"] / "intent.json")["limits"][
            "min_final_battery_frac"
        ]
        margins.append(
            dict(
                case=row["flight"],
                mission=row["mission"],
                policy=row["policy"],
                initial_battery=p["cases"][row["flight"]]["initial_battery"],
                final_battery=row["final_battery"],
                reserve_margin_j=(row["final_battery"] - reserve) * capacity,
                full_success=row["full_success"],
            )
        )
    save(
        root / "reserve_margins.json",
        dict(
            scope="configured_trace_margin_only",
            hardware_robustness_proven=False,
            analysis_sha256=sha(root / "analysis.json"),
            cases=margins,
            source_sha256=sha(Path(__file__)),
        ),
    )
    save(
        root / "paired_mechanisms.json",
        dict(
            scope="descriptive_matched_ablation_on_inspected_recordings",
            analysis_sha256=sha(root / "analysis.json"),
            cases=paired_mechanisms(root, a),
            source_sha256=sha(Path(__file__)),
        ),
    )

    def matrix(key):
        return np.asarray(
            [
                [
                    next(
                        r
                        for r in a["policy_summary"]
                        if r["policy"] == policy and r["initial_battery"] == battery
                    )[key]
                    for battery in BATTERIES
                ]
                for policy in POLICIES
            ]
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True, layout="constrained")
    names = [n.replace("static:CFG_", "Fixed ").replace("_", " ") for n in POLICIES]
    for ax, key, title in zip(
        axes,
        ("successes", "reserve_satisfied"),
        ("Full mission success / 4", "Final reserve satisfied / 4"),
        strict=True,
    ):
        values = matrix(key)
        ax.imshow(values, vmin=0, vmax=4, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3), [f"{100 * b:g}%" for b in BATTERIES])
        ax.set_xlabel("Initial battery")
        ax.set_title(title)
        for i in range(len(POLICIES)):
            for j in range(3):
                ax.text(
                    j,
                    i,
                    str(values[i, j]),
                    ha="center",
                    va="center",
                    color="white" if values[i, j] >= 3 else "black",
                    fontsize=13,
                )
    axes[0].set_yticks(range(len(POLICIES)), names, fontsize=10)
    fig.suptitle(
        "Battery intervention on inspected physical recordings\n"
        "132 policy replays; configured 15 W; evaluator witnesses excluded",
        fontsize=14,
    )
    path = root / "battery_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    save(
        root / "battery_comparison.provenance.json",
        dict(
            source_sha256=sha(Path(__file__)),
            analysis_sha256=sha(root / "analysis.json"),
            image_sha256=sha(path),
            fresh_test=False,
        ),
    )
    for filename, expected in v["source_summaries_sha256"].items():
        if sha(root / filename) != expected:
            raise ValueError("Changed validated summary")
    for row in a["witnesses"]:
        search = root / "witnesses" / row["flight"] / row["mission"] / "search.json"
        if sha(search) != row["search_sha256"]:
            raise ValueError("Changed witness search")
        if "path" in row:
            for filename, key in (
                ("outcome.json", "outcome_sha256"),
                ("execution/events.json", "events_sha256"),
                ("execution/results.json", "results_sha256"),
            ):
                if sha(root / row["path"] / filename) != row[key]:
                    raise ValueError("Changed witness replay")
    artifacts = (
        "protocol.json",
        "validation.json",
        "analysis.json",
        "completion.json",
        "REPORT.md",
        "reserve_margins.json",
        "paired_mechanisms.json",
        "battery_comparison.png",
        "battery_comparison.provenance.json",
    )
    audit = dict(
        complete=True,
        policy_replays=v["policy_cases"],
        witness_replays=v["witness_cases"],
        public_decisions=v["public_decisions_reproduced"],
        witness_decisions=v["witness_decisions_reproduced"],
        new_neural_calls=0,
        new_physical_flights=0,
        fresh_test=False,
        research_readiness_certified=False,
        artifacts_sha256={name: sha(root / name) for name in artifacts},
        source_sha256=sha(Path(__file__)),
    )
    save(root / "release_audit.json", audit)
    print({k: value for k, value in audit.items() if k != "artifacts_sha256"}, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args().output)
