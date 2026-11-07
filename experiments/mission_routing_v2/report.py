"""Report all predeclared policies, including failures and sensitivity conditions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_routing_v2.flights import verify
from experiments.mission_routing_v2.study import POLICIES


def run(root):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = verify(root)
    validation = read(root / "validation.json")
    if not validation["complete"]:
        raise ValueError("Validation must finish first")
    rows = []
    for flight in p["cases"]:
        for power in p["active_cpu_power_sensitivity_w"]:
            file = root / f"summary_{flight}_{power}.json"
            if sha(file) != validation["source_summaries_sha256"][file.name]:
                raise ValueError("Summary changed after validation")
            rows.extend(read(file)["cases"])
    primary = [r for r in rows if r["power_w"] == 15]
    failures = Counter(k for r in primary for k in r["failed_constraints"])
    contrasts = []
    times = []
    for flight in p["cases"]:
        for mission in p["missions"]:
            group = [r for r in primary if r["flight"] == flight and r["mission"] == mission]
            static = [r for r in group if r["policy"].startswith("static:")]
            intent = next(r for r in group if r["policy"] == "intent_history")
            best = max(
                static,
                key=lambda r: (
                    r["full_success"],
                    r["phases_succeeded"],
                    r["report_recall"],
                    r["final_battery"],
                ),
            )
            contrasts.append(
                dict(
                    flight=flight,
                    mission=mission,
                    selection="Best static chosen retrospectively on this test; diagnostic only",
                    best_static=best["policy"],
                    intent_phases=intent["phases_succeeded"],
                    best_static_phases=best["phases_succeeded"],
                    intent_success=intent["full_success"],
                    best_static_success=best["full_success"],
                )
            )
    for row in primary:
        times.extend(
            e["measured_router_wall_clock_s"]
            for e in read(root / row["path"] / "execution/events.json")
        )
    analysis = dict(
        scope=p["scope"],
        independent_map_test=False,
        total_executions=len(rows),
        primary_executions=len(primary),
        primary_successes=sum(r["full_success"] for r in primary),
        failures=dict(failures),
        paired_static_diagnostics=contrasts,
        router_wall_clock_p95_s=float(np.percentile(times, 95)),
        router_wall_clock_max_s=max(times),
        configured_router_latency_s=0.05,
        inference_energy_measured=False,
        validation_sha256=sha(root / "validation.json"),
    )
    save(root / "analysis.json", analysis)
    columns = [(f, m) for f in p["cases"] for m in p["missions"]]
    matrix = np.array(
        [
            [
                next(
                    r["phases_succeeded"]
                    for r in primary
                    if r["policy"] == policy and (r["flight"], r["mission"]) == case
                )
                for case in columns
            ]
            for policy in POLICIES
        ]
    )
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(matrix, vmin=0, vmax=4, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(
        range(len(columns)),
        [f"{f.split('_')[1]} / {m.replace('_', ' ')}" for f, m in columns],
        rotation=25,
        ha="right",
    )
    ax.set_yticks(range(len(POLICIES)), [v.replace("static:CFG_", "static: ") for v in POLICIES])
    for i in range(len(POLICIES)):
        for j in range(len(columns)):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] >= 3 else "black",
            )
    ax.set_title(
        "New flights on a known map: successful phases (out of 4)\n15 W assumed "
        "active compute power; full mission success is scored separately"
    )
    fig.colorbar(im, ax=ax, label="Successful phases")
    fig.tight_layout()
    fig.savefig(root / "phase_comparison.png", dpi=160)
    plt.close(fig)
    lines = [
        "# Mission routing v2: new-flight evaluation",
        "",
        f"{len(rows)} paid fixed-flight replays; {len(primary)} primary comparisons. "
        f"Full mission successes in the primary 15 W condition: "
        f"{analysis['primary_successes']}/{len(primary)}.",
        "",
        "Two new Gazebo recordings use new routes, human placements and seeds on "
        "the same downloaded "
        "textured lake map. Calibration uses only the older day/dusk development flights. "
        "This is a known-map new-flight test, not an unseen-map or field-deployment result. "
        "The two flights are insufficient for a population-level significance claim.",
        "",
        "All policies receive the same physical trajectory, sensor capture bank, "
        "model pool, intent and "
        "cost declaration. Processing latency determines which subsequent capture is available. "
        "The route never depends on model selection. Current RGB, public runtime "
        "state, structured intent "
        "and completed prediction history are the only dynamic routing inputs.",
        "",
        "Full success requires every phase, unique correct reports, deadlines, "
        "false-report limits, "
        "battery reserve, privacy, memory and valid execution. Phase counts below "
        "are secondary outcomes.",
        "",
        "## Primary results",
        "",
        "| Flight | Intent | Policy | Full success | Phases / 4 | Report recall | Final battery |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in primary:
        lines.append(
            f"| {r['flight']} | {r['mission']} | {r['policy']} | {r['full_success']} | "
            f"{r['phases_succeeded']} | {r['report_recall']:.3f} | {r['final_battery']:.3f} |"
        )
    lines += [
        "",
        "## Controller provenance",
        "",
        "`performance_predictor` is a lightweight RGB performance-prediction "
        "adaptation inspired by "
        "ApproxDet and LiteReconfig. `offline_contextual` is full-information offline ridge reward "
        "regression, motivated by learned contextual runtime control such as Chanakya; it does not "
        "reproduce Chanakya's deep bandit, training algorithm or streaming reward. "
        "`intent_history` adds current phase requirements and predicted pending-track support. "
        "Three ablations remove history, continuous RGB context, or phase intent. "
        "All nine fixed configurations, a resource-only controller and training-ranked periodic "
        "alternation are retained. No LLM or free online oracle is used.",
        "",
        "[ApproxDet](https://arxiv.org/abs/2010.10754), "
        "[LiteReconfig](https://github.com/StarsThu2016/LiteReconfig_AE), "
        "[Chanakya](https://github.com/microsoft/chanakya).",
        "",
        "## Cost sensitivity",
        "",
        "Latency profiles are frozen development laptop p95 values. Inference "
        "energy is estimated as "
        "15, 30 or 45 W times latency; these are declared power sensitivities, not "
        "measured energy. "
        "Flight (120 W), routing (50 ms, 0.75 J), tracking (100 ms, 1 J), memory "
        "and load costs are "
        "configured assumptions. Test inference times are diagnostics and never refit the policy. "
        "This models a serial cold/warm pipeline cost convention, not concurrent GPU inference.",
        "",
        "| Assumed compute power | Executions | Full successes | Mean successful phases |",
        "|---:|---:|---:|---:|",
    ]
    for power in p["active_cpu_power_sensitivity_w"]:
        selected = [r for r in rows if r["power_w"] == power]
        lines.append(
            f"| {power} W | {len(selected)} | {sum(r['full_success'] for r in selected)} | "
            f"{np.mean([r['phases_succeeded'] for r in selected]):.3f} |"
        )
    lines += [
        "",
        "## Verification and remaining limits",
        "",
        f"The independent validator rebuilt {validation['public_decisions_reproduced']} selections "
        "and tracker histories from public RGB and cached neural masks. Physical capture poses, "
        "per-event clock, energy, memory and canonical mission outcomes were verified.",
        "",
        "The history consists of predicted tracks; neither a report count nor a pending track is "
        "ground-truth completeness. The benchmark uses static human meshes, ideal odometry and a "
        "perfect known DEM. Tiled/gamma YOLO configurations share weights; CABiNet is trained on "
        "UAVid. Scenarios are authored simulated search and inspection tasks, not "
        "actual drone logs.",
        "",
        "Do not infer a proposed-controller advantage from a heatmap alone. Compare full success, "
        "report quality, battery and all fixed baselines per flight/intent. "
        "Best-static selection on "
        "test is reported only as a retrospective diagnostic. More independently held-out maps, "
        "moving targets, sensor/pose uncertainty and measured onboard costs remain necessary for "
        "a publication-level generalization or deployment claim.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    print(analysis, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
