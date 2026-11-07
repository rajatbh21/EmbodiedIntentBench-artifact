"""All-cell descriptive results for a separately sourced map and fixed policy grid."""

import argparse
from pathlib import Path

import numpy as np

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.report import paired, summarize
from experiments.mission_transfer_comparison_v4.audit import complete_grid
from experiments.mission_transfer_comparison_v4.protocol import verify


def run(root):
    if (root / "completion.json").exists():
        raise FileExistsError("Preserve the completed report")
    p = verify(root)
    validation = read(root / "validation.json")
    if not validation["complete"] or validation["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Complete independent validation required")
    rows, host = [], {}
    for f, c in p["cases"].items():
        audit = root / f"validation_{f}.json"
        if sha(audit) != validation["condition_audits_sha256"][audit.name]:
            raise ValueError("Changed condition audit")
        for name, expected in read(audit)["source_summaries_sha256"].items():
            if sha(root / name) != expected:
                raise ValueError("Changed audited summary")
            for r in read(root / name)["cases"]:
                dest = root / r["path"]
                for n, k in (
                    ("outcome.json", "outcome_sha256"),
                    ("execution/events.json", "events_sha256"),
                    ("execution/results.json", "results_sha256"),
                ):
                    if sha(dest / n) != r[k]:
                        raise ValueError("Changed audited execution")
                result = read(dest / "execution/results.json")
                events = read(dest / "execution/events.json")
                scenario = read(root / f / r["mission"] / "scenario.json")
                intent = read(root / f / r["mission"] / "intent.json")
                capacity = scenario["mission"]["battery_capacity_wh"] * 3600
                reserve = intent["limits"]["min_final_battery_frac"]
                energy = result["cumulative_energy_j"]
                rows.append(
                    dict(
                        r,
                        layout=c["layout_group"],
                        lighting=c["lighting"],
                        battery=c["initial_battery"],
                        energy_j=energy,
                        reserve_margin_j=(c["initial_battery"] - reserve) * capacity - energy,
                        false_components=sum(
                            e["score"]["false_positive_components"] for e in events
                        ),
                        schedule=[
                            (e["observation_id"], e["execution"]["config_id"]) for e in events
                        ],
                    )
                )
                host.setdefault(r["policy"], []).extend(
                    e["measured_router_wall_clock_s"] for e in events
                )
    complete_grid(rows, p)
    n = len(p["cases"]) * len(p["missions"])
    summary = [
        dict(policy=name, **summarize([r for r in rows if r["policy"] == name]))
        for name in p["policies"]
    ]
    cells = []
    for f in p["cases"]:
        for m in p["missions"]:
            selected = [r for r in rows if r["flight"] == f and r["mission"] == m]
            cells.append(
                dict(
                    flight=f,
                    mission=m,
                    successful_policies=[r["policy"] for r in selected if r["full_success"]],
                    success_varies=len({r["full_success"] for r in selected}) > 1,
                    phases_varies=len({r["phases_succeeded"] for r in selected}) > 1,
                    report_recall_varies=len({r["report_recall"] for r in selected}) > 1,
                )
            )
    contrasts = [
        paired(rows, p["primary_policy"], c) for c in p["policies"] if c != p["primary_policy"]
    ]
    strata = {
        key: [
            dict(
                value=value,
                policy=policy,
                **summarize([r for r in rows if r[key] == value and r["policy"] == policy]),
            )
            for value in sorted({r[key] for r in rows})
            for policy in p["policies"]
        ]
        for key in ("layout", "mission", "battery", "lighting")
    }
    analysis = dict(
        complete=True,
        fresh_test=False,
        prospective_policy_evaluation=True,
        geometry_revised_after_rejected_acquisition=True,
        admission_amended_after_capture=True,
        fully_prospective_protocol=False,
        lighting_causal_effect_identified=False,
        routing_admission_sha256=p["routing_admission_sha256"],
        protocol_sha256=sha(root / "protocol.json"),
        validation_sha256=sha(root / "validation.json"),
        policy_cases=rows,
        policy_summary=summary,
        paired_contrasts=contrasts,
        stratified=strata,
        cells=cells,
        independent_layouts=p["independent_evaluation_layouts"],
        evaluation_maps=1,
        maps_unseen_by_training=1,
        maps_unseen_before_sensor_inspection=0,
        statistical_inference=False,
        superiority_claim=False,
        source_sha256=sha(Path(__file__)),
        router_timing={
            c: dict(
                count=len(v),
                p95_ms=float(np.percentile(v, 95) * 1000),
                max_ms=max(v) * 1000,
                above_charged_50ms=sum(t > 0.05 for t in v),
            )
            for c, v in host.items()
        },
    )
    save(root / "analysis.json", analysis)
    plot(root, p, rows, summary, n)
    primary = next(r for r in summary if r["policy"] == p["primary_policy"])
    adapter = next(r for r in summary if r["policy"] == "approxdet_eq5_adapter")
    fixed = max(r["successes"] for r in summary if r["policy"].startswith("static:"))
    lines = [
        "# Prospective routing transfer to the Depot map",
        "",
        f"The complete {len(p['policies'])}-policy, {n}-condition grid contains {len(rows)} "
        "executions. Every predeclared execution, including failure, is retained.",
        "",
        f"Coverage-tail completes **{primary['successes']}/{n}**, the ApproxDet Equation 5 "
        f"scheduler adapter **{adapter['successes']}/{n}**, and the highest fixed configuration "
        f"**{fixed}/{n}**. The best fixed result is selected descriptively after evaluation.",
        "",
        "## Controlled intervention and checks",
        "",
        f"{validation['physical_recordings']} actual Gazebo recordings supply "
        f"{validation['frame_config_pairs']:,} frame/configuration pairs and "
        f"{validation['actual_neural_calls']:,} actual neural forwards. Independent replay "
        f"reconstructs {validation['public_decisions_reproduced']:,} decisions, selected raw "
        "predictions, report support, clocks, complete execution coverage and resource debits. "
        "All policies in each condition share the same fixed physical observations. "
        "Intent and battery mates share exact RGB and poses. Lighting banks have separately "
        "recorded trajectories and are observational strata.",
        "",
        "Methods, coefficients and thresholds are inherited unchanged from the rejected v2 "
        "acquisition. The v3 scenario and implementation were frozen before its new captures "
        "and all neural inference. Raw RGB and geometry ground truth from v2 had already "
        "been inspected to repair collision geometry and route timing. The v2 route/sensor "
        "gates failed before model outputs or policy outcomes existed; all failed recordings "
        "and their rejection are retained. This is prospective policy evaluation after a "
        "disclosed sensor-informed engineering revision, not an untouched sensor holdout. "
        "No policy is fitted to either acquisition's evaluation labels.",
        "",
        "The v3 captures passed every physical-route and target/intent-observation check, "
        "but failed the original requirement for bit-identical poses across lighting. "
        "The original failure and all recordings are retained. Admission scope was then "
        "amended before any model outputs or policy runs: unchanged routers are compared "
        "on exactly the same recording within each condition; intent and battery also "
        "reuse exact recorded inputs. Cross-lighting pose differences are reported without "
        "a new tolerance or isolated lighting-effect claim. This scope amendment occurred "
        "after sensor inspection and is not a wholly prospective protocol.",
        "",
        "## Complete policy comparison",
        "",
        f"| Policy | Full /{n} | Phases /{4 * n} | Report recall "
        f"| False components | Reserve /{n} |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['policy']} | {r['successes']} | {r['phases_succeeded']} | "
        f"{r['mean_report_recall']:.3f} | {r['false_components']} | {r['reserve_passes']} |"
        for r in summary
    ]
    lines += [
        "",
        "## Paired comparisons against coverage-tail",
        "",
        "| Comparator | Wins | Losses | Ties | Phase difference | Changed schedules |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['comparator']} | {r['wins']} | {r['losses']} | {r['ties']} | "
        f"{r['phase_difference']:+d} | {r['changed_schedules']} |"
        for r in contrasts
    ]
    lines += [
        "",
        "## Discrimination and uncertainty",
        "",
        f"Full success varies in {sum(c['success_varies'] for c in cells)}/{n} conditions; "
        f"phase completion varies in {sum(c['phases_varies'] for c in cells)}/{n}; "
        f"report recall varies in {sum(c['report_recall_varies'] for c in cells)}/{n}. "
        f"Every predeclared policy fails {sum(not c['successful_policies'] for c in cells)}/{n}. "
        "Sensor admission is an optimistic observation-opportunity check, not a paid "
        "feasibility proof. All-policy failure alone cannot establish infeasibility.",
        "",
        "Each layout, lighting, intent and battery stratum is retained in `analysis.json`. "
        "There is one independently sourced evaluation map; the two locations are repeated "
        "spatial groups within it. Neither the number of frames nor the factorial's number "
        "of cells is a count of independent maps. No population confidence interval or "
        "statistical superiority claim is made.",
        "",
        "## Interpretation boundaries",
        "",
        "The OpenRobotics Depot environment differs from the previous lake in scenery, "
        "indoor altitude, safe route geometry and illumination. This tests transfer to "
        "another authored setting; it does not isolate a pure map-only causal effect. "
        "People are static, pose and ground plane are ideal, and mission thresholds "
        "are authored. The environment is not a replay of an operational warehouse mission. "
        "The map is absent from development training but was seen by the designers during "
        "the rejected v2 sensor inspection. Fresh captures do not erase that prehistory.",
        "",
        "`approxdet_eq5_adapter` implements the paper's accuracy-under-latency objective "
        "with this benchmark's frozen recall head, nine configurations and shared reserve "
        "guard. It does not reproduce ApproxDet's original accuracy/latency predictors, "
        "tracker branches, contention sensing or switching cadence. These results are "
        "not a head-to-head comparison against the original complete published system.",
        "",
        "Energy and memory remain configured and inference timing is from the historical "
        "CPU calibration. Concurrent wall-clock inference and routing measurements are "
        "diagnostic; they do not silently retime the mission. No measured onboard-energy "
        "or broad field-generalization claim follows.",
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    names = (
        "protocol.json",
        "analysis.json",
        "validation.json",
        "REPORT.md",
        "comparison.png",
        "comparison.pdf",
        "success_matrix.png",
    )
    save(
        root / "completion.json",
        dict(
            complete=True,
            fresh_test=False,
            prospective_policy_evaluation=True,
            geometry_revised_after_rejected_acquisition=True,
            admission_amended_after_capture=True,
            lighting_causal_effect_identified=False,
            routing_admission_sha256=p["routing_admission_sha256"],
            source_sha256=sha(Path(__file__)),
            artifacts_sha256={name: sha(root / name) for name in names},
            research_readiness_certified=False,
        ),
    )
    print(
        dict(
            primary=primary,
            adapter=adapter,
            best_fixed=fixed,
            discriminating_conditions=sum(c["success_varies"] for c in cells),
        ),
        flush=True,
    )


def plot(root, p, rows, summary, n):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = p["policies"]
    labels = [name.replace("static:CFG_", "") for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 10), sharey=True, layout="constrained")
    for ax, key, title, limit in zip(
        axes,
        ("successes", "phases_succeeded", "mean_report_recall"),
        (f"Full success / {n}", f"Completed phases / {4 * n}", "Mean report recall"),
        (n, 4 * n, 1),
        strict=True,
    ):
        colors = [
            "#155e75"
            if name == p["primary_policy"]
            else "#b45309"
            if name == "approxdet_eq5_adapter"
            else "#94a3b8"
            for name in names
        ]
        ax.barh(range(len(names)), [r[key] for r in summary], color=colors)
        ax.set(yticks=range(len(names)), yticklabels=labels, xlim=(0, limit), title=title)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Depot map transfer: identical flights, different routing policies\n"
        "Two spatial groups and lighting strata; paired intents and initial battery"
    )
    fig.savefig(root / "comparison.png", dpi=180)
    fig.savefig(root / "comparison.pdf")
    plt.close(fig)
    cells = [(f, m) for f in p["cases"] for m in p["missions"]]
    by = {(r["flight"], r["mission"], r["policy"]): r for r in rows}
    values = np.array([[by[f, m, name]["full_success"] for f, m in cells] for name in names], int)
    fig, ax = plt.subplots(figsize=(15, 10), layout="constrained")
    ax.imshow(
        values,
        cmap=matplotlib.colors.ListedColormap(["#e2e8f0", "#0f766e"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set(
        yticks=range(len(names)),
        yticklabels=labels,
        xticks=range(len(cells)),
        xticklabels=[f"{f}\n{m}" for f, m in cells],
        title="Every declared condition: green = full success; grey = failure",
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    fig.savefig(root / "success_matrix.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
