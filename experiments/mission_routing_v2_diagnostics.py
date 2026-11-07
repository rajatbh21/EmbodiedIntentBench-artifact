"""Post-hoc evaluator-only feasibility diagnostics for the frozen new-flight test.

Never called by routing policies or used to retune this study. Exact optimistic
phase bounds separate unavailable evidence from scheduling/controller failures.
Passing such a bound does not establish report-required mission feasibility.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from missionbench.v2.evaluation import MissionEvaluator
from experiments.lake_feasibility_inference import read
from experiments.lake_feasibility_metrics import optimistic_bound
from experiments.lake_hindsight_witness import VerifiedMasks
from experiments.lake_missions.assets import save, sha
from experiments.lake_rotation_latency_bound import latency_bound
from experiments.lake_routing_diagnostics import complementarity
from experiments.mission_routing_v2.flights import verify
from experiments.mission_routing_v2.study import scenario_for
from experiments.mission_routing_v2.train import checked_manifest
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame


def run(root):
    p = verify(root)
    results = {}
    for flight in p["cases"]:
        scenario = scenario_for(root, p, flight, "rapid_survey", 15)
        folder = root / flight / "predictions"
        manifest = checked_manifest(folder, root / "protocol.json")
        masks = {c: VerifiedMasks(folder, m) for c, m in manifest["models"].items()}
        scorers = {
            c: MissionEvaluator(
                scenario.matching_iou_threshold,
                len(scenario.targets),
                scenario.observation_interval_s,
            )
            for c in masks
        }
        backend = ReplayBackend(scenario, root / flight / "recording")
        rows, times = [], []
        try:
            for index, timestamp in enumerate(backend.frames):
                backend.advance_to(timestamp)
                frame = backend.capture()
                time = (timestamp - scenario.schedule.epoch_ns) / 1e9
                times.append(time)
                truth = evaluation_frame(frame, index, time, scenario)
                for config in masks:
                    rows.append(
                        dict(
                            capture_id=index,
                            config_id=config,
                            **scorers[config]
                            .update(truth, masks[config][rgb_key(frame.rgb)])
                            .to_dict(),
                        )
                    )
        finally:
            backend.close()
        save(root / flight / "diagnostic_outcomes.json", rows)
        comparison = complementarity(rows, times, list(masks))
        missions = {}
        costs = {s.config_id: s.mission_latency_s for s in scenario.executors}
        for name in p["missions"]:
            mission = read(root / flight / name / "intent.json")["mission"]
            union = optimistic_bound(mission, rows, times, list(masks))
            stages = [
                latency_bound(
                    phase, rows, scenario.schedule, costs, scenario.router_latency_s + 0.1
                )
                for phase in mission["phases"]
            ]
            missions[name] = dict(
                free_union=union,
                serial_latency_relaxation=stages,
                latency_certified_infeasible=any(s["cannot_meet_recall"] for s in stages),
            )
        results[flight] = dict(
            complementarity=comparison,
            missions=missions,
            source_manifest_sha256=sha(folder / "manifest.json"),
            outcomes_sha256=sha(root / flight / "diagnostic_outcomes.json"),
        )
    result = dict(
        schema_version="mission_routing_v2_posthoc_feasibility",
        gt_aware=True,
        deployable_policy=False,
        posthoc=True,
        policy_or_contract_retuned=False,
        flights=results,
        diagnostic_implementation_sha256=sha(Path(__file__)),
        protocol_sha256=sha(root / "protocol.json"),
    )
    save(root / "feasibility_diagnostics.json", result)
    lines = [
        "# Post-hoc feasibility diagnostics",
        "",
        "Evaluator-only analysis; no policy or contract was retuned. These are optimistic bounds, "
        "not a deployable policy and not proof that a complete reporting mission is feasible.",
        "",
        "| Flight | Best static observations | Free best per frame | Free union | "
        "Visible observations |",
        "|---|---:|---:|---:|---:|",
    ]
    for flight, data in results.items():
        c = data["complementarity"]
        lines.append(
            f"| {flight} | {c['best_static_matched_observations']} | "
            f"{c['free_per_frame_best_single_matched_observations']} | "
            f"{c['free_per_frame_union_matched_observations']} | "
            f"{c['visible_target_observations']} |"
        )
    lines += [
        "",
        "| Flight | Intent | Infeasible even with optimistic serial timing? | Failed phases |",
        "|---|---|---|---|",
    ]
    for flight, data in results.items():
        for name, m in data["missions"].items():
            failed = [
                s["phase_id"] for s in m["serial_latency_relaxation"] if s["cannot_meet_recall"]
            ]
            lines.append(
                f"| {flight} | {name} | {m['latency_certified_infeasible']} | "
                f"{', '.join(failed) or 'None'} |"
            )
    lines += [
        "",
        "The serial bound charges inference, routing and tracking latency but ignores energy, "
        "memory, false positives, switching, incoming busy state and reporting. A failed bound "
        "certifies impossibility under this particular fixed pool/flight/contract. Passing leaves "
        "full feasibility unresolved. Counts are repeated target observations, not unique people.",
        "",
    ]
    (root / "FEASIBILITY.md").write_text("\n".join(lines))
    print(
        {
            f: {m: d["latency_certified_infeasible"] for m, d in v["missions"].items()}
            for f, v in results.items()
        },
        flush=True,
    )


def policy_audit(root):
    p = verify(root)
    validation = read(root / "validation.json")
    rows = []
    for filename, expected in validation["source_summaries_sha256"].items():
        if sha(root / filename) != expected:
            raise ValueError("Changed validated summary")
        rows.extend(read(root / filename)["cases"])
    comparisons, groups = [], []
    for flight in p["cases"]:
        for mission in p["missions"]:
            for power in p["active_cpu_power_sensitivity_w"]:
                selected = [
                    r
                    for r in rows
                    if r["flight"] == flight and r["mission"] == mission and r["power_w"] == power
                ]
                groups.append(
                    dict(
                        flight=flight,
                        mission=mission,
                        power_w=power,
                        full_success_range=[
                            min(r["full_success"] for r in selected),
                            max(r["full_success"] for r in selected),
                        ],
                        phases_range=[
                            min(r["phases_succeeded"] for r in selected),
                            max(r["phases_succeeded"] for r in selected),
                        ],
                        report_recall_range=[
                            min(r["report_recall"] for r in selected),
                            max(r["report_recall"] for r in selected),
                        ],
                    )
                )
                schedules = {}
                for r in selected:
                    path = root / r["path"] / "execution/events.json"
                    if sha(path) != r["events_sha256"]:
                        raise ValueError("Changed validated schedule")
                    schedules[r["policy"]] = [
                        (e["observation_id"], e["execution"]["config_id"]) for e in read(path)
                    ]
                for ablation in ("no_history", "no_context", "no_intent"):
                    comparisons.append(
                        dict(
                            flight=flight,
                            mission=mission,
                            power_w=power,
                            ablation=ablation,
                            identical_schedule=schedules["intent_history"] == schedules[ablation],
                        )
                    )
    result = dict(
        posthoc=True,
        policy_or_contract_retuned=False,
        groups=groups,
        ablations=comparisons,
        validation_sha256=sha(root / "validation.json"),
        implementation_sha256=sha(Path(__file__)),
    )
    save(root / "policy_diagnostics.json", result)
    lines = [
        "# Paired policy discrimination and ablations",
        "",
        "Differences are descriptive within matched flights. No frame-level significance or "
        "population generalization claim is made.",
        "",
        "| Flight | Intent | Power | Full-success range | Phases / 4 range | Report recall range |",
        "|---|---|---:|---|---|---|",
    ]
    for g in groups:
        lines.append(
            f"| {g['flight']} | {g['mission']} | {g['power_w']} W | "
            f"{g['full_success_range']} | {g['phases_range']} | "
            f"{g['report_recall_range']} |"
        )
    lines += [
        "",
        "| Removed input | Identical model/capture schedules | Cases |",
        "|---|---:|---:|",
    ]
    for ablation in ("no_history", "no_context", "no_intent"):
        subset = [a for a in comparisons if a["ablation"] == ablation]
        lines.append(
            f"| {ablation} | {sum(a['identical_schedule'] for a in subset)} | {len(subset)} |"
        )
    lines += [
        "",
        "An identical schedule demonstrates no incremental effect of that input in "
        "the named case, even if the interface exposes it. Different schedules alone "
        "do not establish improved mission outcomes.",
        "",
    ]
    (root / "POLICY_DIAGNOSTICS.md").write_text("\n".join(lines))
    print(
        {
            a: sum(c["identical_schedule"] for c in comparisons if c["ablation"] == a)
            for a in ("no_history", "no_context", "no_intent")
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policies-only", action="store_true")
    args = parser.parse_args()
    if args.policies_only:
        policy_audit(args.output)
    else:
        run(args.output)
