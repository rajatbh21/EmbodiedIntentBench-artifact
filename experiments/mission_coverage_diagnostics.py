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
from experiments.mission_coverage_routing.cohort import verify
from experiments.mission_coverage_routing.study import scenario_for
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
