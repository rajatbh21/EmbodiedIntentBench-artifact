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
from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST
from experiments.mission_environment_transfer_v3.cohort import verify
from experiments.mission_environment_transfer_v3.scenario import load_scenario
from experiments.mission_routing_v2.train import checked_manifest
from experiments.mission_transfer_comparison_v4.protocol import (
    check_prediction_admission,
)
from experiments.mission_transfer_comparison_v4.protocol import (
    verify as verify_study,
)
from experiments.mission_transfer_within_recording.admission import verify as verify_admission
from experiments.v4_benchmark.backends import ReplayBackend
from experiments.v4_benchmark.probe_cache import rgb_key
from experiments.v4_benchmark.runner import evaluation_frame


def run(root, study=None):
    p = verify(root)
    study = study or root / "study_paired_scope_v1"
    comparison_protocol = verify_study(study)
    if Path(comparison_protocol["cohort"]).resolve() != root.resolve():
        raise ValueError("Study and diagnostic cohort differ")
    verify_admission(Path(comparison_protocol["admission"]))
    if (root / "feasibility_diagnostics.json").exists():
        raise FileExistsError("Preserve prior feasibility diagnostics")
    results = {}
    for flight in p["cases"]:
        scenario = load_scenario(
            root / flight / "rapid_survey/scenario.json",
            p["cases"][flight]["environment"],
            Path(p["assets"]),
        )
        folder = root / flight / "predictions"
        manifest = checked_manifest(folder, root / "protocol.json")
        check_prediction_admission(manifest, comparison_protocol)
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
        complementarity_summary = complementarity(rows, times, list(masks))
        missions = {}
        costs = {s.config_id: s.mission_latency_s for s in scenario.executors}
        for name in p["missions"]:
            mission = read(root / flight / name / "intent.json")["mission"]
            union = optimistic_bound(mission, rows, times, list(masks))
            stages = [
                latency_bound(
                    phase,
                    rows,
                    scenario.schedule,
                    costs,
                    scenario.router_latency_s + DEFAULT_TRACKING_COST.latency_s,
                )
                for phase in mission["phases"]
            ]
            missions[name] = dict(
                free_union=union,
                serial_latency_relaxation=stages,
                latency_certified_infeasible=any(s["cannot_meet_recall"] for s in stages),
            )
        results[flight] = dict(
            complementarity=complementarity_summary,
            missions=missions,
            source_manifest_sha256=sha(folder / "manifest.json"),
            outcomes_sha256=sha(root / flight / "diagnostic_outcomes.json"),
        )
    result = dict(
        schema_version="independent_map_posthoc_feasibility_v1",
        gt_aware=True,
        deployable_policy=False,
        posthoc=True,
        policy_or_contract_retuned=False,
        flights=results,
        diagnostic_implementation_sha256=sha(Path(__file__)),
        protocol_sha256=sha(root / "protocol.json"),
        comparison_protocol_sha256=sha(study / "protocol.json"),
        routing_admission_sha256=comparison_protocol["routing_admission_sha256"],
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
    parser.add_argument("--study", type=Path)
    args = parser.parse_args()
    run(args.output, args.study)
