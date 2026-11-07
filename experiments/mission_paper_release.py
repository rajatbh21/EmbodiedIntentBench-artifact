"""Assemble English paper evidence from completed, independently checked artifacts."""

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_missions.assets import save, sha
from experiments.mission_coverage_repair.protocol import verify


def completed(root):
    completion = read(root / "completion.json")
    if not completion["complete"]:
        raise ValueError("Incomplete evidence")
    for name, expected in completion["artifacts_sha256"].items():
        if sha(root / name) != expected:
            raise ValueError("Changed completed artifact: " + str(root / name))


def run(study, telemetry, real_images, docs):
    p = verify(study)
    for root in (study, telemetry, real_images):
        completed(root)
    a = read(study / "analysis.json")
    validation = read(study / "validation.json")
    interpretation = read(study / "interpretation.json")
    energy = read(study / "energy_robustness.json")
    timing = read(study / "host_timing.json")
    witnesses = read(study / "witness_validation.json")
    if (
        not interpretation["complete"]
        or not energy["complete"]
        or interpretation["analysis_sha256"] != sha(study / "analysis.json")
        or interpretation["witness_validation_sha256"] != sha(study / "witness_validation.json")
        or energy["analysis_sha256"] != sha(study / "analysis.json")
        or not witnesses["complete"]
        or not timing["complete"]
        or timing["protocol_sha256"] != sha(study / "protocol.json")
    ):
        raise ValueError("Incomplete or changed auxiliary analysis")
    primary = next(r for r in a["policy_summary"] if r["policy"] == p["primary_policy"])
    primary_energy = next(
        r for r in energy["low_battery_summary"] if r["policy"] == p["primary_policy"]
    )
    best_static = max(
        r["successes"] for r in a["policy_summary"] if r["policy"].startswith("static:")
    )
    discriminate = sum(c["success_varies"] for c in a["cells"])
    all_failed = sum(not c["successful_policies"] for c in a["cells"])
    real = read(real_images / "validation.json")
    transfer = read(telemetry / "analysis.json")
    header = [
        "# Mission coverage routing: completed comparison",
        "",
        f"The predeclared primary policy completes **{primary['successes']}/24** conditions. "
        f"The highest fixed-configuration result is **{best_static}/24**, selected "
        f"descriptively after evaluation. Full mission success differs between policies in "
        f"**{discriminate}/24** paired conditions; **{all_failed}/24** conditions defeat every "
        "predeclared policy. The result is a benchmark discrimination finding, not evidence "
        "that the new coverage heuristic is the best router.",
        "",
        "## What was executed and checked",
        "",
        "Six new Gazebo flights supply 360 synchronized observations on three spatial "
        "layouts of one known textured map. Each flight is replayed under two authored "
        "person-survey intents and two battery settings (34% and 65%). The complete "
        "21-policy grid contains 504 executions, including all nine static configurations. "
        "All 3,240 frame/configuration pairs derive from 6,120 actual neural forwards. "
        "Configurations include repeated weights with resolution, tiling and orientation "
        "changes; they are not nine independent learned experts.",
        "",
        f"An independent reconstruction reproduces **{validation['public_decisions_reproduced']:,} "
        "routing decisions**, raw mask scores, predicted reports, integer capture/completion "
        "clocks and configured resource accounting. All lighting/battery mates use identical "
        "physical poses. The input checks exercise masked RGB, intent and battery as well "
        "as rejection of future and GT-marked history.",
        "",
        f"All **{witnesses['full_contract_witnesses']}/24** conditions have paid, full-contract "
        "evaluator witnesses. Their 1,440 actions and raw report evidence were independently "
        "reconstructed. They use hidden ground truth and are excluded from policy rankings. "
        "They establish feasibility under the declared assumptions, not optimality or a "
        "deployable method.",
        "",
        "## Complete comparison",
        "",
        "| Policy | Full successes /24 | Phases /96 | Mean report recall | Reserve passes /24 |",
        "|---|---:|---:|---:|---:|",
    ]
    header += [
        f"| {r['policy']} | {r['successes']} | {r['phases_succeeded']} | "
        f"{r['mean_report_recall']:.3f} | {r['reserve_passes']} |"
        for r in a["policy_summary"]
    ]
    header += [
        "",
        "## Predeclared paired contrasts",
        "",
        "Wins/losses/ties compare the primary with the named method on the same flight, "
        "intent and initial battery. Counts are repeated measures on three layouts.",
        "",
        "| Comparator | Wins | Losses | Ties | Phase difference |",
        "|---|---:|---:|---:|---:|",
    ]
    header += [
        f"| {r['comparator']} | {r['wins']} | {r['losses']} | {r['ties']} | "
        f"{r['phase_difference']:+d} |"
        for r in a["paired_contrasts"]
        if r["comparator"] in p["primary_comparisons"]
    ]
    header += [
        "",
        "## Why full mission scoring matters",
        "",
        "The illustrative central-day, 34%-battery rapid survey compares two failing "
        "policies with one successful ablation. Coverage-tail obtains 4/6 reports and "
        "finishes at 20.0034% battery but fails sector 3. Its battery-blind ablation "
        "obtains 6/6 reports and passes all four phases but ends at 19.2896%, below "
        "the same 20% reserve. The global-prototype ablation passes the full contract "
        "with 5/6 reports and 20.0919% battery. A report-recall ranking alone would "
        "therefore choose a failing method in this cell. This is an illustrative "
        "case, not a substitute for the complete table.",
        "",
        "## Disclosure and scope",
        "",
        "The design, coefficients, contracts and comparison grid were fixed before "
        "acquisition. Four original executions failed after a saturated probability "
        "exceeded one by floating-point roundoff. Their outputs remain in `study`. "
        "A versioned normalization-only correction was frozen, then all 504 cells "
        "were rerun in `study_repair_v1`. No objective, threshold, model or coefficient "
        "was changed based on the evaluation outcomes. This is explicitly not an "
        "untouched implementation holdout.",
        "",
        "The independent grouping unit is layout (three), not capture (360) or "
        "condition (24). The complete leave-one-layout-out descriptive contrasts "
        "and all 210 pairwise comparisons are in `interpretation.json`; no population "
        "p-value or confidence interval is asserted. Energy sensitivity changes "
        "inference power while keeping recorded actions and timings fixed, so it is "
        "neither an adaptive rerun nor an onboard measurement.",
        "",
        f"At 34% initial battery the primary has {primary_energy['reference_successes']}/12 "
        "full successes under the declared 15-W inference cost. Keeping those successful "
        "schedules fixed, only "
        f"{primary_energy['recorded_successes_retaining_reserve']['16']} retain the reserve "
        "at 16 W. This exposes a narrow configured energy margin; it does not predict "
        "what an adaptive rerun would choose under a different power model.",
        "",
        "The recorded host timings under concurrent orchestration are slower than the "
        "frozen calibration. For example, unrotated Mask R-CNN has a 4.88-second host "
        "median versus the 3.47-second declared latency; tiled YOLO has a 0.57-second "
        "median versus 0.42 seconds. The complete distributions are in `HOST_TIMING.md`. "
        "This experiment evaluates a declared timing model, not live onboard scheduling "
        "at the measured concurrent wall-clock rate. No result is silently retimed.",
        "",
        "A separate full-pool recall check finds substantial transfer error: the frozen "
        "RGB head predicts mean YOLO-nano recall of 0.00077, versus observed recall of "
        "0.18333 over 1,320 visible target-observations. For full-frame YOLO-small the "
        "corresponding values are 0.143 and 0.505. This is consistent with undervaluing "
        "fast configurations, but it does not isolate the cause of every mission failure. "
        "No predictor was fitted to the evaluation labels. `RECALL_TRANSFER.md` retains "
        "all nine configurations and JSON retains each complete layout.",
        "",
        "Reproduction details: [guide](mission_coverage_reproduction.md). "
        "The original [method specification](mission_coverage_routing.md) remains frozen. "
        "The [paper evidence and limitations](paper_evidence.md) keep auxiliary real-source "
        "checks separate from the 504-case simulation denominator.",
        "",
    ]
    (docs / "mission_coverage_results.md").write_text("\n".join(header))
    paper = [
        "# Paper evidence, methods and claim boundaries",
        "",
        "Working title: **Mission Routing Benchmark: Fixed-Flight Evaluation of Mission-Aware "
        "Perception Routing**. This is an evidence brief for manuscript development, "
        "not a certification of publication readiness or a claim of acceptance at SenSys.",
        "",
        "## Research claim supported by the current evidence",
        "",
        "A routing policy can be evaluated against a complete mission contract while "
        "holding flight dynamics and potential observations fixed. The benchmark exposes "
        "differences in phase evidence, timely unique reporting and resource compliance "
        "that aggregate perception quality alone does not express. Positive and negative "
        "outcomes, including unsuccessful proposed methods, belong in the comparison.",
        "",
        f"The latest comparison contains 21 policies and 504 executions. Full completion "
        f"varies in {discriminate}/24 conditions; {all_failed} conditions defeat all policies "
        "despite independently checked paid witnesses for every condition. "
        "See [all results and predeclared contrasts](mission_coverage_results.md).",
        "",
        "## Methods suitable for a manuscript",
        "",
        "The unit of intervention is the inference-selection policy. Each condition fixes "
        "a recorded physical trajectory, RGB stream, model weights, structured intent and "
        "initial resource state. At each available capture the policy observes current RGB, "
        "public phase requirements, the remaining budget and completed prediction history. "
        "It selects one declared configuration. Inference, loading, routing and tracking "
        "advance a common integer clock; captures during execution are skipped. Only the "
        "selected output reaches the policy. Dense offline inference provides reproducible "
        "potential outcomes without granting runtime access to unselected outputs. "
        "This is API-level information isolation for the audited implementations, not an "
        "OS sandbox for arbitrary untrusted policies.",
        "",
        "The primary endpoint is a conjunction: all four phase evidence requirements, "
        "correct unique reporting by the deadline, false/duplicate report limits, energy "
        "reserve, memory and privacy compliance. Rapid survey requires two temporally "
        "separated observations and lower coverage; verified inspection requires three "
        "observations and full required-target coverage. Both are authored person-survey "
        "contracts, not distinct inspection object classes or natural-language benchmarks. "
        "Language parsing, path planning and navigation are outside the tested intervention.",
        "",
        "Communication and privacy are checked but are nonbinding here: all configurations "
        "run locally with zero communication. Every configuration has the same declared "
        "2,400-MB model allocation under a 4,096-MB capacity, plus the common tracking "
        "allocation. No claim about bandwidth adaptation, privacy trade-offs or measured "
        "model memory follows from this cohort.",
        "",
        "The new policy learns 18 anonymous response prototypes from 596 development "
        "captures in three different layout groups. It combines current conditional recall "
        "with model complementarity, a per-phase repeated-evidence distribution and a "
        "full-flight resource continuation. Matched controls remove complementarity, "
        "coverage weighting, horizon length, context, intent or battery information. "
        "The surrogate assumes stationary/independent future evidence and is not a "
        "calibrated probability of mission success. Its parameters are frozen on development "
        "data, and its negative evaluation results are not tuned away.",
        "",
        "The six new physical recordings span three authored locations and paired lighting "
        "on one known map. Geometry and per-target visibility gates run before model "
        "inference. Each recording supplies two intents and two battery conditions. "
        "Statistical descriptions retain whole layouts and all failed cells. The disclosed "
        "floating-point normalization repair is recorded separately from the original "
        "pre-acquisition specification.",
        "",
        "## Distinct real-source evidence",
        "",
        "Official SeaDronesSee MOT training annotations supply 27,259 frame records "
        "in 21 observed platform/video groups (20 videos listed in source metadata). "
        "All records are retained; timestamp reversals and gaps split continuous segments. "
        "Two deterministic 120-second windows are translated, without rotation, scale "
        "or time rescaling, onto the existing simulation terrain. These are source-derived "
        "motion paths, not replicas of the original scene, camera or mission.",
        "",
        "| Source platform | Legacy waypoint RMSE (m) | Reference-controller RMSE (m) |",
        "|---|---:|---:|",
    ]
    paper += [
        f"| {r['source_identity']['drone']} | {r['waypoint_rmse_m']:.3f} | "
        f"{r['reference_controller_rmse_m']:.3f} |"
        for r in transfer["comparisons"]
    ]
    best = max(real["summary"], key=lambda r: r["matched"])
    paper += [
        "",
        "The reference controller was developed after observing the legacy "
        "controller's errors. This demonstrates a corrected engineering transfer on "
        "two known traces, not independent controller generalization. Altitude datum "
        "is unverified; motion preserves recorded altitude differences.",
        "",
        f"A separate fixed-time sample applies the same nine configurations to twelve "
        f"real training images (108 pairs, 204 neural forwards). The highest matched "
        f"swimmer count is {best['matched']}/{best['swimmer_observations']} for "
        f"`{best['config_id']}`. These are bounding-box matches of person-mask "
        "components at IoU 0.5, not official dataset AP or mask IoU. Unmatched "
        "components are not independently verified false detections because source "
        "labels focus on people in water. The small sample documents an external-domain "
        "limitation; it does not establish real-world mission performance.",
        "",
        "The [official dataset page](https://seadronessee.cs.uni-tuebingen.de/dataset) "
        "and [source repository](https://github.com/Ben93kie/SeaDronesSee) are recorded "
        "with download hashes. The repository states CC0 1.0 while the annotation "
        "JSON license field says `Unknown`; both statements are preserved.",
        "",
        "## Claim matrix for reviewers",
        "",
        "| Claim | Evidence | Boundary |",
        "|---|---|---|",
        f"| Routing-only discrimination | {discriminate}/24 conditions separate full "
        "success; all 504 executions audited | One known map, three authored layout groups |",
        "| Feasible operating range | 24/24 paid witnesses; 1,440 reconstructed actions | "
        "GT-aware diagnostic, not a deployable policy or optimum |",
        "| Causal inputs and accounting | Selected-output history, pose pairing, raw scores "
        "and integer clocks rebuilt | Shared scorer semantics; ideal pose/terrain |",
        "| Resource evaluation | Declared debits and reserve checked per action | "
        "15-W inference and 120-W flight power are assumptions, not measurements |",
        "| Real telemetry ingestion | Complete source records and two physical path transfers | "
        "No source mission constraints or scene reconstruction |",
        "| Real-image execution | 108 independently rescored image/configuration pairs | "
        "Two training-source windows; external-domain sanity check only |",
        "| Portable endpoint review | Standard-library capsule with exact frozen scorer | "
        "Not clean-machine neural/physics/policy reproduction |",
        "",
        "## Positioning against related work",
        "",
        "[ApproxDet](https://arxiv.org/abs/2010.10754) schedules object-detection "
        "branches from content, resource contention and latency requirements. "
        "[Towards Streaming Perception](https://arxiv.org/abs/2005.10420) evaluates "
        "perception together with the time at which results become available. "
        "[Chanakya](https://proceedings.neurips.cc/paper_files/paper/2023/hash/"
        "ae2d574d2c309f3a45880e4460efd176-Abstract-Conference.html) learns adaptive "
        "runtime decisions from content and system context. "
        "[Dynamic Selection of Perception Models for Robotic Control]"
        "(https://arxiv.org/abs/2207.06390) already studies model sequences for "
        "multi-step robotic control, including drone landing. Accordingly, this work "
        "must not claim to invent adaptive perception or task-aware drone model selection. "
        "The candidate contribution is the controlled mission-contract benchmark and "
        "its interpretable feasibility/discrimination analysis. The local policies "
        "are not reproductions of those published algorithms.",
        "",
        "## Evidence still needed for stronger publication claims",
        "",
        "1. Evaluate frozen policies on independently sourced maps, routes and moving "
        "targets, with enough independent groups to report meaningful uncertainty.",
        "2. Ground operational intent thresholds in documented missions or expert "
        "requirements; combine source imagery/telemetry with those contracts without "
        "pretending the annotations contain requirements they do not provide.",
        "3. Reproduce at least one relevant published scheduling algorithm with explicit "
        "pool, feature and reward adaptations on the same permitted training split.",
        "4. Measure target-device latency, load, memory and energy; assess pose/depth "
        "uncertainty and interference. A fixed-schedule power sensitivity is insufficient.",
        "5. Demonstrate a clean-machine end-to-end artifact setup, including asset/model "
        "acquisition and redistribution terms. The portable endpoint capsule closes "
        "only the scoring-review step.",
        "",
        "These are evidence gaps, not implementation failures that can be removed by "
        "changing a pass threshold. A defensible paper should narrow its claims to "
        "the completed evidence until these gates are closed.",
        "",
    ]
    (docs / "paper_evidence.md").write_text("\n".join(paper))
    save(
        study / "paper_release.json",
        dict(
            complete=True,
            research_readiness_certified=False,
            files_sha256={
                str(docs / name): sha(docs / name)
                for name in ("mission_coverage_results.md", "paper_evidence.md")
            },
            inputs_sha256={
                str(root / "completion.json"): sha(root / "completion.json")
                for root in (study, telemetry, real_images)
            },
            source_sha256=sha(Path(__file__)),
            auxiliary_sha256={
                str(study / name): sha(study / name)
                for name in (
                    "protocol.json", "interpretation.json", "energy_robustness.json",
                    "host_timing.json", "recall_transfer.json", "witness_validation.json",
                )
            },
            claim="Known-map mission-level routing discrimination with explicit limitations",
        ),
    )
    print(
        dict(
            primary=primary["successes"],
            best_fixed=best_static,
            discriminating_conditions=discriminate,
            witnessed_all_policy_failures=all_failed,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--real-images", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    args = parser.parse_args()
    run(args.study, args.telemetry, args.real_images, args.docs)
