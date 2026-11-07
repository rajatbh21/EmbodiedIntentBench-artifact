"""Audit old runs under explicit new intents and smoke-test bound physical replays.

This is a post-hoc migration diagnostic, not a new held-out policy experiment.
No image, model output, trajectory, resource cost or historical result is changed.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import LIMIT_FIELDS, VERSION, IntentContract, evaluate
from experiments.lake_missions.assets import save, sha

TEXT = {
    "sector_rescue": (
        "Sector search and verification",
        "Search the west, prioritize the north shoreline, search the eastern forest edge, "
        "then revisit the west with fresh repeated evidence.",
    ),
    "perimeter_inspection": (
        "Facility perimeter inspection",
        "Inspect all three sectors with repeated observations and limited false alarms, "
        "then recheck the north with fresh evidence.",
    ),
    "reserve_revisit": (
        "Battery-constrained revisit",
        "Collect repeated evidence in the west and north, revisit both with fresh evidence, "
        "and retain at least 20 percent battery.",
    ),
}
PHASE_TEXT = {
    "west_search": "Collect repeated evidence in the west sector",
    "north_priority": "Prioritize the northern shoreline",
    "east_search": "Search the eastern forest edge",
    "west_revisit": "Revisit the west using only fresh observations",
    "west_inspect": "Repeatedly inspect the western access route",
    "north_inspect": "Inspect the northern shoreline walking area",
    "east_inspect": "Inspect the eastern facility boundary",
    "north_recheck": "Recheck the north using fresh evidence",
    "west_first": "Collect initial repeated evidence in the west",
    "north_first": "Collect initial repeated evidence in the north",
    "west_fresh": "Collect new repeated evidence on the western revisit",
    "north_fresh": "Recheck the north while preserving battery reserve",
}


def declare(
    mission: dict, runtime: dict, targets: list[str], mode: str, duration_s: float
) -> IntentContract:
    """Explicit migration template; no inference of requirements from free-form text."""
    mission = copy.deepcopy(mission)
    mission["title"], mission["user_request"] = TEXT[mission["mission_id"]]
    for phase in mission["phases"]:
        phase["instruction"] = PHASE_TEXT[phase["phase_id"]]
    reporting = dict(mode=mode)
    if mode == "report_required":
        old = runtime["mission_requirements"]
        reporting.update(
            target_scope="all_scenario_targets",
            deadline_s=old["confirmed_report_deadline_s"],
            min_report_recall=old["min_report_recall"],
            max_false_reports=old["max_false_reports"],
            max_duplicate_reports=old["max_duplicate_reports"],
        )
        mission["user_request"] += (
            f" Also deliver confirmed reports for at least {reporting['min_report_recall']:.1%} "
            f"of all scenario targets by {reporting['deadline_s']:g} seconds, with at most "
            f"{reporting['max_false_reports']} false and "
            f"{reporting['max_duplicate_reports']} duplicate reports."
        )
    else:
        mission["user_request"] += (
            " This intent requires phase evidence only; reports are diagnostic."
        )
    return IntentContract(
        dict(
            schema_version=VERSION,
            mission=mission,
            duration_s=duration_s,
            limits={key: runtime["contract"][key] for key in LIMIT_FIELDS},
            reporting=reporting,
            evaluator_target_ids=targets,
        )
    )


def cases(prior: Path) -> list[dict]:
    rows = [dict(row, gt_aware=False) for row in read(prior / "summary.json")["comparisons"]]
    for path in sorted((prior / "hindsight").glob("*/*/verified.json")):
        verified = read(path)
        rows.append(
            dict(
                environment=path.parent.parent.name,
                mission_id=path.parent.name,
                policy="GT_AWARE_OFFLINE_WITNESS",
                gt_aware=True,
                output=str(path.parent / "execution"),
                **{key: verified[key] for key in ("events_sha256", "results_sha256")},
            )
        )
    for row in rows:
        for name in ("events", "results"):
            if sha(Path(row["output"]) / f"{name}.json") != row[f"{name}_sha256"]:
                raise ValueError("Historical result hash mismatch")
    return rows


def audit(prior: Path, output: Path, replay_smoke: bool = False) -> dict:
    from experiments.lake_missions.run import freeze

    source = Path(read(prior / "protocol.json")["source"])
    freeze(source)
    missions = {m["mission_id"]: m for m in read(source / "missions.json")["missions"]}
    duration_s = read(source / "scenario.json")["mission"]["duration_s"]
    targets = sorted(
        {t for m in missions.values() for p in m["phases"] for t in p["evaluator_target_ids"]}
    )
    inputs = cases(prior)
    output.mkdir(parents=True, exist_ok=False)
    declarations = {}
    for mission_id, mission in missions.items():
        runtime = read(
            Path(next(r["output"] for r in inputs if r["mission_id"] == mission_id))
            / "results.json"
        )
        for mode in ("evidence_only", "report_required"):
            contract = declare(mission, runtime, targets, mode, duration_s)
            declarations[(mission_id, mode)] = contract
            folder = output / "contracts" / mission_id / mode
            folder.mkdir(parents=True)
            save(folder / "intent.json", contract.document())
            save(folder / "public_intent.json", contract.public_document())
    save(
        output / "protocol.json",
        dict(
            schema_version="lake_intent_migration_v1",
            prior=str(prior.resolve()),
            scope="Post-hoc semantic migration; no historical result is replaced",
            prior_summary_sha256=sha(prior / "summary.json"),
            source_missions_sha256=sha(source / "missions.json"),
            contracts={f"{m}/{mode}": c.sha256 for (m, mode), c in declarations.items()},
            inputs=inputs,
            replay_smoke_requested=replay_smoke,
            implementation_sha256={
                name: sha(Path(__file__).with_name(name))
                for name in (
                    "lake_intent_contract.py",
                    "lake_intent_runner.py",
                    "lake_intent_study.py",
                )
            },
        ),
    )
    rows = []
    for index, row in enumerate(inputs):
        path = Path(row["output"])
        events, result = read(path / "events.json"), read(path / "results.json")
        for mode in ("evidence_only", "report_required"):
            contract = declarations[row["mission_id"], mode]
            outcome = evaluate(contract, events, result)
            outcome.update(
                gt_aware=row["gt_aware"],
                policy=row["policy"],
                environment=row["environment"],
                retrospective=True,
                events_sha256=row["events_sha256"],
                execution_sha256=row["results_sha256"],
            )
            destination = output / "audit" / f"{index:03d}_{mode}.json"
            destination.parent.mkdir(exist_ok=True)
            save(destination, outcome)
            rows.append(
                {
                    key: outcome[key]
                    for key in (
                        "mission_id",
                        "environment",
                        "policy",
                        "gt_aware",
                        "mode",
                        "mission_success",
                        "failed_constraints",
                        "reporting",
                    )
                }
                | {"outcome": str(destination.resolve()), "sha256": sha(destination)}
            )
    smoke = []
    if replay_smoke:
        selected = [
            r for r in inputs if r["environment"] == "day" and r["mission_id"] == "sector_rescue"
        ]
        witness = next(r for r in selected if r["gt_aware"])
        static = next(r for r in selected if r["policy"] == "static:CFG_MASKRCNN_1280_ROT0_90")
        for row, mode in (
            (witness, "evidence_only"),
            (witness, "report_required"),
            (static, "report_required"),
        ):
            smoke.append(replay(prior, source, output, row, declarations[row["mission_id"], mode]))
    groups = []
    for gt in (False, True):
        for mode in ("evidence_only", "report_required"):
            subset = [r for r in rows if r["gt_aware"] == gt and r["mode"] == mode]
            groups.append(
                dict(
                    cohort="offline_witnesses" if gt else "predefined_policies",
                    mode=mode,
                    runs=len(subset),
                    successes=sum(r["mission_success"] for r in subset),
                )
            )
    summary = dict(
        schema_version=VERSION,
        groups=groups,
        rows=rows,
        replay_smoke=smoke,
        protocol_sha256=sha(output / "protocol.json"),
    )
    save(output / "summary.json", summary)
    lines = [
        "# Explicit intent contract migration",
        "",
        "This is a post-hoc audit of fixed development recordings, not held-out policy evidence. "
        "Historical results and requirements remain unchanged. New runs use outcome.json as "
        "their sole primary success result; execution/results.json is a legacy diagnostic.",
        "",
        "| Cohort | Intent | Runs | Successes |",
        "|---|---|---:|---:|",
    ]
    lines += [f"| {r['cohort']} | {r['mode']} | {r['runs']} | {r['successes']} |" for r in groups]
    lines += [
        "",
        "Evidence-only success does not establish tracked reporting success. "
        "Offline witnesses use evaluator ground truth and are not deployable routers. "
        "The report-required contract counts unique correct reports completed by its deadline; "
        "false and duplicate reports count across the entire execution, including late reports. "
        "This is stricter than the legacy first-correct-report deadline rule.",
        "",
        f"Fresh bound replay checks completed: {len(smoke)}. "
        "They reuse verified actual model masks and recorded Gazebo poses; cache lookup speed "
        "does not affect the simulated mission clock. No new live physics or neural inference "
        "is claimed. Energy and memory remain configured assumptions.",
        "",
        "The declared English intent is delivered to every router decision without target IDs. "
        "This verifies the interface, not that static policies consume intent or that a learned "
        "router generalizes. Free-form language parsing and improved predicted tracking remain "
        "separate work. The current tracker and perception pool are unchanged.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    return summary


def replay(prior: Path, source: Path, output: Path, row: dict, contract: IntentContract) -> dict:
    from experiments.lake_fast_evaluation import load_fast_scenario
    from experiments.lake_hindsight_search import HindsightSchedule
    from experiments.lake_hindsight_witness import VerifiedMasks
    from experiments.lake_intent_runner import run_intent_mission
    from experiments.lake_missions.scenario import load
    from experiments.v4_benchmark.backends import ReplayBackend
    from experiments.v4_benchmark.calibrated_router import model_signature
    from experiments.v4_benchmark.probe_cache import CachedExecutor
    from experiments.v4_benchmark.routers import build_router

    env, mission = row["environment"], row["mission_id"]
    suite = read(source / "suite.json")
    base = load(source / "scenario.json", suite["environments"][env], Path(suite["assets"]))
    control = next(s for s in base.executors if s.config_id == "CFG_MASKRCNN_1280")
    scenario = load_fast_scenario(
        prior / env / "missions" / mission / "scenario.json",
        suite["environments"][env],
        Path(suite["assets"]),
        control,
    )
    rotated = Path(read(prior / "protocol.json")["prior"])
    original = Path(read(rotated / "protocol.json")["prior"])
    available = {}
    for root in (original, rotated, prior):
        directory = root / env / "predictions"
        manifest, complete = read(directory / "manifest.json"), read(directory / "completion.json")
        if not complete["complete"] or complete["manifest_sha256"] != sha(
            directory / "manifest.json"
        ):
            raise ValueError("Unverified inference manifest")
        if manifest["recording_identity"] != scenario.recording_identity:
            raise ValueError("Cache does not match fixed flight")
        available.update(
            {config: (directory, model) for config, model in manifest["models"].items()}
        )
    executors = {}
    for spec in scenario.executors:
        directory, model = available[spec.config_id]
        if model["signature"] != model_signature(spec):
            raise ValueError("Cache model signature mismatch")
        executors[spec.config_id] = CachedExecutor(
            spec, VerifiedMasks(directory, model), sha(directory / "manifest.json")
        )
    if row["gt_aware"]:
        router = HindsightSchedule(
            read(prior / "hindsight" / env / mission / "search.json")["schedule"]
        )
    else:
        router = build_router(row["policy"], scenario.profiles, scenario.observation_interval_s)
    destination = (
        output
        / "replay"
        / ("offline" if row["gt_aware"] else "static")
        / contract.document()["reporting"]["mode"]
    )
    backend = ReplayBackend(scenario, source / env / "recording")
    poses = {t: f["body_pose"] for t, f in backend.frames.items()}
    try:
        outcome = run_intent_mission(
            scenario,
            backend,
            router,
            executors,
            contract,
            destination,
            router_name=row["policy"],
            gt_aware=row["gt_aware"],
        )
    finally:
        backend.close()
    events = read(destination / "execution/events.json")
    previous = read(Path(row["output"]) / "events.json")
    keys = (
        "observation_id",
        "capture_timestamp_ns",
        "capture_body_pose",
        "capture_time_s",
        "completion_time_s",
        "cumulative_energy_j",
        "battery_after",
        "score",
        "emitted_reports",
    )
    if len(events) != len(previous) or any(
        any(a[key] != b[key] for key in keys)
        or a["execution"]["config_id"] != b["execution"]["config_id"]
        for a, b in zip(events, previous, strict=True)
    ):
        raise ValueError("Intent binding changed fixed-schedule execution")
    if any(e["capture_body_pose"] != poses[e["capture_timestamp_ns"]] for e in events):
        raise ValueError("Capture pose differs from canonical recording")
    expected = evaluate(contract, previous, read(Path(row["output"]) / "results.json"))
    if any(outcome[key] != value for key, value in expected.items()):
        raise ValueError("Fresh replay and retrospective scorer disagree")
    verification = dict(
        output=str(destination.resolve()),
        gt_aware=row["gt_aware"],
        mode=outcome["mode"],
        mission_success=outcome["mission_success"],
        exact_execution_parity=True,
        exact_capture_poses=True,
        canonical_score_parity=True,
        public_contract_deliveries=outcome["public_contract_deliveries"],
        outcome_sha256=sha(destination / "outcome.json"),
    )
    save(destination / "verification.json", verification)
    print(f"Verified {row['policy']} / {outcome['mode']}: {outcome['mission_success']}", flush=True)
    return verification


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prior", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replay-smoke", action="store_true")
    args = parser.parse_args()
    audit(args.prior, args.output, args.replay_smoke)
