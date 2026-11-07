"""Audit development completion and smoke-test profile compatibility with v5 routing."""

import argparse
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.lake_causal_router import MissionCosts
from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, immutable
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save, sha
from experiments.lake_tracking import TrackingConfig
from experiments.lake_tracking_runtime import DEFAULT_TRACKING_COST
from experiments.mission_routing_v2.flights import load_scenario
from experiments.mission_routing_v5.policy import Policy
from experiments.routing_development_matrix.cohort import verify
from experiments.routing_development_matrix.train import verify_training


def run(root, training):
    p, profile = verify(root), verify_training(training)
    report, validation = read(root / "analysis.json"), read(root / "validation.json")
    preview, coverage = (
        read(root / "physical_preview_provenance.json"),
        read(root / "observability.json"),
    )
    if (
        not report["complete"]
        or not validation["complete"]
        or report["validation_sha256"] != sha(root / "validation.json")
        or report["training_completion_sha256"] != sha(training / "completion.json")
        or report["sensor_assessment_sha256"] != sha(root / "observability.json")
        or report["protocol_sha256"] != sha(root / "protocol.json")
        or validation["protocol_sha256"] != sha(root / "protocol.json")
        or preview["protocol_sha256"] != sha(root / "protocol.json")
        or coverage["protocol_sha256"] != sha(root / "protocol.json")
        or preview["image_sha256"] != sha(root / "physical_preview.png")
    ):
        raise ValueError("Mismatched development release artifacts")
    for artifact, key, filename in (
        (report, "source_sha256", "report.py"),
        (validation, "validator_sha256", "validate.py"),
        (preview, "source_sha256", "preview.py"),
        (coverage, "source_sha256", "observability.py"),
    ):
        if artifact[key] != sha(Path(__file__).with_name(filename)):
            raise ValueError("Changed artifact implementation")
    risk = read(
        Path(read(Path(p["original"]) / "protocol.json")["risk_training"]) / "risk_profile.json"
    )
    smoke = []
    for flight, case in p["cases"].items():
        folder = root / flight / "rapid_survey"
        scenario = load_scenario(folder / "scenario.json", case["environment"], Path(p["assets"]))
        contract = IntentContract(read(folder / "intent.json"))
        scenario = bind_scenario(scenario, contract)
        recording = root / flight / "recording/recording.json"
        if coverage["cases"][flight]["recording_sha256"] != sha(recording):
            raise ValueError("Coverage belongs to changed recording")
        frame = read(recording)["frames"][0]
        path = recording.parent / frame["file"]
        if sha(path) != frame["sha256"]:
            raise ValueError("Changed physical first frame")
        with np.load(path, allow_pickle=False) as data:
            rgb = data["rgb"]
        o = SimpleNamespace(
            rgb=rgb,
            contract=scenario.contract,
            state=SimpleNamespace(
                current_time_s=0,
                frame_id=0,
                current_config_id=None,
                battery_frac=scenario.initial_battery_frac,
            ),
            configs=scenario.catalog,
            resources=scenario.resources,
            intent_contract=immutable(contract.public_document()),
            tracking=immutable(
                dict(
                    configuration=asdict(TrackingConfig(remember_reports=True)),
                    cost=asdict(DEFAULT_TRACKING_COST),
                )
            ),
            evidence_history=immutable(
                dict(
                    available_at_s=0,
                    last_completion_s=None,
                    tracks=[],
                    emitted_prediction_reports=0,
                    counts_are_ground_truth=False,
                )
            ),
        )
        for variant in ("spatial_recall", "base_recall", "global_recall"):
            policy = Policy(
                dict(profile, routing_variant=variant),
                scenario.executors,
                MissionCosts.from_scenario(scenario),
                risk,
            )
            choice = policy.select(o)
            if choice not in scenario.catalog.ids() or policy.last_decision["uses_ground_truth"]:
                raise ValueError("Expanded profile runtime boundary failed")
            smoke.append(
                dict(
                    flight=flight,
                    variant=variant,
                    selected_config_id=choice,
                    gt_used=False,
                    mission_replay=False,
                )
            )
    save(
        root / "profile_runtime_smoke.json",
        dict(
            schema_version="development_profile_smoke_v1",
            checks=smoke,
            mission_performance_claim=False,
            profile_sha256=sha(training / "profile.json"),
        ),
    )
    files = (
        "protocol.json",
        "validation.json",
        "physical_route_validation.json",
        "analysis.json",
        "REPORT.md",
        "observability.json",
        "physical_preview.png",
        "physical_preview_provenance.json",
        "profile_runtime_smoke.json",
    )
    if (root / "coverage_and_transfer_provenance.json").exists():
        figure = read(root / "coverage_and_transfer_provenance.json")
        if (
            figure["analysis_sha256"] != sha(root / "analysis.json")
            or figure["image_sha256"] != sha(root / "coverage_and_transfer.png")
            or figure["source_sha256"] != sha(Path(__file__).with_name("plot.py"))
        ):
            raise ValueError("Changed coverage/transfer figure")
        files = (*files, "coverage_and_transfer.png", "coverage_and_transfer_provenance.json")
    result = dict(
        schema_version="development_matrix_completion_v1",
        complete=True,
        scope="development_only",
        new_physical_flights=validation["new_physical_flights"],
        new_captures=validation["new_captures"],
        new_frame_config_pairs=validation["new_frame_config_pairs"],
        actual_neural_calls=validation["actual_neural_calls"],
        training=read(training / "completion.json"),
        sensor_coverage_passes=sum(c["all_targets_passed"] for c in coverage["cases"].values()),
        promote_as_all_target_mission_test=False,
        new_policy_missions_run=0,
        fresh_evaluation=False,
        source_sha256=sha(Path(__file__)),
        artifacts_sha256={n: sha(root / n) for n in files},
        training_completion_sha256=sha(training / "completion.json"),
    )
    save(root / "completion_audit.json", result)
    print(
        {
            k: result[k]
            for k in ("complete", "new_captures", "actual_neural_calls", "sensor_coverage_passes")
        },
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.training)
