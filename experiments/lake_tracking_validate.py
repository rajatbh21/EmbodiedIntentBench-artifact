"""Verify tracking provenance, fixed-flight timing and complete energy accounting."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_missions.assets import save, sha
from experiments.v4_benchmark.scenario import STEP_NS
from experiments.v4_gazebo_pilot.timing import CaptureSchedule, seconds_to_ns


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root):
    protocol, summary = read(root / "protocol.json"), read(root / "summary.json")
    require(summary["protocol_sha256"] == sha(root / "protocol.json"), "Study protocol changed")
    for name, expected in protocol["implementation_sha256"].items():
        require(sha(Path(__file__).with_name(name)) == expected, "Implementation changed")
    for row in protocol["input_runs"]:
        for name in ("events", "results"):
            require(
                sha(Path(row["output"]) / f"{name}.json") == row[f"{name}_sha256"],
                "Historical result changed",
            )
    source = Path(protocol["source"])
    raw = read(source / "scenario.json")
    mission = raw["mission"]
    epoch = seconds_to_ns(mission["warmup_s"])
    schedule = CaptureSchedule(
        epoch,
        seconds_to_ns(mission["observation_interval_s"]),
        epoch + seconds_to_ns(mission["duration_s"]),
        STEP_NS,
    )
    poses = {}
    pairs = 0
    for env in ("day", "dusk_haze"):
        provenance = read(root / env / "packet_provenance.json")
        pairs += provenance["pairs"]
        for name, expected in provenance["prediction_files_sha256"].items():
            require(sha(Path(name)) == expected, "Prediction mask changed")
        for name, expected in provenance["prediction_manifests_sha256"].items():
            require(sha(Path(name)) == expected, "Prediction manifest changed")
        require(
            sha(root / env / "public_measurements.json")
            == provenance["public_measurements_sha256"],
            "Public packet table changed",
        )
        require(
            sha(root / env / "private_component_matches.json")
            == provenance["private_matches_sha256"],
            "Private scoring table changed",
        )
        recording = source / env / "recording/recording.json"
        require(sha(recording) == provenance["recording_sha256"], "Recording manifest changed")
        poses[env] = {f["timestamp_ns"]: f["body_pose"] for f in read(recording)["frames"]}
    require(
        len(summary["fixed_schedule"]) == protocol["expected_fixed_schedule_comparisons"],
        "Incomplete paired study",
    )
    for row in summary["fixed_schedule"]:
        require(sha(Path(row["output"])) == row["sha256"], "Diagnostic outcome changed")
    require(
        len(summary["paid_replays"]) == protocol["expected_paid_replays"], "Incomplete paid study"
    )
    for row in summary["paid_replays"]:
        folder = Path(row["output"])
        outcome = read(folder / "outcome.json")
        declared = read(folder / "protocol.json")
        result, events = (
            read(folder / "execution/results.json"),
            read(folder / "execution/events.json"),
        )
        require(sha(folder / "outcome.json") == row["outcome_sha256"], "Paid outcome changed")
        for name, file in (
            ("protocol", "protocol.json"),
            ("events", "execution/events.json"),
            ("execution", "execution/results.json"),
        ):
            require(outcome[f"{name}_sha256"] == sha(folder / file), "Outcome binding mismatch")
        contract = IntentContract(read(folder / "intent.json"))
        require(contract.sha256 == declared["contract_sha256"], "Intent mismatch")
        require(
            contract.public_document() == read(folder / "public_intent.json"),
            "Public intent mismatch",
        )
        scored = evaluate(contract, events, result)
        require(
            outcome["mission_success"] == scored["mission_success"]
            and outcome["constraints"] == scored["constraints"],
            "Canonical outcome mismatch",
        )
        require(
            result["tracking_cost"] == protocol["cost"] == declared["tracking_cost"],
            "Undeclared tracking costs",
        )
        require(
            outcome["public_contract_deliveries"] == len(events), "Missing policy contract delivery"
        )
        cumulative_nonflight = 0
        next_capture = 0
        for event in events:
            index = event["observation_id"]
            require(index == next_capture, "Illegal capture cadence")
            require(event["capture_timestamp_ns"] == schedule.at(index), "Capture clock mismatch")
            require(
                event["capture_body_pose"] == poses[row["environment"]][schedule.at(index)],
                "Pose mismatch",
            )
            latency = sum(
                (
                    event["configured_router_latency_s"],
                    event["model_load_latency_s"],
                    event["execution"]["mission_latency_s"],
                    event["configured_tracking_latency_s"],
                )
            )
            completion, next_capture, skipped = schedule.complete(
                index, seconds_to_ns(latency, allow_zero=True)
            )
            require(
                abs(event["completion_time_s"] - (completion - epoch) / 1e9) < 1e-9,
                "Uncharged tracking latency",
            )
            require(event["skipped_observation_ids"] == list(skipped), "Skipped-capture mismatch")
            cumulative_nonflight += (
                event["execution"]["energy_j"]
                + event["execution"]["communication_energy_j"]
                + event["model_load_energy_j"]
                + mission["router_energy_j"]
                + event["tracking_energy_j"]
            )
            expected_energy = (
                event["completion_time_s"] * mission["flight_power_w"] + cumulative_nonflight
            )
            require(
                abs(event["cumulative_energy_j"] - expected_energy) < 1e-6, "Event energy mismatch"
            )
        require(
            abs(result["tracking_energy_j"] - len(events) * protocol["cost"]["energy_j"]) < 1e-7,
            "Tracking energy mismatch",
        )
        require(
            abs(
                result["cumulative_energy_j"]
                - (result["final_time_s"] * mission["flight_power_w"] + cumulative_nonflight)
            )
            < 1e-6,
            "Final energy mismatch",
        )
    validation = dict(
        schema_version="tracking_validation_v1",
        historical_runs=len(protocol["input_runs"]),
        verified_prediction_pairs=pairs,
        fixed_schedule_comparisons=len(summary["fixed_schedule"]),
        paid_replays=len(summary["paid_replays"]),
        causal_capture_schedule_verified=True,
        original_physical_poses_verified=True,
        tracking_costs_charged=True,
        canonical_outcomes_verified=True,
        historical_results_unchanged=True,
        summary_sha256=sha(root / "summary.json"),
        implementation_sha256=sha(Path(__file__)),
    )
    save(root / "validation.json", validation)
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(validate(parser.parse_args().output))
