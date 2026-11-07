"""Sparse opportunities retain the paid runtime, canonical clock and flight costs.

Tiny synthetic RGB/masks exercise the real policy factory, selected executor,
downstream pipeline and scorer; no model weights or Gazebo are required.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from experiments.lake_causal_router import MissionCosts
from experiments.lake_intent_contract import IntentContract
from experiments.lake_tracking import HeightField
from experiments.mission_network_emulation import LinkSegment, LinkTrace
from experiments.mission_network_policies import ActionDescriptor, PolicyCosts
from experiments.mission_network_study_v1.executors import prepare_payload
from experiments.mission_runtime_profile import read, save, sha
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_gazebo_pilot.timing import CaptureSchedule
from experiments.wide_area_routing import adapter, runner
from tests.test_mission_transfer_comparison import synthetic_case
from tests.test_v4_benchmark import _FakeBackend

SECOND = 1_000_000_000


@pytest.mark.parametrize(
    "finish, expected_position, skipped",
    [
        (6 * SECOND, 1, ()),
        (6 * SECOND + 1, 2, (3,)),
        (16 * SECOND, 2, (3,)),
        (16 * SECOND + 1, 3, (3, 8)),
        (40 * SECOND, 5, (3, 8, 9, 12)),
    ],
)
def test_noncontiguous_schedule_skips_only_busy_eligible_captures(finish, expected_position, skipped):
    # Nonzero acquisition epoch protects against accidentally treating relative
    # execution time as an absolute Gazebo timestamp.
    schedule = CaptureSchedule(3 * SECOND, 2 * SECOND, 33 * SECOND, 10_000_000)
    indices = (2, 3, 8, 9, 12)
    completion, position, missed = runner.advance_opportunities(schedule, indices, 0, finish)
    assert completion == ((finish + schedule.step_ns - 1) // schedule.step_ns) * schedule.step_ns
    assert position == expected_position and missed == skipped
    assert not set(missed) & {0, 1, 4, 5, 6, 7, 10, 11, 13, 14}


def prepared_case(tmp_path, monkeypatch, *, budget=8, battery=0.9, capacity_wh=100):
    scenario, original_contract, profiles = synthetic_case(tmp_path)
    document = original_contract.document()
    document["duration_s"] = 30
    document["limits"]["deadline_s"] = 40
    document["mission"]["initial_battery_frac"] = battery
    document["mission"]["phases"][0].update(end_s=30, deadline_s=30)
    document["reporting"]["deadline_s"] = 30
    contract = IntentContract(document)
    scenario = replace(
        scenario,
        duration_s=30,
        battery_capacity_wh=capacity_wh,
        flight_power_w=10,
        initial_battery_frac=battery,
        router_latency_s=0.05,
        router_energy_j=0.75,
        contract=replace(scenario.contract, **document["limits"]),
    )
    actions = tuple(
        ActionDescriptor(
            s.config_id,
            s.config_id,
            "local",
            model_signature(s),
            4.1,
            1.5,
            memory_mb=64,
        )
        for s in scenario.executors
    )
    case = dict(
        scenario=scenario,
        contract=contract,
        actions=actions,
        costs=PolicyCosts(MissionCosts.from_scenario(scenario)),
        trace=LinkTrace((LinkSegment(0, 60 * SECOND, 50_000_000, 100_000_000, 0),)),
        profiles=profiles,
        learned_profile=None,
    )
    monkeypatch.setattr(adapter, "load_case", lambda *_: case)
    indices = [2, 3, 8, 9, 12]
    frames, payloads = {}, {}
    backend = _FakeBackend(scenario)
    for index in indices:
        backend.advance_to(scenario.schedule.at(index))
        frames[index] = backend.capture()
        payloads[index] = prepare_payload(frames[index].rgb)
    bank = dict(
        frames=frames,
        payloads=payloads,
        opportunity_indices=indices,
        provenance=dict(recording_identity=scenario.recording_identity, opportunity_indices=indices,
                        recording_sha256="synthetic_recording_bytes",
                        capture_schedule={name: getattr(scenario.schedule, name)
                                          for name in ("epoch_ns", "interval_ns", "end_ns")}),
    )
    item = dict(
        flight="synthetic_flight", policy="fixed_local_fast", battery=battery,
        link="good", perception_budget_s=budget, run_id="synthetic_run",
    )
    root = tmp_path / "study"
    folder = root / "predictions" / item["flight"]
    folder.mkdir(parents=True)
    mask_path = folder / "mask.npz"
    mask = frames[2].object_ids > 0
    np.savez_compressed(mask_path, prediction_mask=mask)
    entry = dict(file=mask_path.name, sha256=sha(mask_path),
                 mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    manifest = dict(
        recording_identity=scenario.recording_identity,
        recording_sha256="synthetic_recording_bytes",
        models={a.source_config_id: dict(signature=a.source_signature,
                predictions={p.rgb_sha256: entry for p in payloads.values()}) for a in actions},
    )
    plan = dict(fixture="Synthetic inputs; no pretrained inference or physical simulation", grid=[item])
    save(folder / "manifest.json", manifest)
    save(root / "plan.json", plan)
    terrain = HeightField(np.zeros((4, 4)), west=-10, east=10, south=-10, north=10)
    return dict(root=root, plan=plan, item=item, terrain=terrain, bank=bank, manifest=manifest), case


def outputs(settings, row):
    folder = settings["root"] / row["path"]
    return {name: read(folder / (name + ".json"))
            for name in ("protocol", "intent", "events", "results", "outcome")}


def test_paid_sparse_replay_includes_initial_transit_gaps_and_final_tail(tmp_path, monkeypatch):
    settings, case = prepared_case(tmp_path, monkeypatch)
    row = runner.run_case(**settings)
    saved = outputs(settings, row)
    events, result = saved["events"], saved["results"]
    assert [e["observation_id"] for e in events] == [2, 8, 12]
    assert [e["capture_time_s"] for e in events] == [4, 16, 24]
    assert [e["skipped_observation_ids"] for e in events] == [[3], [9], []]
    assert all(e["pipeline"] is not None for e in events)
    assert result["skipped_observation_ids"] == [3, 9]
    assert result["inactive_capture_count"] == 10
    assert result["unprocessed_opportunity_ids"] == []
    assert result["final_time_s"] == 30 > events[-1]["completion_time_s"]
    assert result["flight_energy_j"] == 300  # 30 seconds, not only invocation time.
    assert events[0]["battery_at_decision"] == pytest.approx(
        0.9 - (4.02 * 10 + 0.02 * 15) / 360000
    )
    assert result["cumulative_energy_j"] == pytest.approx(
        300 + sum(e["perception"]["costs"]["charged_energy_j"]
                  + e["pipeline"]["charged_energy_j"] for e in events)
    )
    assert row["nonflight_energy_j"] > 0 and row["total_energy_j"] > 300
    assert row["final_battery_frac"] == pytest.approx(0.9 - row["total_energy_j"] / 360000)
    assert row["decision_count"] == row["perception_latency"]["count"] == 3
    assert row["perception_latency"]["p95_s"] == pytest.approx(4.17)
    assert row["selected_action_counts"] == {"CFG_YOLO11S_960_ROT0": 3}
    assert saved["protocol"]["recording"]["recording_identity"] == case["scenario"].recording_identity
    assert saved["protocol"]["opportunity_indices"] == [2, 3, 8, 9, 12]
    assert saved["protocol"]["prediction_manifest_sha256"] == sha(
        settings["root"] / "predictions/synthetic_flight/manifest.json"
    )
    assert all(sha(path) == expected for path, expected in saved["protocol"]["implementation_sha256"].items())
    assert all(sha(settings["root"] / row["path"] / name) == expected
               for name, expected in row["artifacts_sha256"].items())


def test_late_selected_outputs_are_paid_but_never_read_or_sent_downstream(tmp_path, monkeypatch):
    settings, _ = prepared_case(tmp_path, monkeypatch, budget=2)
    # A cache read would fail hash validation. Existing execute_perception must
    # discard the late result before asking VerifiedMasks for any pixels.
    mask_path = settings["root"] / "predictions/synthetic_flight/mask.npz"
    mask_path.write_bytes(b"deliberately unavailable selected output")
    row = runner.run_case(**settings)
    saved = outputs(settings, row)
    events, result = saved["events"], saved["results"]
    assert len(events) == 3 and result["skipped_observation_ids"] == [3, 9]
    for event in events:
        assert event["perception"]["status"] == "task_deadline_miss"
        assert event["perception"]["costs"]["compute_energy_j"] == 1.5
        assert not event["perception"]["prediction_cache_read"]
        assert event["pipeline"] is None and event["emitted_reports"] == []
        assert event["score"]["matched_target_ids"] == []
        assert event["selected_mask_sha256"] is None
        assert event["downstream_skip_reason"] == "perception_output_not_admitted"
    assert result["flight_energy_j"] == 300 and result["cumulative_nonflight_energy_j"] > 0
    assert row["task_metrics"]["perception"]["deadline_misses"] == 3
    assert saved["outcome"]["constraints"]["execution"]
    assert not row["mission_success"]


@pytest.mark.parametrize("mutation", ["manifest_identity", "recording_hash", "model_signature", "bank_identity", "capture_clock", "pose_clock", "schedule_receipt", "duplicate_index", "missing_payload"])
def test_input_binding_failures_are_rejected_before_run_artifacts(tmp_path, monkeypatch, mutation):
    settings, _ = prepared_case(tmp_path, monkeypatch)
    if mutation == "manifest_identity":
        settings["manifest"]["recording_identity"] = "wrong"
    elif mutation == "recording_hash":
        settings["manifest"]["recording_sha256"] = "wrong"
    elif mutation == "model_signature":
        next(iter(settings["manifest"]["models"].values()))["signature"] = "wrong"
        save(settings["root"] / "predictions/synthetic_flight/manifest.json", settings["manifest"])
    elif mutation == "bank_identity":
        settings["bank"]["provenance"]["recording_identity"] = "wrong"
    elif mutation == "capture_clock":
        settings["bank"]["frames"][2] = replace(settings["bank"]["frames"][2], timestamp_ns=1)
    elif mutation == "pose_clock":
        settings["bank"]["frames"][2].body_pose["timestamp_ns"] = 1
    elif mutation == "schedule_receipt":
        settings["bank"]["provenance"]["capture_schedule"]["interval_ns"] = SECOND
    elif mutation == "duplicate_index":
        settings["bank"]["opportunity_indices"] = [2, 2, 3, 8, 9, 12]
    elif mutation == "missing_payload":
        del settings["bank"]["payloads"][8]
    with pytest.raises(ValueError):
        runner.run_case(**settings)
    assert not (settings["root"] / "runs").exists()


def test_empty_latency_summary_does_not_invent_measurements():
    assert runner.latency_summary([])["count"] == 0
    assert runner.latency_summary([])["p95_s"] is None


def test_battery_can_deplete_in_transit_after_the_final_perception(tmp_path, monkeypatch):
    settings, _ = prepared_case(tmp_path, monkeypatch, capacity_wh=0.095)
    row = runner.run_case(**settings)
    saved = outputs(settings, row)
    assert len(saved["events"]) == 3
    assert saved["events"][-1]["battery_after"] > 0
    assert row["final_time_s"] == 30 and row["flight_energy_j"] == 300
    assert row["final_battery_frac"] == 0 and row["termination_reason"] == "battery_depleted"
    assert not row["mission_completed"] and not row["mission_success"]
