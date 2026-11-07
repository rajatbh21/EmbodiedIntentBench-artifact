import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from missionbench.v2.executors import ImageExecutionResult
from experiments.v4_benchmark.backends import Frame, ReplayBackend, record_world
from experiments.v4_benchmark.routers import build_router, immutable_rgb
from experiments.v4_benchmark.runner import run_mission
from experiments.v4_benchmark.scenario import load_scenario
from experiments.v4_benchmark.suite import run_suite

ROOT = Path(__file__).parents[1]


class _FakeBackend:
    def __init__(self, scenario):
        self.time_ns = scenario.schedule.epoch_ns
        self.provenance = {"backend": "test_fake"}

    def advance_to(self, time_ns):
        self.time_ns = time_ns

    def capture(self):
        ids = np.zeros((32, 32), dtype=np.uint8)
        ids[4:12, 4:12] = 1
        ids[16:24, 12:20] = 2
        ids[8:16, 24:32] = 3
        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[ids > 0] = (240, 20, 10)
        pose = {
            "timestamp_ns": self.time_ns,
            "position_m": [0.0, 0.0, 4.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        return Frame(self.time_ns, rgb, ids, pose)

    def close(self):
        pass


class _PerfectExecutor:
    def __init__(self, config_id, latency):
        self.config_id = config_id
        self.model_strategy_id = f"test_{config_id}"
        self.latency = latency

    def run(self, rgb):
        started = np.zeros(rgb.shape[:2], dtype=bool)
        started[(rgb[..., 0] > 200) & (rgb[..., 1] < 50)] = True
        return ImageExecutionResult(
            config_id=self.config_id,
            model_strategy_id=self.model_strategy_id,
            success=True,
            prediction_mask=started,
            mission_latency_s=self.latency,
            measured_wall_clock_s=0.0,
            energy_j=1.0,
            communication_mb=0.0,
            measurement_provenance={"test": "configured"},
        )


def test_scenario_load_and_router_are_public_only():
    scenario = load_scenario(ROOT / "experiments/v4_benchmark/example_scenario.json")
    router = build_router("static:CFG_V4_LIGHT", scenario.profiles, scenario.observation_interval_s)
    assert scenario.targets == {1: "target_1", 2: "target_2", 3: "target_3"}
    assert router.select  # adapter exposes the public router surface


def test_fixed_world_mission_evaluates_rgb_prediction_and_latency():
    source = load_scenario(ROOT / "experiments/v4_benchmark/example_scenario.json")
    scenario = replace(source, duration_s=2.0)
    backend = _FakeBackend(scenario)
    router = build_router("static:CFG_V4_LIGHT", scenario.profiles, scenario.observation_interval_s)
    executors = {
        "CFG_V4_LIGHT": _PerfectExecutor("CFG_V4_LIGHT", 0.8),
        "CFG_V4_STRONG": _PerfectExecutor("CFG_V4_STRONG", 0.8),
    }
    result = run_mission(scenario, backend, router, executors, router_name="test")
    assert result["status"] == "completed"
    assert result["mission_success"]
    assert result["processed_observations"] == 2
    assert result["quality"]["target_recall"] == 1.0
    assert result["skipped_observation_ids"] == [1, 3]
    assert result["final_time_s"] == 2.0


def test_rgb_is_independent_and_cannot_be_made_writable():
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    public = immutable_rgb(original)
    original[:] = 255
    assert not public.any()
    with pytest.raises(ValueError):
        public.setflags(write=True)


def test_replay_rejects_changed_world_and_tampered_frames(tmp_path):
    scenario = load_scenario(ROOT / "experiments/v4_benchmark/example_scenario.json")
    recording = tmp_path / "recording"
    record_world(scenario, _FakeBackend(scenario), recording)
    with pytest.raises(ValueError, match="differs"):
        ReplayBackend(replace(scenario, duration_s=3.0), recording)
    (recording / "frame_00000.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        ReplayBackend(scenario, recording)


def test_paired_suite_preserves_failures_and_reproducible_inputs(tmp_path):
    scenario = load_scenario(ROOT / "experiments/v4_benchmark/remote_scenario.json")
    recording = tmp_path / "recording"
    record_world(scenario, _FakeBackend(scenario), recording)
    spec = ROOT / "experiments/v4_benchmark/example_suite.json"
    report = run_suite(spec, recording, tmp_path / "suite")
    assert len(report["rows"]) == 40
    assert all(row["status"] == "completed" for row in report["rows"])
    rows = {(r["case_id"], r["router"]): r for r in report["rows"]}
    remote = "static:CFG_V4_REMOTE"
    privacy = rows["local_privacy", remote]
    assert not privacy["mission_success"]
    assert privacy["privacy_violations"] == 16
    assert privacy["communication_mb"] == 0
    assert (
        "communication_constraint_success" in rows["limited_upload", remote]["failed_constraints"]
    )
    assert rows["network_loss", remote]["inference_failures"] == 8
    assert not rows["tight_deadline", "rule_based"]["mission_success"]
    assert rows["nominal", "static:CFG_V4_STRONG"]["skipped_observations"] == 8
    identities = {r["backend"]["recording_identity"] for r in report["rows"]}
    assert len(identities) == 1
    result_folder = tmp_path / "suite" / "high_precision" / "router_00"
    restored = load_scenario(result_folder / "scenario.json")
    assert restored.contract.quality_metric == "detection_precision"
    # Wall-clock diagnostics stay in events; the comparison is reproducible.
    again = run_suite(spec, recording, tmp_path / "again")
    assert report == again
    result = json.loads((result_folder / "results.json").read_text())
    assert (
        result["mission_success"]
        == rows["high_precision", "static:CFG_V4_LIGHT"]["mission_success"]
    )


def test_suite_rejects_typo_before_evaluation(tmp_path):
    source = ROOT / "experiments/v4_benchmark/example_suite.json"
    spec = json.loads(source.read_text())
    spec["base_scenario"] = str(source.parent / "remote_scenario.json")
    spec["cases"][0]["scenario_overrides"] = {"mission": {"battery_typo": 0.5}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="battery_typo"):
        run_suite(path, tmp_path / "unused_recording", tmp_path / "bad_output")
    assert not list((tmp_path / "bad_output").rglob("results.json"))


def test_decision_audit_separates_deadline_state_network_and_intent(tmp_path):
    folder = ROOT / "experiments/v4_benchmark"
    scenario = load_scenario(folder / "audit_scenario.json")
    higher = load_scenario(folder / "higher_altitude_scenario.json")
    recording, high_recording = tmp_path / "normal", tmp_path / "high"
    record_world(scenario, _FakeBackend(scenario), recording)
    record_world(higher, _FakeBackend(higher), high_recording)
    output = tmp_path / "audit"
    run_suite(
        folder / "audit_suite.json",
        recording,
        output,
        case_recordings={"higher_altitude": high_recording},
    )
    audit = json.loads((output / "decision_audit.json").read_text())
    pairs = {(p["case_id"], p["router"]): p for p in audit["pairs"]}
    assert len(pairs) == 50
    assert not any(p["explanation_mismatches"] for p in pairs.values())
    for router in ("rule_based", "budget_planner"):
        assert pairs["nominal", router]["deadline_pressure_decisions"] == 0
        assert pairs["deadline_equal_flight", router]["deadline_pressure_decisions"] == 15
        assert pairs["deadline_equal_flight", router]["changed_action_capture_ids"]
        assert pairs["low_battery", router]["battery_pressure_decisions"] > 0
        assert pairs["network_loss", router]["changed_action_capture_ids"]
        assert pairs["high_precision", router]["changed_action_capture_ids"] == []
        assert not pairs["higher_altitude", router]["same_recording"]
    events = json.loads((output / "nominal/router_03/events.json").read_text())
    assert events[1]["router_decision"]["projected_finish_s"] == 8.0
    assert events[1]["router_decision"]["deadline_trigger_s"] == 9.5
    assert events[1]["router_decision"]["quality_threshold_used_in_selection"] is False


@pytest.mark.parametrize("name", ["rule_based", "budget_planner"])
def test_audit_does_not_change_policy_state_or_decisions(tmp_path, name):
    from missionbench.v2.executors import build_executors

    folder = ROOT / "experiments/v4_benchmark"
    scenario = load_scenario(folder / "audit_scenario.json")
    annotated = build_router(name, scenario.profiles, scenario.observation_interval_s)
    plain = build_router(name, scenario.profiles, scenario.observation_interval_s)

    class WithoutAudit:
        def select(self, observation):
            return plain.policy.select_config(
                observation.contract, observation.state, observation.configs
            )

    a = run_mission(
        scenario,
        _FakeBackend(scenario),
        annotated,
        build_executors(scenario.executors),
        router_name=name,
    )
    b = run_mission(
        scenario,
        _FakeBackend(scenario),
        WithoutAudit(),
        build_executors(scenario.executors),
        router_name=name,
    )
    assert a == b
