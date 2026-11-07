"""Public boundary, geometry, split and paid feedback integration regressions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.lake_causal_router import MissionCosts
from experiments.lake_intent_contract import IntentContract, immutable
from experiments.lake_tracking import TrackingConfig, TrackingPipeline
from experiments.mission_routing_v2.features import NAMES, encode, fit, predict
from experiments.mission_routing_v2.feedback import snapshot
from experiments.mission_routing_v2.flights import contract_document, split_check
from experiments.mission_routing_v2.models import tiled_mask, tiles
from experiments.mission_routing_v2.policy import MODES, Policy, history_need
from experiments.mission_routing_v2.runtime import run_tracking_mission
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.scenario import load_scenario
from tests.test_v4_benchmark import ROOT, _FakeBackend, _PerfectExecutor


@pytest.mark.parametrize("shape", [(7, 11), (720, 960), (2, 2)])
def test_fixed_tiles_cover_sensor_and_preserve_native_mask(shape):
    rgb = np.zeros((*shape, 3), np.uint8)
    rgb[::2, ::3, 0] = 255
    calls = []
    predictor = SimpleNamespace(
        predict=lambda image, turn: (calls.append(turn) or image[..., 0] > 100, 0)
    )
    result = tiled_mask(rgb, predictor, turns=1)
    assert np.array_equal(result, rgb[..., 0] > 100)
    assert calls == [1] * 4
    coverage = np.zeros(shape, bool)
    for top, bottom, left, right in tiles(shape):
        coverage[top:bottom, left:right] = True
    assert coverage.all()


def test_gamma_is_fixed_rgb_only_and_does_not_mutate_input():
    rgb = np.full((8, 8, 3), 64, np.uint8)
    seen = []
    predictor = SimpleNamespace(
        predict=lambda image, turn: (
            seen.append(image.copy()) or np.zeros(image.shape[:2], bool),
            0,
        )
    )
    tiled_mask(rgb, predictor, gamma=0.5)
    assert all(np.all(v == 128) for v in seen)
    assert np.all(rgb == 64)
    with pytest.raises(ValueError):
        tiled_mask(rgb, predictor, gamma=float("nan"))


def test_continuous_context_distinguishes_dark_and_bright_without_metadata():
    a = encode(np.full((8, 8, 3), 30, np.uint8))
    b = encode(np.full((8, 8, 3), 220, np.uint8))
    assert len(a) == len(NAMES) and not np.array_equal(a, b)
    x = [a, b] * 8
    model = fit(x, [0, 1] * 8)
    assert predict(model, a)[0] < predict(model, b)[0]
    assert "target" not in repr(model)


def test_new_flight_split_allows_shared_map_but_rejects_shared_layout_or_route():
    training = dict(recording="r", world="w", scenery="s", layout="l", route="p")
    test = {k: v + "new" for k, v in training.items()}
    test["scenery"] = "s"
    assert split_check([training], test)["overlaps"] == ["scenery"]
    for key in ("recording", "world", "layout", "route"):
        with pytest.raises(ValueError, match="overlaps"):
            split_check([training], {**test, key: training[key]})


def test_feedback_excludes_future_and_stale_tracks_and_is_immutable():
    tracker = TrackingPipeline(TrackingConfig(association="image_iou"))
    mask = np.zeros((12, 12), bool)
    mask[2:7, 2:7] = True
    tracker.update(mask, 0, 1)
    with pytest.raises(ValueError, match="Future"):
        snapshot(tracker, 0.5)
    value = immutable(snapshot(tracker, 2))
    assert value["tracks"][0]["capture_times_s"] == (0,)
    with pytest.raises(TypeError):
        value["tracks"][0]["reported"] = True
    tracker.update(mask, 2, 3)
    assert value["tracks"][0]["capture_times_s"] == (0,)
    assert snapshot(tracker, 12)["tracks"] == []


def test_revisit_history_does_not_count_earlier_phase_evidence():
    history = dict(tracks=[dict(capture_times_s=[0, 2, 20], reported=True)])
    phase = dict(start_s=15, end_s=30, min_hits=2, min_evidence_span_s=2)
    needs = history_need(history, phase, 22)
    assert needs[0]["hits"] == 1 and needs[0]["missing_hits"] == 1
    history["tracks"][0]["capture_times_s"] = [0, 2]
    assert history_need(history, phase, 22) == []


@pytest.mark.parametrize("mission", ["rapid_survey", "verified_inspection", "reserve_patrol"])
def test_all_predeclared_intents_validate_and_redact_targets(mission):
    contract = IntentContract(contract_document([f"hidden{i}" for i in range(6)], mission))
    assert "hidden" not in repr(contract.public_document())
    assert len(contract.document()["mission"]["phases"]) == 4


@pytest.mark.parametrize("mode", MODES)
def test_feedback_controller_runs_through_paid_runtime(tmp_path, mode):
    from tests.test_lake_intent_contract import fixture

    document, _, _ = fixture()
    base = load_scenario(ROOT / "experiments/v4_benchmark/example_scenario.json")
    targets = list(base.targets.values())
    document["evaluator_target_ids"] = targets
    document["mission"]["phases"][0]["evaluator_target_ids"] = targets
    contract = IntentContract(document)
    scenario = replace(
        base,
        duration_s=10,
        initial_battery_frac=0.9,
        contract=replace(
            base.contract,
            deadline_s=10,
            min_final_battery_frac=0.2,
            quality_threshold=0,
            communication_budget_mb=0,
            privacy_level="local_only",
        ),
    )
    feature = encode(np.zeros((8, 8, 3), np.uint8))
    profile = dict(
        feature_names=list(NAMES),
        training_lineage=[{"secret": "excluded"}],
        models={
            s.config_id: dict(
                signature=model_signature(s),
                quality=fit([feature] * 4, [0.5] * 4),
                reward=fit([feature] * 4, [0.5] * 4),
            )
            for s in scenario.executors
        },
    )
    router = Policy(profile, scenario.executors, MissionCosts.from_scenario(scenario), mode)
    assert not hasattr(router, "training_lineage")
    backend = _FakeBackend(scenario)
    backend.provenance["recording_identity"] = scenario.recording_identity
    result = run_tracking_mission(
        scenario,
        backend,
        router,
        {
            s.config_id: _PerfectExecutor(s.config_id, s.mission_latency_s)
            for s in scenario.executors
        },
        contract,
        tmp_path / "run",
        config=TrackingConfig(association="image_iou"),
        terrain=None,
        map_metadata=None,
        router_name=mode,
        gt_aware=False,
    )
    assert result["public_contract_deliveries"] > 1
    assert not result["gt_aware"]
    assert "evaluator_target_ids" not in repr(router.last_decision)
    assert (
        router.last_decision["history"]["last_completion_s"]
        <= router.last_decision["history"]["available_at_s"]
    )


def controller_fixture(mode):
    from tests.test_lake_causal_router import fixture

    _, observation, _, _, specs = fixture()
    dark = encode(np.zeros((8, 8, 3), np.uint8))
    bright = encode(np.full((8, 8, 3), 255, np.uint8))
    profile = dict(
        feature_names=list(NAMES),
        models={
            s.config_id: dict(
                signature=model_signature(s),
                quality=fit([dark, bright] * 16, ([0.9, 0.1] if i == 0 else [0.1, 0.9]) * 16),
                reward=fit([dark, bright] * 16, ([0.9, 0.1] if i == 0 else [0.1, 0.9]) * 16),
            )
            for i, s in enumerate(specs)
        },
    )
    observation.evidence_history = immutable(
        dict(tracks=[], last_completion_s=None, available_at_s=0)
    )
    observation.tracking["configuration"] = dict(max_gap_s=8)
    return Policy(
        profile, specs, MissionCosts(8, 0.5, 1000, 1, 0.01, 0.02, 10000000), mode
    ), observation


def test_public_rgb_and_battery_can_change_selected_model():
    policy, observation = controller_fixture("performance_predictor")
    assert policy.select(observation) == "CFG_V4_LIGHT"
    observation.rgb[:] = 255
    assert policy.select(observation) == "CFG_V4_STRONG"
    observation.state.battery_frac = 0.275
    assert policy.select(observation) == "CFG_V4_LIGHT"


def test_phase_deadline_changes_intent_policy_without_changing_quality_predictions():
    policy, observation = controller_fixture("intent_history")
    for name, model in policy.models.items():
        value = 0.3 if name == "CFG_V4_LIGHT" else 0.9
        model["quality"] = fit([encode(observation.rgb)] * 16, [value] * 16)
    assert policy.select(observation) == "CFG_V4_STRONG"
    before = [c["quality"] for c in policy.last_decision["candidates"]]
    observation.intent_contract["mission"]["phases"][0].update(
        end_s=0.9, deadline_s=0.9, min_evidence_span_s=0.5
    )
    assert policy.select(observation) == "CFG_V4_LIGHT"
    assert [c["quality"] for c in policy.last_decision["candidates"]] == before


@pytest.mark.parametrize("reaches", [True, False])
def test_physical_route_gate_rejects_missing_waypoints_without_model_outputs(
    tmp_path, monkeypatch, reaches
):
    from experiments import mission_routing_v2_route_validation as gate
    from experiments.lake_missions.assets import save

    root = tmp_path / "study"
    folder = root / "flight/recording"
    folder.mkdir(parents=True)
    save(
        folder / "recording.json",
        dict(
            frames=[
                dict(timestamp_ns=int(t * 1e9), body_pose=dict(position_m=xyz))
                for t, xyz in (
                    (0, [0, 0, 10]),
                    (2, [0, 0, 10]),
                    (4, [0, 0, 10]),
                    (6, [4 if reaches else 0, 0, 10]),
                )
            ]
        ),
    )
    monkeypatch.setattr(
        gate, "verify", lambda _: dict(cases={"flight": dict(environment={})}, assets="unused")
    )
    scenario = SimpleNamespace(
        world=dict(
            waypoints=[
                dict(start_s=0, position_m=[0, 0, 10]),
                dict(start_s=4, position_m=[4, 0, 10]),
            ]
        ),
        duration_s=8,
        schedule=SimpleNamespace(epoch_ns=0),
    )
    monkeypatch.setattr(gate, "load_scenario", lambda *args: scenario)
    result = gate.audit(root)
    assert result["all_passed"] is reaches
    assert not result["model_predictions_used"]
    assert not result["ground_truth_labels_used"]


@pytest.mark.parametrize("passed", [True, False])
def test_checked_runner_refuses_failed_physical_routes(monkeypatch, passed):
    from experiments import mission_routing_v2_checked as checked

    monkeypatch.setattr(checked, "verify", lambda _: {})
    monkeypatch.setattr(checked, "audit", lambda _: dict(all_passed=passed))
    if passed:
        assert checked.require_route(None)["all_passed"]
    else:
        with pytest.raises(ValueError, match="Physical route gate failed"):
            checked.require_route(None)


def test_terrain_derivation_aligns_physics_without_changing_original(tmp_path):
    from experiments.lake_missions.assets import save, sha
    from experiments.mission_routing_v2_physics import derive, terrain_geometry, validate_alignment

    source = tmp_path / "original"
    folder = source / "harmonic_terrain"
    folder.mkdir(parents=True)
    sdf = folder / "model.sdf"
    sdf.write_text("""<sdf><model><link>
    <collision><pose>0 -300 -5 0 0 0</pose><geometry><heightmap>
    <uri>map.png</uri><size>1200 1200 600</size>
    </heightmap></geometry></collision>
    <visual><geometry><heightmap><uri>map.png</uri>
    <size>1200 1200 201.66</size><pos>0 -300 -5</pos>
    </heightmap></geometry></visual></link></model></sdf>""")
    original_sha = sha(sdf)
    save(
        source / "manifest.json",
        dict(models=[dict(files={"harmonic_terrain/model.sdf": original_sha})]),
    )
    (source / "ATTRIBUTION.txt").write_text("Original attribution\n")
    with pytest.raises(ValueError, match="mismatch"):
        validate_alignment(source)
    derived = tmp_path / "derived"
    derive(source, derived)
    geometry = validate_alignment(derived)
    assert geometry["visual_size"] == geometry["collision_size"] == [1200, 1200, 201.66]
    assert geometry["visual_position"] == geometry["collision_position"] == [0, -300, -5]
    assert sha(sdf) == original_sha
    assert terrain_geometry(sdf)["collision_size"][-1] == 600
    with pytest.raises(FileExistsError):
        derive(source, derived)
