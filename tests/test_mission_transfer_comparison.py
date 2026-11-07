"""Paid parity and adversarial integrity checks without assets, weights or Gazebo."""

import copy
import json
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from experiments.lake_feasibility_inference import read
from experiments.lake_intent_contract import IntentContract, evaluate
from experiments.lake_intent_runner import bind_scenario
from experiments.lake_missions.assets import save
from experiments.lake_tracking import TrackingConfig
from experiments.literature_routing.policy import NAME
from experiments.mission_coverage_repair.study import POLICIES as ORIGINAL_POLICIES
from experiments.mission_routing_v2.features import NAMES, encode, fit
from experiments.mission_routing_v2.models import POOL
from experiments.mission_routing_v2.runtime import run_tracking_mission
from experiments.mission_transfer_comparison.integrity import check_execution
from experiments.v4_benchmark.calibrated_router import model_signature
from experiments.v4_benchmark.routers import build_router
from experiments.v4_benchmark.scenario import load_scenario
from tests.test_lake_intent_contract import fixture as intent_fixture
from tests.test_mission_routing_v4 import profile_for
from tests.test_mission_routing_v5 import recall_profile
from tests.test_mission_temporal_routing import temporal_profile
from tests.test_v4_benchmark import ROOT, _FakeBackend, _PerfectExecutor


def synthetic_case(tmp_path):
    """Nine named choices, synthetic predictions, and source-free training heads."""
    original = ROOT / "experiments/v4_benchmark/example_scenario.json"
    doc = json.loads(original.read_text())
    doc["contract_path"] = str(original.with_name("example_contract.json"))
    exemplar = doc["executor_configs"][0]
    doc["executor_configs"] = [
        dict(
            exemplar,
            config_id=c,
            model_strategy_id=c.lower(),
            mission_latency_s=0.2 + i * 0.05,
            energy_j_per_call=1,
        )
        for i, c in enumerate(POOL)
    ]
    exemplar = doc["public_profiles"][0]
    doc["public_profiles"] = [dict(exemplar, config_id=c) for c in POOL]
    doc["mission"].update(
        duration_s=10,
        observation_interval_s=2,
        initial_battery_frac=0.9,
        fallback_config_id=POOL[0],
    )
    save(tmp_path / "scenario.json", doc)
    scenario = load_scenario(tmp_path / "scenario.json")
    document, _, _ = intent_fixture()
    document["evaluator_target_ids"] = list(scenario.targets.values())
    document["mission"]["phases"][0].update(
        evaluator_target_ids=list(scenario.targets.values()), end_s=10, deadline_s=10
    )
    contract = IntentContract(document)
    scenario = replace(
        scenario,
        contract=replace(scenario.contract, **document["limits"], quality_threshold=0),
    )
    scenario = bind_scenario(scenario, contract)
    rgb = np.zeros((8, 8, 3), np.uint8)
    features = encode(rgb)
    old = dict(
        feature_names=list(NAMES),
        models={
            s.config_id: dict(signature=model_signature(s), quality=fit([features] * 4, [0.5] * 4))
            for s in scenario.executors
        },
    )
    profiles = dict(
        recall=recall_profile(old, "base_recall"),
        risk=profile_for(old, features),
        temporal=temporal_profile(old, rgb),
        coverage=dict(
            schema_version="anonymous_coverage_v1",
            config_ids=list(POOL),
            rates=[[0.5] * len(POOL)],
            weights=[1],
            signatures={c: m["signature"] for c, m in old["models"].items()},
        ),
    )
    return scenario, contract, profiles


def execute(tmp_path, scenario, contract, policy, name):
    backend = _FakeBackend(scenario)
    backend.provenance["recording_identity"] = scenario.recording_identity
    outcome = run_tracking_mission(
        scenario,
        backend,
        policy,
        {
            s.config_id: _PerfectExecutor(s.config_id, s.mission_latency_s)
            for s in scenario.executors
        },
        contract,
        tmp_path,
        config=TrackingConfig(association="flat_world", remember_reports=True),
        terrain=None,
        map_metadata=None,
        router_name=name,
        gt_aware=False,
    )
    events, result = (
        read(tmp_path / "execution/events.json"),
        read(tmp_path / "execution/results.json"),
    )
    canonical = evaluate(contract, events, result)
    row = dict(
        full_success=outcome["mission_success"],
        phases_succeeded=sum(p["success"] for p in outcome["phases"]),
        report_recall=outcome["reporting"]["recall_by_deadline"],
        false_reports=outcome["reporting"]["false_reports"],
        duplicate_reports=outcome["reporting"]["duplicate_reports"],
        final_battery=result["final_battery_frac"],
        calls=len(events),
        selections=dict(Counter(e["execution"]["config_id"] for e in events)),
        failed_constraints=outcome["failed_constraints"],
    )
    return events, result, row, canonical


@pytest.fixture
def paid_case(tmp_path):
    scenario, contract, _ = synthetic_case(tmp_path)
    name = "static:" + POOL[0]
    policy = build_router(name, scenario.profiles, scenario.observation_interval_s)
    return scenario, *execute(tmp_path / "run", scenario, contract, policy, name)


def test_independent_integrity_accepts_actual_paid_runtime(paid_case):
    checked = check_execution(*paid_case)
    assert checked["events"] == 5 and checked["complete_capture_window"]
    assert checked["final_time_s"] == 10


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_time_s", 9),
        ("final_battery_frac", 0.99),
        ("processed_observations", 4),
        ("config_switches", 3),
        ("config_selection_history", []),
        ("skipped_observation_ids", [1]),
        ("flight_energy_j", 0),
        ("inference_energy_j", 0),
        ("tracking_energy_j", 0),
        ("termination_reason", "battery_depleted"),
        ("mission_completed", False),
        ("decision_stride", 2),
    ],
)
def test_independent_integrity_rejects_corrupted_terminal_fields(paid_case, field, value):
    scenario, events, result, row, canonical = paid_case
    result[field] = value
    with pytest.raises(ValueError):
        check_execution(scenario, events, result, row, canonical)


@pytest.mark.parametrize(
    "section,field,value",
    [
        (None, "capture_time_s", 0.2),
        (None, "capture_timestamp_ns", 0),
        (None, "observation_id", 1),
        (None, "battery_after", 1),
        (None, "model_load_energy_j", 99),
        (None, "configured_tracking_latency_s", 0),
        ("runtime_at_capture", "frame_id", 1),
        ("runtime_at_capture", "current_time_s", 0.2),
        ("runtime_at_capture", "battery_frac", 1),
        ("execution", "energy_j", 0),
        ("execution", "mission_latency_s", 0),
        ("execution", "communication_mb", 1),
    ],
)
def test_independent_integrity_rejects_corrupted_event_fields(paid_case, section, field, value):
    scenario, events, result, row, canonical = paid_case
    target = events[0] if section is None else events[0][section]
    target[field] = value
    with pytest.raises(ValueError):
        check_execution(scenario, events, result, row, canonical)


@pytest.mark.parametrize("field", ["false_reports", "duplicate_reports", "calls", "final_battery"])
def test_independent_integrity_rejects_corrupted_report_table(paid_case, field):
    scenario, events, result, row, canonical = paid_case
    row[field] += 1
    with pytest.raises(ValueError, match="Summary"):
        check_execution(scenario, events, result, row, canonical)


def test_unexplained_prefix_and_empty_execution_are_rejected(paid_case):
    scenario, events, result, row, canonical = paid_case
    with pytest.raises(ValueError, match="truncated"):
        check_execution(scenario, events[:-1], result, row, canonical)
    with pytest.raises(ValueError, match="Empty execution"):
        check_execution(scenario, [], result, row, canonical)


def test_legitimate_battery_failure_remains_a_valid_failed_execution(tmp_path):
    scenario, contract, _ = synthetic_case(tmp_path)
    # Enough charge for one call, depleted during the next flight advance.
    document = contract.document()
    document["mission"]["initial_battery_frac"] = 0.001
    document["mission"]["min_final_battery_frac"] = 0.0001
    document["limits"]["min_final_battery_frac"] = 0.0001
    contract = IntentContract(document)
    scenario = bind_scenario(
        replace(
            scenario,
            initial_battery_frac=0.001,
            contract=replace(scenario.contract, min_final_battery_frac=0.0001),
        ),
        contract,
    )
    name = "static:" + POOL[0]
    policy = build_router(name, scenario.profiles, scenario.observation_interval_s)
    data = execute(tmp_path / "run", scenario, contract, policy, name)
    checked = check_execution(scenario, *data)
    assert checked["terminal_reason"] == "battery_depleted"
    assert not data[3]["mission_success"]


def test_policy_registry_preserves_all_previous_methods_and_adds_one_named_adapter():
    from experiments.mission_transfer_comparison.study import POLICIES

    assert len(POLICIES) == len(set(POLICIES)) == 22
    assert set(POLICIES) == {*ORIGINAL_POLICIES, NAME}


@pytest.mark.parametrize("name", ORIGINAL_POLICIES)
def test_original_policy_factory_paid_replay_parity(tmp_path, name):
    from experiments.mission_coverage_repair.study import make_policy as old_factory
    from experiments.mission_transfer_comparison.study import make_policy as new_factory

    scenario, contract, profiles = synthetic_case(tmp_path)
    before = copy.deepcopy(profiles)
    first = execute(
        tmp_path / "old", scenario, contract, old_factory(name, profiles, scenario), name
    )
    second = execute(
        tmp_path / "new", scenario, contract, new_factory(name, profiles, scenario), name
    )
    check_execution(scenario, *first)
    check_execution(scenario, *second)
    assert profiles == before
    assert first[2:] == second[2:]
    for a, b in zip(first[0], second[0], strict=True):
        for key in ("router_decision", "capture_time_s", "completion_time_s", "battery_after"):
            assert a[key] == b[key]
        assert a["execution"]["config_id"] == b["execution"]["config_id"]


def test_transfer_factory_adapter_runs_through_identical_paid_runtime(tmp_path):
    from experiments.mission_transfer_comparison.study import make_policy

    scenario, contract, profiles = synthetic_case(tmp_path)
    policy = make_policy(NAME, profiles, scenario)
    data = execute(tmp_path / "adapter", scenario, contract, policy, NAME)
    assert check_execution(scenario, *data)["complete_capture_window"]
    assert all(e["router_decision"]["mode"] == NAME for e in data[0])


def test_full_grid_rejects_omitted_failed_or_duplicated_conditions():
    from experiments.mission_transfer_comparison.audit import complete_grid
    from experiments.mission_transfer_comparison.study import POLICIES

    p = dict(cases={"a": {}, "b": {}}, missions=["rapid", "strict"], policies=list(POLICIES))
    rows = [
        dict(flight=f, mission=m, policy=c, full_success=False)
        for f in p["cases"]
        for m in p["missions"]
        for c in p["policies"]
    ]
    complete_grid(rows, p)
    for altered in (rows[1:], [*rows, rows[0]], [{**r, "flight": "unknown"} for r in rows]):
        with pytest.raises(ValueError):
            complete_grid(altered, p)
