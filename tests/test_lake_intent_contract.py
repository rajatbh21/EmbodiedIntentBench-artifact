"""Protect intent semantics and public boundaries without external assets or models."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from experiments.lake_intent_contract import (
    LIMIT_FIELDS,
    RESOURCE_FLAGS,
    VERSION,
    IntentContract,
    evaluate,
    immutable,
)


def fixture(mode="report_required"):
    reporting = dict(mode=mode)
    if mode == "report_required":
        reporting.update(
            target_scope="all_scenario_targets",
            deadline_s=7,
            min_report_recall=1,
            max_false_reports=0,
            max_duplicate_reports=0,
        )
    limits = dict(
        deadline_s=10,
        communication_budget_mb=0,
        min_final_battery_frac=0.2,
        privacy_level="local_only",
    )
    document = dict(
        schema_version=VERSION,
        duration_s=10,
        limits=limits,
        reporting=reporting,
        evaluator_target_ids=["hidden_a", "hidden_b"],
        mission=dict(
            mission_id="example",
            title="Example",
            user_request="Repeat and report",
            initial_battery_frac=0.9,
            min_final_battery_frac=0.2,
            phases=[
                dict(
                    phase_id="search",
                    instruction="Search",
                    start_s=0,
                    end_s=6,
                    deadline_s=6,
                    required_recall=1,
                    min_hits=2,
                    min_evidence_span_s=2,
                    max_false_components=0,
                    weight=1,
                    evaluator_target_ids=["hidden_a", "hidden_b"],
                )
            ],
        ),
    )
    events = [
        dict(
            capture_time_s=t,
            completion_time_s=t + 0.1,
            execution=dict(success=True),
            score=dict(matched_target_ids=["hidden_a", "hidden_b"], false_positive_components=0),
        )
        for t in (0, 2)
    ]
    result = dict(
        contract={
            **limits,
            "quality_metric": "target_recall",
            "quality_operator": ">=",
            "quality_threshold": 0,
        },
        constraints=dict.fromkeys(RESOURCE_FLAGS, True),
        status="completed",
        error=None,
        mission_completed=True,
        termination_reason="mission_window_complete",
        final_time_s=10,
        final_battery_frac=0.8,
        cumulative_communication_mb=0,
        router_errors=0,
        inference_failures=0,
        invalid_actions=0,
        privacy_violations=0,
        processed_observations=2,
        report_quality=dict(
            reports=[dict(target_id=t, completion_time_s=4) for t in ("hidden_a", "hidden_b")]
        ),
    )
    return document, events, result


def test_explicit_mode_changes_only_declared_success_scope():
    document, events, result = fixture()
    result["report_quality"]["reports"] = []
    result["mission_success"] = True  # Untrusted legacy success cannot override the contract.
    required = evaluate(IntentContract(document), events, result)
    assert not required["mission_success"]
    assert required["failed_constraints"] == ["report_recall_by_deadline"]
    document["reporting"] = {"mode": "evidence_only"}
    result["mission_success"] = False
    assert evaluate(IntentContract(document), events, result)["mission_success"]


def test_all_required_unique_reports_must_arrive_by_deadline():
    document, events, result = fixture()
    result["report_quality"]["reports"][1]["completion_time_s"] = 8
    outcome = evaluate(IntentContract(document), events, result)
    assert not outcome["mission_success"]
    assert outcome["reporting"]["recall_by_deadline"] == 0.5
    assert outcome["reporting"]["late_reports"] == 1
    result["report_quality"]["reports"][1]["completion_time_s"] = 7
    assert evaluate(IntentContract(document), events, result)["mission_success"]


@pytest.mark.parametrize(
    "target,key", [(None, "false_report_limit"), ("hidden_a", "duplicate_report_limit")]
)
def test_late_bad_reports_cannot_evade_limits(target, key):
    document, events, result = fixture()
    result["report_quality"]["reports"].append(dict(target_id=target, completion_time_s=9))
    assert evaluate(IntentContract(document), events, result)["failed_constraints"] == [key]


@pytest.mark.parametrize(
    "failure", ["partial", "errors", "failed_call", "short", "missing_events", "comm", "privacy"]
)
def test_execution_and_all_resource_constraints_are_required(failure):
    document, events, result = fixture("evidence_only")
    if failure == "partial":
        result["mission_completed"] = False
        result["termination_reason"] = "memory_capacity_exceeded"
    elif failure == "errors":
        result["router_errors"] = 1
    elif failure == "failed_call":
        events[0]["execution"]["success"] = False
    elif failure == "short":
        result["final_time_s"] = 5
    elif failure == "missing_events":
        result["processed_observations"] = 3
    elif failure == "comm":
        result["cumulative_communication_mb"] = 1
    else:
        result["privacy_violations"] = 1
    assert not evaluate(IntentContract(document), events, result)["mission_success"]


@pytest.mark.parametrize("key", RESOURCE_FLAGS)
def test_resource_failure_is_never_dropped(key):
    document, events, result = fixture()
    result["constraints"][key] = False
    assert key in evaluate(IntentContract(document), events, result)["failed_constraints"]


def test_late_evidence_and_revisit_freshness_preserve_frozen_semantics():
    document, events, result = fixture("evidence_only")
    events[-1]["completion_time_s"] = 6.1
    assert not evaluate(IntentContract(document), events, result)["mission_success"]
    events[-1]["completion_time_s"] = 2.1
    document["mission"]["phases"].append(
        {
            **document["mission"]["phases"][0],
            "phase_id": "revisit",
            "start_s": 6,
            "end_s": 10,
            "deadline_s": 10,
        }
    )
    assert evaluate(IntentContract(document), events, result)["failed_constraints"] == [
        "phase:revisit"
    ]


@pytest.mark.parametrize(
    "failure", ["unknown", "mode", "contradiction", "nan", "bool", "battery", "targets", "version"]
)
def test_invalid_contracts_fail_closed(failure):
    document, _, _ = fixture()
    if failure == "unknown":
        document["mission"]["hidden_hint"] = "forbidden"
    elif failure == "mode":
        document["reporting"]["mode"] = "automatic"
    elif failure == "contradiction":
        document["reporting"]["mode"] = "evidence_only"
    elif failure == "nan":
        document["limits"]["deadline_s"] = float("nan")
    elif failure == "bool":
        document["reporting"]["max_false_reports"] = True
    elif failure == "battery":
        document["limits"]["min_final_battery_frac"] = 0.3
    elif failure == "targets":
        document["evaluator_target_ids"] = ["hidden_a"]
    else:
        document["schema_version"] = "future"
    with pytest.raises(ValueError):
        IntentContract(document)


def test_public_view_and_hash_do_not_depend_on_hidden_ids():
    document, _, _ = fixture()
    contract = IntentContract(document)
    for owner in (document, document["mission"]["phases"][0]):
        owner["evaluator_target_ids"] = ["secret_x", "secret_y"]
    other = IntentContract(document)
    assert contract.sha256 != other.sha256
    assert contract.public_sha256 == other.public_sha256
    public = immutable(contract.public_document())
    assert "hidden_a" not in repr(public) and "evaluator_target_ids" not in repr(public)
    with pytest.raises(TypeError):
        public["mission"]["phases"][0]["min_hits"] = 1
    document["mission"]["phases"][0]["min_hits"] = 4
    assert contract.document()["mission"]["phases"][0]["min_hits"] == 2


def test_runtime_budget_or_hidden_aggregate_quality_cannot_drift():
    document, events, result = fixture()
    for key, value in (("deadline_s", 20), ("quality_threshold", 0.9)):
        modified = copy.deepcopy(result)
        modified["contract"][key] = value
        with pytest.raises(ValueError):
            evaluate(IntentContract(document), events, modified)


@pytest.mark.parametrize("mode", ["evidence_only", "report_required"])
def test_real_runner_receives_public_intent_and_writes_one_primary_outcome(tmp_path, mode):
    from experiments.lake_intent_runner import run_intent_mission
    from experiments.v4_benchmark.scenario import load_scenario
    from tests.test_v4_benchmark import ROOT, _FakeBackend, _PerfectExecutor

    scenario = load_scenario(ROOT / "experiments/v4_benchmark/example_scenario.json")
    scenario = replace(
        scenario,
        duration_s=10,
        contract=replace(
            scenario.contract,
            quality_metric="target_recall",
            quality_threshold=0,
            deadline_s=10,
        ),
    )
    document, _, _ = fixture(mode)
    document["limits"] = {key: asdict(scenario.contract)[key] for key in LIMIT_FIELDS}
    document["mission"]["initial_battery_frac"] = scenario.initial_battery_frac
    document["mission"]["min_final_battery_frac"] = scenario.contract.min_final_battery_frac
    document["evaluator_target_ids"] = list(scenario.targets.values())
    document["mission"]["phases"][0]["evaluator_target_ids"] = list(scenario.targets.values())
    contract = IntentContract(document)
    seen = []

    def select(observation):
        seen.append(observation)
        assert observation.intent_contract["reporting"]["mode"] == mode
        assert observation.public_contract_sha256 == contract.public_sha256
        assert "evaluator_target_ids" not in repr(observation.intent_contract)
        assert (observation.requirements.min_report_recall is not None) == (
            mode == "report_required"
        )
        return scenario.catalog.ids()[0]

    backend = _FakeBackend(scenario)
    backend.provenance = {**backend.provenance, "recording_identity": scenario.recording_identity}
    outcome = run_intent_mission(
        scenario,
        backend,
        SimpleNamespace(select=select),
        {c: _PerfectExecutor(c, 0.1) for c in scenario.catalog.ids()},
        contract,
        tmp_path / mode,
        router_name="test",
        gt_aware=False,
    )
    assert seen and outcome["public_contract_deliveries"] == len(seen)
    assert outcome["constraints"]["execution"]
    assert outcome["mission_success"] == all(outcome["constraints"].values())
    assert (tmp_path / mode / "outcome.json").exists()
    assert (tmp_path / mode / "execution/results.json").exists()
