"""Integrated task-budget design, recovery endpoint and paid replay regressions."""

import copy
from dataclasses import replace

import numpy as np
import pytest

from experiments.lake_causal_router import MissionCosts
from experiments.lake_intent_contract import IntentContract
from experiments.lake_tracking import HeightField
from experiments.mission_network_policies import ActionDescriptor, PolicyCosts
from experiments.mission_network_study_v1.executors import prepare_payload
from experiments.mission_network_study_v1.study import authored_traces
from experiments.mission_runtime_profile import read, save
from experiments.mission_task_deadlines import study
from experiments.mission_task_deadlines.contract import DeadlineProfile
from experiments.v4_benchmark.calibrated_router import model_signature
from tests.test_lake_intent_contract import fixture as intent_fixture
from tests.test_mission_transfer_comparison import synthetic_case
from tests.test_v4_benchmark import _FakeBackend


def test_default_design_crosses_task_budget_independently_of_intent():
    spec = study.specification()
    assert spec["expected_runs"] == 192
    assert len({r["run_id"] for r in spec["grid"]}) == 192
    for mission in spec["factors"]["missions"]:
        for profile in ("tight", "relaxed"):
            assert (
                len(
                    [
                        r
                        for r in spec["grid"]
                        if r["mission"] == mission and r["deadline_profile"] == profile
                    ]
                )
                == 48
            )
    assert study.StudyConfig.from_dict(study.StudyConfig().to_dict()) == study.StudyConfig()


@pytest.mark.parametrize(
    "field,value",
    [
        ("flights", ("unseen_test",)),
        ("missions", ("unknown",)),
        ("batteries", (True,)),
        ("links", ("good", "good")),
        ("policies", ()),
        ("deadline_profiles", (DeadlineProfile("../escape", 1, 1, 1),)),
        ("deadline_profiles", (DeadlineProfile("same", 1, 1, 1), DeadlineProfile("same", 2, 2, 2))),
    ],
)
def test_design_rejects_unknown_duplicate_empty_or_unsafe_levels(field, value):
    with pytest.raises(ValueError):
        replace(study.StudyConfig(), **{field: value})


def test_explicit_configuration_has_no_silent_defaults():
    for changed in ({}, {**study.StudyConfig().to_dict(), "task_deadline": 2}):
        with pytest.raises(ValueError):
            study.StudyConfig.from_dict(changed)
    small = replace(
        study.StudyConfig(), flights=("east_shore_day",), policies=("fixed_local_fast",)
    )
    assert study.specification(small)["expected_runs"] == 24


def event_fields(event, *, status="completed", admitted=True, deadline_missed=False):
    event.update(
        selected_action_id="model",
        emitted_reports=[],
        pipeline=None,
        perception=dict(
            task="perception",
            status=status,
            admitted=admitted,
            task_deadline_met=admitted,
            deadline_missed=deadline_missed,
            on_time=True,
            lateness_ns=0,
        ),
    )
    return event


def test_cancellation_at_deadline_is_a_miss_despite_zero_timestamp_lateness():
    event = event_fields({}, status="task_deadline_miss", admitted=False, deadline_missed=True)
    metrics = study.task_metrics([event])["perception"]
    assert metrics == dict(
        attempts=1, on_time=0, deadline_misses=1, discarded_outputs=1, lateness_ns=0
    )


def test_transport_failure_before_task_deadline_is_not_successful_task_completion():
    event = event_fields({}, status="transport_timeout", admitted=False)
    metrics = study.task_metrics([event])["perception"]
    assert metrics["on_time"] == metrics["deadline_misses"] == 0
    assert metrics["discarded_outputs"] == metrics["attempts"] == 1


def test_recovered_miss_does_not_erase_valid_intent_evidence_or_fail_whole_mission():
    document, events, result = intent_fixture()
    for event in events:
        event_fields(event)
    missed = event_fields(
        dict(
            capture_time_s=4,
            completion_time_s=5,
            execution=dict(success=False),
            score=dict(matched_target_ids=[], false_positive_components=0),
        ),
        status="task_deadline_miss",
        admitted=False,
        deadline_missed=True,
    )
    events.append(missed)
    result.update(processed_observations=3, inference_failures=1)
    outcome = study.evaluate_recoverable(IntentContract(document), events, result)
    assert outcome["mission_success"]
    assert not outcome["strict_original_endpoint_success"]
    assert not outcome["all_dispatched_task_deadlines_met"]
    changed = copy.deepcopy(result)
    changed["final_battery_frac"] = 0.19
    assert not study.evaluate_recoverable(IntentContract(document), events, changed)[
        "mission_success"
    ]
    changed = copy.deepcopy(events)
    changed[-1]["perception"]["status"] = "execution_failed"
    assert not study.evaluate_recoverable(IntentContract(document), changed, result)[
        "mission_success"
    ]


@pytest.mark.parametrize(
    "profile_name,deadline",
    [("ordinary", 2_000_000_000), ("impossible", 1), ("depleted", 2_000_000_000)],
)
def test_integrated_replay_pays_misses_and_suppresses_all_downstream_work(
    tmp_path, monkeypatch, profile_name, deadline
):
    from experiments.mission_network_emulation import LinkSegment, LinkTrace

    scenario, contract, profiles = synthetic_case(tmp_path)
    if profile_name == "depleted":
        scenario = replace(scenario, battery_capacity_wh=0.000001)
    root, parent, cohort = tmp_path / "study", tmp_path / "parent", tmp_path / "cohort"
    profile = DeadlineProfile(profile_name, deadline, 150_000_000, 30_000_000)
    plan = dict(parent_root=str(parent), factors=dict(deadline_profiles=[profile.to_dict()]))
    save(root / "plan.json", plan)
    save(parent / "plan.json", dict(cohort=str(cohort)))
    save(parent / "missions/cell/rapid_survey/intent.json", contract.document())
    manifest = dict(
        recording_identity=scenario.recording_identity,
        models={s.config_id: {} for s in scenario.executors},
    )
    save(cohort / "flight/predictions/manifest.json", manifest)
    descriptors = []
    for spec in scenario.executors:
        for placement in ("local", "remote"):
            descriptors.append(
                ActionDescriptor(
                    ("REMOTE__" if placement == "remote" else "") + spec.config_id,
                    spec.config_id,
                    placement,
                    model_signature(spec),
                    spec.mission_latency_s,
                    spec.energy_j_per_call if placement == "local" else 0,
                    memory_mb=64,
                )
            )
    bank, payloads = {}, {}
    fake = _FakeBackend(scenario)
    for index in range(5):
        fake.advance_to(scenario.schedule.at(index))
        bank[index] = fake.capture()
        payloads[index] = prepare_payload(bank[index].rgb)
    mask_reads = []

    class Masks:
        def __init__(self, *_):
            pass

        def __getitem__(self, key):
            mask_reads.append(key)
            return bank[0].object_ids > 0

    monkeypatch.setattr(study, "scenario_for", lambda *_: scenario)
    monkeypatch.setattr(study, "actions_for", lambda *_: tuple(descriptors))
    monkeypatch.setattr(study, "load_profiles", lambda *_: profiles)
    monkeypatch.setattr(
        study, "policy_costs", lambda *_: PolicyCosts(MissionCosts.from_scenario(scenario))
    )
    monkeypatch.setattr(
        study,
        "trace_for",
        lambda *_: LinkTrace(tuple(LinkSegment(**s) for s in authored_traces()["good"])),
    )
    monkeypatch.setattr(study, "VerifiedMasks", Masks)
    terrain = HeightField(np.zeros((4, 4)), west=-10, east=10, south=-10, north=10)
    item = dict(
        run_id="run_0000",
        cell="cell",
        flight="flight",
        mission="rapid_survey",
        initial_battery_frac=0.9,
        link="good",
        deadline_profile=profile_name,
        policy="fixed_local_fast",
    )
    row = study.execute_case(
        root,
        plan,
        item,
        terrain,
        (bank, payloads, {"recording_identity": scenario.recording_identity}),
        manifest,
    )
    events = read(root / row["path"] / "events.json")
    result = read(root / row["path"] / "results.json")
    assert len(events) == (1 if profile_name == "depleted" else 5)
    assert result["cumulative_energy_j"] > result["flight_energy_j"]
    assert all(e["perception"]["costs"]["charged_energy_j"] > 0 for e in events)
    if profile_name == "impossible":
        assert not mask_reads
        assert all(e["pipeline"] is None and e["emitted_reports"] == [] for e in events)
        assert row["task_metrics"]["perception"]["deadline_misses"] == 5
        assert row["task_metrics"]["association"]["attempts"] == 0
    elif profile_name == "depleted":
        assert len(mask_reads) == 1
        assert events[0]["pipeline"] is None
        assert events[0]["downstream_skip_reason"] == "battery_depleted_before_association_release"
        assert result["termination_reason"] == "battery_depleted"
        assert result["final_time_s"] < 2
    else:
        assert len(mask_reads) == 5
        assert all(e["pipeline"] is not None for e in events)
        assert row["task_metrics"]["association"]["attempts"] == 5
    with pytest.raises(FileExistsError):
        study.execute_case(
            root,
            plan,
            item,
            terrain,
            (bank, payloads, {"recording_identity": scenario.recording_identity}),
            manifest,
        )
