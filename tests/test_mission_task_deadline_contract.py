"""Independent boundary cases for per-task release budgets and usable output."""

from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType

import pytest

from experiments.mission_task_deadlines.contract import (
    CompletionAssessment,
    DeadlineProfile,
    TaskDeadline,
    TaskKind,
    assess_completion,
)


def profile():
    return DeadlineProfile("responsive", 2_000_000_000, 50_000_000, 20_000_000)


def test_profile_has_independent_explicit_budgets_and_round_trips():
    value = profile()
    assert [value.budget_for(t) for t in TaskKind] == [2_000_000_000, 50_000_000, 20_000_000]
    assert DeadlineProfile.from_dict(value.to_dict()) == value
    assert DeadlineProfile.from_dict(MappingProxyType(value.to_dict())) == value
    exported = value.to_dict()
    exported["perception_ns"] = 1
    assert value.perception_ns == 2_000_000_000
    with pytest.raises(FrozenInstanceError):
        value.perception_ns = 1


@pytest.mark.parametrize("field", ["perception_ns", "association_ns", "verification_ns"])
@pytest.mark.parametrize("bad", [0, -1, True, False, 1.0, float("inf"), float("nan"), "1", None])
def test_budgets_are_positive_integer_nanoseconds(field, bad):
    with pytest.raises(ValueError):
        replace(profile(), **{field: bad})


@pytest.mark.parametrize("bad", [None, "", " \t", 1, True])
def test_profile_name_is_explicit(bad):
    with pytest.raises(ValueError):
        replace(profile(), name=bad)


@pytest.mark.parametrize("mutation", ["extra", "missing", "legacy_unit", "not_mapping"])
def test_profile_rejects_silent_defaulting_or_unknown_fields(mutation):
    source = profile().to_dict()
    if mutation == "extra":
        source["whole_mission_ns"] = 130_000_000_000
    elif mutation == "missing":
        del source["association_ns"]
    elif mutation == "legacy_unit":
        source["perception_s"] = source.pop("perception_ns") / 1e9
    else:
        source = list(source.items())
    with pytest.raises(ValueError):
        DeadlineProfile.from_dict(source)


@pytest.mark.parametrize("task", ["perception", "segmentation", None, 0])
def test_in_memory_task_selection_requires_the_typed_enum(task):
    with pytest.raises(ValueError):
        profile().budget_for(task)
    with pytest.raises(ValueError):
        TaskDeadline(task, 0, 1)


def test_request_round_trip_and_absolute_deadline_from_nonzero_release():
    request = TaskDeadline(TaskKind.ASSOCIATION, 13, 27)
    assert request.deadline_ns == 40
    assert TaskDeadline.from_dict(request.to_dict()) == request
    with pytest.raises(FrozenInstanceError):
        request.release_ns = 0
    for source in (
        {**request.to_dict(), "deadline_ns": 40},
        {"task": "association", "budget_ns": 27},
        {**request.to_dict(), "task": "unknown"},
        {**request.to_dict(), "task": 1},
    ):
        with pytest.raises(ValueError):
            TaskDeadline.from_dict(source)


@pytest.mark.parametrize("release", [-1, True, False, 1.0, "1", None])
def test_request_release_uses_nonnegative_integer_nanoseconds(release):
    with pytest.raises(ValueError):
        TaskDeadline(TaskKind.PERCEPTION, release, 1)


@pytest.mark.parametrize("budget", [0, -1, True, False, 1.0, "1", None])
def test_request_budget_uses_positive_integer_nanoseconds(budget):
    with pytest.raises(ValueError):
        TaskDeadline(TaskKind.PERCEPTION, 0, budget)


def test_exact_boundary_is_accepted_but_one_nanosecond_late_is_not():
    request = TaskDeadline(TaskKind.PERCEPTION, 7, 23)
    exact = assess_completion(request, 30, step_ns=1)
    late = assess_completion(request, 31, step_ns=1)
    assert exact.on_time and exact.lateness_ns == 0 and exact.status == "completed"
    assert not late.on_time and late.lateness_ns == 1 and late.status == "task_deadline_miss"


def test_rounding_uses_absolute_mission_ticks_not_time_since_release():
    request = TaskDeadline(TaskKind.ASSOCIATION, 13, 27)
    result = assess_completion(request, 35, step_ns=10)
    assert result.usable_finish_ns == 40
    assert result.on_time
    assert result.to_dict() == dict(
        task="association",
        release_ns=13,
        budget_ns=27,
        deadline_ns=40,
        raw_finish_ns=35,
        usable_finish_ns=40,
        step_ns=10,
        on_time=True,
        lateness_ns=0,
        status="completed",
    )


def test_raw_completion_within_budget_may_be_late_at_usable_tick():
    request = TaskDeadline(TaskKind.VERIFICATION, 13, 26)
    result = assess_completion(request, 39, step_ns=10)
    assert result.raw_finish_ns == request.deadline_ns
    assert result.usable_finish_ns == 40
    assert not result.on_time and result.lateness_ns == 1


def test_default_tick_and_nonpreemptive_late_time_are_preserved():
    request = TaskDeadline(TaskKind.PERCEPTION, 1_000_000_000, 2_000_000_000)
    result = assess_completion(request, 5_123_000_001)
    assert result.step_ns == 10_000_000
    assert result.raw_finish_ns == 5_123_000_001
    assert result.usable_finish_ns == 5_130_000_000
    assert result.lateness_ns == 2_130_000_000
    assert result.status == "task_deadline_miss"
    with pytest.raises(FrozenInstanceError):
        result.raw_finish_ns = 3_000_000_000


def test_zero_duration_completion_and_large_integer_time_are_exact():
    assert assess_completion(TaskDeadline(TaskKind.PERCEPTION, 0, 1), 0).on_time
    origin = 10**22
    request = TaskDeadline(TaskKind.VERIFICATION, origin + 1, 9)
    result = assess_completion(request, origin + 1, step_ns=10)
    assert result.usable_finish_ns == origin + 10
    assert result.on_time


@pytest.mark.parametrize("finish", [-1, True, False, 1.0, "1", None])
def test_finish_requires_nonnegative_integer_nanoseconds(finish):
    with pytest.raises(ValueError):
        assess_completion(TaskDeadline(TaskKind.PERCEPTION, 0, 1), finish)


@pytest.mark.parametrize("step", [0, -1, True, False, 1.0, "1", None])
def test_tick_requires_positive_integer_nanoseconds(step):
    with pytest.raises(ValueError):
        assess_completion(TaskDeadline(TaskKind.PERCEPTION, 0, 1), 0, step)


def test_assessment_rejects_reversed_time_or_untyped_request():
    with pytest.raises(ValueError, match="precede"):
        assess_completion(TaskDeadline(TaskKind.PERCEPTION, 10, 1), 9)
    with pytest.raises(ValueError, match="TaskDeadline"):
        assess_completion({"release_ns": 0, "budget_ns": 1}, 0)
    with pytest.raises(TypeError):
        CompletionAssessment(TaskDeadline(TaskKind.PERCEPTION, 0, 1), 0, on_time=True)
