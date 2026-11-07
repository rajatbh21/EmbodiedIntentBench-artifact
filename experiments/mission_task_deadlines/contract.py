"""Public per-invocation deadlines, independent of mission and transport limits.

All timestamps use integer nanoseconds relative to the mission origin. A task's
release is when its input becomes available, before the routing, loading and
execution overhead assigned to that task. A downstream task is released only
when its predecessor's admitted output becomes usable. Deadlines are inclusive.

Availability is rounded upward to the next absolute physics tick. The release
itself need not lie on that grid; rounding never restarts a task's budget.

Assessment implements non-preemptive late discard: it classifies a completed
attempt, without truncating its busy interval or refunding computation, loading,
routing, communication or flight costs. A caller must suppress late outputs and
their temporal-state changes. Transport cancellation, if supported separately,
must retain its actual paid time and bytes; this contract does not simulate
preemption. Mission deadlines, task deadlines and transport timeouts remain
separate recorded reasons and must not be silently substituted for one another.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


def _integer(value: object, name: str, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer number of nanoseconds")


def _exact_fields(value: object, names: set[str], kind: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != names:
        raise ValueError(f"{kind} requires exactly these fields: {', '.join(sorted(names))}")
    return value


class TaskKind(StrEnum):
    PERCEPTION = "perception"
    ASSOCIATION = "association"
    VERIFICATION = "verification"


@dataclass(frozen=True)
class DeadlineProfile:
    """An explicitly named budget for every supported task, with no defaults."""

    name: str
    perception_ns: int
    association_ns: int
    verification_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Deadline profile name must be a nonempty string")
        for name in ("perception_ns", "association_ns", "verification_ns"):
            _integer(getattr(self, name), name, positive=True)

    def budget_for(self, task: TaskKind) -> int:
        if not isinstance(task, TaskKind):
            raise ValueError("A typed TaskKind is required")
        return {
            TaskKind.PERCEPTION: self.perception_ns,
            TaskKind.ASSOCIATION: self.association_ns,
            TaskKind.VERIFICATION: self.verification_ns,
        }[task]

    def to_dict(self) -> dict:
        return dict(
            name=self.name,
            perception_ns=self.perception_ns,
            association_ns=self.association_ns,
            verification_ns=self.verification_ns,
        )

    @classmethod
    def from_dict(cls, value: Mapping) -> DeadlineProfile:
        fields = _exact_fields(
            value,
            {"name", "perception_ns", "association_ns", "verification_ns"},
            "DeadlineProfile",
        )
        return cls(**fields)


@dataclass(frozen=True)
class TaskDeadline:
    """A public budget anchored to one task invocation's input release."""

    task: TaskKind
    release_ns: int
    budget_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.task, TaskKind):
            raise ValueError("A typed TaskKind is required")
        _integer(self.release_ns, "release_ns")
        _integer(self.budget_ns, "budget_ns", positive=True)

    @property
    def deadline_ns(self) -> int:
        return self.release_ns + self.budget_ns

    def to_dict(self) -> dict:
        return dict(task=self.task.value, release_ns=self.release_ns, budget_ns=self.budget_ns)

    @classmethod
    def from_dict(cls, value: Mapping) -> TaskDeadline:
        fields = _exact_fields(value, {"task", "release_ns", "budget_ns"}, "TaskDeadline")
        if not isinstance(fields["task"], str):
            raise ValueError("Serialized task must name a supported TaskKind")
        return cls(TaskKind(fields["task"]), fields["release_ns"], fields["budget_ns"])


@dataclass(frozen=True)
class CompletionAssessment:
    """Immutable derived decision; no caller-supplied status or rounded finish."""

    request: TaskDeadline
    raw_finish_ns: int
    step_ns: int = 10_000_000

    def __post_init__(self) -> None:
        if not isinstance(self.request, TaskDeadline):
            raise ValueError("A typed TaskDeadline is required")
        _integer(self.raw_finish_ns, "raw_finish_ns")
        _integer(self.step_ns, "step_ns", positive=True)
        if self.raw_finish_ns < self.request.release_ns:
            raise ValueError("Task completion cannot precede its input release")

    @property
    def usable_finish_ns(self) -> int:
        return ((self.raw_finish_ns + self.step_ns - 1) // self.step_ns) * self.step_ns

    @property
    def on_time(self) -> bool:
        return self.usable_finish_ns <= self.request.deadline_ns

    @property
    def lateness_ns(self) -> int:
        return max(0, self.usable_finish_ns - self.request.deadline_ns)

    @property
    def status(self) -> str:
        return "completed" if self.on_time else "task_deadline_miss"

    def to_dict(self) -> dict:
        return dict(
            **self.request.to_dict(),
            deadline_ns=self.request.deadline_ns,
            raw_finish_ns=self.raw_finish_ns,
            usable_finish_ns=self.usable_finish_ns,
            step_ns=self.step_ns,
            on_time=self.on_time,
            lateness_ns=self.lateness_ns,
            status=self.status,
        )


def assess_completion(
    request: TaskDeadline, raw_finish_ns: int, step_ns: int = 10_000_000
) -> CompletionAssessment:
    """Classify actual usable completion; retain full late-work time and cost."""
    return CompletionAssessment(request, raw_finish_ns, step_ns)
