"""Per-episode metrics, computed from the episode record and the task's verdict.

Mission success is the conjunction of five hard constraints. The quality one arrives as a
standardised ``TaskEvaluationResult``, so nothing here names ``target_f1`` or knows what task
ran -- adding a task metric must not require editing this file.

Violation margins are reported alongside the booleans because a run that missed the deadline
by half a second and one that missed it by five minutes are both "false", and the difference
is the whole story when comparing policies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Final

from missionbench.schemas.contract import Contract
from missionbench.simulator.records import EpisodeRecord
from missionbench.tasks.base import TaskEvaluationResult

__all__ = ["CONSTRAINT_NAMES", "ConstraintOutcome", "EpisodeMetrics", "compute_episode_metrics"]

#: The five hard constraints, in reporting order. ``constraint_violation_count`` counts how
#: many of these categories failed -- not how many time steps a single one stayed violated.
CONSTRAINT_NAMES: Final = ("quality", "deadline", "battery", "communication", "privacy")

_MS_PER_S: Final = 1000.0


@dataclass(frozen=True, slots=True)
class ConstraintOutcome:
    """Whether one constraint held, and by how much it was missed if not."""

    name: str
    satisfied: bool
    #: How far past the limit the episode went. Zero when satisfied; never negative, so it
    #: reads as "the violation", not "the headroom".
    violation: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "satisfied": self.satisfied, "violation": self.violation}


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """The scored result of one episode."""

    episode_id: str
    contract_id: str
    policy: str
    executor_id: str

    mission_success: bool
    quality: TaskEvaluationResult

    quality_success: bool
    deadline_success: bool
    battery_constraint_success: bool
    communication_constraint_success: bool
    privacy_constraint_success: bool

    final_battery_fraction: float
    mission_completion_time_s: float
    mean_end_to_end_inference_latency_ms: float
    total_communication_mb: float
    total_energy_j: float
    flight_energy_j: float
    compute_energy_j: float
    communication_energy_j: float

    configuration_switch_count: int
    constraint_violation_count: int
    deadline_violation_s: float
    battery_violation_frac: float
    communication_violation_mb: float
    invalid_action_count: int
    failed_inference_count: int

    termination_reason: str
    processed_frames: int
    step_count: int
    frames_skipped_total: int

    @property
    def quality_value(self) -> float:
        return self.quality.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "contract_id": self.contract_id,
            "policy": self.policy,
            "executor_id": self.executor_id,
            "mission_success": self.mission_success,
            "quality": self.quality.to_dict(),
            "constraints": {
                "quality_success": self.quality_success,
                "deadline_success": self.deadline_success,
                "battery_constraint_success": self.battery_constraint_success,
                "communication_constraint_success": self.communication_constraint_success,
                "privacy_constraint_success": self.privacy_constraint_success,
            },
            "violations": {
                "constraint_violation_count": self.constraint_violation_count,
                "deadline_violation_s": self.deadline_violation_s,
                "battery_violation_frac": self.battery_violation_frac,
                "communication_violation_mb": self.communication_violation_mb,
            },
            "resources": {
                "final_battery_fraction": self.final_battery_fraction,
                "mission_completion_time_s": self.mission_completion_time_s,
                "mean_end_to_end_inference_latency_ms": self.mean_end_to_end_inference_latency_ms,
                "total_communication_mb": self.total_communication_mb,
                "total_energy_j": self.total_energy_j,
                "flight_energy_j": self.flight_energy_j,
                "compute_energy_j": self.compute_energy_j,
                "communication_energy_j": self.communication_energy_j,
            },
            "behaviour": {
                "configuration_switch_count": self.configuration_switch_count,
                "invalid_action_count": self.invalid_action_count,
                "failed_inference_count": self.failed_inference_count,
                "processed_frames": self.processed_frames,
                "step_count": self.step_count,
                "frames_skipped_total": self.frames_skipped_total,
                "termination_reason": self.termination_reason,
            },
        }


def compute_episode_metrics(
    record: EpisodeRecord,
    evaluation: TaskEvaluationResult,
    contract: Contract,
) -> EpisodeMetrics:
    """Score one episode from its record and the task's verdict."""
    # The task reports false positives per minute of examined footage; the mission wall-clock
    # rate needs the completion time, which lives on the record, so it is added here. Guarded
    # on the empirical detection count, so profile/legacy results are untouched.
    evaluation = _with_mission_false_positive_rate(evaluation, record.final_time_s)

    deadline = _at_most(record.final_time_s, contract.deadline_s, name="deadline")
    battery = _at_least(record.final_battery_frac, contract.min_final_battery_frac, name="battery")
    communication = _at_most(
        record.cumulative_communication_mb, contract.communication_budget_mb, name="communication"
    )
    # A privacy-violating selection is blocked before it can execute, so this asks whether
    # one was ever *attempted*. Choosing a forbidden configuration is the violation; that
    # the simulator refused to carry it out does not make the attempt acceptable.
    privacy = ConstraintOutcome(
        name="privacy",
        satisfied=record.privacy_violation_count == 0,
        violation=float(record.privacy_violation_count),
    )
    quality = ConstraintOutcome(
        name="quality",
        satisfied=evaluation.success,
        violation=abs(evaluation.threshold - evaluation.value) if not evaluation.success else 0.0,
    )

    outcomes = (quality, deadline, battery, communication, privacy)
    latencies = record.successful_execution_latencies_s
    energy = record.cumulative_energy

    return EpisodeMetrics(
        episode_id=record.episode_id,
        contract_id=record.contract_id,
        policy=record.policy_name,
        executor_id=record.executor_id,
        mission_success=all(outcome.satisfied for outcome in outcomes),
        quality=evaluation,
        quality_success=quality.satisfied,
        deadline_success=deadline.satisfied,
        battery_constraint_success=battery.satisfied,
        communication_constraint_success=communication.satisfied,
        privacy_constraint_success=privacy.satisfied,
        final_battery_fraction=record.final_battery_frac,
        mission_completion_time_s=record.final_time_s,
        mean_end_to_end_inference_latency_ms=(
            sum(latencies) / len(latencies) * _MS_PER_S if latencies else 0.0
        ),
        total_communication_mb=record.cumulative_communication_mb,
        total_energy_j=energy.total_j,
        flight_energy_j=energy.flight_j,
        compute_energy_j=energy.compute_j,
        communication_energy_j=energy.communication_j,
        configuration_switch_count=record.configuration_switch_count,
        constraint_violation_count=sum(not outcome.satisfied for outcome in outcomes),
        deadline_violation_s=deadline.violation,
        battery_violation_frac=battery.violation,
        communication_violation_mb=communication.violation,
        invalid_action_count=record.invalid_action_count,
        failed_inference_count=record.failed_inference_count,
        termination_reason=record.termination_reason.value,
        processed_frames=getattr(record.evidence, "processed_frames", 0),
        step_count=record.step_count,
        frames_skipped_total=record.frames_skipped_total,
    )


_SECONDS_PER_MINUTE: Final = 60.0


def _with_mission_false_positive_rate(
    evaluation: TaskEvaluationResult, mission_time_s: float
) -> TaskEvaluationResult:
    """Add ``false_positives_per_mission_minute`` (wall-clock) to an empirical evaluation.

    The numerator is the frame-level ``false_positive_detections`` the task reported; the
    denominator is the mission's wall-clock completion time in minutes. A zero-duration mission
    yields ``0.0`` rather than dividing by zero. Non-empirical evaluations (no detection count)
    are returned untouched, so profile and legacy results do not gain the field.
    """
    false_positives = evaluation.details.get("false_positive_detections")
    if false_positives is None:
        return evaluation
    minutes = mission_time_s / _SECONDS_PER_MINUTE
    rate = false_positives / minutes if minutes > 0.0 else 0.0
    return replace(
        evaluation,
        details={**evaluation.details, "false_positives_per_mission_minute": rate},
    )


def _at_most(value: float, limit: float, *, name: str) -> ConstraintOutcome:
    return ConstraintOutcome(name=name, satisfied=value <= limit, violation=max(0.0, value - limit))


def _at_least(value: float, limit: float, *, name: str) -> ConstraintOutcome:
    return ConstraintOutcome(name=name, satisfied=value >= limit, violation=max(0.0, limit - value))
