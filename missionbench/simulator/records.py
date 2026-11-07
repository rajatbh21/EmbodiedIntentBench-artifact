"""Structured records: what the runner emits, and what everything downstream reads.

Evaluators and metrics compute from records, never from live simulator state. That
separation is what lets a metric be added later without rerunning anything, provided the
step log already carried the data -- which is why the step log is generous rather than
minimal. Logging a field costs a dictionary entry; discovering it was needed after a
benchmark campaign costs the campaign.

The records are deliberately plain and JSON-serialisable. Nothing here interprets a
prediction payload or knows what task produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from missionbench.simulator.battery_model import EnergyUsage
from missionbench.simulator.termination import TerminationReason

__all__ = ["EpisodeRecord", "StepRecord"]


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One decision step, from the observation offered to the resulting state."""

    step_index: int
    #: Clock at the *start* of the step, which is the time the observation described.
    time_s: float
    frame_id: int
    #: The policy-visible observation, exactly as the policy saw it.
    state: dict[str, Any]
    #: What the policy asked for, what actually ran, and why they differ.
    action: dict[str, Any]
    executed_config_id: str
    #: Execution outcome: success, latency, energy, communication, failure reason.
    execution: dict[str, Any]
    #: Energy consumed over the step, broken down by component.
    energy: dict[str, float]
    #: Wall-clock duration, i.e. ``max(decision interval, inference latency)``.
    elapsed_s: float
    #: Frames that passed unprocessed because inference outran the interval.
    frames_skipped: int
    is_switch: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "time_s": self.time_s,
            "frame_id": self.frame_id,
            "state": self.state,
            "action": self.action,
            "executed_config_id": self.executed_config_id,
            "execution": self.execution,
            "energy": self.energy,
            "elapsed_s": self.elapsed_s,
            "frames_skipped": self.frames_skipped,
            "is_switch": self.is_switch,
        }


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    """Everything one episode did. The sole input to evaluation and metrics."""

    episode_id: str
    contract_id: str
    policy_name: str
    executor_id: str
    steps: tuple[StepRecord, ...]
    termination_reason: TerminationReason
    final_time_s: float
    final_battery_frac: float
    final_path_progress: float
    cumulative_energy: EnergyUsage
    cumulative_communication_mb: float
    #: The task's own evidence record. Opaque here; only its evaluator interprets it.
    evidence: Any
    invalid_action_count: int
    privacy_violation_count: int
    failed_inference_count: int
    configuration_switch_count: int
    frames_skipped_total: int
    #: Times at which network conditions changed. Not used by any V1 metric; recorded so
    #: adaptation latency can be computed later without rerunning the campaign, once
    #: "appropriately adapted" is formally defined.
    network_change_times_s: tuple[float, ...] = ()
    #: Times at which the battery crossed a notable threshold, for the same reason.
    battery_event_times_s: tuple[float, ...] = ()
    seed: int = 0
    config_selection_history: tuple[str, ...] = field(default_factory=tuple)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def successful_execution_latencies_s(self) -> tuple[float, ...]:
        """Latencies of executions that produced a result.

        Failed executions are excluded on purpose. A timeout is a fixed penalty, not a
        measurement of how long inference takes, and averaging it in would blend the
        latency metric with the failure rate -- two things the benchmark reports separately.
        """
        return tuple(
            step.execution["latency_s"] for step in self.steps if step.execution["success"]
        )

    def to_dict(self, *, include_detail: bool = False) -> dict[str, Any]:
        """Serialise the record.

        The step log and the raw evidence are omitted by default, for two reasons. A 900-step
        episode logs 900 steps and several thousand predicted instances, which is a large
        file for what is usually read as a summary. More importantly, raw evidence carries
        ``ground_truth_track_id`` and ``mask_iou`` on every instance -- publishing a result
        file with those in it would publish the answers.

        The evaluator's own ``details`` already report the counts a reader needs, so the
        default remains informative without disclosing anything.
        """
        payload: dict[str, Any] = {
            "episode_id": self.episode_id,
            "contract_id": self.contract_id,
            "policy": self.policy_name,
            "executor_id": self.executor_id,
            "seed": self.seed,
            "step_count": self.step_count,
            "termination_reason": self.termination_reason.value,
            "final_time_s": self.final_time_s,
            "final_battery_frac": self.final_battery_frac,
            "final_path_progress": self.final_path_progress,
            "energy": self.cumulative_energy.to_dict(),
            "cumulative_communication_mb": self.cumulative_communication_mb,
            "invalid_action_count": self.invalid_action_count,
            "privacy_violation_count": self.privacy_violation_count,
            "failed_inference_count": self.failed_inference_count,
            "configuration_switch_count": self.configuration_switch_count,
            "frames_skipped_total": self.frames_skipped_total,
            "evidence_summary": {
                "processed_frames": getattr(self.evidence, "processed_frames", 0),
                "instance_count": len(getattr(self.evidence, "instances", ())),
            },
            "adaptation_log": {
                "network_change_times_s": list(self.network_change_times_s),
                "battery_event_times_s": list(self.battery_event_times_s),
                "config_selection_history": list(self.config_selection_history),
            },
        }
        if include_detail:
            payload["steps"] = [step.to_dict() for step in self.steps]
            payload["evidence"] = self.evidence.to_dict() if self.evidence is not None else None
        return payload
