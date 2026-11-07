"""The policy-visible observation.

``RuntimeState`` is the **entire** boundary between the simulator and a policy. It is
constructed fresh by the runner at every decision step and is frozen, so a policy can
neither retain a mutable handle on simulator state nor influence the run except through
its returned configuration ID.

What must never appear here
---------------------------
Any ground-truth-derived quantity: true recall, the hidden target count, whether a
prediction actually matched. Any future knowledge: the remainder of the network trace,
upcoming frames. Any simulator handle: the ``Episode``, the runner, the energy ledger.

A leak here does not produce a wrong number -- it silently invalidates every result the
benchmark reports, because the policy would be scored on information no deployed system
could have. ``tests/test_runtime_state.py`` pins the exact field set for that reason.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from missionbench.schemas.network_trace import NetworkObservation

__all__ = ["EvidenceSummary", "RuntimeState"]


@dataclass(frozen=True, slots=True)
class EvidenceSummary:
    """What the policy is told about the evidence gathered so far.

    Every field is derived from the policy's *own predictions*, never from ground truth.
    ``predicted_unique_targets`` is what the system believes it has found -- which may be
    wrong, and the policy is given no way to learn that it is.
    """

    predicted_unique_targets: int
    processed_frames: int
    mean_prediction_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicted_unique_targets": self.predicted_unique_targets,
            "processed_frames": self.processed_frames,
            "mean_prediction_confidence": self.mean_prediction_confidence,
        }


@dataclass(frozen=True, slots=True)
class RuntimeState:
    """A single decision step's observation, as handed to ``Policy.select_config``."""

    current_time_s: float
    frame_id: int
    battery_frac: float
    power_mode: str
    network: NetworkObservation
    current_config_id: str | None
    remaining_deadline_s: float
    cumulative_energy_j: float
    cumulative_communication_mb: float
    path_progress: float
    evidence_summary: EvidenceSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot, as embedded in step-level log records."""
        return {
            "current_time_s": self.current_time_s,
            "frame_id": self.frame_id,
            "battery_frac": self.battery_frac,
            "power_mode": self.power_mode,
            "network": self.network.to_dict(),
            "current_config_id": self.current_config_id,
            "remaining_deadline_s": self.remaining_deadline_s,
            "cumulative_energy_j": self.cumulative_energy_j,
            "cumulative_communication_mb": self.cumulative_communication_mb,
            "path_progress": self.path_progress,
            "evidence_summary": self.evidence_summary.to_dict(),
        }
