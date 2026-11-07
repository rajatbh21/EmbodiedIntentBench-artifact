"""Mission contract: what the mission must achieve, and under which hard constraints.

The contract states the *goal*. It deliberately does not state how the task is scored --
the rule that decides whether a predicted mask counts as a found target belongs to the
task specification, so that changing a matching threshold does not mean editing every
contract that uses the task.

All V1 contract constraints are hard constraints.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from missionbench.schemas.common import ComparisonOperator
from missionbench.schemas.loading import open_document

__all__ = ["Contract", "PrivacyLevel", "load_contract"]


class PrivacyLevel(StrEnum):
    """How far mission data is permitted to travel.

    The rule mapping a level to permitted configurations is *not* defined here: it is an
    action-validation concern and lives in ``missionbench.simulator.action_validator``,
    where it can consider the full configuration metadata rather than just placement.
    """

    #: No data leaves the vehicle; only on-board execution is permitted.
    LOCAL_ONLY = "local_only"
    #: Derived features may be transmitted, but raw sensor data may not.
    FEATURES_ONLY = "features_only"
    #: Raw input may be transmitted to a remote endpoint.
    REMOTE_ALLOWED = "remote_allowed"


_FIELDS: Final = (
    "contract_id",
    "task_id",
    "evidence_type",
    "quality_metric",
    "quality_operator",
    "quality_threshold",
    "deadline_s",
    "communication_budget_mb",
    "min_final_battery_frac",
    "privacy_level",
)


@dataclass(frozen=True, slots=True)
class Contract:
    """A mission contract. Immutable: the simulator never rewrites the goal mid-episode."""

    contract_id: str
    task_id: str
    evidence_type: str
    quality_metric: str
    quality_operator: ComparisonOperator
    quality_threshold: float
    deadline_s: float
    communication_budget_mb: float
    min_final_battery_frac: float
    privacy_level: PrivacyLevel

    def quality_satisfied(self, value: float) -> bool:
        """Return whether a measured quality value meets the contract's threshold."""
        return self.quality_operator.compare(value, self.quality_threshold)


def load_contract(path: Path) -> Contract:
    """Load and validate a contract specification file."""
    reader = open_document(path, document_type="Contract", allowed_fields=_FIELDS)
    return Contract(
        contract_id=reader.get_str("contract_id"),
        task_id=reader.get_str("task_id"),
        evidence_type=reader.get_str("evidence_type"),
        quality_metric=reader.get_str("quality_metric"),
        quality_operator=reader.get_enum("quality_operator", ComparisonOperator),
        quality_threshold=reader.get_float("quality_threshold"),
        deadline_s=reader.get_float("deadline_s", exclusive_minimum=0.0),
        communication_budget_mb=reader.get_float("communication_budget_mb", minimum=0.0),
        min_final_battery_frac=reader.get_fraction("min_final_battery_frac"),
        privacy_level=reader.get_enum("privacy_level", PrivacyLevel),
    )
