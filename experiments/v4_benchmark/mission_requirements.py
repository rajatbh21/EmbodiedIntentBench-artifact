"""Optional mission-specific V4 requirements; frozen V1 contracts stay unchanged.

Bounds are additive to the primary contract. Correct-report timing is evaluated
only after inference completes, with capture-time GT; it is never public evidence.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from missionbench.schemas.loading import SchemaValidationError


@dataclass(frozen=True)
class MissionRequirements:
    min_target_recall: float | None = None
    min_detection_precision: float | None = None
    require_detection: bool = False
    first_correct_report_deadline_s: float | None = None

    min_report_recall: float | None = None
    max_false_reports: int | None = None
    max_duplicate_reports: int | None = None
    confirmed_report_deadline_s: float | None = None

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) - set(cls.__dataclass_fields__):
            raise SchemaValidationError("unknown mission requirements or non-object input")
        for key, value in data.items():
            if key == "require_detection":
                if type(value) is not bool:
                    raise SchemaValidationError("require_detection must be boolean")
            elif key in ("max_false_reports", "max_duplicate_reports"):
                if type(value) is not int or value < 0:
                    raise SchemaValidationError(f"{key} must be a nonnegative integer")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise SchemaValidationError(f"{key} must be a finite number")
            elif key.startswith("min_") and not 0 <= value <= 1:
                raise SchemaValidationError(f"{key} must be in [0,1]")
            elif key.endswith("_s") and value <= 0:
                raise SchemaValidationError(f"{key} must be positive")
        return cls(**data)

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}

    def evaluate(self, quality, events, report_quality=None):
        completions = [
            event["completion_time_s"]
            for event in events
            if event["score"]["matched_components"] > 0
        ]
        first = min(completions) if completions else None
        constraints = {}
        if self.min_target_recall is not None:
            constraints["required_recall_success"] = (
                quality["target_recall"] >= self.min_target_recall
            )
        if self.min_detection_precision is not None:
            constraints["required_precision_success"] = (
                quality["detection_precision"] >= self.min_detection_precision
            )
        if self.require_detection:
            # A spurious prediction alone must not satisfy a human-search requirement.
            constraints["required_detection_success"] = quality["matched_detections"] > 0
        if self.first_correct_report_deadline_s is not None:
            constraints["first_report_deadline_success"] = (
                first is not None and first <= self.first_correct_report_deadline_s
            )
        if any(
            value is not None
            for value in (
                self.min_report_recall,
                self.max_false_reports,
                self.max_duplicate_reports,
                self.confirmed_report_deadline_s,
            )
        ):
            if report_quality is None:
                raise ValueError("report requirements need report evaluation")
            for field, metric in (
                ("min_report_recall", "report_recall"),
                ("max_false_reports", "false_reports"),
                ("max_duplicate_reports", "duplicate_correct_reports"),
            ):
                bound = getattr(self, field)
                if bound is not None:
                    constraints[field + "_success"] = (
                        report_quality[metric] >= bound
                        if field.startswith("min_")
                        else report_quality[metric] <= bound
                    )
            if self.confirmed_report_deadline_s is not None:
                confirmed = report_quality["first_confirmed_correct_report_s"]
                constraints["confirmed_report_deadline_success"] = (
                    confirmed is not None and confirmed <= self.confirmed_report_deadline_s
                )
        return constraints, first
