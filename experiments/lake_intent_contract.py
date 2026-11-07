"""Versioned lake intent scoring; hidden identities never enter the public contract.

This adds an explicit primary outcome without changing frozen study scorers. Phase
evidence reuses their exact matching semantics. Reporting, when requested, requires
unique correct reports to arrive by the declared deadline. It is not an NLP parser.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from experiments.lake_missions.objectives import public_phases, score, validate

VERSION = "lake_intent_v1"
MISSION_FIELDS = {
    "mission_id",
    "title",
    "user_request",
    "initial_battery_frac",
    "min_final_battery_frac",
    "phases",
}
LIMIT_FIELDS = {"deadline_s", "communication_budget_mb", "min_final_battery_frac", "privacy_level"}
REPORT_FIELDS = {
    "mode",
    "target_scope",
    "deadline_s",
    "min_report_recall",
    "max_false_reports",
    "max_duplicate_reports",
}
RESOURCE_FLAGS = (
    "battery_constraint_success",
    "communication_constraint_success",
    "deadline_success",
    "memory_constraint_success",
    "privacy_constraint_success",
)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def immutable(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: immutable(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(immutable(item) for item in value)
    return value


def fields(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Unexpected {label} fields")


def number(value: Any, minimum: float, maximum: float = math.inf) -> None:
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError("Expected a finite number in range")


@dataclass(frozen=True, init=False)
class IntentContract:
    """Own an immutable JSON copy; callers cannot mutate validated requirements."""

    _json: str

    def __init__(self, document: dict[str, Any]):
        fields(
            document,
            {
                "schema_version",
                "mission",
                "duration_s",
                "limits",
                "reporting",
                "evaluator_target_ids",
            },
            "intent contract",
        )
        if document["schema_version"] != VERSION:
            raise ValueError("Unsupported intent contract version")
        mission, limits, reports = (document[k] for k in ("mission", "limits", "reporting"))
        fields(mission, MISSION_FIELDS, "mission")
        fields(limits, LIMIT_FIELDS, "limits")
        for key in ("mission_id", "title", "user_request"):
            if not isinstance(mission[key], str) or not mission[key].strip():
                raise ValueError("Mission text must be nonempty")
        number(document["duration_s"], 0)
        if document["duration_s"] == 0:
            raise ValueError("Positive duration required")
        targets = document["evaluator_target_ids"]
        if (
            not isinstance(targets, list)
            or not targets
            or any(not isinstance(t, str) or not t for t in targets)
            or len(targets) != len(set(targets))
        ):
            raise ValueError("Unique evaluator targets required")
        number(mission["initial_battery_frac"], 0, 1)
        number(mission["min_final_battery_frac"], 0, 1)
        validate(mission, document["duration_s"], targets)
        for phase in mission["phases"]:
            for key in ("phase_id", "instruction"):
                if not isinstance(phase[key], str) or not phase[key].strip():
                    raise ValueError("Phase text must be nonempty")
        number(limits["deadline_s"], document["duration_s"])
        number(limits["communication_budget_mb"], 0)
        number(limits["min_final_battery_frac"], 0, 1)
        if limits["min_final_battery_frac"] != mission["min_final_battery_frac"]:
            raise ValueError("Conflicting battery reserves")
        if limits["privacy_level"] not in ("local_only", "features_only", "remote_allowed"):
            raise ValueError("Unknown privacy level")
        if reports == {"mode": "evidence_only"}:
            pass
        else:
            fields(reports, REPORT_FIELDS, "reporting")
            if (
                reports["mode"] != "report_required"
                or reports["target_scope"] != "all_scenario_targets"
            ):
                raise ValueError("Unsupported reporting mode or scope")
            number(reports["deadline_s"], 0, limits["deadline_s"])
            number(reports["min_report_recall"], 0, 1)
            if reports["deadline_s"] == 0 or reports["min_report_recall"] == 0:
                raise ValueError("Reporting requires positive deadline and recall")
            for key in ("max_false_reports", "max_duplicate_reports"):
                if type(reports[key]) is not int or reports[key] < 0:
                    raise ValueError("Report limits must be nonnegative integers")
        object.__setattr__(self, "_json", canonical(document))

    def document(self) -> dict[str, Any]:
        return json.loads(self._json)

    def public_document(self) -> dict[str, Any]:
        data = self.document()
        del data["evaluator_target_ids"]
        data["mission"]["phases"] = public_phases(data["mission"])
        return data

    @property
    def public_sha256(self) -> str:
        # Hash only public fields: hidden target identities cannot affect policy input.
        return digest(self.public_document())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self._json.encode()).hexdigest()


def validate_runtime(contract: IntentContract, runtime: dict[str, Any]) -> None:
    """Fail closed when the frozen runner carries an undeclared quality requirement."""
    for key, expected in contract.document()["limits"].items():
        if runtime[key] != expected:
            raise ValueError(f"Runtime contract mismatch: {key}")
    if (runtime["quality_metric"], runtime["quality_operator"], runtime["quality_threshold"]) != (
        "target_recall",
        ">=",
        0,
    ):
        raise ValueError("Phase contract requires neutral legacy aggregate quality")


def evaluate(contract: IntentContract, events: list[dict], result: dict) -> dict:
    """The sole primary outcome for this version; old success flags are not inputs."""
    data = contract.document()
    validate_runtime(contract, result["contract"])
    phases = score(data["mission"], events, result)
    checks = {"phase:" + p["phase_id"]: p["success"] for p in phases["phases"]}
    checks["execution"] = (
        result["status"] == "completed"
        and result["error"] is None
        and result["mission_completed"] is True
        and result["termination_reason"] == "mission_window_complete"
        and result["final_time_s"] >= data["duration_s"]
        and result["processed_observations"] == len(events)
        and not any(result[k] for k in ("router_errors", "inference_failures", "invalid_actions"))
        and all(e["execution"]["success"] is True for e in events)
    )
    checks.update({key: result["constraints"][key] is True for key in RESOURCE_FLAGS})
    checks["battery_constraint_success"] &= (
        result["final_battery_frac"] >= data["limits"]["min_final_battery_frac"]
    )
    checks["deadline_success"] &= result["final_time_s"] <= data["limits"]["deadline_s"]
    checks["communication_constraint_success"] &= (
        result["cumulative_communication_mb"] <= data["limits"]["communication_budget_mb"]
    )
    checks["privacy_constraint_success"] &= result["privacy_violations"] == 0
    reporting = {"required": False}
    required = data["reporting"]
    if required["mode"] == "report_required":
        reports = result["report_quality"]["reports"]
        known = set(data["evaluator_target_ids"])
        if any(r["target_id"] is not None and r["target_id"] not in known for r in reports):
            raise ValueError("Report target outside evaluator scope")
        for report in reports:
            number(report["completion_time_s"], 0, result["final_time_s"])
        timely = {
            r["target_id"]
            for r in reports
            if r["target_id"] is not None and r["completion_time_s"] <= required["deadline_s"]
        }
        correct = [r["target_id"] for r in reports if r["target_id"] is not None]
        false = len(reports) - len(correct)
        duplicates = len(correct) - len(set(correct))
        recall = len(timely) / len(known)
        checks.update(
            {
                "report_recall_by_deadline": recall >= required["min_report_recall"],
                "false_report_limit": false <= required["max_false_reports"],
                "duplicate_report_limit": duplicates <= required["max_duplicate_reports"],
            }
        )
        reporting = dict(
            required=True,
            recall_by_deadline=recall,
            timely_unique_reports=len(timely),
            denominator=len(known),
            false_reports=false,
            duplicate_reports=duplicates,
            late_reports=sum(r["completion_time_s"] > required["deadline_s"] for r in reports),
        )
    return dict(
        schema_version=VERSION,
        contract_sha256=contract.sha256,
        public_contract_sha256=contract.public_sha256,
        mission_id=data["mission"]["mission_id"],
        mode=required["mode"],
        mission_success=all(checks.values()),
        constraints=checks,
        failed_constraints=[key for key, passed in checks.items() if not passed],
        phases=phases["phases"],
        reporting=reporting,
        semantics=(
            "Phase evidence uses evaluator associations; required reports use predicted tracks"
        ),
    )
