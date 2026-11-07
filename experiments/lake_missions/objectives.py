"""Deadline-aware, repeated-evidence objectives; evaluator-only target identities.

This scores perception evidence across mission stages. It does not invent a
predicted identity tracker or claim that the drone issued identity-level reports.
"""

import math

PUBLIC = {
    "phase_id",
    "instruction",
    "start_s",
    "end_s",
    "deadline_s",
    "required_recall",
    "min_hits",
    "min_evidence_span_s",
    "max_false_components",
    "weight",
}


def validate(mission, duration, known_targets):
    phases = mission["phases"]
    if not phases or len({p["phase_id"] for p in phases}) != len(phases):
        raise ValueError("Nonempty, uniquely named phases required")
    previous = 0
    for p in phases:
        if set(p) != PUBLIC | {"evaluator_target_ids"}:
            raise ValueError("Unexpected phase fields")
        for key in PUBLIC - {"phase_id", "instruction"}:
            value = p[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError("Phase values must be finite numbers")
        if not previous <= p["start_s"] < p["end_s"] <= p["deadline_s"] <= duration:
            raise ValueError("Phase windows must be ordered, non-overlapping and bounded")
        previous = p["end_s"]
        if type(p["min_hits"]) is not int or p["min_hits"] < 2:
            raise ValueError("At least two distinct evidence captures are required")
        if (
            not 0 < p["required_recall"] <= 1
            or not 0 <= p["min_evidence_span_s"] < p["end_s"] - p["start_s"]
        ):
            raise ValueError("Invalid recall or evidence span")
        if (
            type(p["max_false_components"]) is not int
            or p["max_false_components"] < 0
            or p["weight"] <= 0
        ):
            raise ValueError("Invalid false-positive limit or phase weight")
        targets = p["evaluator_target_ids"]
        if (
            not targets
            or len(targets) != len(set(targets))
            or not set(targets) <= set(known_targets)
        ):
            raise ValueError("Invalid evaluator targets")
    if not 0 < mission["min_final_battery_frac"] < mission["initial_battery_frac"] <= 1:
        raise ValueError("Invalid battery requirements")


def public_phases(mission):
    return [{k: p[k] for k in PUBLIC} for p in mission["phases"]]


def phase_at(phases, time_s):
    return next((p for p in phases if p["start_s"] <= time_s < p["end_s"]), None)


def enough_evidence(times, phase):
    times = sorted(set(times))
    return len(times) >= phase["min_hits"] and times[-1] - times[0] >= phase["min_evidence_span_s"]


def score(mission, events, result):
    rows = []
    for phase in mission["phases"]:
        # A late result is not evidence available before a mission deadline.
        eligible = [
            e
            for e in events
            if phase["start_s"] <= e["capture_time_s"] < phase["end_s"]
            and e["completion_time_s"] <= phase["deadline_s"]
            and e["execution"]["success"]
        ]
        evidence = {
            target: [
                e["capture_time_s"] for e in eligible if target in e["score"]["matched_target_ids"]
            ]
            for target in phase["evaluator_target_ids"]
        }
        supported = [target for target, times in evidence.items() if enough_evidence(times, phase)]
        recall = len(supported) / len(evidence)
        # Charge every false output to its capture stage, including late results.
        # Delaying a bad result past a deadline cannot hide it from the score.
        false = sum(
            e["score"]["false_positive_components"]
            for e in events
            if phase["start_s"] <= e["capture_time_s"] < phase["end_s"]
        )
        passed = recall >= phase["required_recall"] and false <= phase["max_false_components"]
        rows.append(
            dict(
                phase_id=phase["phase_id"],
                success=passed,
                evidence_recall=recall,
                supported_targets=supported,
                required_targets=phase["evaluator_target_ids"],
                evidence_capture_times=evidence,
                false_components=false,
                weight=phase["weight"],
                required_recall=phase["required_recall"],
                max_false_components=phase["max_false_components"],
                late_results=sum(
                    phase["start_s"] <= e["capture_time_s"] < phase["end_s"]
                    and e["completion_time_s"] > phase["deadline_s"]
                    for e in events
                ),
            )
        )
    resource_ok = (
        result["final_battery_frac"] >= mission["min_final_battery_frac"]
        and result["constraints"]["deadline_success"]
        and result["constraints"]["memory_constraint_success"]
        and result["constraints"]["privacy_constraint_success"]
    )
    execution_ok = result["status"] == "completed" and not any(
        result[key] for key in ("router_errors", "inference_failures", "invalid_actions")
    )
    return dict(
        mission_id=mission["mission_id"],
        mission_success=all(p["success"] for p in rows) and resource_ok and execution_ok,
        phase_success_rate=sum(p["success"] for p in rows) / len(rows),
        weighted_evidence_recall=sum(p["evidence_recall"] * p["weight"] for p in rows)
        / sum(p["weight"] for p in rows),
        resource_success=resource_ok,
        execution_success=execution_ok,
        phases=rows,
        scoring="phase_local_repeated_mask_iou_evidence_v1",
        caveat="Evaluator GT associations; not predicted tracking or identity-level reports.",
    )
