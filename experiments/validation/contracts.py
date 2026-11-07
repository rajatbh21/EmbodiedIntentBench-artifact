"""Declared mission-contract variants and a parametrised outcome scorer.

Responsibility: express each validation contract as a parameter setting of the
existing ``LocalInspectionIntent``/``MissionIntent`` schema (phase windows,
``min_hits``, ``min_evidence_span_s``, ``required_recall``, false-component caps,
phase/report deadlines, report recall and report caps) and score an execution
history against it. No new requirement type is introduced.

Boundaries: the scorer reads only receipts (events + result) and the hidden
contract document; it never re-executes anything. For the frozen reference
contract (``confirm``) it must reproduce ``experiments/intent_mission_v2/
runtime.py::outcome`` check-for-check (``study.py parity`` verifies all 648).

Variant definitions are loaded from ``config.json`` (frozen before any
variant outcome was computed); see docs/VALIDATION.md for their sources.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from experiments.intent_mission_v2.protocol import WINDOWS, MissionIntent
from experiments.lake_intent_contract import canonical
from experiments.lake_missions.objectives import score as phase_score
from experiments.mission_task_deadlines.study import task_metrics

CONFIG = json.loads((Path(__file__).with_name("config.json")).read_text())
VARIANTS = CONFIG["contracts"]
REFERENCE = "confirm"
RELAXATIONS = CONFIG["relaxations"]


def variant_document(base_document, name):
    """Apply one declared variant to the frozen v2 contract document."""
    spec = VARIANTS[name]
    doc = copy.deepcopy(base_document)
    if name == REFERENCE:
        return doc
    (base_phase,) = doc["mission"]["phases"]
    phases = spec.get("phases", [{}])
    doc["mission"]["phases"] = [dict(copy.deepcopy(base_phase), **override) for override in phases]
    for phase in doc["mission"]["phases"]:
        if "phase_deadline_s" in spec:
            phase["deadline_s"] = spec["phase_deadline_s"]
    if "report_deadline_s" in spec:
        doc["reporting"]["deadline_s"] = spec["report_deadline_s"]
    doc["schema_version"] = "validation_contract_variant_v1:" + name
    doc["mission"]["title"] = spec["title"]
    doc["mission"]["user_request"] = spec["user_request"]
    return doc


class VariantIntent(MissionIntent):
    """Same public/hidden split as ``MissionIntent``; only declared parameters differ."""

    def __init__(self, case, battery, name):
        super().__init__(case, battery)
        if name not in VARIANTS:
            raise ValueError("Undeclared contract variant: " + name)
        self.name = name
        if name != REFERENCE:
            self._json = canonical(variant_document(self.document(), name))


def contract_for(case, battery, name):
    return MissionIntent(case, battery) if name == REFERENCE else VariantIntent(case, battery, name)


def _relaxed(doc, relax):
    doc = copy.deepcopy(doc)
    for key in relax or ():
        if key not in RELAXATIONS:
            raise ValueError("Undeclared relaxation: " + key)
        for phase in doc["mission"]["phases"]:
            if key == "repeated_evidence":
                phase["min_hits"], phase["min_evidence_span_s"] = 1, 0.0
            elif key == "evidence_recall":
                phase["required_recall"] = 1 / 3
            elif key == "raw_false_components":
                phase["max_false_components"] = math.inf
        if key == "report_recall":
            doc["reporting"]["min_report_recall"] = 1 / 3
        elif key == "report_deadline":
            doc["reporting"]["deadline_s"] = math.inf
        elif key == "report_caps":
            doc["reporting"]["max_false_reports"] = doc["reporting"]["max_duplicate_reports"] = (
                math.inf
            )
        elif key == "battery_reserve":
            doc["mission"]["min_final_battery_frac"] = 0.0
    return doc


def sweep_settings():
    """The 144 predeclared sensitivity settings, in declared order."""
    import itertools

    spec = CONFIG["sensitivity_map"]["parameters"]
    names = (
        "people_required",
        "sightings_per_person",
        "false_alarm_cap",
        "report_required",
        "deadline_s",
    )
    return [
        dict(zip(names, values, strict=True))
        for values in itertools.product(*(spec[n]["values"] for n in names))
    ]


def sweep_document(base_document, setting):
    """Express one sensitivity setting in the existing contract schema."""
    doc = copy.deepcopy(base_document)
    (phase,) = doc["mission"]["phases"]
    k, h = setting["people_required"], setting["sightings_per_person"]
    cap, deadline = setting["false_alarm_cap"], setting["deadline_s"]
    phase.update(
        min_hits=h,
        min_evidence_span_s=2.0 * (h - 1),
        required_recall=k / 3,
        max_false_components=math.inf if cap is None else cap,
        deadline_s=deadline,
    )
    reporting = doc["reporting"]
    reporting["deadline_s"] = deadline
    if setting["report_required"]:
        reporting.update(min_report_recall=k / 3, max_false_reports=2, max_duplicate_reports=2)
    else:
        reporting.update(
            min_report_recall=0.0, max_false_reports=math.inf, max_duplicate_reports=math.inf
        )
    return doc


def outcome(contract, events, result, relax=None, document=None):
    """Generalised ``dynamic_outcome`` for one continuous report window.

    ``relax`` names evaluation-only counterfactual relaxations (Experiment 4);
    ``document`` replaces the hidden contract document (sensitivity map).
    The execution history is never changed by a relaxation.
    """
    doc = _relaxed(contract.document() if document is None else document, relax)
    focal = set(doc["evaluator_target_ids"])
    (window,) = WINDOWS
    start_s, end_s = window
    phases = phase_score(doc["mission"], events, result)["phases"]
    reporting = doc["reporting"]
    deadline_s = reporting["deadline_s"]
    reports = result["report_quality"]["reports"]
    current = [r for r in reports if r["window_id"] == 0]
    correct = [r["target_id"] for r in current if r["target_id"] is not None]
    valid = [
        r
        for r in current
        if r["target_id"] in focal
        and r["completion_time_s"] <= deadline_s
        and all(start_s <= e["capture_s"] < end_s for e in r["evidence"])
    ]
    ids = {r["target_id"] for r in valid}
    false = len(current) - len(correct)
    duplicates = len(correct) - len(set(correct))
    selected = [e for e in events if e["window_id"] == 0]
    first = min((r["completion_time_s"] for r in valid), default=None)
    checks = dict(
        repeated_evidence=all(p["success"] for p in phases),
        fresh_report_recall=len(ids) / len(focal) >= reporting["min_report_recall"],
        false_report_limit=false <= reporting["max_false_reports"],
        duplicate_report_limit=duplicates <= reporting["max_duplicate_reports"],
    )
    if len(phases) > 1:
        # Multi-phase variants keep the mission-wide raw false-component cap of the reference.
        cap = min(p["max_false_components"] for p in doc["mission"]["phases"])
        checks["mission_false_components"] = sum(p["false_components"] for p in phases) <= cap
    rows = [
        dict(
            window_id=0,
            start_s=start_s,
            end_s=end_s,
            deadline_s=deadline_s,
            checks=checks,
            success=all(checks.values()),
            focal_correct_reports=len(ids),
            evidence_recall=min(p["evidence_recall"] for p in phases),
            world_false_reports=false,
            world_duplicate_reports=duplicates,
            timely_targets=sorted(ids),
            first_valid_report_s=first,
            first_valid_report_wait_s=None if first is None else first - start_s,
            first_decision_wait_s=None if not selected else selected[0]["capture_time_s"] - start_s,
            requests=len(selected),
            abstentions=sum(e["selected_action_id"] is None for e in selected),
            deadline_misses=sum(e["perception"]["deadline_missed"] for e in selected),
        )
    ]
    allowed = {
        "completed",
        "task_deadline_miss",
        "transport_timeout",
        "mission_deadline_exceeded",
        "abstained",
        "timeout",
        "deadline",
    }
    execution = (
        result["mission_completed"]
        and result["termination_reason"] == "mission_window_complete"
        and result["error"] is None
        and result["final_time_s"] >= doc["duration_s"]
        and all(
            e["perception"]["status"] in allowed
            and (
                e["perception"]["admitted"] or (e["pipeline"] is None and not e["emitted_reports"])
            )
            for e in events
        )
    )
    constraints = dict(result["constraints"])
    if relax and "battery_reserve" in relax:
        constraints["battery_constraint_success"] = (
            result["final_battery_frac"] >= doc["mission"]["min_final_battery_frac"]
        )
    checks_all = dict(
        execution=execution, all_windows=all(w["success"] for w in rows), **constraints
    )
    success = all(checks_all.values())
    return dict(
        schema_version="cumulative_inspection_dynamic_urgency_outcome_v2",
        contract_variant=getattr(contract, "name", REFERENCE),
        relaxations=sorted(relax or ()),
        dynamic_mission_success=success,
        local_inspection_success=success,
        full_wide_mission_evaluated=False,
        checks=checks_all,
        failed_constraints=[k for k, v in checks_all.items() if not v],
        windows=rows,
        window_successes=sum(w["success"] for w in rows),
        phases=phases,
        reporting=dict(
            world_false_reports=false,
            world_duplicates=duplicates,
            correct_nonfocal_reports=sum(
                r["target_id"] is not None and r["target_id"] not in focal for r in reports
            ),
            timely_focal_reports=[(0, t) for t in sorted(ids)],
            recall_by_deadline=len(ids) / len(focal),
        ),
        task_metrics=task_metrics(events),
        abstentions=sum(e["selected_action_id"] is None for e in events),
    )
