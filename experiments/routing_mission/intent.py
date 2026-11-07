"""Deterministic user-request compiler. No LLM is required or evaluated.

Supported Korean form: 사람 수색; 기한=400초; 재현율=80%; 오탐=1; 예비배터리=20%
Unknown/ambiguous text is rejected, never silently interpreted as a different mission.
"""

import math
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Intent:
    task: str = "person_search"
    deadline_s: float = 400.0
    min_report_recall: float = 0.5
    max_false_reports: int = 2
    max_duplicate_reports: int = 2
    confirmed_report_deadline_s: float = 300.0
    min_final_battery_frac: float = 0.2
    communication_budget_mb: float = 0.0
    privacy_level: str = "local_only"

    def __post_init__(self):
        if self.task != "person_search" or self.privacy_level != "local_only":
            raise ValueError("this release supports local person-search inference")
        for key in (
            "deadline_s",
            "confirmed_report_deadline_s",
            "communication_budget_mb",
            "min_report_recall",
            "min_final_battery_frac",
        ):
            v = getattr(self, key)
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v < 0
            ):
                raise ValueError(f"invalid {key}")
        if self.deadline_s <= 0 or not 0 < self.confirmed_report_deadline_s <= self.deadline_s:
            raise ValueError("report deadline must be positive and within mission deadline")
        if not 0 < self.min_report_recall <= 1 or not 0 <= self.min_final_battery_frac < 1:
            raise ValueError("invalid recall/reserve fraction")
        for key in ("max_false_reports", "max_duplicate_reports"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"invalid {key}")

    def compile(self):
        contract = dict(
            schema_version="1.0",
            contract_id="USER_INTENT",
            task_id="HUMAN_SEARCH_SEGMENTATION",
            evidence_type="instance_mask_set",
            quality_metric="target_recall",
            quality_operator=">=",
            quality_threshold=0,
            deadline_s=self.deadline_s,
            communication_budget_mb=self.communication_budget_mb,
            min_final_battery_frac=self.min_final_battery_frac,
            privacy_level=self.privacy_level,
        )
        requirements = {
            k: getattr(self, k)
            for k in (
                "min_report_recall",
                "max_false_reports",
                "max_duplicate_reports",
                "confirmed_report_deadline_s",
            )
        }
        requirements["require_detection"] = True
        return contract, requirements


def parse(text, *, duration_s):
    defaults = asdict(
        Intent(
            deadline_s=round(duration_s + 30, 1),
            confirmed_report_deadline_s=round(duration_s * 0.8, 1),
        )
    )
    clauses = [x.strip() for x in text.split(";") if x.strip()]
    if not clauses or clauses.pop(0) not in ("사람 수색", "실종자 수색", "person search"):
        raise ValueError(
            "start with 사람 수색, 실종자 수색, or person search; use ; key=value fields"
        )
    keys = {
        "기한": "deadline_s",
        "보고기한": "confirmed_report_deadline_s",
        "재현율": "min_report_recall",
        "오탐": "max_false_reports",
        "중복보고": "max_duplicate_reports",
        "예비배터리": "min_final_battery_frac",
    }
    explicit = {}
    for clause in clauses:
        key, sep, value = clause.partition("=")
        key = keys.get(key.strip(), key.strip())
        if not sep or key not in set(keys.values()) or key in explicit:
            raise ValueError("unknown, duplicate, or malformed intent clause: " + clause)
        units = "초|s" if key.endswith("_s") else "%" if key.startswith("min_") else ""
        m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(" + units + r")?\s*", value)
        if not m:
            raise ValueError("invalid value/unit: " + clause)
        v = float(m[1])
        if key.startswith("min_") and m[2] == "%":
            v /= 100
        if key.startswith("max_"):
            if not v.is_integer():
                raise ValueError("report count must be integer")
            v = int(v)
        explicit[key] = v
    if "deadline_s" in explicit and "confirmed_report_deadline_s" not in explicit:
        defaults["confirmed_report_deadline_s"] = min(
            defaults["confirmed_report_deadline_s"], explicit["deadline_s"]
        )
    result = Intent(**{**defaults, **explicit})
    return result, dict(
        user_input=text,
        compiler="controlled_form_v1",
        explicit=explicit,
        defaulted={k: v for k, v in asdict(result).items() if k not in explicit},
        interpretation="only routing policy is scored; intent compilation is fixed",
    )
