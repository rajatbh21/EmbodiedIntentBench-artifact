"""Distinct routing objectives with common causal resource admission.

These are transparent benchmark baselines, not reproductions of complete
ApproxDet, Chanakya, or FrugalML systems. The learned policies are frozen at
evaluation time. No evaluator reward, candidate mask, or future link is an input.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np

from experiments.mission_routing_v2.features import encode
from experiments.mission_task_deadlines.policy import POLICIES as ORIGINAL_POLICIES
from experiments.mission_task_deadlines.policy import DeadlineAwarePolicy

ADAPTIVE_POLICIES = (
    "resource_threshold",
    "accuracy_latency",
    "supervised_utility",
    "linucb_frozen",
)
POLICIES = (*ORIGINAL_POLICIES, *ADAPTIVE_POLICIES)
LEARNED_POLICIES = {"supervised_utility": "supervised", "linucb_frozen": "bandit"}
THRESHOLDS = dict(battery_headroom=0.20, upload_bps=10_000_000, rtt_ns=100_000_000)
HEAVY = "CFG_MASKRCNN_1280_ROT0_90"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def tie_key(row):
    """Stable preference for lower latency, lower immediate cost, then action ID."""
    return (row["predicted_usable_finish_ns"], row["estimate"]["energy_j"], row["action_id"])


class AdaptivePolicy:
    def __init__(
        self,
        name,
        actions,
        costs,
        profile,
        quality_profile,
        risk_profile,
        *,
        learned_profile=None,
        **kwargs,
    ):
        if name not in ADAPTIVE_POLICIES:
            raise ValueError("Unknown adaptive perception policy")
        self.name = name
        self.forecaster = DeadlineAwarePolicy(
            "deadline_aware", actions, costs, profile, quality_profile, risk_profile, **kwargs
        )
        self.learned = None
        self.learned_sha256 = None
        if name in LEARNED_POLICIES:
            from experiments.perception_router_suite.learning import FEATURES, SCHEMA_VERSION

            if not isinstance(learned_profile, dict):
                raise ValueError("Frozen development-trained profile required")
            if learned_profile.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Wrong learned profile schema")
            if learned_profile.get("feature_names") != list(FEATURES):
                raise ValueError("Wrong learned context feature schema")
            signatures = {a.action_id: a.source_signature for a in self.forecaster.actions.values()}
            if learned_profile.get("action_signatures") != signatures:
                raise ValueError("Learned action signature mismatch")
            section = learned_profile.get(LEARNED_POLICIES[name])
            if not isinstance(section, dict) or set(section) != set(signatures):
                raise ValueError("Complete learned action pool required")
            # Only numerical prediction heads survive into the running policy.
            self.learned = {}
            for action_id, head in section.items():
                retained = {}
                for key in ("mean", "scale", "coefficient"):
                    values = np.asarray(head.get(key), dtype=float)
                    if values.shape != (len(FEATURES),) or not np.isfinite(values).all():
                        raise ValueError("Finite learned coefficients required")
                    if key == "scale" and np.any(values <= 0):
                        raise ValueError("Positive learned feature scales required")
                    retained[key] = values.tolist()
                self.learned[action_id] = retained
            self.learned_sha256 = digest(learned_profile)

    def select(self, **observation):
        decision = self.forecaster.select(**observation)
        decision["policy"] = self.name
        decision["common_resource_guard"] = "deadline_aware_v1"
        admitted = [r for r in decision["candidates"] if r["feasible"]]
        for row in decision["candidates"]:
            # Preserve the original intent score as a diagnostic, never as this
            # policy's ranking objective.
            row["reference_intent_score"] = row.pop("score")
            row["routing_score"] = None
        decision["learned_profile_sha256"] = self.learned_sha256
        decision["evaluation_learning_updates"] = 0
        if not admitted:
            decision.update(action_id=None, reason="no_feasible_action")
            return decision

        if self.name == "resource_threshold":
            limits, link = observation["intent"]["limits"], observation["link"]
            margin = observation["battery_frac"] - limits["min_final_battery_frac"]
            if margin <= THRESHOLDS["battery_headroom"]:
                selected = min(
                    admitted,
                    key=lambda r: (r["estimate"]["energy_j"], *tie_key(r)),
                )
                reason = "low_battery_choose_minimum_service_energy"
            else:
                good_link = (
                    not link.outage
                    and link.upload_bps >= THRESHOLDS["upload_bps"]
                    and link.rtt_ns <= THRESHOLDS["rtt_ns"]
                )
                preferred = "REMOTE__" + HEAVY if good_link else HEAVY
                selected = next((r for r in admitted if r["action_id"] == preferred), None)
                reason = (
                    "healthy_resources_prefer_remote_union" if good_link else "prefer_local_union"
                )
                if selected is None:
                    selected = min(admitted, key=tie_key)
                    reason = "preferred_union_infeasible_choose_fastest"
            decision["thresholds"] = copy.deepcopy(THRESHOLDS)
        elif self.name == "accuracy_latency":
            for row in admitted:
                row["routing_score"] = row["predicted_recall"]
            selected = min(admitted, key=lambda r: (-r["routing_score"], *tie_key(r)))
            reason = "highest_predicted_recall_under_common_constraints"
        else:
            from experiments.perception_router_suite.learning import context_vector, predict

            context = dict(
                perception_budget_s=self.forecaster.profile.perception_ns / 1e9,
                battery_frac=observation["battery_frac"],
                reserve_available_j=decision["reserve_available_j"],
                **{
                    k: decision["intent_requirements"][k]
                    for k in ("min_hits", "min_evidence_span_s", "max_false_components")
                },
            )
            rgb_features = encode(observation["rgb"])
            for row in admitted:
                candidate = dict(
                    row,
                    latency_s=(row["predicted_usable_finish_ns"] - observation["release_ns"]) / 1e9,
                )
                vector = context_vector(rgb_features, candidate, context)
                value = predict(self.learned, row["action_id"], vector)
                if not math.isfinite(value):
                    raise ValueError("Finite learned routing score required")
                row["routing_score"] = value
                row["learned_context"] = vector.tolist()
            selected = min(admitted, key=lambda r: (-r["routing_score"], *tie_key(r)))
            reason = (
                "highest_supervised_utility"
                if self.name == "supervised_utility"
                else "highest_frozen_linucb_mean_no_test_updates"
            )
        decision.update(action_id=selected["action_id"], reason=reason)
        return decision


def make_policy(
    name,
    actions,
    costs,
    profile,
    quality_profile,
    risk_profile,
    learned_profile=None,
    **kwargs,
):
    if name in ORIGINAL_POLICIES:
        return DeadlineAwarePolicy(
            name, actions, costs, profile, quality_profile, risk_profile, **kwargs
        )
    return AdaptivePolicy(
        name,
        actions,
        costs,
        profile,
        quality_profile,
        risk_profile,
        learned_profile=learned_profile,
        **kwargs,
    )
