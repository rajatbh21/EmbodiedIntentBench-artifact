"""Intent-level expected false-component budget with pooled development priors.

This separately declared heuristic preserves the common resource guard. It
projects a candidate's mean development false components over a complete public
phase at that candidate's forecast cadence. It is not a probability bound, a
guarantee, a remaining-budget ledger, or feedback from evaluator correctness.
"""
from __future__ import annotations

import math

from experiments.strong_routing_20261030 import calibration_v2 as pooled

NAME = "mission_risk_budget"


class MissionRiskBudgetPolicy:
    def __init__(self, actions, costs, profile, calibration):
        self.delegate = pooled.PoolCalibratedPolicy(
            "calibrated_quality", actions, costs, profile, calibration)

    def select(self, **observation):
        decision = self.delegate.select(**observation)
        release_ns = observation["release_ns"]
        capture_s = release_ns / 1e9
        phase = next(p for p in observation["intent"]["mission"]["phases"]
                     if p["start_s"] <= capture_s < p["end_s"])
        forecaster = self.delegate.forecaster
        interval_ns = round(forecaster.costs.mission.interval_s * 1e9)
        duration_s = phase["end_s"] - phase["start_s"]
        allowed = []
        for row in decision["candidates"]:
            row["reference_calibrated_quality_score"] = row["score"]
            row.update(mission_risk_feasible=False, full_phase_expected_calls=None,
                       full_phase_expected_false_components=None,
                       forecast_capture_cadence_s=None)
            row["score"] = None
            if not row["feasible"]:
                continue
            # Exact same completion-to-capture cadence convention as the guard.
            downstream_end_ns = row["predicted_usable_finish_ns"] + forecaster.downstream_latency_ns
            cadence_ns = max(interval_ns,
                math.ceil((downstream_end_ns - release_ns) / interval_ns) * interval_ns)
            cadence_s = cadence_ns / 1e9
            calls = math.ceil(duration_s / cadence_s)
            expected_false = calls * row["predicted_false_components"]
            risk_feasible = expected_false <= phase["max_false_components"] + 1e-12
            row.update(mission_risk_feasible=risk_feasible,
                       full_phase_expected_calls=calls,
                       full_phase_expected_false_components=expected_false,
                       forecast_capture_cadence_s=cadence_s)
            if risk_feasible:
                row["score"] = row["predicted_recall"]
                allowed.append(row)
        selected = min(allowed, key=lambda row: (-row["predicted_recall"],
            row["predicted_usable_finish_ns"], row["estimate"]["energy_j"], row["action_id"])) if allowed else None
        resource_feasible = any(row["feasible"] for row in decision["candidates"])
        decision.update(policy=NAME, action_id=selected["action_id"] if selected else None,
            reason="highest_calibrated_recall_within_resource_and_expected_phase_risk" if selected else
                   ("no_model_within_expected_phase_false_component_budget" if resource_feasible else "no_feasible_action"),
            mission_risk_guard=dict(scope="Entire current public phase, assuming one candidate and its forecast cadence",
                phase_duration_s=duration_s, max_false_components=phase["max_false_components"],
                uses_actual_feedback=False, guarantees_false_component_cap=False,
                formula="ceil(full_phase_duration_s / forecast_capture_cadence_s) * development_mean_false_components_per_capture"),
            score_interpretation="Expected-count feasibility heuristic; no probabilistic guarantee or actual-error feedback")
        return decision


def make_policy(name, actions, costs, profile, quality, risk, learned_profile=None):
    if name != NAME:
        raise ValueError("Only the separately declared mission_risk_budget policy is accepted")
    return MissionRiskBudgetPolicy(actions, costs, profile, learned_profile)
