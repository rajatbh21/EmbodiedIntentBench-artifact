"""Transparent experimental heuristic over public intent, RGB and resource state.

Image contrast/edge density are uncalibrated difficulty proxies, not target detection
or predicted accuracy. Ordinal model tiers cannot certify a numeric quality contract.
No labels, scene objects, future network or executor internals enter this module.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from missionbench.policies.base import estimated_latency_s
from missionbench.schemas.profile import PublicProfileView, QualityTier
from missionbench.simulator.action_validator import privacy_permits

if TYPE_CHECKING:
    from experiments.v4_benchmark.routers import RouterInput


@dataclass(frozen=True)
class Settings:
    demanding_precision: float = 0.9
    demanding_recall: float = 0.8
    low_contrast: float = 0.05
    edge_threshold: float = 0.15
    dense_edges: float = 0.2
    battery_margin_frac: float = 0.02
    communication_reserve_frac: float = 0.1

    def __post_init__(self):
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite fraction")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")


def visual_features(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bounded-cost, uniform RGB sampling; no task/color/label-specific thresholds."""
    h, w = rgb.shape[:2]
    if not h or not w:
        raise ValueError("empty RGB observation")
    small = rgb[:: max(1, math.ceil(h / 64)), :: max(1, math.ceil(w / 64))]
    gray = small.astype(np.float32).mean(axis=2) / 255.0
    edges = np.concatenate(
        (np.abs(np.diff(gray, axis=0)).ravel(), np.abs(np.diff(gray, axis=1)).ravel())
    )
    return gray, edges


class IntentVisualRouter:
    def __init__(
        self,
        profiles: PublicProfileView,
        settings: Settings,
        *,
        use_intent: bool = True,
        use_visual: bool = True,
    ):
        self.profiles = profiles
        self.settings = settings
        self.use_intent = use_intent
        self.use_visual = use_visual
        self.last_decision = None

    def select(self, observation: RouterInput) -> str:
        contract, state = observation.contract, observation.state
        settings = self.settings
        features = None
        visually_hard = False
        if self.use_visual:
            gray, edges = visual_features(observation.rgb)
            contrast = float(gray.std())
            density = float(np.mean(edges > settings.edge_threshold)) if edges.size else 0.0
            features = {"contrast_std": contrast, "edge_density": density}
            visually_hard = contrast < settings.low_contrast or density > settings.dense_edges
        demanding = False
        if self.use_intent:
            if str(contract.quality_operator) not in (">=", ">"):
                raise ValueError("intent_visual supports lower-bound quality contracts only")
            thresholds = {
                "target_recall": settings.demanding_recall,
                "detection_precision": settings.demanding_precision,
            }
            if contract.quality_metric not in thresholds:
                raise ValueError("unsupported intent quality metric")
            demanding = contract.quality_threshold >= thresholds[contract.quality_metric]
            extra = observation.requirements
            demanding = (
                demanding
                or (
                    extra.min_target_recall is not None
                    and extra.min_target_recall >= settings.demanding_recall
                )
                or (
                    extra.min_detection_precision is not None
                    and extra.min_detection_precision >= settings.demanding_precision
                )
            )
        legal = [c for c in observation.configs if privacy_permits(contract.privacy_level, c)]
        options = []
        estimates = {}
        for config in legal:
            profile = self.profiles.get(config.config_id)
            latency = estimated_latency_s(config, profile, state.network)
            estimates[config.config_id] = latency
            if config.strategy.is_remote:
                if state.network.is_disconnected or profile is None:
                    continue
                available = (
                    contract.communication_budget_mb * (1 - settings.communication_reserve_frac)
                    - state.cumulative_communication_mb
                )
                if profile.expected_upload_mb > available:
                    continue
            options.append(config)
        reason = "fastest_for_ordinary_observation"
        battery_low = state.battery_frac <= (
            contract.min_final_battery_frac + settings.battery_margin_frac
        )
        deadline_pressure = False
        # The benchmark owns invalid-action fallback for an impossible catalog.
        if not options:
            locals_ = [c for c in legal if not c.strategy.is_remote]
            selected = (locals_ or legal or list(observation.configs))[0]
            reason = "no_feasible_candidate"
        else:

            def speed(config):
                value = estimates[config.config_id]
                return math.inf if value is None else value

            deadline_budget = state.remaining_deadline_s
            report_deadline = observation.requirements.first_correct_report_deadline_s
            if (
                self.use_intent
                and report_deadline is not None
                and observation.detections_so_far == 0
                and report_deadline > state.current_time_s
            ):
                deadline_budget = min(deadline_budget, report_deadline - state.current_time_s)
            fits = [c for c in options if speed(c) <= deadline_budget]
            deadline_pressure = not fits
            options = fits or options
            if battery_low or not fits:
                reason = "resource_pressure_fastest"
                selected = min(options, key=speed)
            elif demanding or visually_hard:
                reason = "higher_tier_for_intent_or_visual_difficulty"

                def rank(config):
                    profile = self.profiles.get(config.config_id)
                    return QualityTier(profile.quality_tier).rank if profile else -1

                selected = max(options, key=lambda c: (rank(c), -speed(c)))
            else:
                selected = min(options, key=speed)
        self.last_decision = {
            "selected_config_id": selected.config_id,
            "reason": reason,
            "uses_rgb": self.use_visual,
            "uses_quality_intent": self.use_intent,
            "mission_requirements_consumed": self.use_intent,
            "visual_features": features,
            "visual_difficulty_trigger": visually_hard,
            "demanding_intent_trigger": demanding,
            "battery_pressure": battery_low,
            "deadline_pressure": deadline_pressure,
            "estimated_latency_s": estimates,
            "quality_estimate_calibrated": False,
            "settings": asdict(settings),
        }
        return selected.config_id
