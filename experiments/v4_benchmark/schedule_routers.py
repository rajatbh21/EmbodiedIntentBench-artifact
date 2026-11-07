"""Non-adaptive controls: fixed time cycling and seeded random legal selection.

These are experimental controls, not reproductions of learned routing papers.
They test whether input-aware routers improve on a cheap selection schedule.
"""

from __future__ import annotations

import math
import random

from missionbench.simulator.action_validator import privacy_permits
from experiments.v4_benchmark.routers import RouterInput


class ScheduleRouter:
    def __init__(self, *, mode: str, config_ids: list[str], period_s: float = 2, seed: int = 0):
        if mode not in ("periodic", "seeded_random"):
            raise ValueError("unsupported schedule control")
        if (
            not isinstance(config_ids, list)
            or not config_ids
            or any(not isinstance(k, str) or not k for k in config_ids)
            or len(set(config_ids)) != len(config_ids)
        ):
            raise ValueError("config_ids must be a nonempty unique string list")
        if (
            isinstance(period_s, bool)
            or not isinstance(period_s, (int, float))
            or not math.isfinite(period_s)
            or period_s <= 0
        ):
            raise ValueError("period_s must be positive and finite")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        self.mode, self.config_ids, self.period_s = mode, tuple(config_ids), period_s
        self.random = random.Random(seed)
        self.last_decision = None

    def select(self, observation: RouterInput) -> str:
        if any(k not in observation.configs for k in self.config_ids):
            raise ValueError("scheduled configuration absent from catalog")
        legal = [
            k
            for k in self.config_ids
            if privacy_permits(observation.contract.privacy_level, observation.configs.get(k))
        ]
        if not legal:
            raise ValueError("no privacy-legal scheduled configuration")
        if self.mode == "periodic":
            selected = legal[
                math.floor(observation.state.current_time_s / self.period_s) % len(legal)
            ]
        else:
            selected = self.random.choice(legal)
        self.last_decision = {
            "reason": self.mode,
            "selected_config_id": selected,
            "uses_rgb": False,
            "quality_intent_used": False,
            "privacy_filter_applied": True,
        }
        return selected
