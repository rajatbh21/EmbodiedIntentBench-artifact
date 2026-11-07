"""A budget-planning baseline: pace the contract's budgets over the remaining mission.

Where :class:`~missionbench.policies.rule_based.RuleBasedPolicy` reacts to
*instantaneous* pressure (battery at the floor, budget nearly spent), this policy
*projects*: it estimates how much mission remains from the observed path progress,
paces the communication budget over the captures still to come, and projects the
final battery from the drain rate it has itself observed. The intent is a stronger —
but still honest — baseline between the reactive rules and the GT-aware skyline
upper bound (``missionbench/v2/skyline.py``).

Epistemic position, unchanged from V1: the policy sees only the contract, the frozen
``RuntimeState``, the episode's configurations, and (when disclosed) public profiles.
The drain rate is estimated from *successive observed battery fractions* — its own
history, never the simulator's internals — so the instance is deliberately stateful
across steps of one episode. Reusing one instance across episodes without ``reset()``
would leak an old drain estimate into a new mission; the composition root constructs
policies per run, which is why this is a documented caveat rather than a hidden bug.

The rule, in order:

1. **Privacy** — drop forbidden configurations (a selection would be a recorded
   violation).
2. **Reachability** — drop remote configurations while the link is down.
3. **Communication pacing** — the cumulative spend may run at most a bounded burst
   ahead of the *pro-rata schedule* (spendable budget x projected mission fraction).
   Early remote use is possible (the burst), sustained overuse throttles before the
   budget cliff. Zero-upload configurations always pass.
4. **Battery projection** — from the EMA drain rate, project the final battery at the
   projected mission end; below the floor plus margin, keep only the fastest
   candidates (latency is the only public proxy for energy, as in V1).
5. **Deadline projection** — as in the reactive baseline: a projected overrun keeps
   only the fastest candidates.
6. **Quality** — highest disclosed tier among what remains, ties toward lower
   estimated latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from missionbench.policies.base import estimated_latency_s
from missionbench.schemas.configuration import ConfigCatalog, Configuration, Placement
from missionbench.schemas.contract import Contract
from missionbench.schemas.profile import PublicProfileView, QualityTier
from missionbench.schemas.runtime_state import RuntimeState
from missionbench.simulator.action_validator import privacy_permits

__all__ = ["BudgetPlannerPolicy", "BudgetPlannerSettings"]

_DEFAULT_TIER_RANK: Final = -1


@dataclass(frozen=True, slots=True)
class BudgetPlannerSettings:
    """Named planning parameters."""

    #: Fraction of the communication budget never planned for spending.
    communication_reserve_frac: float = 0.05
    #: Fraction of the budget the cumulative spend may run ahead of the pro-rata
    #: schedule — the pacing rule's burst capacity.
    communication_burst_frac: float = 0.15
    #: Seconds between decisions, for projecting the remaining capture count.
    decision_interval_s: float = 1.0
    #: Battery headroom above the contract floor the projection must preserve.
    battery_margin_frac: float = 0.03
    #: EMA weight of the newest drain-rate sample (1.0 = last sample only).
    drain_ema_alpha: float = 0.5
    #: Projected overrun beyond this fraction of the deadline triggers the fast path.
    deadline_pressure_frac: float = 0.95
    #: Estimated latency above this many intervals is dropped when avoidable.
    max_latency_intervals: float = 1.5


class BudgetPlannerPolicy:
    """Paces communication and projects battery instead of reacting at thresholds."""

    __slots__ = ("_drain_per_s", "_last_battery_frac", "_last_time_s", "_profiles", "_settings")

    def __init__(
        self,
        *,
        public_profiles: PublicProfileView | None = None,
        settings: BudgetPlannerSettings | None = None,
    ) -> None:
        self._profiles = (
            public_profiles if public_profiles is not None else PublicProfileView.hidden()
        )
        self._settings = settings or BudgetPlannerSettings()
        self._last_time_s: float | None = None
        self._last_battery_frac: float | None = None
        self._drain_per_s: float | None = None

    def reset(self) -> None:
        """Forget the drain estimate (call between episodes if reusing an instance)."""
        self._last_time_s = None
        self._last_battery_frac = None
        self._drain_per_s = None

    def select_config(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str:
        self._observe_drain(state)

        candidates = [
            config for config in configs if privacy_permits(contract.privacy_level, config)
        ]
        if not candidates:
            return next(iter(configs)).config_id  # impossible pool; validator records it

        candidates = self._reachable(candidates, state) or candidates
        candidates = self._within_pace(candidates, contract, state) or candidates

        if self._battery_projection_low(contract, state) or self._deadline_pressure(
            contract, state
        ):
            return self._fastest(candidates, state).config_id

        candidates = self._within_latency_budget(candidates, state) or candidates
        return self._best_quality(candidates, state).config_id

    # -- projections ------------------------------------------------------------------

    def _observe_drain(self, state: RuntimeState) -> None:
        """Update the EMA drain estimate from consecutive observed battery fractions."""
        if self._last_time_s is not None and self._last_battery_frac is not None:
            dt = state.current_time_s - self._last_time_s
            if dt > 0.0:
                sample = max(0.0, self._last_battery_frac - state.battery_frac) / dt
                if self._drain_per_s is None:
                    self._drain_per_s = sample
                else:
                    alpha = self._settings.drain_ema_alpha
                    self._drain_per_s = alpha * sample + (1.0 - alpha) * self._drain_per_s
        self._last_time_s = state.current_time_s
        self._last_battery_frac = state.battery_frac

    def _remaining_time_s(self, contract: Contract, state: RuntimeState) -> float:
        """Projected seconds to mission end, from progress so far (deadline-capped)."""
        if state.path_progress > 0.0 and state.current_time_s > 0.0:
            projected_total = state.current_time_s / state.path_progress
            remaining = projected_total - state.current_time_s
        else:
            remaining = contract.deadline_s - state.current_time_s
        return max(self._settings.decision_interval_s, min(remaining, state.remaining_deadline_s))

    def _battery_projection_low(self, contract: Contract, state: RuntimeState) -> bool:
        if self._drain_per_s is None:
            return False  # no observation yet; the pace filter still applies
        projected_final = state.battery_frac - self._drain_per_s * self._remaining_time_s(
            contract, state
        )
        return (
            projected_final < contract.min_final_battery_frac + self._settings.battery_margin_frac
        )

    def _deadline_pressure(self, contract: Contract, state: RuntimeState) -> bool:
        if state.path_progress <= 0.0 or state.current_time_s <= 0.0:
            return False
        projected_total_s = state.current_time_s / state.path_progress
        return projected_total_s > contract.deadline_s * self._settings.deadline_pressure_frac

    # -- filters ----------------------------------------------------------------------

    def _reachable(
        self, candidates: list[Configuration], state: RuntimeState
    ) -> list[Configuration]:
        if not state.network.is_disconnected:
            return candidates
        return [c for c in candidates if c.strategy.placement is Placement.LOCAL]

    def _within_pace(
        self, candidates: list[Configuration], contract: Contract, state: RuntimeState
    ) -> list[Configuration]:
        """Keep configurations whose upload stays within the paced cumulative cap.

        The cap is the pro-rata schedule — the spendable budget scaled by the fraction
        of the mission projected to have elapsed after this step — plus a bounded
        burst, so a good early link can be exploited without racing to the cliff.
        """
        settings = self._settings
        spendable = contract.communication_budget_mb * (1.0 - settings.communication_reserve_frac)
        elapsed_after_step = state.current_time_s + settings.decision_interval_s
        projected_total = elapsed_after_step + self._remaining_time_s(contract, state)
        schedule_frac = min(1.0, elapsed_after_step / max(projected_total, elapsed_after_step))
        cap = (
            spendable * schedule_frac
            + contract.communication_budget_mb * settings.communication_burst_frac
        )
        within = []
        for config in candidates:
            profile = self._profiles.get(config.config_id)
            upload = 0.0 if profile is None else profile.expected_upload_mb
            if upload <= 0.0 or state.cumulative_communication_mb + upload <= cap:
                within.append(config)
        return within

    def _within_latency_budget(
        self, candidates: list[Configuration], state: RuntimeState
    ) -> list[Configuration]:
        ceiling = self._settings.max_latency_intervals * self._settings.decision_interval_s
        within = []
        for config in candidates:
            latency = estimated_latency_s(
                config, self._profiles.get(config.config_id), state.network
            )
            if latency is None or latency <= ceiling:
                within.append(config)
        return within

    # -- choices ----------------------------------------------------------------------

    def _fastest(self, candidates: list[Configuration], state: RuntimeState) -> Configuration:
        return min(candidates, key=lambda config: self._latency_key(config, state))

    def _best_quality(self, candidates: list[Configuration], state: RuntimeState) -> Configuration:
        return max(
            candidates,
            key=lambda config: (self._tier_rank(config), -self._latency_key(config, state)),
        )

    def _tier_rank(self, config: Configuration) -> int:
        profile = self._profiles.get(config.config_id)
        return _DEFAULT_TIER_RANK if profile is None else QualityTier(profile.quality_tier).rank

    def _latency_key(self, config: Configuration, state: RuntimeState) -> float:
        latency = estimated_latency_s(config, self._profiles.get(config.config_id), state.network)
        return float("inf") if latency is None else latency

    def __repr__(self) -> str:
        return f"BudgetPlannerPolicy(profiles_available={self._profiles.is_available})"
