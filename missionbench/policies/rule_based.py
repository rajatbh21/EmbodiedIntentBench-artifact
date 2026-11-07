"""A hand-written heuristic baseline.

**Not optimal, and not claimed to be.** It is a readable reference point that consults every
signal the contract makes binding, so that a learned or optimisation-based policy has
something meaningful to beat.

The rule, in order. Each step narrows the candidates; the last one that leaves anything
standing decides.

1. **Privacy.** Drop configurations the contract forbids. Selecting one would be recorded as
   a violation, and a baseline should not spend its budget on those.
2. **Reachability.** Drop remote configurations while the link is dead. They would fail,
   costing a timeout and producing nothing.
3. **Communication budget.** Drop configurations whose upload would breach the remaining
   budget, keeping a reserve so the last steps are not left with no legal option.
4. **Deadline pressure.** If the mission is projected to overrun, keep only the fastest
   candidates -- quality is worthless if the contract's deadline fails.
5. **Battery pressure.** Near the required reserve, prefer the fastest candidate. Latency is
   a proxy here: public profiles disclose no energy, and a longer inference means more time
   airborne. Weak, and deliberately so -- see docs/v1_spec.md §11 on why battery is a guard.
6. **Latency budget.** Drop candidates whose estimated latency exceeds a multiple of the
   decision interval, *if any candidate survives*. This is what makes the policy abandon a
   remote configuration as bandwidth collapses: at 2 Mbps one upload spans six frames, and
   every skipped frame is a target that may never be seen again.
7. **Quality.** Among what remains, take the highest quality tier, breaking ties toward
   lower estimated latency.

Without public profiles the policy still runs: steps 1-4 need no profile, and step 7 falls
back to catalog order. It degrades rather than failing, because whether profiles are
disclosed is a property of the run, not something a policy may assume.
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

__all__ = ["RuleBasedPolicy", "RuleBasedSettings"]

_DEFAULT_TIER_RANK: Final = -1


@dataclass(frozen=True, slots=True)
class RuleBasedSettings:
    """Tunable thresholds, named rather than scattered as literals."""

    #: Fraction of the communication budget held back, so a late step is never left with a
    #: budget too small for any remote option it might legitimately want.
    communication_reserve_frac: float = 0.05
    #: Estimated latency above this many decision intervals is treated as too expensive,
    #: because the frames it skips cost more quality than the configuration's tier buys.
    max_latency_intervals: float = 1.5
    decision_interval_s: float = 1.0
    #: Battery headroom above the contract's reserve, below which the policy plays safe.
    battery_margin_frac: float = 0.05
    #: Projected overrun beyond this fraction of the deadline triggers the fast path.
    deadline_pressure_frac: float = 0.95


class RuleBasedPolicy:
    """Selects a configuration from the contract's constraints and the observed state."""

    __slots__ = ("_profiles", "_settings")

    def __init__(
        self,
        *,
        public_profiles: PublicProfileView | None = None,
        settings: RuleBasedSettings | None = None,
    ) -> None:
        self._profiles = (
            public_profiles if public_profiles is not None else PublicProfileView.hidden()
        )
        self._settings = settings or RuleBasedSettings()

    def select_config(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str:
        candidates = [
            config for config in configs if privacy_permits(contract.privacy_level, config)
        ]
        if not candidates:
            # Every option violates privacy. Returning the first is honest: the validator
            # will record the violation, which is the correct report for an impossible pool.
            return next(iter(configs)).config_id

        candidates = self._reachable(candidates, state) or candidates
        candidates = self._affordable(candidates, contract, state) or candidates

        if self._under_deadline_pressure(contract, state) or self._under_battery_pressure(
            contract, state
        ):
            return self._fastest(candidates, state).config_id

        candidates = self._within_latency_budget(candidates, state) or candidates
        return self._best_quality(candidates, state).config_id

    # -- filters ----------------------------------------------------------------------

    def _reachable(
        self, candidates: list[Configuration], state: RuntimeState
    ) -> list[Configuration]:
        if not state.network.is_disconnected:
            return candidates
        return [c for c in candidates if c.strategy.placement is Placement.LOCAL]

    def _affordable(
        self, candidates: list[Configuration], contract: Contract, state: RuntimeState
    ) -> list[Configuration]:
        remaining = contract.communication_budget_mb - state.cumulative_communication_mb
        spendable = (
            remaining - contract.communication_budget_mb * self._settings.communication_reserve_frac
        )
        affordable = []
        for config in candidates:
            profile = self._profiles.get(config.config_id)
            upload = 0.0 if profile is None else profile.expected_upload_mb
            # Transmitting nothing can never breach a budget, so a zero-upload configuration
            # stays affordable even once the reserve has driven `spendable` negative. Testing
            # `upload <= spendable` alone would rule out every option at that point, and the
            # empty-result fallback would then hand remote back -- defeating the filter
            # exactly when it matters most.
            if upload <= 0.0 or upload <= spendable:
                affordable.append(config)
        return affordable

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

    # -- pressure signals -------------------------------------------------------------

    def _under_deadline_pressure(self, contract: Contract, state: RuntimeState) -> bool:
        """Project the finish time from progress so far, and compare against the deadline."""
        if state.path_progress <= 0.0 or state.current_time_s <= 0.0:
            return False
        projected_total_s = state.current_time_s / state.path_progress
        return projected_total_s > contract.deadline_s * self._settings.deadline_pressure_frac

    def _under_battery_pressure(self, contract: Contract, state: RuntimeState) -> bool:
        return (
            state.battery_frac
            <= contract.min_final_battery_frac + self._settings.battery_margin_frac
        )

    # -- choices ------------------------------------------------------------------------

    def _fastest(self, candidates: list[Configuration], state: RuntimeState) -> Configuration:
        return min(candidates, key=lambda config: self._latency_key(config, state))

    def _best_quality(self, candidates: list[Configuration], state: RuntimeState) -> Configuration:
        return max(
            candidates,
            key=lambda config: (
                self._tier_rank(config),
                -self._latency_key(config, state),
            ),
        )

    def _tier_rank(self, config: Configuration) -> int:
        profile = self._profiles.get(config.config_id)
        return _DEFAULT_TIER_RANK if profile is None else QualityTier(profile.quality_tier).rank

    def _latency_key(self, config: Configuration, state: RuntimeState) -> float:
        """Estimated latency, with unknowns sorted last so they are never chosen as 'fastest'."""
        latency = estimated_latency_s(config, self._profiles.get(config.config_id), state.network)
        return float("inf") if latency is None else latency

    def __repr__(self) -> str:
        return f"RuleBasedPolicy(profiles_available={self._profiles.is_available})"
