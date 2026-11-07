"""Time-varying network conditions for V2/V3 missions (V3 P1).

A mission's link is a deterministic, piecewise-constant sequence of **named regimes**
(``5g_good``, ``lte_degraded``, ``disconnected`` …). The runner samples the regime at
defined mission times; the policy sees only the current sample, projected into the
frozen V1 ``NetworkObservation`` (bandwidth/RTT/loss) — never the regime name, the
segment boundaries, or anything about the future. Regime identity exists for
diagnostics and replay only.

Conceptually this is V1's ``TraceBasedNetworkModel`` adapted to V2: the same
piecewise-constant, sampled-at-decision-time semantics, but with separate uplink and
downlink (a remote inference request is upload-heavy) and a regime label. V2 scenarios
without a ``network_trace`` keep their constant network, unchanged.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Final

from missionbench.schemas.loading import SchemaValidationError
from missionbench.schemas.network_trace import NetworkObservation

__all__ = ["NetworkRegime", "V2NetworkModel", "constant_network_model"]

#: The regime id used when a scenario declares only a constant network.
CONSTANT_REGIME_ID: Final = "constant"


@dataclass(frozen=True, slots=True)
class NetworkRegime:
    """One named, piecewise-constant network condition.

    ``start_s`` is when the regime begins; it lasts until the next regime starts (the
    last regime lasts for the rest of the mission). All values are configured
    simulation parameters, not radio measurements.
    """

    regime_id: str
    start_s: float
    uplink_mbps: float
    downlink_mbps: float
    rtt_ms: float
    packet_loss_frac: float

    @property
    def reachable(self) -> bool:
        """Whether a remote endpoint can be reached at all under this regime."""
        return self.uplink_mbps > 0.0 and self.downlink_mbps > 0.0

    def to_observation(self) -> NetworkObservation:
        """The policy-visible projection (the frozen V1 observation shape).

        V1 carries a single ``bandwidth_mbps``; the conservative projection is the
        uplink, because every V2 remote request is upload-dominated. The regime name
        and the downlink stay evaluator/diagnostics-side.
        """
        return NetworkObservation(
            bandwidth_mbps=self.uplink_mbps,
            rtt_ms=self.rtt_ms,
            packet_loss_frac=self.packet_loss_frac,
        )

    def to_dict(self) -> dict[str, float | str | bool]:
        return {
            "regime_id": self.regime_id,
            "start_s": self.start_s,
            "uplink_mbps": self.uplink_mbps,
            "downlink_mbps": self.downlink_mbps,
            "rtt_ms": self.rtt_ms,
            "packet_loss_frac": self.packet_loss_frac,
            "reachable": self.reachable,
        }


class V2NetworkModel:
    """Samples the network regime at mission times, deterministically.

    Sampling semantics are half-open: a regime starting at ``t`` owns ``[t, next_t)``,
    so a boundary time belongs to the regime that starts there. Times before the first
    regime clamp to it (traces must start at 0 anyway); times after the last regime
    stay in it.
    """

    __slots__ = ("_regimes",)

    def __init__(self, regimes: tuple[NetworkRegime, ...]) -> None:
        if not regimes:
            raise SchemaValidationError("a network model needs at least one regime")
        if regimes[0].start_s != 0.0:
            raise SchemaValidationError(
                f"the first network regime must start at 0.0 s, got {regimes[0].start_s}"
            )
        for earlier, later in itertools.pairwise(regimes):
            if later.start_s <= earlier.start_s:
                raise SchemaValidationError(
                    "network regimes must have strictly increasing start times; "
                    f"{later.regime_id!r} at {later.start_s} follows {earlier.regime_id!r} "
                    f"at {earlier.start_s}"
                )
        self._regimes = regimes

    @property
    def regimes(self) -> tuple[NetworkRegime, ...]:
        return self._regimes

    def state_at(self, time_s: float) -> NetworkRegime:
        current = self._regimes[0]
        for regime in self._regimes:
            if regime.start_s <= time_s:
                current = regime
            else:
                break
        return current

    def observation_at(self, time_s: float) -> NetworkObservation:
        """The policy-visible network sample for a decision at ``time_s``."""
        return self.state_at(time_s).to_observation()


def constant_network_model(
    bandwidth_mbps: float, rtt_ms: float, packet_loss_frac: float
) -> V2NetworkModel:
    """The backwards-compatible model for scenarios without a ``network_trace``."""
    return V2NetworkModel(
        (
            NetworkRegime(
                regime_id=CONSTANT_REGIME_ID,
                start_s=0.0,
                uplink_mbps=bandwidth_mbps,
                downlink_mbps=bandwidth_mbps,
                rtt_ms=rtt_ms,
                packet_loss_frac=packet_loss_frac,
            ),
        )
    )
