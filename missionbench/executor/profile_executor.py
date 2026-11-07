"""Profile-driven execution: costs come from a per-platform profile, not from real work.

This is the V1 default. It needs no GPU, no model weights, and no network, and it is exactly
reproducible -- the same request always yields the same result.

Placement drives the cost model, read from typed strategy metadata:

- **local**: latency is the profiled compute time; nothing is transmitted.
- **remote**: ``upload + RTT + server compute + download``, where transfer time depends on
  the bandwidth observed *at this instant*. This is the whole point of a remote
  configuration being available -- it turns a network condition into a mission cost.

Nothing here inspects ``config_id``. A configuration is remote because
``strategy.placement`` says so.
"""

from __future__ import annotations

from typing import Final

from missionbench.executor.base import (
    ExecutionRequest,
    ExecutionResult,
    FailureReason,
    PredictionSource,
)
from missionbench.schemas.configuration import Placement
from missionbench.schemas.profile import ProfileCatalog

__all__ = ["DEFAULT_REMOTE_TIMEOUT_S", "ProfileExecutor", "remote_latency_s", "transfer_time_s"]

#: How long the vehicle waits before abandoning a remote execution it cannot complete.
#: A V1 simplification: one flat timeout rather than a modelled retry policy. The episode
#: continues afterwards -- a failed inference is never a termination condition.
DEFAULT_REMOTE_TIMEOUT_S: Final = 2.0

_MS_PER_S: Final = 1000.0
_BITS_PER_MEGABYTE: Final = 8.0


def transfer_time_s(size_mb: float, bandwidth_mbps: float) -> float:
    """Seconds to move ``size_mb`` megabytes over a ``bandwidth_mbps`` megabit link.

    Megabytes to megabits is the factor of 8. Zero bandwidth is not handled here -- callers
    must decide what an unusable link means, and for V1 that decision is "the execution
    fails", not "the transfer takes forever".
    """
    return _BITS_PER_MEGABYTE * size_mb / bandwidth_mbps


def remote_latency_s(
    *,
    upload_mb: float,
    download_mb: float,
    server_latency_ms: float,
    bandwidth_mbps: float,
    rtt_ms: float,
) -> float:
    """End-to-end latency of a remote execution."""
    return (
        transfer_time_s(upload_mb, bandwidth_mbps)
        + rtt_ms / _MS_PER_S
        + server_latency_ms / _MS_PER_S
        + transfer_time_s(download_mb, bandwidth_mbps)
    )


class ProfileExecutor:
    """Synthesises execution results from profiled costs and the observed network."""

    #: Provenance, equal to this backend's registry name.
    executor_id: Final = "profile"

    __slots__ = ("_predictions", "_profiles", "_remote_timeout_s")

    def __init__(
        self,
        profiles: ProfileCatalog,
        *,
        predictions: PredictionSource | None = None,
        remote_timeout_s: float = DEFAULT_REMOTE_TIMEOUT_S,
    ) -> None:
        """Args:
        profiles: Per-configuration costs for the platform this episode runs on.
        predictions: Supplies prediction payloads. ``None`` runs the executor in
            resource-only mode, which is what the simulator tests use -- costs and
            failures are exercised without any task being involved.
        remote_timeout_s: Latency charged when remote execution cannot proceed.
        """
        self._profiles = profiles
        self._predictions = predictions
        self._remote_timeout_s = remote_timeout_s

    def execute(self, request: ExecutionRequest) -> ExecutionResult:
        profile = self._profiles.get(request.configuration.config_id)

        if request.configuration.strategy.placement is Placement.LOCAL:
            return ExecutionResult(
                success=True,
                latency_s=profile.compute_latency_ms / _MS_PER_S,
                onboard_energy_j=profile.onboard_energy_j,
                prediction=self._prediction_for(request),
                metadata={"placement": Placement.LOCAL.value},
            )

        if request.network.is_disconnected:
            # Nothing reaches the server, so no bytes are counted -- but the vehicle still
            # captured and encoded the frame before discovering that, and it still waited.
            return ExecutionResult(
                success=False,
                latency_s=self._remote_timeout_s,
                onboard_energy_j=profile.onboard_energy_j,
                failure_reason=FailureReason.NETWORK_UNAVAILABLE,
                metadata={
                    "placement": Placement.REMOTE.value,
                    "bandwidth_mbps": request.network.bandwidth_mbps,
                    "timeout_s": self._remote_timeout_s,
                },
            )

        latency_s = remote_latency_s(
            upload_mb=profile.upload_mb,
            download_mb=profile.download_mb,
            server_latency_ms=profile.compute_latency_ms,
            bandwidth_mbps=request.network.bandwidth_mbps,
            rtt_ms=request.network.rtt_ms,
        )
        return ExecutionResult(
            success=True,
            latency_s=latency_s,
            onboard_energy_j=profile.onboard_energy_j,
            upload_mb=profile.upload_mb,
            download_mb=profile.download_mb,
            prediction=self._prediction_for(request),
            metadata={
                "placement": Placement.REMOTE.value,
                "bandwidth_mbps": request.network.bandwidth_mbps,
                "rtt_ms": request.network.rtt_ms,
                # Recorded, not modelled: V1 applies no retransmission penalty, and saying
                # so in the log is better than leaving the omission implicit.
                "packet_loss_frac": request.network.packet_loss_frac,
            },
        )

    def _prediction_for(self, request: ExecutionRequest):
        return None if self._predictions is None else self._predictions.prediction_for(request)
