"""The policy interface, and the estimates a policy is allowed to make.

A policy is the subject under evaluation. Its entire input is the contract, the
policy-visible ``RuntimeState``, and the configurations the episode allows -- plus, when the
benchmark run discloses them, public profiles injected at construction. It may not read
ground truth, the future network trace, the episode object, or the simulator.

The action is a configuration ID. A policy that returns something invalid is not an error
condition to crash on: the action validator records it and substitutes, and the invalid
action count is part of the result.

Estimation, not knowledge
-------------------------
:func:`estimated_latency_s` deliberately produces an *approximation*. A public profile
discloses upload size but not download, and compute time but not the server's actual load,
so a policy's estimate of a remote configuration's latency is close but not exact. That is
the intended epistemic position: a deployed system estimates from what it can observe. A
policy that could compute the executor's number exactly would be planning, not adapting.
"""

from __future__ import annotations

from typing import Final, Protocol

from missionbench.registry import Registry
from missionbench.schemas.configuration import ConfigCatalog, Configuration, Placement
from missionbench.schemas.contract import Contract
from missionbench.schemas.network_trace import NetworkObservation
from missionbench.schemas.profile import PublicProfile, PublicProfileView
from missionbench.schemas.runtime_state import RuntimeState

__all__ = ["Policy", "estimated_latency_s", "policy_registry"]

_MS_PER_S: Final = 1000.0
_BITS_PER_MEGABYTE: Final = 8.0


class Policy(Protocol):
    """Selects one configuration per decision step."""

    def select_config(
        self,
        contract: Contract,
        state: RuntimeState,
        configs: ConfigCatalog,
    ) -> str:
        """Return the ID of the configuration to run next.

        Args:
            contract: What the mission must achieve, and its hard constraints.
            state: The current policy-visible observation.
            configs: The configurations this episode allows. Already restricted, so a
                policy cannot select -- or see -- one the episode excludes.
        """
        ...


def estimated_latency_s(
    configuration: Configuration,
    profile: PublicProfile | None,
    network: NetworkObservation,
) -> float | None:
    """Estimate end-to-end latency from public information alone.

    Returns ``None`` when profiles are not disclosed, or when a remote configuration faces a
    dead link -- in the latter case there is no latency to estimate, because the execution
    will not happen at all.

    The download leg is not disclosed and so is not modelled; on the shipped fixtures it is
    a small fraction of the upload, and a policy has to work with what it is told.
    """
    if profile is None:
        return None
    compute_s = profile.expected_latency_ms / _MS_PER_S
    if configuration.strategy.placement is Placement.LOCAL:
        return compute_s
    if network.is_disconnected:
        return None
    upload_s = _BITS_PER_MEGABYTE * profile.expected_upload_mb / network.bandwidth_mbps
    return upload_s + network.rtt_ms / _MS_PER_S + compute_s


def public_profile_for(
    profiles: PublicProfileView, configuration: Configuration
) -> PublicProfile | None:
    """Look up a configuration's public profile, or ``None`` if profiles are hidden."""
    return profiles.get(configuration.config_id)


#: Name-to-policy mapping for the composition root, so ``--policy rule_based`` resolves
#: without the runner importing any policy. Populated in ``policies/registry.py``.
policy_registry: Registry[Policy] = Registry("policy")
