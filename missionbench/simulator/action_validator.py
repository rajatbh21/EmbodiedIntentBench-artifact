"""Validation of the configuration a policy selected.

A policy is the subject under evaluation, so it is treated as untrusted input. It may name
a configuration that does not exist, one the episode disallows, one the contract's privacy
level forbids, or something that is not a configuration ID at all. None of that may end the
run: an episode that crashed on a bad action would report nothing, when what the benchmark
wants to report is precisely that the policy behaved badly.

So an invalid action is recorded and replaced, never raised:

1. Keep the current configuration if there is one and it is still acceptable.
2. Otherwise use the configured safe fallback.

A privacy-violating selection is *blocked*, not executed and then penalised. The simulator
must not model data leaving the vehicle in violation of its contract even hypothetically;
the attempt is recorded, and the episode's privacy constraint fails on the record.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from missionbench.schemas.configuration import ConfigCatalog, Configuration, Placement
from missionbench.schemas.contract import PrivacyLevel

__all__ = [
    "DEFAULT_SAFE_FALLBACK_CONFIG_ID",
    "LEGACY_FALLBACK_CONFIG_ID",
    "TRANSMITTED_PAYLOAD_PARAMETER",
    "ActionOutcome",
    "ActionValidator",
    "TransmittedPayload",
    "ValidatedAction",
    "fallback_rejection_reason",
    "is_switch",
    "privacy_permits",
]

#: The configuration named as a *last-resort* legacy fallback, for a data root built around
#: the shipped V1 catalog. It is not required to exist: the composition root resolves a
#: fallback by explicit precedence (see ``benchmark.resolve_fallback_config_id``) and only
#: falls back to this name when it is present and usable, so a bundle with different config
#: IDs is not obliged to contain it.
LEGACY_FALLBACK_CONFIG_ID: Final = "CFG_LOCAL_LIGHT"

#: Backwards-compatible alias. Callers that constructed an ``ActionValidator`` without naming
#: a fallback used to get this default; the resolver now supplies an explicit resolved id, but
#: the constructor keeps the same default so existing direct callers are unaffected.
DEFAULT_SAFE_FALLBACK_CONFIG_ID: Final = LEGACY_FALLBACK_CONFIG_ID

#: Strategy parameter declaring what a remote configuration puts on the wire. Lives in
#: ``strategy.parameters`` because it is exactly the controlled extension point for a
#: dimension the core reasons about but has not earned a typed field: only the
#: ``features_only`` privacy level reads it today.
TRANSMITTED_PAYLOAD_PARAMETER: Final = "transmitted_payload"


class TransmittedPayload(StrEnum):
    """What a remote configuration transmits off the vehicle."""

    #: Sensor data, however compressed. Compression is not de-identification.
    RAW_INPUT = "raw_input"
    #: Derived representations from which the raw input is not intended to be recoverable.
    FEATURES = "features"


class ActionOutcome(StrEnum):
    """The result of validating one policy action."""

    VALID = "valid"
    #: Not a string at all -- the policy returned something that is not a configuration ID.
    MALFORMED = "malformed"
    #: A well-formed ID that no configuration in the catalog uses.
    UNKNOWN_CONFIG = "unknown_config"
    #: A real configuration that this episode's allowed pool excludes.
    NOT_ALLOWED = "not_allowed"
    #: A real, allowed configuration that the contract's privacy level forbids.
    PRIVACY_VIOLATION = "privacy_violation"

    @property
    def is_valid(self) -> bool:
        return self is ActionOutcome.VALID


@dataclass(frozen=True, slots=True)
class ValidatedAction:
    """What the policy asked for, what will actually run, and why they differ."""

    requested: object
    config_id: str
    outcome: ActionOutcome
    reason: str = ""

    @property
    def is_valid(self) -> bool:
        return self.outcome.is_valid

    @property
    def was_substituted(self) -> bool:
        return not self.is_valid

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_config_id": self.requested if isinstance(self.requested, str) else None,
            "requested_repr": repr(self.requested),
            "config_id": self.config_id,
            "outcome": self.outcome.value,
            "reason": self.reason,
        }


def privacy_permits(privacy_level: PrivacyLevel, configuration: Configuration) -> bool:
    """Return whether ``privacy_level`` allows running ``configuration``.

    - ``local_only``: nothing leaves the vehicle, so only local placement qualifies.
    - ``features_only``: local placement, or remote placement that declares it transmits
      features rather than raw input.
    - ``remote_allowed``: no restriction.

    A remote configuration that does not declare :data:`TRANSMITTED_PAYLOAD_PARAMETER` is
    treated as transmitting raw input. The conservative default matters: an undeclared
    payload under ``features_only`` must fail closed, or forgetting the parameter would
    quietly grant permission.
    """
    if privacy_level is PrivacyLevel.REMOTE_ALLOWED:
        return True
    if configuration.strategy.placement is Placement.LOCAL:
        return True
    if privacy_level is PrivacyLevel.LOCAL_ONLY:
        return False
    declared = configuration.strategy.parameters.get(
        TRANSMITTED_PAYLOAD_PARAMETER, TransmittedPayload.RAW_INPUT.value
    )
    return declared == TransmittedPayload.FEATURES.value


def fallback_rejection_reason(
    config_id: str,
    *,
    catalog: ConfigCatalog,
    allowed_config_ids: tuple[str, ...],
    privacy_level: PrivacyLevel,
) -> str | None:
    """Return why ``config_id`` may not serve as a fallback, or ``None`` if it may.

    A usable fallback must exist in the catalog, be in the episode's allowed pool, and be
    permitted by the contract's privacy level -- the same three structural requirements the
    validator imposes on any configuration it would run. Exposed so the composition root can
    resolve and validate a fallback *before* the episode starts, without depending on an
    ``ActionValidator`` instance or on any particular configuration name.
    """
    if config_id not in catalog:
        return f"no configuration {config_id!r} in the catalog"
    if config_id not in allowed_config_ids:
        return f"{config_id!r} is not in the episode's allowed pool {list(allowed_config_ids)}"
    if not privacy_permits(privacy_level, catalog.get(config_id)):
        return f"{config_id!r} is forbidden by privacy level {privacy_level.value!r}"
    return None


def is_switch(previous_config_id: str | None, current_config_id: str) -> bool:
    """Return whether moving to ``current_config_id`` counts as a configuration switch.

    Re-selecting the same configuration is not a switch. Starting from no configuration is
    not a switch either -- there is nothing to switch away from. Counted on the
    configuration that actually ran, so a substituted invalid action does not inflate the
    count with a change the policy never achieved.
    """
    return previous_config_id is not None and previous_config_id != current_config_id


class ActionValidator:
    """Validates policy actions against the catalog, the allowed pool, and privacy."""

    __slots__ = ("_allowed_config_ids", "_catalog", "_fallback_config_id", "_privacy_level")

    def __init__(
        self,
        *,
        catalog: ConfigCatalog,
        allowed_config_ids: tuple[str, ...],
        privacy_level: PrivacyLevel,
        fallback_config_id: str = DEFAULT_SAFE_FALLBACK_CONFIG_ID,
    ) -> None:
        self._catalog = catalog
        self._allowed_config_ids = allowed_config_ids
        self._privacy_level = privacy_level
        self._fallback_config_id = fallback_config_id
        self._check_fallback_is_usable()

    def _check_fallback_is_usable(self) -> None:
        """Reject an unusable fallback at construction.

        The fallback is the last line of defence against a misbehaving policy. Discovering
        mid-episode that it is unknown, disallowed, or itself a privacy violation would
        leave the runner with no legal action, so it is checked before the episode starts.
        """
        problem = self._rejection_reason(self._fallback_config_id)
        if problem is not None:
            _, reason = problem
            raise ValueError(
                f"fallback configuration {self._fallback_config_id!r} is unusable: {reason}. "
                "Pass fallback_config_id for episodes whose allowed pool or privacy level "
                "excludes the default."
            )

    @property
    def fallback_config_id(self) -> str:
        return self._fallback_config_id

    def _rejection_reason(self, config_id: str) -> tuple[ActionOutcome, str] | None:
        """Return why ``config_id`` may not run, or ``None`` if it may."""
        if config_id not in self._catalog:
            return (
                ActionOutcome.UNKNOWN_CONFIG,
                f"no configuration {config_id!r} in the catalog",
            )
        if config_id not in self._allowed_config_ids:
            return (
                ActionOutcome.NOT_ALLOWED,
                f"{config_id!r} is not in this episode's allowed pool "
                f"{list(self._allowed_config_ids)}",
            )
        if not privacy_permits(self._privacy_level, self._catalog.get(config_id)):
            return (
                ActionOutcome.PRIVACY_VIOLATION,
                f"{config_id!r} is forbidden by privacy level {self._privacy_level.value!r}",
            )
        return None

    def validate(self, requested: object, *, current_config_id: str | None) -> ValidatedAction:
        """Validate one action and resolve the configuration that will actually run.

        Args:
            requested: Whatever the policy returned. Not assumed to be a string.
            current_config_id: The configuration in force, or ``None`` on the first step.
        """
        if not isinstance(requested, str):
            return self._substitute(
                requested,
                ActionOutcome.MALFORMED,
                f"policy returned {type(requested).__name__}, expected a configuration ID string",
                current_config_id,
            )

        problem = self._rejection_reason(requested)
        if problem is None:
            return ValidatedAction(
                requested=requested, config_id=requested, outcome=ActionOutcome.VALID
            )

        outcome, reason = problem
        return self._substitute(requested, outcome, reason, current_config_id)

    def _substitute(
        self,
        requested: object,
        outcome: ActionOutcome,
        reason: str,
        current_config_id: str | None,
    ) -> ValidatedAction:
        """Choose the configuration to run in place of an invalid action.

        The current configuration is re-checked rather than trusted: an episode's declared
        ``initial_config_id`` never passed through validation, so it could itself be
        disallowed or privacy-violating.
        """
        if current_config_id is not None and self._rejection_reason(current_config_id) is None:
            replacement = current_config_id
            resolution = f"kept current configuration {replacement!r}"
        else:
            replacement = self._fallback_config_id
            resolution = f"used safe fallback {replacement!r}"
        return ValidatedAction(
            requested=requested,
            config_id=replacement,
            outcome=outcome,
            reason=f"{reason}; {resolution}",
        )
