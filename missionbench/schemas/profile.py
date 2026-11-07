"""Per-platform configuration profiles, and the public view a policy may be shown.

Configuration *identity* lives in ``configuration.py``; what a configuration **costs** lives
here, keyed by platform. The split is what lets the same catalog fly on different hardware,
and what lets a real measured profile replace a synthetic one without touching any
configuration, policy, or simulator code.

Two views of the same data
--------------------------
- :class:`ConfigurationProfile` is the simulator's view: exact costs, used by executors.
- :class:`PublicProfile` is the policy's view: rounded expectations plus an ordinal quality
  tier, never the exact numbers the executor will produce.

The distinction matters. A policy handed exact profiles could compute the optimal schedule
in advance, which turns a benchmark about *adapting under uncertainty* into a planning
exercise. A policy handed nothing cannot reason about quality at all, and would have to
infer it by parsing configuration IDs -- which the architecture forbids for good reason.
The public view is the middle: enough to reason, not enough to solve.

Visibility is an explicit benchmark setting (:meth:`PublicProfileView.hidden`), not an
accidental import.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final

from missionbench.schemas.loading import (
    DocumentReader,
    SchemaValidationError,
    open_document,
)

__all__ = [
    "ConfigurationProfile",
    "ProfileCatalog",
    "PublicProfile",
    "PublicProfileView",
    "QualityTier",
    "load_profile_catalog",
]


class QualityTier(StrEnum):
    """Coarse, ordinal expectation of a configuration's output quality.

    Ordinal on purpose. A policy needs to know that one configuration is more accurate than
    another to reason about a quality threshold at all; it does not need -- and must not be
    given -- a number close enough to the ground-truth score to plan against.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def rank(self) -> int:
        """0, 1, 2 -- so policies can compare tiers without hardcoding the names."""
        return _TIER_RANKS[self]


_TIER_RANKS: Final[dict[QualityTier, int]] = {
    QualityTier.LOW: 0,
    QualityTier.MEDIUM: 1,
    QualityTier.HIGH: 2,
}

_CATALOG_FIELDS: Final = ("profile_id", "platform_id", "profiles")
_PROFILE_FIELDS: Final = (
    "compute_latency_ms",
    "onboard_energy_j",
    "upload_mb",
    "download_mb",
    "quality_tier",
)


@dataclass(frozen=True, slots=True)
class ConfigurationProfile:
    """What one configuration costs on one platform."""

    #: Time spent computing, wherever that happens: on-vehicle inference for a local
    #: configuration, server-side inference for a remote one. It is *not* the end-to-end
    #: latency of a remote configuration -- transfer and RTT depend on the network at the
    #: moment of execution, so the executor adds them.
    compute_latency_ms: float
    #: Energy drawn from the vehicle battery. Full inference locally; the smaller
    #: capture-and-encode cost remotely.
    onboard_energy_j: float
    upload_mb: float
    download_mb: float
    quality_tier: QualityTier

    @property
    def communication_mb(self) -> float:
        return self.upload_mb + self.download_mb


class ProfileCatalog:
    """Configuration profiles for one platform."""

    __slots__ = ("_platform_id", "_profile_id", "_profiles")

    def __init__(
        self,
        profiles: Mapping[str, ConfigurationProfile],
        *,
        platform_id: str,
        profile_id: str = "",
    ) -> None:
        self._profiles = MappingProxyType(dict(profiles))
        self._platform_id = platform_id
        self._profile_id = profile_id

    @property
    def platform_id(self) -> str:
        return self._platform_id

    @property
    def profile_id(self) -> str:
        return self._profile_id

    def __contains__(self, config_id: object) -> bool:
        return config_id in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def __iter__(self) -> Iterator[str]:
        return iter(self._profiles)

    def get(self, config_id: str) -> ConfigurationProfile:
        try:
            return self._profiles[config_id]
        except KeyError:
            raise SchemaValidationError(
                f"platform {self._platform_id!r} has no profile for configuration "
                f"{config_id!r}; profiled configurations are {list(self._profiles)}"
            ) from None

    def public_view(self) -> PublicProfileView:
        """Derive the policy-visible view of these profiles."""
        return PublicProfileView(
            {
                config_id: PublicProfile(
                    config_id=config_id,
                    expected_latency_ms=profile.compute_latency_ms,
                    expected_upload_mb=profile.upload_mb,
                    quality_tier=profile.quality_tier,
                )
                for config_id, profile in self._profiles.items()
            }
        )


@dataclass(frozen=True, slots=True)
class PublicProfile:
    """What a policy is told about a configuration's cost and quality.

    ``expected_latency_ms`` is the *compute* latency only. For a remote configuration the
    real end-to-end latency also depends on bandwidth and RTT, which the policy already
    observes in its ``RuntimeState`` -- so it can estimate the total itself, and gets no
    advance knowledge of a network it has not yet seen.
    """

    config_id: str
    expected_latency_ms: float
    expected_upload_mb: float
    quality_tier: QualityTier


class PublicProfileView:
    """The profile information a policy may consult, or nothing at all.

    Constructed by the composition root and injected into policies that want it. A policy
    must treat an empty view as legitimate: whether profiles are public is a property of
    the benchmark run, and a policy that crashes without them is not a valid submission.
    """

    __slots__ = ("_profiles",)

    def __init__(self, profiles: Mapping[str, PublicProfile] | None = None) -> None:
        self._profiles = MappingProxyType(dict(profiles or {}))

    @classmethod
    def hidden(cls) -> PublicProfileView:
        """An empty view, for runs where profiles are not disclosed to policies."""
        return cls()

    @property
    def is_available(self) -> bool:
        return bool(self._profiles)

    def __contains__(self, config_id: object) -> bool:
        return config_id in self._profiles

    def __len__(self) -> int:
        return len(self._profiles)

    def get(self, config_id: str) -> PublicProfile | None:
        """Return the public profile, or ``None`` if profiles are hidden or absent.

        Returns ``None`` rather than raising: "I was not told" is an ordinary situation for
        a policy, not an error, and forcing every call site into a try block would make
        profile-blind policies harder to write than they should be.
        """
        return self._profiles.get(config_id)

    def __repr__(self) -> str:
        return f"PublicProfileView(config_ids={list(self._profiles)})"


def load_profile_catalog(path: Path) -> ProfileCatalog:
    """Load and validate a per-platform configuration profile file."""
    reader = open_document(path, document_type="ProfileCatalog", allowed_fields=_CATALOG_FIELDS)

    profiles = {
        config_id: _read_profile(entry)
        for config_id, entry in reader.get_id_keyed_object("profiles", value_fields=_PROFILE_FIELDS)
    }
    if not profiles:
        raise SchemaValidationError(f"{path.name} -> ProfileCatalog: 'profiles' must not be empty")

    return ProfileCatalog(
        profiles,
        platform_id=reader.get_str("platform_id"),
        profile_id=reader.get_optional_str("profile_id") or "",
    )


def _read_profile(reader: DocumentReader) -> ConfigurationProfile:
    return ConfigurationProfile(
        compute_latency_ms=reader.get_float("compute_latency_ms", minimum=0.0),
        onboard_energy_j=reader.get_float("onboard_energy_j", minimum=0.0),
        upload_mb=reader.get_float("upload_mb", minimum=0.0),
        download_mb=reader.get_float("download_mb", minimum=0.0),
        quality_tier=reader.get_enum("quality_tier", QualityTier),
    )


def check_catalog_is_profiled(
    profiles: ProfileCatalog, *, config_ids: tuple[str, ...], platform_id: str
) -> None:
    """Raise unless every configuration the episode may select has a profile.

    Cross-document check: a missing profile would surface as a failure on whichever step
    first selected that configuration, which is a confusing way to learn that a fixture is
    incomplete.
    """
    if profiles.platform_id != platform_id:
        raise SchemaValidationError(
            f"profile catalog is for platform {profiles.platform_id!r}, "
            f"but the episode runs on {platform_id!r}"
        )
    missing = [config_id for config_id in config_ids if config_id not in profiles]
    if missing:
        raise SchemaValidationError(
            f"platform {platform_id!r} has no profile for configuration(s) {missing}; "
            f"profiled configurations are {list(profiles)}"
        )
