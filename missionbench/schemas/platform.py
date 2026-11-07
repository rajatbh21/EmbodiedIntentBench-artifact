"""UAV platform profile: the vehicle's energy characteristics.

Describes the platform, not any particular configuration. Per-configuration cost lives in
a separate profile file keyed by platform, so the same configuration catalog can be flown
on different hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from missionbench.schemas.loading import SchemaValidationError, open_document

__all__ = ["PlatformProfile", "check_episode_power_mode", "load_platform_profile"]

#: Joules per watt-hour, used to convert the declared battery capacity into the joule
#: budget the battery model actually spends.
JOULES_PER_WATT_HOUR: Final = 3600.0

_FIELDS: Final = (
    "platform_id",
    "battery_capacity_wh",
    "flight_power_w",
    "communication_energy_j_per_mb",
    "supported_power_modes",
)


@dataclass(frozen=True, slots=True)
class PlatformProfile:
    """Energy characteristics of one UAV platform."""

    platform_id: str
    battery_capacity_wh: float
    flight_power_w: float
    communication_energy_j_per_mb: float
    supported_power_modes: tuple[str, ...]

    @property
    def battery_capacity_j(self) -> float:
        """Battery capacity in joules, the unit the battery model works in."""
        return self.battery_capacity_wh * JOULES_PER_WATT_HOUR

    def supports_power_mode(self, power_mode: str) -> bool:
        return power_mode in self.supported_power_modes


def load_platform_profile(path: Path) -> PlatformProfile:
    """Load and validate a platform profile file."""
    reader = open_document(path, document_type="PlatformProfile", allowed_fields=_FIELDS)
    return PlatformProfile(
        platform_id=reader.get_str("platform_id"),
        battery_capacity_wh=reader.get_float("battery_capacity_wh", exclusive_minimum=0.0),
        flight_power_w=reader.get_float("flight_power_w", minimum=0.0),
        communication_energy_j_per_mb=reader.get_float(
            "communication_energy_j_per_mb", minimum=0.0
        ),
        supported_power_modes=reader.get_str_tuple("supported_power_modes"),
    )


def check_episode_power_mode(profile: PlatformProfile, *, power_mode: str) -> None:
    """Raise if an episode requests a power mode this platform does not support.

    Cross-document check: neither file can see the other at load time, so the runner
    calls this once the pair is resolved.
    """
    if not profile.supports_power_mode(power_mode):
        raise SchemaValidationError(
            f"platform {profile.platform_id!r} does not support power mode {power_mode!r}; "
            f"supported modes are {list(profile.supported_power_modes)}"
        )
