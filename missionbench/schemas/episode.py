"""Initial episode state: everything an episode starts from.

An :class:`Episode` is simulator-internal. It is never handed to a policy -- it names the
network trace, which would reveal future conditions the policy is supposed to adapt to
without foreknowledge. The policy sees a ``RuntimeState`` built fresh at each step.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from missionbench.schemas.loading import SchemaValidationError, open_document

__all__ = ["Episode", "load_episode"]

_FIELDS: Final = (
    "episode_id",
    "platform_id",
    "path_id",
    "frame_stream_id",
    "initial_altitude_m",
    "velocity_mps",
    "initial_battery_frac",
    "power_mode",
    "network_trace_id",
    "allowed_config_ids",
    "initial_config_id",
    "fallback_config_id",
    "seed",
)


@dataclass(frozen=True, slots=True)
class Episode:
    """The initial conditions of one benchmark episode.

    ``seed`` makes an episode reproducible: the same fixtures, seed, and policy must
    produce byte-identical metrics.
    """

    episode_id: str
    platform_id: str
    path_id: str
    frame_stream_id: str
    initial_altitude_m: float
    velocity_mps: float
    initial_battery_frac: float
    power_mode: str
    network_trace_id: str
    allowed_config_ids: tuple[str, ...]
    initial_config_id: str | None
    seed: int
    #: Optional per-episode safe fallback, used when a policy's action is invalid and there is
    #: no current configuration to keep. It replaces the benchmark-wide legacy assumption that
    #: a ``CFG_LOCAL_LIGHT`` exists, so a data root built with its own configuration names can
    #: declare its own fallback. Must be in ``allowed_config_ids`` if present.
    fallback_config_id: str | None = None


def load_episode(path: Path) -> Episode:
    """Load and validate an episode specification file."""
    reader = open_document(path, document_type="Episode", allowed_fields=_FIELDS)

    allowed_config_ids = reader.get_str_tuple("allowed_config_ids")
    initial_config_id = reader.get_optional_str("initial_config_id")
    if initial_config_id is not None and initial_config_id not in allowed_config_ids:
        raise SchemaValidationError(
            f"{reader.context}: initial_config_id {initial_config_id!r} is not in "
            f"allowed_config_ids {list(allowed_config_ids)}"
        )
    fallback_config_id = reader.get_optional_str("fallback_config_id")
    if fallback_config_id is not None and fallback_config_id not in allowed_config_ids:
        raise SchemaValidationError(
            f"{reader.context}: fallback_config_id {fallback_config_id!r} is not in "
            f"allowed_config_ids {list(allowed_config_ids)}"
        )

    return Episode(
        episode_id=reader.get_str("episode_id"),
        platform_id=reader.get_str("platform_id"),
        path_id=reader.get_str("path_id"),
        frame_stream_id=reader.get_str("frame_stream_id"),
        initial_altitude_m=reader.get_float("initial_altitude_m", minimum=0.0),
        velocity_mps=reader.get_float("velocity_mps", exclusive_minimum=0.0),
        initial_battery_frac=reader.get_fraction("initial_battery_frac"),
        power_mode=reader.get_str("power_mode"),
        network_trace_id=reader.get_str("network_trace_id"),
        allowed_config_ids=allowed_config_ids,
        initial_config_id=initial_config_id,
        seed=reader.get_int("seed"),
        fallback_config_id=fallback_config_id,
    )
