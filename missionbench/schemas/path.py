"""Predefined flight path.

V1 evaluates configuration selection, not navigation, so a path is reduced to the one
property the benchmark actually consumes: how far the vehicle must travel. Combined with
the episode's fixed velocity that yields ``path_progress``, which drives the primary
termination condition.

Waypoints, altitude profiles, and turn dynamics are deliberately absent. They would imply
a flight model V1 does not have, and nothing in the decision loop would read them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from missionbench.schemas.loading import open_document

__all__ = ["PathSpec", "load_path_spec"]

_FIELDS: Final = ("path_id", "length_m", "description")


@dataclass(frozen=True, slots=True)
class PathSpec:
    """A predefined path, described by its total length."""

    path_id: str
    length_m: float
    description: str = ""


def load_path_spec(path: Path) -> PathSpec:
    """Load and validate a path specification file."""
    reader = open_document(path, document_type="PathSpec", allowed_fields=_FIELDS)
    return PathSpec(
        path_id=reader.get_str("path_id"),
        length_m=reader.get_float("length_m", exclusive_minimum=0.0),
        description=reader.get_optional_str("description") or "",
    )
