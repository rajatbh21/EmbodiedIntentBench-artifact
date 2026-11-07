"""Path progress model.

V1 flies a predefined path at a fixed nominal velocity, so progress is a function of
elapsed time alone. The vehicle keeps moving regardless of what the policy selects -- a
slow inference does not pause the aircraft, it just means fewer frames get processed over
the same ground.
"""

from __future__ import annotations

from dataclasses import dataclass

from missionbench.schemas.path import PathSpec

__all__ = ["ConstantVelocityPath"]


@dataclass(frozen=True, slots=True)
class ConstantVelocityPath:
    """Progress along a fixed-length path flown at a constant velocity."""

    length_m: float
    velocity_mps: float

    @classmethod
    def from_spec(cls, spec: PathSpec, *, velocity_mps: float) -> ConstantVelocityPath:
        return cls(length_m=spec.length_m, velocity_mps=velocity_mps)

    @property
    def duration_s(self) -> float:
        """Time to fly the whole path at the nominal velocity."""
        return self.length_m / self.velocity_mps

    def distance_at(self, elapsed_s: float) -> float:
        """Distance travelled after ``elapsed_s``, capped at the path length."""
        return min(self.length_m, self.velocity_mps * max(0.0, elapsed_s))

    def progress_at(self, elapsed_s: float) -> float:
        """Fraction of the path flown, clamped to [0, 1].

        Clamped rather than allowed to exceed 1.0 so that ``path_progress`` keeps the
        meaning its schema promises. Overrun is visible in elapsed time instead.
        """
        return self.distance_at(elapsed_s) / self.length_m
