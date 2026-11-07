"""Physical sensor/environment identity, independent of policy and mission intent."""

import math
from dataclasses import dataclass, field
from pathlib import Path

from experiments.v4_benchmark.scenario import Scenario, digest, load_scenario


@dataclass(frozen=True)
class Environment:
    width_px: int = 960
    height_px: int = 720
    ambient: float = 0.5
    sunlight: float = 0.9
    fog_density: float = 0.0
    rgb_noise_stddev: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if (
            type(self.width_px) is not int
            or type(self.height_px) is not int
            or not 320 <= self.width_px <= 1920
            or not 240 <= self.height_px <= 1440
            or self.width_px * 3 != self.height_px * 4
        ):
            raise ValueError("camera must be bounded 4:3; projection calibration assumes HFOV=1.2")
        for name in ("ambient", "sunlight", "fog_density", "rgb_noise_stddev"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError("environment values must be finite numbers")
            if not 0 <= value <= (0.1 if name in ("fog_density", "rgb_noise_stddev") else 1):
                raise ValueError("environment value out of range")
        if type(self.seed) is not int or not 0 <= self.seed <= 2147483647:
            raise ValueError("invalid seed")


@dataclass(frozen=True)
class PhysicalScenario(Scenario):
    environment: Environment = field(default_factory=Environment)

    @property
    def recording_identity(self):
        return digest(
            dict(
                base=super().recording_identity,
                environment=vars(self.environment),
                builder="routing_mission_v1",
            )
        )


def load(path, environment):
    base = load_scenario(Path(path))
    return PhysicalScenario(**vars(base), environment=Environment(**environment))
