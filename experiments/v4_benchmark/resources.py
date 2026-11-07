"""Explicit configured costs for a single resident inference configuration.

This simulates deployment residency; Python executors may all be preloaded on the
host. Values are assumptions, never process-RSS or electrical measurements.
"""

import math
from dataclasses import asdict, dataclass

from missionbench.schemas.loading import SchemaValidationError


@dataclass(frozen=True)
class ModelResources:
    config_id: str
    memory_mb: float
    load_latency_s: float
    load_energy_j: float


@dataclass(frozen=True)
class ResourceConfig:
    memory_capacity_mb: float | None = None
    models: tuple[ModelResources, ...] = ()

    @classmethod
    def parse(cls, data, config_ids):
        if not isinstance(data, dict) or set(data) - {"memory_capacity_mb", "models"}:
            raise SchemaValidationError("unknown resource settings")
        if not data:
            return cls()
        capacity = data.get("memory_capacity_mb")
        models = data.get("models")
        if not isinstance(models, dict) or set(models) != set(config_ids):
            raise SchemaValidationError("resources must specify every configuration exactly once")

        def finite(value):
            return (
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(value)
                and value >= 0
            )

        if not finite(capacity) or capacity == 0:
            raise SchemaValidationError("memory capacity must be positive and finite")
        parsed = []
        for key, costs in models.items():
            if (
                not isinstance(costs, dict)
                or set(costs) != {"memory_mb", "load_latency_s", "load_energy_j"}
                or not all(finite(v) for v in costs.values())
            ):
                raise SchemaValidationError(
                    "model resources need nonnegative finite memory/load costs"
                )
            parsed.append(ModelResources(key, **costs))
        return cls(capacity, tuple(parsed))

    def for_config(self, key):
        return next((m for m in self.models if m.config_id == key), ModelResources(key, 0, 0, 0))

    def to_dict(self):
        if not self.models:
            return {}
        return dict(
            memory_capacity_mb=self.memory_capacity_mb,
            models={
                m.config_id: {k: v for k, v in asdict(m).items() if k != "config_id"}
                for m in self.models
            },
        )
