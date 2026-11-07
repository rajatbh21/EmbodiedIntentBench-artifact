"""Inference configuration identity and the configuration catalog.

A configuration says *what* to run and *how it is arranged* -- never how fast it is. Its
measured or simulated cost lives in a separate per-platform profile, so that connecting a
real model later changes profiles without touching configuration identity.

Two rules that the rest of the benchmark depends on:

- **``config_id`` is an identifier only.** No component may infer behaviour by parsing it.
  A configuration named ``CFG_LOCAL_LIGHT`` is local because ``strategy.placement`` says
  so, and for no other reason.
- **Power mode is not part of a configuration in V1.** It is fixed by the episode.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
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
    "ConfigCatalog",
    "Configuration",
    "ParameterValue",
    "Placement",
    "Precision",
    "Strategy",
    "load_config_catalog",
]

ParameterValue = str | int | float | bool | None


class Placement(StrEnum):
    """Where inference runs."""

    LOCAL = "local"
    REMOTE = "remote"


class Precision(StrEnum):
    """Numeric precision of the deployed model."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


_STRATEGY_FIELDS: Final = ("placement", "precision", "input_compression", "parameters")
_CONFIG_FIELDS: Final = ("config_id", "model_id", "strategy")
_CATALOG_FIELDS: Final = ("catalog_id", "configs")


@dataclass(frozen=True, slots=True)
class Strategy:
    """How a configuration is executed.

    ``placement`` and ``precision`` are the typed V1 dimensions. ``parameters`` is the
    controlled extension point for future strategy dimensions -- pruning ratio, split
    point, input resolution, frame sampling rate, early-exit index, accelerator -- so that
    adding one does not scatter new fields through the simulator. Values are JSON scalars,
    which keeps a parameter inspectable metadata rather than arbitrary structure.

    A dimension graduates from ``parameters`` to a typed field once the benchmark core
    actually reasons about it.
    """

    placement: Placement
    precision: Precision
    input_compression: str | None = None
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Configurations are policy-visible. A plain dict here would let a policy mutate
        # the catalog it was handed, so the map is made read-only on construction.
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    @property
    def is_remote(self) -> bool:
        return self.placement is Placement.REMOTE


@dataclass(frozen=True, slots=True)
class Configuration:
    """One selectable inference configuration."""

    config_id: str
    model_id: str
    strategy: Strategy


class ConfigCatalog:
    """An immutable, ordered collection of configurations addressed by ID.

    Ordered by declaration so that iteration is deterministic; a policy that scans the
    catalog must not depend on dictionary insertion luck differing between runs.
    """

    __slots__ = ("_by_id", "_catalog_id")

    def __init__(self, configs: Iterable[Configuration], *, catalog_id: str = "") -> None:
        by_id: dict[str, Configuration] = {}
        for config in configs:
            if config.config_id in by_id:
                raise SchemaValidationError(
                    f"configuration catalog contains duplicate config_id {config.config_id!r}"
                )
            by_id[config.config_id] = config
        self._by_id = MappingProxyType(by_id)
        self._catalog_id = catalog_id

    @property
    def catalog_id(self) -> str:
        return self._catalog_id

    def __contains__(self, config_id: object) -> bool:
        return config_id in self._by_id

    def __iter__(self) -> Iterator[Configuration]:
        return iter(self._by_id.values())

    def __len__(self) -> int:
        return len(self._by_id)

    def __repr__(self) -> str:
        return f"ConfigCatalog(catalog_id={self._catalog_id!r}, config_ids={list(self._by_id)})"

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def get(self, config_id: str) -> Configuration:
        """Return the configuration with this ID.

        Raises:
            KeyError: No such configuration. Callers validating a policy action should
                use ``config_id in catalog`` instead of catching this.
        """
        try:
            return self._by_id[config_id]
        except KeyError:
            raise KeyError(
                f"unknown config_id {config_id!r}; catalog contains {list(self._by_id)}"
            ) from None

    def subset(self, config_ids: Iterable[str]) -> ConfigCatalog:
        """Return a catalog restricted to ``config_ids``, preserving this catalog's order.

        The runner uses this to hand a policy exactly the episode's allowed pool, so a
        policy cannot select -- or even see -- a configuration the episode disallows.
        """
        requested = tuple(config_ids)
        missing = [config_id for config_id in requested if config_id not in self._by_id]
        if missing:
            raise SchemaValidationError(
                f"cannot build catalog subset: unknown config_id(s) {missing}; "
                f"catalog contains {list(self._by_id)}"
            )
        selected = set(requested)
        return ConfigCatalog(
            (config for config in self._by_id.values() if config.config_id in selected),
            catalog_id=self._catalog_id,
        )


def load_config_catalog(path: Path) -> ConfigCatalog:
    """Load and validate a configuration catalog file."""
    reader = open_document(path, document_type="ConfigCatalog", allowed_fields=_CATALOG_FIELDS)
    configs = [
        Configuration(
            config_id=entry.get_str("config_id"),
            model_id=entry.get_str("model_id"),
            strategy=_read_strategy(entry.get_object("strategy", allowed_fields=_STRATEGY_FIELDS)),
        )
        for entry in reader.get_object_list("configs", allowed_fields=_CONFIG_FIELDS)
    ]
    return ConfigCatalog(configs, catalog_id=reader.get_optional_str("catalog_id") or "")


def _read_strategy(reader: DocumentReader) -> Strategy:
    return Strategy(
        placement=reader.get_enum("placement", Placement),
        precision=reader.get_enum("precision", Precision),
        input_compression=reader.get_optional_str("input_compression"),
        parameters=reader.get_scalar_mapping("parameters"),
    )
