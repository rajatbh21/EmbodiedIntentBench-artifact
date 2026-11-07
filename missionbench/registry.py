"""A small explicit registry, so new components need no edit to the benchmark core.

Deliberately not a plugin framework. No entry points, no import scanning, no dynamic
discovery -- those buy indirection the benchmark does not need and cost the ability to see
at a glance what is available. A registry here is a name-to-factory mapping that a module
populates at import time, and nothing more.

Arrives with executors because that is the first component type with several
implementations to choose between. Policies and tasks reuse it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

__all__ = ["Registry", "RegistryError"]

T = TypeVar("T")


class RegistryError(KeyError):
    """A registry lookup or registration failed.

    Subclasses ``KeyError`` so that a missing name still reads as a lookup failure, while
    carrying a message that lists what *is* available -- a typo in a CLI flag should say
    what to type instead.
    """

    def __str__(self) -> str:
        # KeyError.__str__ reprs its argument, which turns a helpful sentence into a
        # quoted blob. Return the message as written.
        return self.args[0] if self.args else ""


class Registry(Generic[T]):
    """Maps a name to a factory for one kind of component."""

    __slots__ = ("_factories", "_kind")

    def __init__(self, kind: str) -> None:
        """Args: kind: What is being registered, used in error messages, e.g. ``"executor"``."""
        self._kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T]) -> Callable[..., T]:
        """Register ``factory`` under ``name``, returning it so this can be used as a decorator.

        Re-registering a name is an error rather than an overwrite: silently replacing an
        implementation because two modules chose the same name would be near-impossible to
        debug from a result file that only records the name.
        """
        if name in self._factories:
            raise RegistryError(
                f"{self._kind} {name!r} is already registered; "
                "names must be unique so a result file identifies one implementation"
            )
        self._factories[name] = factory
        return factory

    def create(self, name: str, /, **kwargs: Any) -> T:
        """Construct the component registered under ``name``."""
        try:
            factory = self._factories[name]
        except KeyError:
            raise RegistryError(
                f"unknown {self._kind} {name!r}; available: {list(self.names())}"
            ) from None
        return factory(**kwargs)

    def names(self) -> tuple[str, ...]:
        """Registered names, sorted for deterministic help output."""
        return tuple(sorted(self._factories))

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __len__(self) -> int:
        return len(self._factories)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Registry(kind={self._kind!r}, names={list(self.names())})"
