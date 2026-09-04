"""Explicit registries for configurable runtime components."""

from collections.abc import Mapping
from typing import Generic, TypeVar

T = TypeVar("T")


class ComponentRegistry(Generic[T]):
    """Resolve a supported component by its public configuration name."""

    def __init__(self, components: Mapping[str, T], *, kind: str):
        self._components = dict(components)
        self.kind = kind

    def resolve(self, name: str) -> T:
        try:
            return self._components[name]
        except KeyError as error:
            available = ", ".join(sorted(self._components))
            raise ValueError(
                f"Unknown {self.kind} {name!r}. Available: {available}."
            ) from error
