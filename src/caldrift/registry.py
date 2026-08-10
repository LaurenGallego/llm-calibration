"""A minimal name -> object registry, used by every pluggable package.

Registration is by decorator so that `orchestration/` never hardcodes a list of
signals, benchmarks, or metrics -- it iterates the registry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator


class Registry[T]:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator: `@registry.register("ece")`."""

        def decorator(obj: T) -> T:
            if name in self._items:
                raise ValueError(f"{self._kind} {name!r} is already registered")
            self._items[name] = obj
            return obj

        return decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"unknown {self._kind} {name!r}; registered: {known}") from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return iter(sorted(self._items.items()))

    def __len__(self) -> int:
        return len(self._items)
