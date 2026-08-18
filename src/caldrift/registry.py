"""A minimal name -> object registry, used by every pluggable package.

Registration is by decorator so that `orchestration/` never hardcodes a list of
signals, benchmarks, or metrics -- it iterates the registry.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import cast


class Registry[T]:
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, T] = {}

    def register[U](self, name: str) -> Callable[[U], U]:
        """Decorator: `@registry.register("ece")`.

        Generic in U rather than T. Returning `Callable[[T], T]` would rebind the
        decorated name to the registry's declared type -- `@register_benchmark`
        would erase MMLU down to `type[Benchmark]`, and every constructor argument
        would look undefined to a type checker.

        U cannot be bounded by T (PEP 695 forbids a generic bound), so this no
        longer checks that the registered object fits the registry. That check
        lives in tests/test_protocol_conformance.py instead.
        """

        def decorator(obj: U) -> U:
            if name in self._items:
                raise ValueError(f"{self._kind} {name!r} is already registered")
            self._items[name] = cast(T, obj)
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
