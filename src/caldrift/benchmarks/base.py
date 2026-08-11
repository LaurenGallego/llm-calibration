"""Benchmark protocol, registry, and the shared `Question` model.

Contracts implementations must honour, since neither is expressible in a signature:

- `load` sampling is deterministic given `seed`, so reruns stay resumable.
- `extract_answer` returns `None` when nothing can be parsed -- never a guess,
  never a placeholder. An unparseable response is a different event from a wrong
  answer, and collapsing the two changes both accuracy and ECE.
- `grade` receives the already-extracted answer, not the raw response.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from caldrift.registry import Registry

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TaskFormat(StrEnum):
    MCQ = "mcq"
    SHORT_FACTOID = "short_factoid"
    MULTI_HOP = "multi_hop"
    MATH = "math"
    CODE = "code"


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    benchmark: str
    body: str
    answer: str
    task_format: TaskFormat
    choices: tuple[str, ...] | None = None
    metadata: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if not self.id or not self.body:
            raise ValueError("Question id and body cannot be empty")
        if self.choices is not None and len(self.choices) < 2:
            raise ValueError("Question choices must have at least 2 options")
        if self.choices is not None and self.answer not in LETTERS[: len(self.choices)]:
            raise ValueError(
                f"Question answer must be one of {LETTERS[: len(self.choices)]}, got {self.answer}"
            )
        if self.choices is None and not self.answer:
            raise ValueError("Question answer cannot be empty when choices are not provided")

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class Benchmark(Protocol):
    name: str
    task_format: TaskFormat

    def load(self, limit: int | None = None, seed: int | None = None) -> Sequence[Question]: ...

    def extract_answer(self, response: str) -> str | None: ...

    def grade(self, question: Question, extracted: str) -> bool: ...


benchmark_registry: Registry[type[Benchmark]] = Registry("benchmark")


def register_benchmark(name: str):
    return benchmark_registry.register(name)


def answer_letter(index: int) -> str:
    if index < 0 or index >= len(LETTERS):
        raise ValueError(f"Index must be between 0 and {len(LETTERS) - 1}, got {index}")
    return LETTERS[index]
