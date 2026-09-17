"""Benchmark protocol, registry, and the shared `Question` model.

Contracts implementations must honour, since neither is expressible in a signature:

- `load` sampling is deterministic given `seed`, so reruns stay resumable.
- `extract_answer` returns `None` when nothing can be parsed -- never a guess,
  never a placeholder. An unparseable response is a different event from a wrong
  answer, and collapsing the two changes both accuracy and ECE.
- `grade` receives the already-extracted answer, not the raw response.
- `exemplars` returns the few-shot examples this benchmark's official protocol uses for
  a given question -- for MMLU, the first `n_shots` of the same subject's `dev` split.
  Which pool is canonical is benchmark-specific knowledge and belongs here rather than
  in a config field: it is part of running the benchmark as published. Raise if the pool
  is too small rather than rendering a shorter prompt than the config asked for.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from types import MappingProxyType
from typing import Protocol

from madcal.registry import Registry

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class TaskFormat(StrEnum):
    MCQ = "mcq"
    MATH = "math"


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

    def exemplars(self, question: Question, n_shots: int) -> Sequence[Question]: ...

    def answer_instruction(self, question: Question) -> str: ...

    def extract_answer(self, response: str, n_choices: int | None) -> str | None: ...

    def grade(self, question: Question, extracted: str) -> bool: ...


benchmark_registry: Registry[type[Benchmark]] = Registry("benchmark")


def register_benchmark(name: str):
    return benchmark_registry.register(name)


def answer_letter(index: int) -> str:
    if index < 0 or index >= len(LETTERS):
        raise ValueError(f"Index must be between 0 and {len(LETTERS) - 1}, got {index}")
    return LETTERS[index]


def choice_answer_instruction(n_choices: int) -> str:
    """Return the instruction asking for a final choice letter among `n_choices` options."""
    letters = ", ".join(_choice_letters(n_choices))
    return f"State your final answer on its own line as 'Answer: X', where X is one of {letters}."


def extract_choice_letter(response: str, n_choices: int) -> str | None:
    """Return the letter of the last line consisting only of `Answer: X`, or None."""
    matches = _choice_pattern(n_choices).findall(response)
    if not matches:
        return None
    return matches[-1].upper()


@cache
def _choice_pattern(n_choices: int) -> re.Pattern[str]:
    letters = _choice_letters(n_choices)
    bold = r"(?:\*\*)?"
    space = r"[ \t]*"
    letter = rf"\$?([{letters}])\$?"
    return re.compile(
        rf"(?im)^{space}{bold}Answer{space}:{space}{bold}{space}{letter}{space}{bold}[ \t\r]*$"
    )


def _choice_letters(n_choices: int) -> str:
    if not 2 <= n_choices <= len(LETTERS):
        raise ValueError(f"n_choices must be in 2..{len(LETTERS)}, got {n_choices}")
    return LETTERS[:n_choices]
