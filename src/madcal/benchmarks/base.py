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

import random
import re
import string
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
    OPEN = "open"


@dataclass(frozen=True, slots=True)
class Question:
    id: str
    benchmark: str
    body: str
    answer: str
    task_format: TaskFormat
    choices: tuple[str, ...] | None = None
    aliases: tuple[str, ...] = ()
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
        if self.task_format is TaskFormat.OPEN and self.answer not in self.aliases:
            raise ValueError("an open question's answer must be one of its aliases")

        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class Benchmark(Protocol):
    name: str
    task_format: TaskFormat

    def load(self, limit: int | None = None, seed: int | None = None) -> Sequence[Question]: ...

    def exemplars(self, question: Question, n_shots: int) -> Sequence[Question]: ...

    def answer_instruction(self, question: Question) -> str: ...

    def extract_answer(self, response: str, n_choices: int | None) -> str | None: ...

    def answer_slot(
        self, response: str, n_choices: int | None
    ) -> tuple[str, str, tuple[str, ...]] | None: ...

    def grade(self, question: Question, extracted: str) -> bool: ...


benchmark_registry: Registry[type[Benchmark]] = Registry("benchmark")


def register_benchmark(name: str):
    return benchmark_registry.register(name)


def sample_questions(
    questions: Sequence[Question], limit: int | None, seed: int | None
) -> list[Question]:
    """Return all questions, the first `limit`, or a seeded sample of `limit` sorted by id."""
    if limit is None or limit >= len(questions):
        return list(questions)
    if seed is None:
        return list(questions[:limit])
    return sorted(random.Random(seed).sample(list(questions), limit), key=lambda q: q.id)


def normalize_answer(text: str) -> str:
    """Return `text` normalised as the official TriviaQA evaluation script does."""
    text = text.replace("_", " ").lower()
    text = "".join(" " if char in _PUNCTUATION else char for char in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


_PUNCTUATION = frozenset(string.punctuation + "\u2018\u2019\u00b4`")


def answer_letter(index: int) -> str:
    if index < 0 or index >= len(LETTERS):
        raise ValueError(f"Index must be between 0 and {len(LETTERS) - 1}, got {index}")
    return LETTERS[index]


def choice_answer_instruction(n_choices: int) -> str:
    """Return the instruction asking for a final choice letter among `n_choices` options."""
    letters = ", ".join(_choice_letters(n_choices))
    return (
        "Think step by step, then state your final answer on its own line as 'Answer: X', "
        f"where X is one of {letters}."
    )


def direct_answer_instruction(n_choices: int) -> str:
    """Return the instruction asking for the choice letter alone, with no reasoning."""
    letters = ", ".join(_choice_letters(n_choices))
    return (
        f"Respond with only your final answer on a single line as 'Answer: X', where X is one "
        f"of {letters}. Do not explain or reason."
    )


def extract_choice_letter(response: str, n_choices: int) -> str | None:
    """Return the letter of the last line consisting only of `Answer: X`, or None."""
    match = _last_choice_match(response, n_choices)
    return None if match is None else match.group(1).upper()


def choice_answer_slot(response: str, n_choices: int) -> tuple[str, str, tuple[str, ...]] | None:
    """Return the text before the answer letter, the lead-in to it, and the candidate letters."""
    match = _last_choice_match(response, n_choices)
    if match is None:
        return None
    before = response[: match.start(1)]
    context = before.rstrip(" \t")
    return context, before[len(context) :], tuple(_choice_letters(n_choices))


def _last_choice_match(response: str, n_choices: int) -> re.Match[str] | None:
    matches = list(_choice_pattern(n_choices).finditer(response))
    return matches[-1] if matches else None


@cache
def _choice_pattern(n_choices: int) -> re.Pattern[str]:
    letters = _choice_letters(n_choices)
    bold = r"(?:\*\*)?"
    fence = r"`{0,3}"
    space = r"[ \t]*"
    letter = rf"\$?([{letters}])\$?"
    return re.compile(
        rf"(?im)^{space}{fence}{bold}(?:Final{space})?Answer{space}:{space}{bold}{space}{letter}"
        rf"{space}{bold}{fence}[ \t\r]*$"
    )


def _choice_letters(n_choices: int) -> str:
    if not 2 <= n_choices <= len(LETTERS):
        raise ValueError(f"n_choices must be in 2..{len(LETTERS)}, got {n_choices}")
    return LETTERS[:n_choices]
