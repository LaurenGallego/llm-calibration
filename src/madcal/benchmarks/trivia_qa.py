"""TriviaQA, closed-book short-answer questions graded by alias match."""

import json
from collections.abc import Sequence
from pathlib import Path

from madcal.benchmarks.base import (
    Question,
    TaskFormat,
    extract_short_answer,
    normalize_answer,
    register_benchmark,
    sample_questions,
)

HF_DATASET = "mandarjoshi/trivia_qa"
HF_CONFIG = "rc.nocontext"
REQUIRED_FIELDS = ("question_id", "question", "answer")
ANSWER_INSTRUCTION = (
    "Respond with only the exact answer, as a single short phrase on a single line. "
    "Do not explain or reason."
)


@register_benchmark("trivia_qa")
class TriviaQA:
    name = "trivia_qa"
    task_format = TaskFormat.OPEN

    def __init__(
        self,
        source: Path | None = None,
        split: str = "validation",
        revision: str | None = None,
        exemplar_source: Path | None = None,
        exemplar_split: str = "train",
    ) -> None:
        self.source = source
        self.split = split
        self.revision = revision
        self.exemplar_source = exemplar_source
        self.exemplar_split = exemplar_split
        self._questions: list[Question] | None = None
        self._exemplars: list[Question] | None = None
        self.conflicting_ids: dict[str, tuple[str, ...]] = {}

    def load(self, limit: int | None = None, seed: int | None = None) -> Sequence[Question]:
        if self._questions is None:
            questions = self._build(self.source, self.split)
            self._questions = sorted(questions, key=lambda q: q.id)
        return sample_questions(self._questions, limit, seed)

    def exemplars(self, question: Question, n_shots: int) -> Sequence[Question]:
        if n_shots < 0:
            raise ValueError(f"n_shots must be non-negative, got {n_shots}")
        if n_shots == 0:
            return ()
        if self._exemplars is None:
            self._exemplars = self._build(self.exemplar_source, self.exemplar_split)
        if len(self._exemplars) < n_shots:
            raise ValueError(
                f"TriviaQA {self.exemplar_split!r} has {len(self._exemplars)} exemplars, "
                f"but n_shots={n_shots} were requested"
            )
        return self._exemplars[:n_shots]

    def answer_instruction(self, question: Question) -> str:
        return ANSWER_INSTRUCTION

    def extract_answer(self, response: str, n_choices: int | None) -> str | None:
        return extract_short_answer(response)

    def answer_slot(
        self, response: str, n_choices: int | None
    ) -> tuple[str, str, tuple[str, ...]] | None:
        return None

    def grade(self, question: Question, extracted: str) -> bool:
        if not isinstance(extracted, str):
            raise TypeError(f"grade expects an extracted answer string, got {extracted!r}")
        gold = {normalize_answer(alias) for alias in question.aliases}
        return normalize_answer(extracted) in gold

    def _build(self, source: Path | None, split: str) -> list[Question]:
        rows = _read_jsonl(source) if source is not None else self._read_hub(split)
        copies: dict[str, list[Question]] = {}
        for row in rows:
            question = self._to_question(row, split)
            copies.setdefault(question.id, []).append(question)
        conflicting = {
            question_id
            for question_id, versions in copies.items()
            if len({(q.body, q.answer, frozenset(q.aliases)) for q in versions}) > 1
        }
        self.conflicting_ids[split] = tuple(sorted(conflicting))
        return [
            versions[0]
            for question_id, versions in copies.items()
            if question_id not in conflicting
        ]

    def _read_hub(self, split: str) -> list[dict]:
        from datasets import load_dataset

        dataset = load_dataset(HF_DATASET, HF_CONFIG, split=split, revision=self.revision)
        return [dict(row) for row in dataset]

    def _to_question(self, raw: dict, split: str) -> Question:
        missing = [name for name in REQUIRED_FIELDS if name not in raw]
        if missing:
            raise ValueError(f"TriviaQA row is missing fields {missing}: {raw}")
        answer = raw["answer"]
        value = answer.get("value")
        if not value:
            raise ValueError(f"TriviaQA question {raw['question_id']!r} has no answer value")
        return Question(
            id=f"{self.name}/{split}/{raw['question_id']}",
            benchmark=self.name,
            body=raw["question"],
            answer=value,
            task_format=self.task_format,
            aliases=tuple(dict.fromkeys([value, *answer.get("aliases", ())])),
            metadata={"split": split},
        )


def _read_jsonl(source: Path) -> list[dict]:
    with source.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]
