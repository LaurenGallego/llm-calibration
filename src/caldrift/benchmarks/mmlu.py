import json
import random
import re
from collections.abc import Sequence
from pathlib import Path

from caldrift.benchmarks.base import (
    Question,
    TaskFormat,
    answer_letter,
    register_benchmark,
)

HF_DATASET = "cais/mmlu"
REQUIRED_FIELDS = ("question", "subject", "choices", "answer")

# OpenAI simple-evals' multiple-choice pattern. [ \t] rather than \s keeps the
# letter on the same line as "Answer:", and \$? tolerates LaTeX-wrapped letters.
ANSWER_PATTERN = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?")


@register_benchmark("mmlu")
class MMLU:
    name = "mmlu"
    task_format = TaskFormat.MCQ

    def __init__(
        self,
        subjects: Sequence[str] | None = None,
        source: Path | None = None,
        split: str = "test",
        revision: str | None = None,
    ) -> None:
        self.subjects = tuple(subjects) if subjects is not None else None
        self.source = source
        self.split = split
        self.revision = revision
        self._questions: list[Question] | None = None

    def load(self, limit: int | None = None, seed: int | None = None) -> Sequence[Question]:
        questions = self._all_questions()
        if limit is None or limit >= len(questions):
            return list(questions)
        if seed is None:
            return questions[:limit]
        return sorted(random.Random(seed).sample(questions, limit), key=lambda q: q.id)

    def extract_answer(self, response: str) -> str | None:
        matches = ANSWER_PATTERN.findall(response)
        if not matches:
            return None
        return matches[-1].upper()

    def grade(self, question: Question, extracted: str) -> bool:
        if not isinstance(extracted, str):
            raise TypeError(f"grade expects an extracted answer string, got {extracted!r}")
        return extracted.strip().upper() == question.answer

    def _all_questions(self) -> list[Question]:
        if self._questions is None:
            counters: dict[str, int] = {}
            questions = []
            for raw in self._load_raw():
                subject = raw["subject"]
                index = counters.get(subject, 0)
                counters[subject] = index + 1
                questions.append(self._to_question(raw, index))

            if self.subjects is not None:
                missing = sorted(set(self.subjects) - counters.keys())
                if missing:
                    raise ValueError(f"no MMLU questions found for subjects: {missing}")

            self._questions = sorted(questions, key=lambda q: q.id)
        return self._questions

    def _load_raw(self) -> list[dict]:
        rows = self._read_jsonl() if self.source is not None else self._read_hub()
        for row in rows:
            missing = [f for f in REQUIRED_FIELDS if f not in row]
            if missing:
                raise ValueError(f"MMLU row is missing fields {missing}: {row}")
        return rows

    def _read_jsonl(self) -> list[dict]:
        assert self.source is not None
        with self.source.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if self.subjects is None:
            return rows
        return [row for row in rows if row.get("subject") in self.subjects]

    def _read_hub(self) -> list[dict]:
        from datasets import load_dataset

        configs = self.subjects if self.subjects is not None else ("all",)
        rows: list[dict] = []
        for config in configs:
            dataset = load_dataset(HF_DATASET, config, split=self.split, revision=self.revision)
            rows.extend(dict(row) for row in dataset)
        return rows

    def _to_question(self, raw: dict, index: int) -> Question:
        answer = raw["answer"]
        if not isinstance(answer, int):
            raise ValueError(f"MMLU answer must be a choice index, got {answer!r}")
        subject = raw["subject"]
        return Question(
            id=f"{self.name}/{subject}/{self.split}/{index}",
            benchmark=self.name,
            body=raw["question"],
            answer=answer_letter(answer),
            task_format=self.task_format,
            choices=tuple(raw["choices"]),
            metadata={"subject": subject, "split": self.split},
        )
