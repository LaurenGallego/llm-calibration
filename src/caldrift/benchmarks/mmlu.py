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
        exemplar_source: Path | None = None,
        exemplar_split: str = "dev",
    ) -> None:
        self.subjects = tuple(subjects) if subjects is not None else None
        self.source = source
        self.split = split
        self.revision = revision
        # `dev` is MMLU's canonical few-shot pool: exactly five per subject, and what
        # lm-evaluation-harness uses. Not a config knob -- it is part of the published
        # protocol, and varying it silently changes what the benchmark measures.
        self.exemplar_source = exemplar_source
        self.exemplar_split = exemplar_split
        self._questions: list[Question] | None = None
        self._exemplars: dict[str, tuple[Question, ...]] | None = None

    def load(self, limit: int | None = None, seed: int | None = None) -> Sequence[Question]:
        questions = self._all_questions()
        if limit is None or limit >= len(questions):
            return list(questions)
        if seed is None:
            return questions[:limit]
        return sorted(random.Random(seed).sample(questions, limit), key=lambda q: q.id)

    def exemplars(self, question: Question, n_shots: int) -> Sequence[Question]:
        if n_shots < 0:
            raise ValueError(f"n_shots must be non-negative, got {n_shots}")
        if n_shots == 0:
            return ()

        subject = question.metadata.get("subject")
        pool = self._exemplar_pool().get(str(subject), ())
        if len(pool) < n_shots:
            raise ValueError(
                f"MMLU {self.exemplar_split!r} has {len(pool)} exemplars for subject "
                f"{subject!r}, but n_shots={n_shots} were requested"
            )
        return pool[:n_shots]

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
            questions = self._build(self.source, self.split)
            self._questions = sorted(questions, key=lambda q: q.id)
        return self._questions

    def _exemplar_pool(self) -> dict[str, tuple[Question, ...]]:
        if self._exemplars is None:
            pool: dict[str, list[Question]] = {}
            # Dataset order, deliberately unsorted: "first n" means the first the
            # dataset offers, which is what lm-evaluation-harness takes.
            for question in self._build(self.exemplar_source, self.exemplar_split):
                pool.setdefault(str(question.metadata["subject"]), []).append(question)
            self._exemplars = {subject: tuple(items) for subject, items in pool.items()}
        return self._exemplars

    def _build(self, source: Path | None, split: str) -> list[Question]:
        counters: dict[str, int] = {}
        questions = []
        for raw in self._load_raw(source, split):
            subject = raw["subject"]
            index = counters.get(subject, 0)
            counters[subject] = index + 1
            questions.append(self._to_question(raw, index, split))

        if self.subjects is not None:
            missing = sorted(set(self.subjects) - counters.keys())
            if missing:
                raise ValueError(f"no MMLU questions found for subjects: {missing}")
        return questions

    def _load_raw(self, source: Path | None, split: str) -> list[dict]:
        rows = self._read_jsonl(source) if source is not None else self._read_hub(split)
        for row in rows:
            missing = [f for f in REQUIRED_FIELDS if f not in row]
            if missing:
                raise ValueError(f"MMLU row is missing fields {missing}: {row}")
        return rows

    def _read_jsonl(self, source: Path) -> list[dict]:
        with source.open() as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if self.subjects is None:
            return rows
        return [row for row in rows if row.get("subject") in self.subjects]

    def _read_hub(self, split: str) -> list[dict]:
        from datasets import load_dataset

        configs = self.subjects if self.subjects is not None else ("all",)
        rows: list[dict] = []
        for config in configs:
            dataset = load_dataset(HF_DATASET, config, split=split, revision=self.revision)
            rows.extend(dict(row) for row in dataset)
        return rows

    def _to_question(self, raw: dict, index: int, split: str) -> Question:
        answer = raw["answer"]
        if not isinstance(answer, int):
            raise ValueError(f"MMLU answer must be a choice index, got {answer!r}")
        subject = raw["subject"]
        return Question(
            id=f"{self.name}/{subject}/{split}/{index}",
            benchmark=self.name,
            body=raw["question"],
            answer=answer_letter(answer),
            task_format=self.task_format,
            choices=tuple(raw["choices"]),
            metadata={"subject": subject, "split": split},
        )
