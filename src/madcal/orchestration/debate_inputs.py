"""Answer-free debate inputs built from benchmark questions."""

from madcal.benchmarks import Benchmark, Question
from madcal.debate import DebateQuestion
from madcal.prompts import question_with_choices


def debate_question(question: Question, benchmark: Benchmark) -> DebateQuestion:
    """Return the question with its choices and the benchmark's answer instruction."""
    if question.benchmark != benchmark.name:
        raise ValueError(
            f"question {question.id!r} belongs to {question.benchmark!r}, not {benchmark.name!r}"
        )
    text = f"{question_with_choices(question)}\n\n{benchmark.answer_instruction(question)}"
    n_choices = len(question.choices) if question.choices is not None else None
    return DebateQuestion(question.id, text, n_choices)
