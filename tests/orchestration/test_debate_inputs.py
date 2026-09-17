from pathlib import Path

import pytest

from madcal.benchmarks import MMLU, Question, TaskFormat
from madcal.orchestration import debate_question

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mmlu_sample.jsonl"


@pytest.fixture
def mmlu() -> MMLU:
    return MMLU(source=FIXTURE)


def test_debate_question_carries_choices_instruction_and_option_count(mmlu):
    question = mmlu.load()[0]
    result = debate_question(question, mmlu)
    assert result.question_id == question.id
    assert result.n_choices == 4
    lines = result.text.split("\n")
    assert lines[0] == question.body.strip()
    choices = question.choices or ()
    assert lines[1:5] == [
        f"{letter}. {choice}" for letter, choice in zip("ABCD", choices, strict=True)
    ]
    assert lines[5] == ""
    assert lines[6] == mmlu.answer_instruction(question)
    assert len(lines) == 7


def test_debate_question_has_no_field_for_the_answer():
    assert set(debate_question.__annotations__) == {"question", "benchmark", "return"}
    from madcal.debate import DebateQuestion

    assert set(DebateQuestion.__dataclass_fields__) == {"question_id", "text", "n_choices"}


def test_question_from_another_benchmark_is_refused(mmlu):
    foreign = Question("gsm8k/test/0", "gsm8k", "What is 2+2?", "4", TaskFormat.MATH)
    with pytest.raises(ValueError, match="belongs to"):
        debate_question(foreign, mmlu)
