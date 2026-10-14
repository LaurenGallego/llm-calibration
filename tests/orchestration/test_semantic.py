import math
from pathlib import Path

import pytest

from madcal.benchmarks import TriviaQA
from madcal.models import Generation
from madcal.orchestration.semantic import semantic_rows, signal_name
from madcal.storage import AnswerNull, Level

FIXTURE = Path(__file__).parent.parent / "fixtures" / "trivia_qa_sample.jsonl"
E = math.exp


@pytest.fixture
def trivia() -> TriviaQA:
    return TriviaQA(source=FIXTURE)


@pytest.fixture
def question(trivia):
    return next(q for q in trivia.load() if q.id == "trivia_qa/validation/tc_3")


def generation(text: str, *logprobs: float, finish: str = "stop") -> Generation:
    return Generation(text, logprobs, tuple(f"{text}:{i}" for i in range(len(logprobs))), finish)


SAMPLES = [
    generation("Sunset Boulevard.", -0.5, -0.5),
    generation("sunset boulevard", -1.0, -1.0),
    generation("Evita", -1.0, -1.0),
    generation("", -1.0),
]


def values(signal_rows) -> dict[str, float | None]:
    return {row.signal: row.value for row in signal_rows}


def test_samples_are_agent_rows_and_the_reference_is_the_graded_system_row(trivia, question):
    rows, _ = semantic_rows(question, trivia, generation("Sunset Boulevard", -0.1), SAMPLES, None)
    agents = [row for row in rows if row.level is Level.AGENT]
    system = [row for row in rows if row.level is Level.SYSTEM]
    assert [row.agent_id for row in agents] == ["agent_0", "agent_1", "agent_2", "agent_3"]
    assert [row.correct for row in agents] == [True, True, False, None]
    assert agents[3].null_reason is AnswerNull.PARSE_FAILED
    assert len(system) == 1
    assert system[0].predicted == "Sunset Boulevard" and system[0].correct is True


def test_measures_are_system_signals_with_hand_computed_values(trivia, question):
    _, signals = semantic_rows(
        question, trivia, generation("Sunset Boulevard", -0.1), SAMPLES, None
    )
    assert all(row.level is Level.SYSTEM for row in signals)
    stored = values(signals)
    assert stored[signal_name("e_sc")] == pytest.approx(2 / 3)
    sunset = E(-1.0) + E(-2.0)
    assert stored[signal_name("l_sc")] == pytest.approx(sunset / (sunset + E(-2.0)))
    assert stored[signal_name("n_parsed")] == 3.0
    assert stored[signal_name("n_clusters")] == 2.0
    assert stored[signal_name("matched")] == 1.0
    assert stored[signal_name("reference_truncated")] == 0.0


def test_an_unparsed_reference_is_a_system_null_and_its_measures_say_why(trivia, question):
    rows, signals = semantic_rows(question, trivia, generation(" ", -0.1), SAMPLES, None)
    system = next(row for row in rows if row.level is Level.SYSTEM)
    assert system.predicted is None
    assert system.null_reason is AnswerNull.NO_VALID_ANSWERS
    measure = next(row for row in signals if row.signal == signal_name("e_sc"))
    assert measure.value is None and measure.reason == "parse_failed"
    assert values(signals)[signal_name("n_parsed")] == 3.0


def test_samples_without_logprobs_give_nulls_not_numbers(trivia, question):
    bare = [Generation(sample.text, None, None, "stop") for sample in SAMPLES]
    _, signals = semantic_rows(question, trivia, generation("Sunset Boulevard", -0.1), bare, None)
    measure = next(row for row in signals if row.signal == signal_name("l_sc"))
    assert measure.value is None and measure.reason == "no_logprobs"


def test_a_truncated_reference_is_recorded(trivia, question):
    reference = generation("Sunset Boulevard", -0.1, finish="length")
    _, signals = semantic_rows(question, trivia, reference, SAMPLES, None)
    assert values(signals)[signal_name("reference_truncated")] == 1.0
