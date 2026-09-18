from pathlib import Path

import pytest

from madcal.benchmarks import MMLU
from madcal.debate import AgentTurn, DebateTranscript, SystemAnswer, SystemNull, agent_id
from madcal.models import Generation
from madcal.orchestration import STATED_CONFIDENCE, SYSTEM_CONFIDENCE, transcript_rows
from madcal.signals import SignalNull, SignalValue
from madcal.storage import AnswerNull, Level, SignalKind

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mmlu_sample.jsonl"
N_AGENTS = 3
N_ROUNDS = 1


@pytest.fixture
def mmlu() -> MMLU:
    return MMLU(source=FIXTURE)


@pytest.fixture
def question(mmlu: MMLU):
    return mmlu.load()[0]


def turn(agent: int, round_: int, answer: str | None, confidence: float | None = None) -> AgentTurn:
    value = (
        SignalValue(confidence)
        if confidence is not None
        else SignalValue(None, SignalNull.PARSE_FAILED)
    )
    generation = Generation(text="text", token_logprobs=None, tokens=None, finish_reason="stop")
    return AgentTurn(agent_id(agent), round_, generation, answer, value)


def make_transcript(question_id: str, answers, system: SystemAnswer) -> DebateTranscript:
    turns = tuple(
        turn(agent, round_, answers[round_][agent], 0.5 + 0.1 * agent)
        for round_ in range(N_ROUNDS + 1)
        for agent in range(N_AGENTS)
    )
    return DebateTranscript(question_id, N_AGENTS, N_ROUNDS, turns, system)


def test_every_turn_and_the_system_answer_become_rows(question, mmlu):
    answers = [["A", "B", "A"], ["A", "A", "A"]]
    transcript = make_transcript(question.id, answers, SystemAnswer("A", 1.0))
    questions, signals = transcript_rows(transcript, question, mmlu)

    assert len(questions) == N_AGENTS * (N_ROUNDS + 1) + 1
    assert len(signals) == len(questions)
    assert [row.level for row in questions[-1:]] == [Level.SYSTEM]
    assert questions[0].subject == "anatomy"
    assert questions[0].task_format == "mcq"


def test_rows_keep_the_agent_and_round_of_the_turn_they_came_from(question, mmlu):
    answers = [["A", "B", "A"], ["A", "A", "A"]]
    transcript = make_transcript(question.id, answers, SystemAnswer("A", 1.0))
    questions, signals = transcript_rows(transcript, question, mmlu)

    expected = [(turn.agent_id, turn.round) for turn in transcript.turns]
    assert [(row.agent_id, row.round) for row in questions[:-1]] == expected
    assert [(row.agent_id, row.round) for row in signals[:-1]] == expected
    assert [row.predicted for row in questions[:-1]] == [turn.answer for turn in transcript.turns]
    assert (questions[-1].agent_id, questions[-1].round) == (None, None)


def test_grading_follows_the_benchmark(question, mmlu):
    assert question.answer == "A"
    answers = [["A", "B", "A"], ["A", "B", "A"]]
    transcript = make_transcript(question.id, answers, SystemAnswer("A", 2 / 3))
    questions, _ = transcript_rows(transcript, question, mmlu)

    agents = [row for row in questions if row.level is Level.AGENT]
    assert [row.correct for row in agents] == [True, False, True, True, False, True]
    assert questions[-1].correct is True


def test_an_unparsed_answer_is_a_null_with_a_reason_never_a_wrong_answer(question, mmlu):
    answers = [["A", None, "A"], ["A", None, "A"]]
    transcript = make_transcript(question.id, answers, SystemAnswer("A", 2 / 3))
    questions, _ = transcript_rows(transcript, question, mmlu)

    unparsed = [row for row in questions if row.predicted is None]
    assert len(unparsed) == 2
    assert all(row.correct is None for row in unparsed)
    assert all(row.null_reason is AnswerNull.PARSE_FAILED for row in unparsed)


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (SystemNull.TIE, AnswerNull.TIE),
        (SystemNull.NO_VALID_ANSWERS, AnswerNull.NO_VALID_ANSWERS),
    ],
)
def test_a_system_null_keeps_its_reason(question, mmlu, reason, expected):
    answers = [["A", "B", "C"], ["A", "B", "C"]]
    transcript = make_transcript(question.id, answers, SystemAnswer(None, None, reason))
    questions, signals = transcript_rows(transcript, question, mmlu)

    system = questions[-1]
    assert system.predicted is None
    assert system.correct is None
    assert system.null_reason is expected
    assert signals[-1].value is None
    assert signals[-1].reason == str(expected)


def test_confidence_becomes_a_signal_row_at_every_level(question, mmlu):
    answers = [["A", "B", "A"], ["A", "A", "A"]]
    transcript = make_transcript(question.id, answers, SystemAnswer("A", 1.0))
    _, signals = transcript_rows(transcript, question, mmlu)

    agents = [row for row in signals if row.level is Level.AGENT]
    assert {row.signal for row in agents} == {STATED_CONFIDENCE}
    assert all(row.kind is SignalKind.SIGNAL for row in agents)
    assert [row.value for row in agents[:3]] == [0.5, 0.6, 0.7]
    assert signals[-1].signal == SYSTEM_CONFIDENCE
    assert signals[-1].kind is SignalKind.AGGREGATE
    assert signals[-1].value == 1.0


def test_an_unparsed_confidence_is_a_null_with_a_reason(question, mmlu):
    turns = tuple(
        turn(agent, round_, "A") for round_ in range(N_ROUNDS + 1) for agent in range(N_AGENTS)
    )
    transcript = DebateTranscript(question.id, N_AGENTS, N_ROUNDS, turns, SystemAnswer("A", 1.0))
    _, signals = transcript_rows(transcript, question, mmlu)

    agents = [row for row in signals if row.level is Level.AGENT]
    assert all(row.value is None for row in agents)
    assert all(row.reason == str(SignalNull.PARSE_FAILED) for row in agents)


def test_no_signal_row_carries_a_grade(question, mmlu):
    answers = [["A", "B", "A"], ["A", "A", "A"]]
    transcript = make_transcript(question.id, answers, SystemAnswer("A", 1.0))
    _, signals = transcript_rows(transcript, question, mmlu)

    fields = set(type(signals[0]).__dataclass_fields__)
    assert "correct" not in fields and "predicted" not in fields and "answer" not in fields


def test_a_transcript_for_another_question_is_refused(question, mmlu):
    answers = [["A", "B", "A"], ["A", "A", "A"]]
    transcript = make_transcript("mmlu/anatomy/test/99", answers, SystemAnswer("A", 1.0))
    with pytest.raises(ValueError, match="but the question is"):
        transcript_rows(transcript, question, mmlu)
