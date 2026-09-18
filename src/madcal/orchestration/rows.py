"""Flattening a graded debate transcript into storage rows."""

from collections.abc import Sequence

from madcal.benchmarks import Benchmark, Question
from madcal.debate import (
    AgentTurn,
    DebateTranscript,
    SystemAnswer,
    SystemNull,
    confidence_mode_registry,
)
from madcal.models import ModelAdapter
from madcal.signals import Signal, SignalValue, applicable, signal_registry
from madcal.storage import AnswerNull, Level, QuestionRow, SignalKind, SignalRow

STATED_CONFIDENCE = "stated_confidence"
SYSTEM_CONFIDENCE = "system_confidence"

SYSTEM_NULLS = {
    SystemNull.TIE: AnswerNull.TIE,
    SystemNull.NO_VALID_ANSWERS: AnswerNull.NO_VALID_ANSWERS,
}


def debate_signals(adapter: ModelAdapter, confidence_mode: str) -> tuple[type[Signal], ...]:
    """Return the registered signals a generative debate on `adapter` computes per agent turn."""
    mode = confidence_mode_registry.get(confidence_mode)
    return tuple(
        signal
        for name, signal in signal_registry
        if signal.requires.needs_generations and name != mode.signal and applicable(signal, adapter)
    )


def transcript_rows(
    transcript: DebateTranscript,
    question: Question,
    benchmark: Benchmark,
    signals: Sequence[type[Signal]],
) -> tuple[list[QuestionRow], list[SignalRow]]:
    """Grade one transcript and return its question rows and signal rows."""
    if transcript.question_id != question.id:
        raise ValueError(
            f"transcript is for {transcript.question_id!r} but the question is {question.id!r}"
        )
    subject = question.metadata.get("subject")
    questions = [_turn_row(turn, question, benchmark, subject) for turn in transcript.turns]
    questions.append(_system_row(transcript.system, question, benchmark, subject))

    signal_rows: list[SignalRow] = []
    for turn in transcript.turns:
        signal_rows.append(_stated_confidence(turn, question.id))
        signal_rows.extend(_computed_signals(turn, question.id, signals))
    signal_rows.append(_system_signal(transcript.system, question.id))
    return questions, signal_rows


def _turn_row(
    turn: AgentTurn, question: Question, benchmark: Benchmark, subject: str | None
) -> QuestionRow:
    return QuestionRow(
        question_id=question.id,
        level=Level.AGENT,
        agent_id=turn.agent_id,
        round=turn.round,
        subject=subject,
        task_format=str(question.task_format),
        predicted=turn.answer,
        correct=_grade(question, benchmark, turn.answer),
        null_reason=None if turn.answer is not None else AnswerNull.PARSE_FAILED,
        finish_reason=turn.finish_reason,
    )


def _system_row(
    system: SystemAnswer, question: Question, benchmark: Benchmark, subject: str | None
) -> QuestionRow:
    return QuestionRow(
        question_id=question.id,
        level=Level.SYSTEM,
        agent_id=None,
        round=None,
        subject=subject,
        task_format=str(question.task_format),
        predicted=system.answer,
        correct=_grade(question, benchmark, system.answer),
        null_reason=None if system.answer is not None else _system_null(system.reason),
        finish_reason=None,
    )


def _stated_confidence(turn: AgentTurn, question_id: str) -> SignalRow:
    return _signal_row(turn, question_id, STATED_CONFIDENCE, turn.confidence)


def _computed_signals(
    turn: AgentTurn, question_id: str, signals: Sequence[type[Signal]]
) -> list[SignalRow]:
    return [
        _signal_row(turn, question_id, signal.name, signal().compute(turn.evidence))
        for signal in signals
    ]


def _signal_row(turn: AgentTurn, question_id: str, name: str, value: SignalValue) -> SignalRow:
    return SignalRow(
        question_id=question_id,
        level=Level.AGENT,
        agent_id=turn.agent_id,
        round=turn.round,
        signal=name,
        kind=SignalKind.SIGNAL,
        value=value.value,
        reason=None if value.reason is None else str(value.reason),
    )


def _system_signal(system: SystemAnswer, question_id: str) -> SignalRow:
    return SignalRow(
        question_id=question_id,
        level=Level.SYSTEM,
        agent_id=None,
        round=None,
        signal=SYSTEM_CONFIDENCE,
        kind=SignalKind.AGGREGATE,
        value=system.confidence,
        reason=None if system.confidence is not None else str(_system_null(system.reason)),
    )


def _grade(question: Question, benchmark: Benchmark, predicted: str | None) -> bool | None:
    if predicted is None:
        return None
    return benchmark.grade(question, predicted)


def _system_null(reason: SystemNull | None) -> AnswerNull:
    if reason is None:
        raise ValueError("a system answer without a value must carry a reason")
    if reason not in SYSTEM_NULLS:
        raise ValueError(f"no stored null reason for {reason!r}")
    return SYSTEM_NULLS[reason]
