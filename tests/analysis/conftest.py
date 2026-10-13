from collections.abc import Mapping
from pathlib import Path

import pytest

from madcal.storage import (
    AGENT_CONFIDENCE,
    SYSTEM_CONFIDENCE,
    AnswerNull,
    Level,
    QuestionRow,
    ResultStore,
    RunManifest,
    RunStatus,
    SignalKind,
    SignalRow,
    Variant,
)

type Turn = tuple[int, int, bool | None, float | None]
type Debate = tuple[list[Turn], tuple[bool | None, float | None]]
type Debates = Mapping[str, Debate]


@pytest.fixture
def store(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / "results")


def make_manifest(
    run_id: str,
    n_agents: int = 1,
    n_rounds: int = 0,
    variant: Variant = Variant.BASE,
    protocol: str = "vanilla",
    config_hash: str = "cfg-a",
    n_questions: int = 1,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        config_hash=config_hash,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        variant=variant,
        benchmark="mmlu",
        protocol=protocol,
        n_agents=n_agents,
        n_rounds=n_rounds,
        confidence_mode="verbalized",
        aggregation="majority",
        temperature=0.7,
        max_tokens=768,
        prompt_protocol="chat_template",
        checkpoint_sha="a" * 40,
        code_sha="b" * 40,
        dtype="bfloat16",
        batch_size=8,
        backend="transformers",
        backend_version="5.15.1",
        n_questions_planned=n_questions,
        seed=0,
    )


def agent_pair(
    question_id: str, agent: int, round_: int, correct: bool | None, confidence: float | None
) -> tuple[QuestionRow, SignalRow]:
    question = QuestionRow(
        question_id=question_id,
        level=Level.AGENT,
        agent_id=f"agent_{agent}",
        round=round_,
        subject="anatomy",
        task_format="mcq",
        predicted=None if correct is None else "B",
        correct=correct,
        null_reason=None if correct is not None else AnswerNull.PARSE_FAILED,
        finish_reason="stop",
    )
    signal = SignalRow(
        question_id=question_id,
        level=Level.AGENT,
        agent_id=f"agent_{agent}",
        round=round_,
        signal=AGENT_CONFIDENCE,
        kind=SignalKind.SIGNAL,
        value=confidence,
        reason=None if confidence is not None else "parse_failed",
    )
    return question, signal


def system_pair(
    question_id: str, correct: bool | None, confidence: float | None
) -> tuple[QuestionRow, SignalRow]:
    question = QuestionRow(
        question_id=question_id,
        level=Level.SYSTEM,
        agent_id=None,
        round=None,
        subject="anatomy",
        task_format="mcq",
        predicted=None if correct is None else "B",
        correct=correct,
        null_reason=None if correct is not None else AnswerNull.NO_VALID_ANSWERS,
        finish_reason=None,
    )
    signal = SignalRow(
        question_id=question_id,
        level=Level.SYSTEM,
        agent_id=None,
        round=None,
        signal=SYSTEM_CONFIDENCE,
        kind=SignalKind.AGGREGATE,
        value=confidence,
        reason=None if confidence is not None else "no_valid_answers",
    )
    return question, signal


def write_run(
    store: ResultStore,
    run_id: str,
    debates: Debates,
    n_agents: int = 1,
    n_rounds: int = 0,
    variant: Variant = Variant.BASE,
    protocol: str = "vanilla",
    config_hash: str = "cfg-a",
    status: RunStatus = RunStatus.COMPLETE,
) -> None:
    manifest = make_manifest(
        run_id,
        n_agents=n_agents,
        n_rounds=n_rounds,
        variant=variant,
        protocol=protocol,
        config_hash=config_hash,
        n_questions=len(debates),
    )
    questions: list[QuestionRow] = []
    signals: list[SignalRow] = []
    for question_id, (turns, system) in debates.items():
        for round_, agent, correct, confidence in turns:
            question, signal = agent_pair(question_id, agent, round_, correct, confidence)
            questions.append(question)
            signals.append(signal)
        question, signal = system_pair(question_id, *system)
        questions.append(question)
        signals.append(signal)

    store.begin_run(manifest)
    store.write_chunk(manifest, questions, signals, seq=0)
    if status is not RunStatus.RUNNING:
        store.finish_run(run_id, status, n_written=len(debates))


def single_agent_debates(outcomes: list[tuple[bool | None, float | None]]) -> Debates:
    return {
        f"q{index}": ([(0, 0, correct, confidence)], (correct, confidence))
        for index, (correct, confidence) in enumerate(outcomes)
    }
