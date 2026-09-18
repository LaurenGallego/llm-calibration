"""Row invariants and the partition slug.

A wrong slug is not a crash: it writes a condition's rows into a directory named after
a different condition, and `analysis/` then groups them together. The row invariants
are the same class -- a parse failure recorded as a wrong answer changes accuracy and
ECE without changing anything a reader would notice.
"""

import pytest

from madcal.storage import (
    AnswerNull,
    Level,
    QuestionRow,
    SignalKind,
    SignalRow,
    model_slug,
)


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("mistralai/Mistral-7B-v0.1", "mistralai__Mistral-7B-v0.1"),
        ("allenai/Llama-3.1-Tulu-3-8B-SFT", "allenai__Llama-3.1-Tulu-3-8B-SFT"),
        ("stub", "stub"),
        ("my org/model v2", "my_org__model_v2"),
    ],
)
def test_slug_is_partition_safe_and_still_readable(model_id: str, expected: str):
    slug = model_slug(model_id)
    assert slug == expected
    # The two characters that break a Hive path: one invents a directory level, the
    # other looks like a second partition key.
    assert "/" not in slug
    assert "=" not in slug


def test_slug_is_stable():
    # The slug is a storage key. If it moved between runs, a resumed job would write
    # into a new directory and the old rows would silently drop out of the condition.
    assert model_slug("Qwen/Qwen2.5-0.5B") == model_slug("Qwen/Qwen2.5-0.5B")


@pytest.mark.parametrize("model_id", ["", "   ", "...", "/", "./x"])
def test_unusable_ids_raise(model_id: str):
    with pytest.raises(ValueError):
        model_slug(model_id)


def test_parse_failure_is_not_a_wrong_answer():
    with pytest.raises(ValueError, match="correct is present"):
        QuestionRow(
            question_id="mmlu/anatomy/test/0",
            level=Level.AGENT,
            agent_id="agent_0",
            round=0,
            subject="anatomy",
            task_format="mcq",
            predicted=None,
            correct=False,
            null_reason=AnswerNull.PARSE_FAILED,
            finish_reason="stop",
        )


def question_row(**overrides):
    fields = {
        "question_id": "mmlu/anatomy/test/0",
        "level": Level.AGENT,
        "agent_id": "agent_0",
        "round": 0,
        "subject": "anatomy",
        "task_format": "mcq",
        "predicted": "B",
        "correct": True,
        "null_reason": None,
        "finish_reason": "stop",
    }
    return QuestionRow(**{**fields, **overrides})


def signal_row(**overrides):
    fields = {
        "question_id": "mmlu/anatomy/test/0",
        "level": Level.AGENT,
        "agent_id": "agent_0",
        "round": 0,
        "signal": "verbalized_confidence",
        "kind": SignalKind.SIGNAL,
        "value": 0.5,
        "reason": None,
    }
    return SignalRow(**{**fields, **overrides})


@pytest.mark.parametrize("row", [question_row, signal_row])
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"level": Level.SYSTEM}, "carry no agent"),
        ({"level": Level.SYSTEM, "agent_id": None}, "carry no agent"),
        ({"agent_id": None}, "need an agent"),
        ({"round": None}, "need an agent"),
        ({"round": -1}, "round must be"),
    ],
)
def test_level_agent_and_round_must_agree(row, overrides, message):
    extra = {"finish_reason": None} if row is question_row else {}
    with pytest.raises(ValueError, match=message):
        row(**overrides, **extra)


def test_a_system_row_needs_no_agent_and_no_round():
    row = question_row(level=Level.SYSTEM, agent_id=None, round=None, finish_reason=None)
    assert row.agent_id is None and row.round is None


@pytest.mark.parametrize(
    ("level", "null_reason"),
    [
        (Level.AGENT, AnswerNull.TIE),
        (Level.AGENT, AnswerNull.NO_VALID_ANSWERS),
        (Level.SYSTEM, AnswerNull.PARSE_FAILED),
    ],
)
def test_null_reasons_belong_to_one_level(level, null_reason):
    grain = (
        {"agent_id": None, "round": None, "finish_reason": None} if level is Level.SYSTEM else {}
    )
    with pytest.raises(ValueError, match="cannot occur at level"):
        question_row(level=level, predicted=None, correct=None, null_reason=null_reason, **grain)


def test_a_system_row_records_no_finish_reason():
    with pytest.raises(ValueError, match="finish_reason"):
        question_row(level=Level.SYSTEM, agent_id=None, round=None, finish_reason="stop")


def test_an_agent_row_must_record_a_finish_reason():
    with pytest.raises(ValueError, match="finish_reason"):
        question_row(finish_reason=None)


def test_signal_value_and_reason_are_exclusive():
    with pytest.raises(ValueError):
        signal_row(value=0.5, reason="no_choice_scores")
    with pytest.raises(ValueError):
        signal_row(value=None, reason=None)
