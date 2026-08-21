"""Row invariants and the partition slug.

A wrong slug is not a crash: it writes a condition's rows into a directory named after
a different condition, and `analysis/` then groups them together. The row invariants
are the same class -- a parse failure recorded as a wrong answer changes accuracy and
ECE without changing anything a reader would notice.
"""

import pytest

from caldrift.storage import QuestionRow, SignalKind, SignalRow, model_slug


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
    with pytest.raises(ValueError):
        QuestionRow(
            question_id="mmlu/anatomy/test/0",
            subject="anatomy",
            task_format="mcq",
            predicted=None,
            correct=False,
            parse_failed=True,
        )


def test_signal_value_and_reason_are_exclusive():
    with pytest.raises(ValueError):
        SignalRow("q", "confidence", SignalKind.SIGNAL, 0.5, "no_choice_scores")
    with pytest.raises(ValueError):
        SignalRow("q", "confidence", SignalKind.SIGNAL, None, None)


def test_nan_never_reaches_parquet():
    # A NaN here is a NaN in every aggregate computed from the column afterwards.
    with pytest.raises(ValueError):
        SignalRow("q", "confidence", SignalKind.SIGNAL, float("nan"), None)
