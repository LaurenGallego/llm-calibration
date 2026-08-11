"""MMLU loader and extraction.

Covers the four things in this benchmark that break silently: a question paired
with the wrong answer, a change in what the extraction regex accepts, sampling
that stops being reproducible, and ids shifting under a subject filter. All run
against a committed fixture -- nothing here touches the Hub.
"""

from pathlib import Path

import pytest

from caldrift.benchmarks import MMLU, TaskFormat

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mmlu_sample.jsonl"


@pytest.fixture
def mmlu() -> MMLU:
    return MMLU(source=FIXTURE)


def test_fixture_row_maps_end_to_end(mmlu):
    question = mmlu.load()[0]
    assert question.id == "mmlu/anatomy/test/0"
    assert question.answer == "A"
    assert question.choices[0] == "paralysis of the facial muscles."
    assert question.metadata["subject"] == "anatomy"
    assert question.task_format is TaskFormat.MCQ


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Answer: A", "A"),
        ("answer: c", "C"),
        ("Answer: A\nWait, reconsidering.\nAnswer: D", "D"),
        ("It's not A, so... Answer: C", "C"),
        # Deliberately strict: both are readable to a human and still rejected,
        # so relaxing the pattern later has to fail a test first.
        ("The answer is D", None),
        ("Answer:\nB", None),
    ],
)
def test_extract_answer(mmlu, response, expected):
    assert mmlu.extract_answer(response) == expected


def test_sampling_is_deterministic(mmlu):
    first = [q.id for q in mmlu.load(limit=5, seed=0)]
    second = [q.id for q in MMLU(source=FIXTURE).load(limit=5, seed=0)]
    assert len(first) == 5
    assert first == second == sorted(first)
    assert [q.id for q in mmlu.load(limit=3)] == [q.id for q in mmlu.load()][:3]


def test_ids_are_per_subject_and_stable_under_filtering(mmlu):
    from_full = [q.id for q in mmlu.load() if q.metadata["subject"] == "anatomy"]
    from_filtered = [q.id for q in MMLU(subjects=["anatomy"], source=FIXTURE).load()]
    assert from_full == from_filtered
    assert from_full[0] == "mmlu/anatomy/test/0"


def test_grade_normalises_input_and_rejects_non_strings(mmlu):
    question = mmlu.load()[0]
    assert mmlu.grade(question, " a ") is True
    assert mmlu.grade(question, "B") is False
    with pytest.raises(TypeError, match="expects an extracted answer string"):
        mmlu.grade(question, None)
