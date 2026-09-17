"""MMLU loader and extraction.

Covers the four things in this benchmark that break silently: a question paired
with the wrong answer, a change in what the extraction regex accepts, sampling
that stops being reproducible, and ids shifting under a subject filter. All run
against a committed fixture -- nothing here touches the Hub.
"""

from pathlib import Path

import pytest

from madcal.benchmarks import MMLU, TaskFormat

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
    assert mmlu.extract_answer(response, 4) == expected


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


def test_exemplars_come_from_the_dev_pool_of_the_same_subject():
    # Which pool is canonical is part of running MMLU as published, so it lives on the
    # benchmark rather than in config. A cross-subject exemplar would also break
    # FewShotCompletion's own check.
    mmlu = MMLU(
        source=Path("tests/fixtures/mmlu_sample.jsonl"),
        exemplar_source=Path("tests/fixtures/mmlu_dev_sample.jsonl"),
    )
    question = mmlu.load()[0]
    shots = mmlu.exemplars(question, 3)

    assert len(shots) == 3
    assert all(s.metadata["subject"] == question.metadata["subject"] for s in shots)
    assert all(s.metadata["split"] == "dev" for s in shots)
    # first_n in dataset order, as lm-evaluation-harness takes them
    assert [s.id for s in shots] == [
        f"mmlu/{question.metadata['subject']}/dev/{i}" for i in range(3)
    ]


def test_exemplars_are_not_the_evaluation_questions():
    # A leaked target would raise accuracy toward 100% and raise nothing else.
    mmlu = MMLU(
        source=Path("tests/fixtures/mmlu_sample.jsonl"),
        exemplar_source=Path("tests/fixtures/mmlu_dev_sample.jsonl"),
    )
    evaluation = {q.id for q in mmlu.load()}
    for question in mmlu.load():
        assert not evaluation & {s.id for s in mmlu.exemplars(question, 5)}


def test_too_few_exemplars_raises_rather_than_shortening_the_prompt():
    # Silently rendering 5 shots as 3 would make the condition something other than
    # what the config asked for, with nothing recording the difference.
    mmlu = MMLU(
        source=Path("tests/fixtures/mmlu_sample.jsonl"),
        exemplar_source=Path("tests/fixtures/mmlu_dev_sample.jsonl"),
    )
    with pytest.raises(ValueError, match="n_shots=6"):
        mmlu.exemplars(mmlu.load()[0], 6)


def test_zero_shot_needs_no_pool():
    assert (
        MMLU(source=Path("tests/fixtures/mmlu_sample.jsonl")).exemplars(
            MMLU(source=Path("tests/fixtures/mmlu_sample.jsonl")).load()[0], 0
        )
        == ()
    )


def test_answer_instruction_names_every_valid_letter(mmlu):
    question = mmlu.load()[0]
    assert mmlu.answer_instruction(question) == (
        "State your final answer on its own line as 'Answer: X', where X is one of A, B, C, D."
    )


def test_extraction_without_a_choice_count_is_refused(mmlu):
    with pytest.raises(ValueError, match="n_choices"):
        mmlu.extract_answer("Answer: A", None)
