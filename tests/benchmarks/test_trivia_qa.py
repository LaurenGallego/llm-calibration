import json
from pathlib import Path

import pytest

from madcal.benchmarks import Question, TaskFormat, TriviaQA, normalize_answer

FIXTURE = Path(__file__).parent.parent / "fixtures" / "trivia_qa_sample.jsonl"
SUNSET = "trivia_qa/validation/tc_3"


@pytest.fixture
def trivia() -> TriviaQA:
    return TriviaQA(source=FIXTURE, exemplar_source=FIXTURE)


def by_id(benchmark: TriviaQA, question_id: str) -> Question:
    return next(question for question in benchmark.load() if question.id == question_id)


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_fixture_row_maps_end_to_end(trivia):
    question = by_id(trivia, SUNSET)
    assert question.body.startswith("Which Lloyd Webber musical")
    assert question.answer == "Sunset Boulevard"
    assert question.aliases[0] == "Sunset Boulevard"
    assert question.aliases.count("Sunset Boulevard") == 1
    assert "Sunset Blvd" in question.aliases
    assert question.choices is None
    assert question.task_format is TaskFormat.OPEN


def test_questions_are_sorted_by_id(trivia):
    ids = [question.id for question in trivia.load()]
    assert ids == sorted(ids)


def test_sampling_is_deterministic(trivia):
    first = [question.id for question in trivia.load(limit=3, seed=0)]
    second = [question.id for question in TriviaQA(source=FIXTURE).load(limit=3, seed=0)]
    assert first == second
    assert len(set(first)) == 3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The  Beatles!", "beatles"),
        ("Brian_Epstein", "brian epstein"),
        ("\u2018Exile\u2019", "exile"),
        ("Sunset Blvd.", "sunset blvd"),
        ("A Tale of Two Cities", "tale of two cities"),
    ],
)
def test_normalize_answer(text, expected):
    assert normalize_answer(text) == expected


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Sunset Boulevard.", "Sunset Boulevard"),
        ("Answer: Sunset Boulevard", "Sunset Boulevard"),
        ("  Exile\nThey were an American band.", "Exile"),
        ("St. Louis", "St. Louis"),
        ("", None),
        (" \n ", None),
        (".", None),
    ],
)
def test_extract_answer(trivia, response, expected):
    assert trivia.extract_answer(response, None) == expected


@pytest.mark.parametrize(
    ("extracted", "expected"),
    [
        ("Sunset Boulevard", True),
        ("sunset blvd", True),
        ("the Sunset Boulevard", True),
        ("Sunset", False),
        ("Evita", False),
    ],
)
def test_grade_matches_any_normalised_alias(trivia, extracted, expected):
    assert trivia.grade(by_id(trivia, SUNSET), extracted) is expected


def test_answer_instruction_asks_for_no_reasoning(trivia):
    assert "Do not explain or reason" in trivia.answer_instruction(trivia.load()[0])


def test_open_answers_have_no_slot_to_score(trivia):
    assert trivia.answer_slot("Exile", None) is None


def test_exemplars_are_the_first_of_the_train_pool_in_dataset_order(trivia):
    examples = trivia.exemplars(trivia.load()[0], 2)
    assert [example.id for example in examples] == ["trivia_qa/train/tc_3", "trivia_qa/train/tc_1"]


def test_too_few_exemplars_is_refused(trivia):
    with pytest.raises(ValueError, match="n_shots=6"):
        trivia.exemplars(trivia.load()[0], 6)


def test_a_row_without_an_answer_value_is_refused(tmp_path):
    rows = [{"question_id": "tc_9", "question": "Q?", "answer": {"value": "", "aliases": []}}]
    with pytest.raises(ValueError, match="no answer value"):
        TriviaQA(source=write_rows(tmp_path / "rows.jsonl", rows)).load()


def test_a_row_missing_a_field_is_refused(tmp_path):
    rows = [{"question_id": "tc_9", "question": "Q?"}]
    with pytest.raises(ValueError, match="missing fields"):
        TriviaQA(source=write_rows(tmp_path / "rows.jsonl", rows)).load()


def test_identical_copies_of_a_question_collapse_to_one(tmp_path):
    row = {"question_id": "tc_9", "question": "Q?", "answer": {"value": "A", "aliases": ["A"]}}
    benchmark = TriviaQA(source=write_rows(tmp_path / "rows.jsonl", [row, row]))
    assert [question.id for question in benchmark.load()] == ["trivia_qa/validation/tc_9"]
    assert benchmark.conflicting_ids["validation"] == ()


def test_copies_that_disagree_are_excluded_and_listed(tmp_path):
    kept = {"question_id": "tc_1", "question": "Q?", "answer": {"value": "A", "aliases": ["A"]}}
    first = {"question_id": "tc_9", "question": "Q?", "answer": {"value": "B", "aliases": ["B"]}}
    second = {"question_id": "tc_9", "question": "Q?", "answer": {"value": "C", "aliases": ["C"]}}
    benchmark = TriviaQA(source=write_rows(tmp_path / "rows.jsonl", [kept, first, second]))
    assert [question.id for question in benchmark.load()] == ["trivia_qa/validation/tc_1"]
    assert benchmark.conflicting_ids["validation"] == ("trivia_qa/validation/tc_9",)


def test_an_open_question_answer_must_be_one_of_its_aliases():
    with pytest.raises(ValueError, match="aliases"):
        Question(
            id="q",
            benchmark="trivia_qa",
            body="Q?",
            answer="Paris",
            task_format=TaskFormat.OPEN,
            aliases=("London",),
        )
