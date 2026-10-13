import pytest

from madcal.benchmarks import choice_answer_instruction, choice_answer_slot, extract_choice_letter


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("Reasoning.\nAnswer: B", "B"),
        ("Reasoning.\n  Answer: B  \n\nThanks.", "B"),
        ("Reasoning.\n**Answer: B**", "B"),
        ("Reasoning.\n**Answer:** B", "B"),
        ("Reasoning.\r\nAnswer: B\r\n", "B"),
        ("Answer: A\nOn reflection.\nAnswer: C", "C"),
        ('Both agents stated "Answer: B". I cannot say.', None),
        ("- Answer: B (Correct)", None),
        ("Answer: B because it is the pancreas", None),
        ("The correct answer is B.", None),
        ("Answer:\nB", None),
        ('My peer wrote "Answer: A".\nAnswer: C', "C"),
        ("Thus, the correct answer is:\n```Answer: D```", "D"),
        ("Reasoning.\n`Answer: D`", "D"),
        ("Reasoning.\n**Final Answer: B**", "B"),
        ("Reasoning.\nFinal Answer: B", "B"),
        ("Reasoning.\nThe Final Answer: B", None),
        ("```Answer: D``` is what I think", None),
    ],
)
def test_answer_must_sit_alone_on_its_line(response, expected):
    assert extract_choice_letter(response, 4) == expected


@pytest.mark.parametrize(
    ("response", "n_choices", "expected"),
    [
        ("Answer: C", 3, "C"),
        ("Answer: D", 3, None),
        ("Answer: E", 4, None),
        ("Answer: E", 5, "E"),
        ("answer: e", 5, "E"),
        ("Answer: B\nAnswer: E", 5, "E"),
        ("Answer: B\nAnswer: E", 4, "B"),
        ("Answer: $C$", 3, "C"),
    ],
)
def test_only_letters_valid_for_the_question_are_extracted(response, n_choices, expected):
    assert extract_choice_letter(response, n_choices) == expected


def test_instruction_lists_exactly_the_valid_letters():
    assert choice_answer_instruction(5) == (
        "Think step by step, then state your final answer on its own line as 'Answer: X', "
        "where X is one of A, B, C, D, E."
    )


@pytest.mark.parametrize("n_choices", [0, 1, 27])
def test_choice_count_out_of_range_is_refused(n_choices):
    with pytest.raises(ValueError, match="n_choices"):
        extract_choice_letter("Answer: A", n_choices)
    with pytest.raises(ValueError, match="n_choices"):
        choice_answer_instruction(n_choices)


@pytest.mark.parametrize(
    ("response", "context", "lead"),
    [
        ("Reasoning.\nAnswer: B", "Reasoning.\nAnswer:", " "),
        ("Reasoning.\n**Answer: B**", "Reasoning.\n**Answer:", " "),
        ("Reasoning.\n**Answer:** B", "Reasoning.\n**Answer:", "** "),
        ("Reasoning.\nAnswer: $B$", "Reasoning.\nAnswer:", " $"),
        ("Reasoning.\nAnswer:B", "Reasoning.\nAnswer:", ""),
        ("Reasoning.\n```Answer: D```", "Reasoning.\n```Answer:", " "),
        ("Reasoning.\n**Final Answer: B**", "Reasoning.\n**Final Answer:", " "),
        ("Answer: A\nOn reflection.\nAnswer: B", "Answer: A\nOn reflection.\nAnswer:", " "),
    ],
)
def test_answer_slot_splits_the_response_where_the_final_letter_is_written(response, context, lead):
    assert choice_answer_slot(response, 4) == (context, lead, ("A", "B", "C", "D"))


def test_answer_slot_and_extraction_agree_on_which_line_is_the_answer():
    response = "Answer: A\nNo wait.\nAnswer: C\nBecause B is wrong."
    slot = choice_answer_slot(response, 4)
    assert slot is not None
    context, lead, _ = slot
    assert response.startswith(f"{context}{lead}{extract_choice_letter(response, 4)}\n")
    assert extract_choice_letter(response, 4) == "C"


def test_answer_slot_is_none_without_an_answer_line():
    assert choice_answer_slot("I think it is B.", 4) is None
