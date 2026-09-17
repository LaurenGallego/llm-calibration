import pytest

from madcal.benchmarks import choice_answer_instruction, extract_choice_letter


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
        "State your final answer on its own line as 'Answer: X', where X is one of A, B, C, D, E."
    )


@pytest.mark.parametrize("n_choices", [0, 1, 27])
def test_choice_count_out_of_range_is_refused(n_choices):
    with pytest.raises(ValueError, match="n_choices"):
        extract_choice_letter("Answer: A", n_choices)
    with pytest.raises(ValueError, match="n_choices"):
        choice_answer_instruction(n_choices)
