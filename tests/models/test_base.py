import math

import pytest

from madcal.models import ChoiceScores, Generation


def generation(token_logprobs, tokens=None) -> Generation:
    return Generation(
        text="Answer: A", token_logprobs=token_logprobs, tokens=tokens, finish_reason="stop"
    )


def test_absent_logprobs_are_allowed():
    assert generation(None).token_logprobs is None


def test_non_finite_token_logprobs_are_refused():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            generation((-0.5, bad))


def test_positive_token_logprobs_are_refused():
    with pytest.raises(ValueError, match="<= 0"):
        generation((-0.5, 0.5))


def test_zero_token_logprobs_are_allowed():
    assert generation((0.0, 0.0)).token_logprobs == (0.0, 0.0)


def test_an_unknown_finish_reason_is_refused():
    with pytest.raises(ValueError, match="finish_reason"):
        Generation(text="x", token_logprobs=None, tokens=None, finish_reason="truncated")


def test_logprobs_that_do_not_align_with_tokens_are_refused():
    with pytest.raises(ValueError, match="does not match"):
        generation((-0.5, -0.5), ("A",))


def test_token_ids_that_do_not_align_with_tokens_are_refused():
    with pytest.raises(ValueError, match="token_ids length"):
        Generation(
            text="A", token_logprobs=None, tokens=("A",), finish_reason="stop", token_ids=(1, 2)
        )


def test_float32_rounding_on_a_near_certain_answer_is_not_overlapping_options():
    rest = math.log(2e-9)
    scores = ChoiceScores(choices=("A", "B", "C", "D"), logprobs=(0.0, rest, rest, rest))
    assert scores.option_mass() == pytest.approx(1 + 6e-9, abs=1e-12)


def test_option_mass_clearly_above_one_is_refused():
    with pytest.raises(ValueError, match="option mass"):
        ChoiceScores(choices=("A", "B"), logprobs=(math.log(0.6), math.log(0.41)))
