import pytest

from madcal.models import Generation


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
