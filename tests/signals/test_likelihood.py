import math

import pytest

from madcal.models import Generation
from madcal.signals import Evidence, LengthNormalisedLikelihood, SignalNull, SignalValue


def generation(token_logprobs: tuple[float, ...] | None) -> Generation:
    return Generation(
        text="Answer: A", token_logprobs=token_logprobs, tokens=None, finish_reason="stop"
    )


def evidence(token_logprobs: tuple[float, ...] | None) -> Evidence:
    return Evidence(generations=(generation(token_logprobs),))


def compute(token_logprobs: tuple[float, ...] | None) -> SignalValue:
    return LengthNormalisedLikelihood().compute(evidence(token_logprobs))


def test_two_token_response_is_the_geometric_mean_of_its_token_probabilities():
    value = compute((math.log(0.5), math.log(0.25)))
    assert value.value == pytest.approx(math.sqrt(0.125))


def test_it_is_neither_the_arithmetic_mean_nor_the_joint_probability():
    value = compute((math.log(0.5), math.log(0.25)))
    assert value.value is not None
    assert value.value != pytest.approx(0.375)
    assert value.value != pytest.approx(0.125)


def test_a_single_token_response_is_that_token_probability():
    assert compute((math.log(0.4),)).value == pytest.approx(0.4)


def test_length_does_not_change_the_value_when_every_token_agrees():
    short = compute((math.log(0.6),) * 2)
    long = compute((math.log(0.6),) * 40)
    assert short.value == pytest.approx(0.6)
    assert long.value == pytest.approx(0.6)


def test_a_certain_response_is_one():
    assert compute((0.0, 0.0)).value == pytest.approx(1.0)


def test_absent_logprobs_are_a_null_with_a_reason_not_a_zero():
    assert compute(None) == SignalValue(None, SignalNull.NO_LOGPROBS)


def test_an_empty_token_sequence_is_a_null_with_a_reason():
    assert compute(()) == SignalValue(None, SignalNull.NO_LOGPROBS)


def test_absent_generations_are_a_null_with_a_reason():
    value = LengthNormalisedLikelihood().compute(Evidence())
    assert value == SignalValue(None, SignalNull.NO_GENERATIONS)


def test_several_generations_are_refused_rather_than_silently_reduced():
    several = Evidence(generations=(generation((0.0,)), generation((-1.0,))))
    with pytest.raises(ValueError, match="single generation"):
        LengthNormalisedLikelihood().compute(several)


def test_it_declares_the_evidence_it_needs():
    requires = LengthNormalisedLikelihood.requires
    assert requires.needs_generations is True
    assert requires.needs_logprobs is True
    assert requires.needs_choice_scores is False
