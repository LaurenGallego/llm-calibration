"""SignalValue's invariants.

The type exists so that "no signal could be computed" cannot be mistaken for "the
signal was zero". Every test here is a case where the mistake would otherwise be
silent and would land in a stored row.
"""

import pytest

from madcal.models import Generation, StubAdapter
from madcal.signals import (
    Confidence,
    Evidence,
    LengthNormalisedLikelihood,
    SignalNull,
    SignalValue,
    VerbalizedConfidence,
    applicable,
    sole_generation,
)


def test_a_value_and_a_reason_together_are_rejected():
    with pytest.raises(ValueError):
        SignalValue(0.5, SignalNull.NO_CHOICE_SCORES)


def test_neither_a_value_nor_a_reason_is_rejected():
    with pytest.raises(ValueError):
        SignalValue(None)


def test_non_finite_values_are_rejected():
    for bad_value in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError):
            SignalValue(bad_value)


def generation() -> Generation:
    return Generation(text="Answer: A", token_logprobs=None, tokens=None, finish_reason="stop")


def test_an_empty_generation_tuple_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        Evidence(generations=())


def test_sole_generation_refuses_evidence_that_carries_none():
    with pytest.raises(ValueError, match="no generations"):
        sole_generation(Evidence())


def test_sole_generation_returns_the_one_generation():
    only = generation()
    assert sole_generation(Evidence(generations=(only,))) is only


class _NoLogprobs(StubAdapter):
    supports_logprobs = False


class _NoScoring(StubAdapter):
    supports_scoring = False


def test_a_logprob_signal_is_not_applicable_to_an_adapter_without_logprobs():
    assert applicable(LengthNormalisedLikelihood, StubAdapter()) is True
    assert applicable(LengthNormalisedLikelihood, _NoLogprobs()) is False


def test_a_scoring_signal_is_not_applicable_to_an_adapter_without_scoring():
    assert applicable(Confidence, StubAdapter()) is True
    assert applicable(Confidence, _NoScoring()) is False


def test_a_generation_signal_needs_nothing_of_the_adapter():
    assert applicable(VerbalizedConfidence, _NoLogprobs()) is True
    assert applicable(VerbalizedConfidence, _NoScoring()) is True
