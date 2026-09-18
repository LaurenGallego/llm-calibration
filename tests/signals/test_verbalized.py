import pytest

from madcal.models import Generation
from madcal.signals import (
    Evidence,
    SignalNull,
    SignalValue,
    VerbalizedConfidence,
    parse_verbalized_confidence,
)

PARSE_FAILED = SignalValue(None, SignalNull.PARSE_FAILED)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Answer: B\nConfidence: 85%", 0.85),
        ("confidence:72.5 %", 0.725),
        ("Confidence: 0%", 0.0),
        ("Confidence: 100%", 1.0),
        ("Confidence: 40%. On reflection, Confidence: 90%", 0.9),
    ],
)
def test_stated_percentage_becomes_a_probability(text, expected):
    assert parse_verbalized_confidence(text) == SignalValue(expected)


@pytest.mark.parametrize(
    "text",
    [
        "Answer: B",
        "Confidence: 0.8",
        "Confidence: high",
        "Confidence: 120%",
        "Confidence:\n85%",
    ],
)
def test_unparseable_confidence_is_a_null_not_a_number(text):
    assert parse_verbalized_confidence(text) == PARSE_FAILED


def generation(text: str) -> Generation:
    return Generation(text=text, token_logprobs=None, tokens=None, finish_reason="stop")


def test_the_signal_reads_the_confidence_out_of_its_generation():
    evidence = Evidence(generations=(generation("Answer: B\nConfidence: 85%"),))
    assert VerbalizedConfidence().compute(evidence) == SignalValue(0.85)


def test_an_unparseable_response_is_a_null_with_a_reason_not_a_zero():
    evidence = Evidence(generations=(generation("Answer: B"),))
    assert VerbalizedConfidence().compute(evidence) == PARSE_FAILED


def test_absent_generations_are_a_null_with_a_reason():
    assert VerbalizedConfidence().compute(Evidence()) == SignalValue(
        None, SignalNull.NO_GENERATIONS
    )


def test_several_generations_are_refused_rather_than_silently_reduced():
    several = Evidence(generations=(generation("Confidence: 10%"), generation("Confidence: 90%")))
    with pytest.raises(ValueError, match="single generation"):
        VerbalizedConfidence().compute(several)


def test_it_needs_a_generation_and_no_logprobs():
    requires = VerbalizedConfidence.requires
    assert requires.needs_generations is True
    assert requires.needs_logprobs is False
    assert requires.needs_choice_scores is False
