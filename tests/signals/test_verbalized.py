import pytest

from madcal.signals import SignalNull, SignalValue, parse_verbalized_confidence

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
