import pytest

from madcal.debate import ConfidenceMode, confidence_mode_registry
from madcal.models import Generation
from madcal.signals import Evidence, SignalNull, SignalValue


def evidence(text: str) -> Evidence:
    return Evidence(
        generations=(Generation(text=text, token_logprobs=None, tokens=None, finish_reason="stop"),)
    )


def test_verbalized_reads_the_registered_signal():
    mode = confidence_mode_registry.get("verbalized")
    assert mode.signal == "verbalized_confidence"
    assert mode.read(evidence("Answer: B\nConfidence: 85%")) == SignalValue(0.85)


def test_verbalized_returns_a_parse_failed_null_never_a_zero():
    mode = confidence_mode_registry.get("verbalized")
    assert mode.read(evidence("Answer: B")) == SignalValue(None, SignalNull.PARSE_FAILED)


def test_none_asks_for_nothing_and_reads_nothing_back():
    mode = confidence_mode_registry.get("none")
    assert mode.instruction is None
    assert mode.signal is None
    assert mode.read(evidence("Confidence: 85%")) == SignalValue(None, SignalNull.NOT_APPLICABLE)


def test_a_mode_naming_an_unregistered_signal_is_refused():
    with pytest.raises(ValueError, match="unknown signal"):
        ConfidenceMode(name="token", instruction="state it", signal="token_confidence")


@pytest.mark.parametrize(
    ("instruction", "signal"),
    [("state it", None), (None, "verbalized_confidence")],
)
def test_a_mode_must_state_an_instruction_and_name_a_signal_or_neither(instruction, signal):
    with pytest.raises(ValueError, match="or neither"):
        ConfidenceMode(name="half", instruction=instruction, signal=signal)
