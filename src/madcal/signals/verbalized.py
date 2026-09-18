"""Verbalized confidence parsed from generated text."""

import re

from madcal.signals.base import (
    Evidence,
    Requirements,
    SignalNull,
    SignalValue,
    register_signal,
    sole_generation,
)

CONFIDENCE_PATTERN = re.compile(r"(?i)confidence[ \t]*:[ \t]*(\d+(?:\.\d+)?)[ \t]*%")


def parse_verbalized_confidence(text: str) -> SignalValue:
    """Return the last stated `Confidence: N%` as a probability, or a parse-failed null."""
    matches = CONFIDENCE_PATTERN.findall(text)
    if not matches:
        return SignalValue(None, SignalNull.PARSE_FAILED)
    percent = float(matches[-1])
    if percent > 100.0:
        return SignalValue(None, SignalNull.PARSE_FAILED)
    return SignalValue(percent / 100.0)


@register_signal("verbalized_confidence")
class VerbalizedConfidence:
    """The probability an agent stated in words, read back from its own response."""

    name = "verbalized_confidence"
    requires = Requirements(needs_generations=True)

    def compute(self, evidence: Evidence) -> SignalValue:
        if evidence.generations is None:
            return SignalValue(None, SignalNull.NO_GENERATIONS)
        return parse_verbalized_confidence(sole_generation(evidence).text)
