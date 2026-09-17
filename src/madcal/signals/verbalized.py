"""Verbalized confidence parsed from generated text."""

import re

from madcal.signals.base import SignalNull, SignalValue

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
