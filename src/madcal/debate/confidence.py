"""Confidence modes."""

from madcal.debate.base import ConfidenceMode, confidence_mode_registry
from madcal.signals import SignalNull, SignalValue, parse_verbalized_confidence

VERBALIZED_INSTRUCTION = (
    "End your response with your confidence that your answer is correct, "
    "in the format 'Confidence: N%' where N is between 0 and 100."
)


def _not_requested(text: str) -> SignalValue:
    return SignalValue(None, SignalNull.NOT_APPLICABLE)


NONE = confidence_mode_registry.register("none")(
    ConfidenceMode(name="none", instruction=None, parse=_not_requested)
)

VERBALIZED = confidence_mode_registry.register("verbalized")(
    ConfidenceMode(
        name="verbalized",
        instruction=VERBALIZED_INSTRUCTION,
        parse=parse_verbalized_confidence,
    )
)
