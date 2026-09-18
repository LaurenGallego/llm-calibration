"""Confidence modes."""

from madcal.debate.base import ConfidenceMode, confidence_mode_registry

VERBALIZED_INSTRUCTION = (
    "End your response with your confidence that your answer is correct, "
    "in the format 'Confidence: N%' where N is between 0 and 100."
)


NONE = confidence_mode_registry.register("none")(
    ConfidenceMode(name="none", instruction=None, signal=None)
)

VERBALIZED = confidence_mode_registry.register("verbalized")(
    ConfidenceMode(
        name="verbalized",
        instruction=VERBALIZED_INSTRUCTION,
        signal="verbalized_confidence",
    )
)
