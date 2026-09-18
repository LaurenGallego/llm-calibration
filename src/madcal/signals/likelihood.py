"""Length-normalised likelihood of a generated response."""

import math

from madcal.signals.base import (
    Evidence,
    Requirements,
    SignalNull,
    SignalValue,
    register_signal,
    sole_generation,
)


@register_signal("length_normalised_likelihood")
class LengthNormalisedLikelihood:
    """The per-token geometric mean probability of a response, `exp(mean(token_logprobs))`."""

    name = "length_normalised_likelihood"
    requires = Requirements(needs_generations=True, needs_logprobs=True)

    def compute(self, evidence: Evidence) -> SignalValue:
        if evidence.generations is None:
            return SignalValue(None, SignalNull.NO_GENERATIONS)
        logprobs = sole_generation(evidence).token_logprobs
        if not logprobs:
            return SignalValue(None, SignalNull.NO_LOGPROBS)
        return SignalValue(math.exp(math.fsum(logprobs) / len(logprobs)))
