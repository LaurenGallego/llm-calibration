"""Signal 2 -- the shape of the distribution the answer was drawn from.

Confidence reports only the height of the winning option; entropy reports whether the
rest of the mass was concentrated on one rival or spread evenly across all of them.
Both come free from the same forward pass.

Two distributions can be built from one `ChoiceScores`, and they are not the same signal:

    raw option probabilities   A 0.55  B 0.15  C 0.06  D 0.04      -> option_mass 0.80
    renormalised               A 0.6875  B 0.1875  C 0.075  D 0.05 -> H = 0.9155 nats
    residual-inclusive         A 0.55  B 0.15  C 0.06  D 0.04  + 0.20 -> H = 1.2328 nats

The gap between them *is* the option mass -- the probability the model put somewhere
outside the candidate set. That quantity moves with format compliance, and format
compliance moves with alignment stage, which is the primary drift axis. So the residual
version may carry information the renormalised one deletes, or may just be measuring
instruction-following in an entropy costume.

**The renormalised version is the registered signal** (`DEVLOG.md`): it is what prior
work reports, and the choice had to be made before any data existed rather than after
seeing which correlates better with ECE. The residual version is stored as a diagnostic
column, computed by `orchestration/` as
`shannon_entropy(residual_distribution(scores))` -- which is why both helpers are
module-level and public. The arithmetic exists once; a second copy in orchestration
could disagree with this one and nothing downstream would notice.

Natural log throughout, unnormalised. Dividing by log k would bake in an assumption
that k never varies, and k is fixed at 4 only for as long as MMLU is the only benchmark.
"""

import math
from collections.abc import Sequence

from caldrift.models import PROBABILITY_TOLERANCE, ChoiceScores
from caldrift.signals.base import (
    Evidence,
    Requirements,
    SignalNull,
    SignalValue,
    register_signal,
)


def shannon_entropy(probabilities: Sequence[float]) -> float:
    if len(probabilities) == 0:
        raise ValueError("Probabilities must contain at least one entry.")

    if any(not math.isfinite(p) or p < 0.0 for p in probabilities):
        raise ValueError(
            f"Probabilities must be finite & non-negative, got {tuple(probabilities)}."
        )

    total = math.fsum(probabilities)
    if not math.isclose(total, 1.0, abs_tol=PROBABILITY_TOLERANCE):
        raise ValueError(f"Probabilities must sum to 1, got {total}.")

    # p == 0 is a legal outcome with zero surprise (the 0*log0 convention), and
    # residual_distribution produces one whenever a model puts all its mass on the
    # options. math.log would raise on it, so the term is skipped rather than computed.
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)


def residual_distribution(scores: ChoiceScores) -> tuple[float, ...]:
    residual = 1.0 - scores.option_mass()

    # A mass over 1 is rejected by ChoiceScores itself, which is where an adapter
    # scoring overlapping events should be caught. What survives to here is float error
    # of a few ulps, and clamping that is not the same concession.
    if residual < -PROBABILITY_TOLERANCE:
        raise ValueError(f"option mass exceeds 1 beyond tolerance, residual {residual}.")

    options = tuple(math.exp(logprob) for logprob in scores.logprobs)
    return (*options, max(residual, 0.0))


@register_signal("confidence_entropy")
class ConfidenceEntropy:
    name = "confidence_entropy"
    requires = Requirements(needs_choice_scores=True)

    def compute(self, evidence: Evidence) -> SignalValue:
        if evidence.scores is None:
            return SignalValue(None, SignalNull.NO_CHOICE_SCORES)

        return SignalValue(shannon_entropy(evidence.scores.probabilities()))
