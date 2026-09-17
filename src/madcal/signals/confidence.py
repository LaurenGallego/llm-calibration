"""Signal 1 -- the probability the model put on the answer it actually gave.

The cheapest signal in the set: a byproduct of the same forward pass that produced the
answer, with no extra generation and no judge. It is also the one a practitioner is most
likely to already be watching, which is why `purpose.md` 5.2 frames the experiment
around whether it is enough.

Two decisions this file depends on, both recorded in `DEVLOG.md`:

- **Renormalised, not raw.** The value is read from `ChoiceScores.probabilities()`,
  which is conditional on the answer being one of the candidates. Whatever mass the
  model put elsewhere in the vocabulary is discarded here, and is recorded separately as
  the `option_mass` diagnostic. Renormalising is what lm-evaluation-harness does and
  what every published MCQ ECE number assumes, so this is the comparability choice.

- **This is the same number `metrics/` receives.** ECE is computed against the model's
  confidence in the answer it gave, which is exactly this value. Orchestration computes
  it once and hands it to both sides; if this file and the metrics input ever diverge,
  the stored row is internally inconsistent and nothing downstream can detect it.

The value is per-question. `purpose.md` calls the signal "mean confidence" because the
comparison in 6.3 is made on a condition-level mean -- that mean is taken in `analysis/`,
over this column, once.
"""

from madcal.signals.base import (
    Evidence,
    Requirements,
    SignalNull,
    SignalValue,
    register_signal,
)


@register_signal("confidence")
class Confidence:
    name = "confidence"
    requires = Requirements(needs_choice_scores=True)

    def compute(self, evidence: Evidence) -> SignalValue:
        if evidence.scores is None:
            return SignalValue(None, SignalNull.NO_CHOICE_SCORES)

        return SignalValue(max(evidence.scores.probabilities()))
