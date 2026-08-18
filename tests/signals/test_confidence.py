"""Hand-computed expected values for the confidence signal.

This number is also what `metrics/` receives as the ECE input, so an error here is
wrong twice over and visible neither time.
"""

import math

import pytest

from caldrift.models import ChoiceScores
from caldrift.signals import Confidence, Evidence, SignalNull

# The worked example from DEVLOG: raw option probabilities 0.55 / 0.15 / 0.06 / 0.04,
# which put 0.80 of the model's mass on the four options and 0.20 elsewhere.
WORKED_RAW = (0.55, 0.15, 0.06, 0.04)
WORKED = ChoiceScores(
    choices=("A", "B", "C", "D"),
    logprobs=tuple(math.log(p) for p in WORKED_RAW),
)


def test_worked_example_is_the_renormalised_top_probability():
    # option mass  = 0.55 + 0.15 + 0.06 + 0.04 = 0.80
    # renormalised = 0.6875 / 0.1875 / 0.075 / 0.05
    # confidence   = max = 0.6875   (raw 0.55 would be the un-renormalised answer)
    result = Confidence().compute(Evidence(scores=WORKED))
    assert result.value == pytest.approx(0.6875)
    assert result.reason is None


def test_missing_scores_is_a_null_not_a_number():
    result = Confidence().compute(Evidence())
    assert result.value is None
    assert result.reason == SignalNull.NO_CHOICE_SCORES
