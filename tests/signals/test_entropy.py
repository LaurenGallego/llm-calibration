"""Hand-computed expected values for the entropy signal and its diagnostic twin.

Every expected number is worked out in the comment above it. The two entropies differ
only in whether the off-option mass is included, so a mix-up returns a plausible number
rather than an error -- which is exactly the failure this file exists to catch.
"""

import math

import pytest

from madcal.models import ChoiceScores
from madcal.signals import (
    ConfidenceEntropy,
    Evidence,
    SignalNull,
    residual_distribution,
    shannon_entropy,
)

WORKED_RAW = (0.55, 0.15, 0.06, 0.04)
WORKED = ChoiceScores(
    choices=("A", "B", "C", "D"),
    logprobs=tuple(math.log(p) for p in WORKED_RAW),
)


def test_registered_signal_uses_the_renormalised_distribution():
    # renormalised: 0.6875 / 0.1875 / 0.075 / 0.05
    #   0.6875 * ln 0.6875 = -0.2576017
    #   0.1875 * ln 0.1875 = -0.3138706
    #   0.075  * ln 0.075  = -0.1942700
    #   0.05   * ln 0.05   = -0.1497866
    #                 sum  = -0.9155290  ->  H = 0.9155290 nats
    result = ConfidenceEntropy().compute(Evidence(scores=WORKED))
    assert result.value == pytest.approx(0.9155289788717947)
    assert result.reason is None


def test_residual_entropy_is_a_different_number():
    # residual-inclusive: 0.55 / 0.15 / 0.06 / 0.04, plus 0.20 for the mass the model
    # put outside the option set.
    #   0.55 * ln 0.55 = -0.3288104
    #   0.15 * ln 0.15 = -0.2845680
    #   0.06 * ln 0.06 = -0.1688046
    #   0.04 * ln 0.04 = -0.1287550
    #   0.20 * ln 0.20 = -0.3218876
    #             sum  = -1.2328256  ->  H = 1.2328256 nats
    diagnostic = shannon_entropy(residual_distribution(WORKED))
    assert diagnostic == pytest.approx(1.2328256066356236)

    # The guard that matters: the registered signal must not silently become this one.
    registered = ConfidenceEntropy().compute(Evidence(scores=WORKED)).value
    assert registered != pytest.approx(diagnostic)


def test_uniform_distribution_is_log_k():
    # Four equal options: H = ln 4 = 1.3862943611198906. Anchors both the log base and
    # the sign -- a base-2 implementation returns 2.0 and a sign-flipped one returns
    # -1.386, and both look reasonable until compared against a published number. The
    # literal is written out rather than computed as math.log(4), so the test does not
    # agree with a wrong implementation by using the same call it does.
    assert shannon_entropy([0.25] * 4) == pytest.approx(1.3862943611198906)


def test_degenerate_distribution_is_zero():
    # All mass on one option -> H = 0, and the 0*log0 convention in its simplest form:
    # three zero terms that must be skipped rather than raising. Note this case does
    # *not* anchor the sign, since -0.0 == approx(0.0) -- the uniform case above is
    # what catches a dropped minus.
    assert shannon_entropy([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_residual_zero_bucket_does_not_raise():
    # Two options at 0.5 carry all the mass, so residual_distribution's final bucket is
    # exactly 0.0 and the entropy is ln 2 = 0.6931471805599453 rather than a math
    # domain error. The zero has to *arrive* from the clamp rather than be typed in --
    # that is the part a hand-written [0.5, 0.5, 0.0] does not exercise.
    full = ChoiceScores(choices=("A", "B"), logprobs=(math.log(0.5), math.log(0.5)))
    distribution = residual_distribution(full)
    assert distribution == pytest.approx((0.5, 0.5, 0.0))
    assert shannon_entropy(distribution) == pytest.approx(0.6931471805599453)


def test_rejects_a_distribution_that_is_not_one():
    # `match` is load-bearing, not decoration. A NaN makes the sum NaN, so a NaN input
    # raises "must sum to 1" even with the finite check deleted -- pinning the message
    # is the only way this test notices that the finite guard is gone.
    with pytest.raises(ValueError, match="sum to 1"):
        shannon_entropy([0.5, 0.2])

    # Sums to exactly 1.0, so it can only be caught by the non-negative guard.
    with pytest.raises(ValueError, match="finite & non-negative"):
        shannon_entropy([0.5, -0.2, 0.7])

    with pytest.raises(ValueError, match="finite & non-negative"):
        shannon_entropy([0.5, float("nan"), 0.5])


def test_missing_scores_is_a_null_not_a_number():
    assert ConfidenceEntropy().compute(Evidence()).value is None
    assert ConfidenceEntropy().compute(Evidence()).reason == SignalNull.NO_CHOICE_SCORES
