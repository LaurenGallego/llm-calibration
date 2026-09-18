"""Regression test for the dropped-confidence-zero bin bug in the DMAD reference code.

An implementation that bins on `0 < conf <= upper` never opens a bin for confidence
exactly 0, so ten confident-at-zero answers vanish from the sum and a badly calibrated
system reports a calibration error of 0.0. Every estimator here must see it.
"""

import pytest

from madcal.metrics import (
    brier_score,
    ece_debiased,
    ece_equal_mass,
    expected_calibration_error,
    smooth_ece,
)

# Ten answers given at confidence 0.0 that are all correct, and ten given at confidence
# 1.0 that are all correct. Half the questions are maximally underconfident.
CONF = [0.0] * 10 + [1.0] * 10
CORRECT = [1] * 20


def test_equal_width_ece_sees_the_confidence_zero_bin():
    # bin 0:  mean confidence 0.0, accuracy 1.0 -> gap 1.0, weight 10/20 -> 0.5
    # bin 14: mean confidence 1.0, accuracy 1.0 -> gap 0.0, weight 10/20 -> 0.0
    assert expected_calibration_error(CONF, CORRECT) == pytest.approx(0.5)


def test_equal_mass_ece_sees_the_confidence_zero_bin():
    # The quantile edges collapse to [0.0, 1.0], so one bin holds every question:
    # mean confidence 0.5, accuracy 1.0, gap 0.5.
    assert ece_equal_mass(CONF, CORRECT) == pytest.approx(0.5)


def test_debiased_ece_sees_the_confidence_zero_bin():
    # One bin of twenty: gap^2 is 0.25 and accuracy 1.0 has no sampling variance.
    assert ece_debiased(CONF, CORRECT) == pytest.approx(0.5)


def test_smooth_ece_sees_the_confidence_zero_bin():
    # Residual +1 on ten of twenty questions, so half the mass is unexplained.
    assert smooth_ece(CONF, CORRECT) == pytest.approx(0.5, abs=1e-6)


def test_brier_sees_the_confidence_zero_answers():
    # Ten squared errors of 1.0 and ten of 0.0, over twenty questions.
    assert brier_score(CONF, CORRECT) == pytest.approx(0.5)


def test_no_estimator_reports_this_system_as_calibrated():
    for value in (
        expected_calibration_error(CONF, CORRECT),
        ece_equal_mass(CONF, CORRECT),
        ece_debiased(CONF, CORRECT),
        smooth_ece(CONF, CORRECT),
    ):
        assert value > 0.4
