import pytest

from madcal.metrics import brier_score

from .test_ece import REF_CONF, REF_CORRECT


def test_reference_case_brier():
    # (0.05-0)^2 = 0.0025   (0.55-0)^2 = 0.3025
    # (0.15-0)^2 = 0.0225   (0.65-1)^2 = 0.1225
    # (0.25-0)^2 = 0.0625   (0.75-1)^2 = 0.0625
    # (0.35-1)^2 = 0.4225   (0.85-1)^2 = 0.0225
    # (0.45-1)^2 = 0.3025   (0.95-1)^2 = 0.0025
    # sum = 1.3250; mean over 10 = 0.13250
    assert brier_score(REF_CONF, REF_CORRECT) == pytest.approx(0.1325)


def test_perfect_prediction_is_zero():
    assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


def test_maximally_wrong_prediction_is_one():
    assert brier_score([1.0, 0.0], [0, 1]) == pytest.approx(1.0)


def test_uninformative_half_confidence_is_a_quarter():
    # Every squared error is (0.5)^2 = 0.25 regardless of the label.
    assert brier_score([0.5] * 6, [1, 0, 1, 0, 1, 0]) == pytest.approx(0.25)
