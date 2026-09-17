"""Hand-computed expected values for the binned-ECE estimator.

Every expected number in this file is worked out by hand in the comment above
it. "The code ran" is not evidence of correctness for this package -- a wrong
ECE is silent and invalidates every downstream result.
"""

import numpy as np
import pytest

from madcal.metrics import bin_statistics, confidence_accuracy_gap, expected_calibration_error

# Reference case, reused across tests. Ten questions, one per half-decile, with
# five equal-width bins (edges 0.0 / 0.2 / 0.4 / 0.6 / 0.8 / 1.0), so exactly two
# questions land in each bin.
REF_CONF = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
REF_CORRECT = [0, 0, 0, 1, 1, 0, 1, 1, 1, 1]


def test_reference_case_ece():
    # bin 0 [0.0,0.2): conf 0.10, acc 0/2 = 0.0 -> gap 0.10, weight 0.2 -> 0.020
    # bin 1 [0.2,0.4): conf 0.30, acc 1/2 = 0.5 -> gap 0.20, weight 0.2 -> 0.040
    # bin 2 [0.4,0.6): conf 0.50, acc 1/2 = 0.5 -> gap 0.00, weight 0.2 -> 0.000
    # bin 3 [0.6,0.8): conf 0.70, acc 2/2 = 1.0 -> gap 0.30, weight 0.2 -> 0.060
    # bin 4 [0.8,1.0]: conf 0.90, acc 2/2 = 1.0 -> gap 0.10, weight 0.2 -> 0.020
    #                                                             total -> 0.140
    assert expected_calibration_error(REF_CONF, REF_CORRECT, n_bins=5) == pytest.approx(0.140)


def test_reference_case_bin_statistics():
    stats = bin_statistics(REF_CONF, REF_CORRECT, n_bins=5)
    assert stats["count"].tolist() == [2, 2, 2, 2, 2]
    assert stats["mean_confidence"] == pytest.approx([0.10, 0.30, 0.50, 0.70, 0.90])
    assert stats["accuracy"] == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0])
    assert stats["count"].sum() == len(REF_CONF)


def test_reference_case_confidence_accuracy_gap():
    # mean confidence = 5.0 / 10 = 0.50; accuracy = 6 / 10 = 0.60; gap = -0.10
    assert confidence_accuracy_gap(REF_CONF, REF_CORRECT) == pytest.approx(-0.10)


def test_perfect_calibration_is_zero():
    # All confidence 0.5, exactly half correct: the single occupied bin has
    # conf 0.5 and accuracy 0.5, so the gap is 0.
    assert expected_calibration_error([0.5] * 4, [1, 1, 0, 0], n_bins=5) == pytest.approx(0.0)


def test_maximally_overconfident_is_one():
    # Confidence 1.0 on every question, every one wrong: gap 1.0 in one bin.
    assert expected_calibration_error([1.0] * 5, [0] * 5, n_bins=10) == pytest.approx(1.0)


def test_maximally_confident_and_correct_is_zero():
    assert expected_calibration_error([1.0] * 5, [1] * 5, n_bins=10) == pytest.approx(0.0)


def test_empty_bins_contribute_nothing():
    # Two questions, 100 bins: 98 bins are empty and must not dilute the result.
    # 0.85 correct   -> gap |1.0 - 0.85| = 0.15, weight 0.5 -> 0.075
    # 0.95 incorrect -> gap |0.0 - 0.95| = 0.95, weight 0.5 -> 0.475
    #                                                 total -> 0.550
    assert expected_calibration_error([0.85, 0.95], [1, 0], n_bins=100) == pytest.approx(0.550)


def test_empty_bins_report_nan_not_zero():
    stats = bin_statistics([0.85, 0.95], [1, 0], n_bins=10)
    assert np.isnan(stats["accuracy"][:8]).all()
    assert np.isnan(stats["mean_confidence"][:8]).all()
    assert stats["count"][:8].tolist() == [0] * 8


@pytest.mark.parametrize(
    ("confidence", "expected_bin"),
    [
        (0.0, 0),  # lower edge of the first bin
        (0.199, 0),
        (0.2, 1),  # bin boundary belongs to the upper bin (half-open on the right)
        (0.8, 4),
        (1.0, 4),  # upper edge folds into the final bin rather than overflowing
    ],
)
def test_bin_edges_are_half_open_with_1_folded_into_the_last_bin(confidence, expected_bin):
    stats = bin_statistics([confidence], [1], n_bins=5)
    assert stats["count"][expected_bin] == 1
    assert stats["count"].sum() == 1


def test_single_bin_reduces_to_absolute_confidence_accuracy_gap():
    # With one bin, ECE is |mean(conf) - accuracy| by construction.
    conf, correct = REF_CONF, REF_CORRECT
    assert expected_calibration_error(conf, correct, n_bins=1) == pytest.approx(
        abs(confidence_accuracy_gap(conf, correct))
    )


def test_ece_is_at_least_the_absolute_gap():
    # The binned estimator can never fall below the single-bin value; this is the
    # inequality that makes mean-confidence monitoring a lower bound on ECE.
    lower_bound = abs(confidence_accuracy_gap(REF_CONF, REF_CORRECT))
    for n_bins in (1, 5, 15, 50):
        assert (
            expected_calibration_error(REF_CONF, REF_CORRECT, n_bins=n_bins) >= lower_bound - 1e-12
        )


def test_bin_count_is_validated():
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error(REF_CONF, REF_CORRECT, n_bins=0)
