"""Hand-computed expected values for the quantile-binned ECE estimator."""

import numpy as np
import pytest

from madcal.metrics import (
    ece_equal_mass,
    equal_mass_bin_statistics,
    expected_calibration_error,
    quantile_edges,
)

# Ten questions whose confidence piles into the top fifth of the range, which is what a
# verbalized-confidence distribution looks like. With two bins the median sits at
# (0.92 + 0.95) / 2 = 0.935, so five questions fall each side.
SKEWED_CONF = [0.80, 0.85, 0.88, 0.90, 0.92, 0.95, 0.96, 0.98, 0.99, 1.00]
SKEWED_CORRECT = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]


def test_quantile_edges_sit_at_the_empirical_median():
    edges = quantile_edges(np.asarray(SKEWED_CONF, dtype=float), 2)
    assert edges == pytest.approx([0.80, 0.935, 1.00])


def test_skewed_case_bin_statistics():
    stats = equal_mass_bin_statistics(SKEWED_CONF, SKEWED_CORRECT, n_bins=2)
    # bin 0: 0.80 0.85 0.88 0.90 0.92 -> sum 4.35, mean 0.870, accuracy 5/5 = 1.0
    # bin 1: 0.95 0.96 0.98 0.99 1.00 -> sum 4.88, mean 0.976, accuracy 2/5 = 0.4
    assert stats["count"].tolist() == [5, 5]
    assert stats["mean_confidence"] == pytest.approx([0.870, 0.976])
    assert stats["accuracy"] == pytest.approx([1.0, 0.4])


def test_skewed_case_equal_mass_ece():
    # bin 0: gap |1.0 - 0.870| = 0.130, weight 0.5 -> 0.065
    # bin 1: gap |0.4 - 0.976| = 0.576, weight 0.5 -> 0.288
    #                                        total -> 0.353
    assert ece_equal_mass(SKEWED_CONF, SKEWED_CORRECT, n_bins=2) == pytest.approx(0.353)


def test_equal_width_hides_what_equal_mass_resolves():
    # Every question lands in the upper equal-width bin, where an underconfident half and
    # an overconfident half cancel: |0.7 - 0.923| = 0.223 against the 0.353 above.
    assert expected_calibration_error(SKEWED_CONF, SKEWED_CORRECT, n_bins=2) == pytest.approx(0.223)


def test_uniform_confidences_put_equal_counts_in_every_bin():
    conf = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    stats = equal_mass_bin_statistics(conf, [0, 0, 0, 1, 1, 0, 1, 1, 1, 1], n_bins=5)
    assert stats["count"].tolist() == [2, 2, 2, 2, 2]


def test_equal_confidences_are_never_split_across_bins():
    conf = [0.9] * 6 + [0.5] * 4
    stats = equal_mass_bin_statistics(conf, [1] * 6 + [0] * 4, n_bins=5)
    occupied = stats["count"][stats["count"] > 0]
    assert sorted(occupied.tolist()) == [4, 6]


def test_a_single_repeated_confidence_collapses_to_one_bin():
    stats = equal_mass_bin_statistics([0.7] * 8, [1, 1, 1, 1, 1, 0, 0, 0], n_bins=5)
    assert stats["count"].sum() == 8
    assert (stats["count"] > 0).sum() == 1
    # mean confidence 0.7, accuracy 5/8 = 0.625, gap 0.075
    assert ece_equal_mass([0.7] * 8, [1, 1, 1, 1, 1, 0, 0, 0], n_bins=5) == pytest.approx(0.075)


def test_perfect_calibration_is_zero():
    assert ece_equal_mass([0.5] * 4, [1, 1, 0, 0], n_bins=2) == pytest.approx(0.0)


def test_maximally_overconfident_is_one():
    assert ece_equal_mass([1.0] * 5, [0] * 5, n_bins=5) == pytest.approx(1.0)


def test_bin_count_below_one_is_refused():
    with pytest.raises(ValueError, match="n_bins"):
        ece_equal_mass([0.5, 0.6], [1, 0], n_bins=0)
