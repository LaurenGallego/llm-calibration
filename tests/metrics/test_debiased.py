"""Hand-computed expected values for the bias-corrected binned calibration error."""

import numpy as np
import pytest

from madcal.metrics import (
    debiased_squared_calibration_error,
    ece_debiased,
    ece_equal_mass,
    plugin_squared_calibration_error,
)


def test_a_calibrated_cell_reports_a_negative_squared_error():
    # One bin: mean confidence 0.5, accuracy 2/4 = 0.5, so the plugin squared gap is 0.
    # The sampling variance of that accuracy is 0.5 * 0.5 / (4 - 1) = 1/12, and the
    # debiased estimate is 0 - 1/12.
    assert debiased_squared_calibration_error([0.5] * 4, [1, 1, 0, 0], n_bins=1) == pytest.approx(
        -1 / 12
    )


def test_a_negative_squared_error_reports_as_zero_calibration_error():
    assert ece_debiased([0.5] * 4, [1, 1, 0, 0], n_bins=1) == pytest.approx(0.0)


def test_a_deterministic_cell_needs_no_correction():
    # Accuracy 0 has zero sampling variance, so the squared gap (1 - 0)^2 stands.
    assert debiased_squared_calibration_error([1.0] * 4, [0] * 4, n_bins=1) == pytest.approx(1.0)
    assert ece_debiased([1.0] * 4, [0] * 4, n_bins=1) == pytest.approx(1.0)


def test_two_small_bins_of_pure_noise_are_corrected_away():
    conf = [0.2] * 4 + [0.8] * 4
    correct = [0, 0, 0, 1, 1, 1, 1, 0]
    # bin 0: conf 0.2, accuracy 1/4 = 0.25 -> gap^2 0.0025, variance 0.25*0.75/3 = 0.0625
    # bin 1: conf 0.8, accuracy 3/4 = 0.75 -> gap^2 0.0025, variance 0.75*0.25/3 = 0.0625
    #                          each contributes 0.0025 - 0.0625 = -0.06 at weight 0.5
    assert debiased_squared_calibration_error(conf, correct, n_bins=2) == pytest.approx(-0.06)
    assert ece_debiased(conf, correct, n_bins=2) == pytest.approx(0.0)
    # The plugin estimator instead reports a calibration error that is entirely noise.
    assert ece_equal_mass(conf, correct, n_bins=2) == pytest.approx(0.05)


def test_it_stays_below_the_plugin_estimator_of_the_same_quantity():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.0, 1.0, 400)
    correct = rng.uniform(0.0, 1.0, 400) < conf * 0.5
    debiased = debiased_squared_calibration_error(conf, correct, n_bins=10)
    plugin = plugin_squared_calibration_error(conf, correct, n_bins=10)
    assert 0.0 < debiased < plugin


def test_the_squared_family_is_not_on_the_same_scale_as_the_absolute_family():
    # The root of a mean square never falls below a mean absolute value, so a table
    # putting ece_debiased beside ece_equal_mass compares two different norms.
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.0, 1.0, 400)
    correct = rng.uniform(0.0, 1.0, 400) < conf * 0.5
    root_mean_square = np.sqrt(plugin_squared_calibration_error(conf, correct, n_bins=10))
    assert root_mean_square >= ece_equal_mass(conf, correct, n_bins=10)


def test_a_singleton_bin_is_refused_rather_than_left_uncorrected():
    # Its sampling variance is undefined, and leaving it uncorrected reinstates the bias
    # the estimator exists to remove.
    with pytest.raises(ValueError, match="at least 2 questions"):
        debiased_squared_calibration_error([0.1, 0.5, 0.9], [1, 0, 1], n_bins=3)


def test_bin_count_below_one_is_refused():
    with pytest.raises(ValueError, match="n_bins"):
        ece_debiased([0.5, 0.6], [1, 0], n_bins=0)
