"""SmoothECE against hand-computed limits and an independent reference implementation."""

import numpy as np
import pytest

from madcal.metrics import smooth_ece, smooth_ece_at
from madcal.metrics.smooth import GRID_SIZE


def reference_smooth_ece(confidences, correct, bandwidth, images=3):
    """Direct summation over reflected Gaussians: no grid convolution, no FFT, no padding."""
    conf = np.asarray(confidences, dtype=float)
    residual = np.asarray(correct, dtype=float) - conf
    grid = (np.arange(GRID_SIZE) + 0.5) / GRID_SIZE

    def gaussian(distance):
        return np.exp(-0.5 * (distance / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))

    total = np.zeros(GRID_SIZE)
    for image in range(-images, images + 1):
        total += residual @ gaussian(grid[None, :] - conf[:, None] + 2 * image)
        total += residual @ gaussian(grid[None, :] + conf[:, None] + 2 * image)
    return float(np.abs(total / conf.size).sum() / GRID_SIZE)


def test_a_perfectly_calibrated_set_is_exactly_zero():
    assert smooth_ece([0.5] * 100, [1, 0] * 50) == pytest.approx(0.0)


def test_total_confidence_that_is_always_wrong_is_one():
    # Every residual is -1 and sits on the boundary, where reflection folds the half of
    # the kernel that leaves [0, 1] back inside, so no mass is lost.
    assert smooth_ece([1.0] * 50, [0] * 50) == pytest.approx(1.0, abs=1e-6)


def test_no_confidence_that_is_always_right_is_one():
    assert smooth_ece([0.0] * 50, [1] * 50) == pytest.approx(1.0, abs=1e-6)


def test_total_confidence_that_is_always_right_is_zero():
    assert smooth_ece([1.0] * 50, [1] * 50) == pytest.approx(0.0)


def test_a_residual_at_the_centre_carries_its_full_mass():
    # Ten questions at confidence 0.5 that are all correct: the residual is +0.5 per
    # question, spread by the kernel but not lost.
    assert smooth_ece_at([0.5] * 10, [1] * 10, 0.3) == pytest.approx(0.5)


@pytest.mark.parametrize("bandwidth", [0.02, 0.05, 0.15, 0.3])
def test_it_matches_an_independent_reference_implementation(bandwidth):
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.0, 1.0, 300)
    correct = rng.integers(0, 2, 300)
    assert smooth_ece_at(conf, correct, bandwidth) == pytest.approx(
        reference_smooth_ece(conf, correct, bandwidth), abs=2e-4
    )


@pytest.mark.parametrize("bandwidth", [0.02, 0.15])
def test_it_matches_the_reference_when_the_mass_sits_on_a_boundary(bandwidth):
    conf = np.concatenate([np.full(80, 0.02), np.full(80, 0.98)])
    correct = np.concatenate([np.ones(80), np.zeros(80)])
    assert smooth_ece_at(conf, correct, bandwidth) == pytest.approx(
        reference_smooth_ece(conf, correct, bandwidth), abs=2e-4
    )


def test_the_reported_value_is_its_own_bandwidth():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.0, 1.0, 500)
    correct = rng.uniform(0.0, 1.0, 500) < conf * 0.6
    value = smooth_ece(conf, correct)
    assert smooth_ece_at(conf, correct, value) == pytest.approx(value, abs=1e-5)


def test_more_smoothing_never_reports_more_error():
    rng = np.random.default_rng(2)
    conf = rng.uniform(0.0, 1.0, 400)
    correct = rng.integers(0, 2, 400)
    values = [smooth_ece_at(conf, correct, b) for b in (0.01, 0.05, 0.2, 0.5)]
    assert values == sorted(values, reverse=True)


def test_worse_calibration_reports_more_error():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0.0, 1.0, 600)
    draw = rng.uniform(0.0, 1.0, 600)
    calibrated = draw < conf
    overconfident = draw < conf * 0.4
    assert smooth_ece(conf, calibrated) < smooth_ece(conf, overconfident)


@pytest.mark.parametrize("bandwidth", [0.0, -0.1, 1.5])
def test_a_bandwidth_outside_the_unit_interval_is_refused(bandwidth):
    with pytest.raises(ValueError, match="bandwidth"):
        smooth_ece_at([0.5, 0.6], [1, 0], bandwidth)
