"""Bias-corrected binned calibration error, for cells too small for the plugin estimator.

These are squared-gap (L2) quantities. `ece` and `ece_equal_mass` are mean-absolute-gap
(L1) quantities, and the root of a mean square is never below a mean absolute value, so
the two families are not comparable figures and must not be tabulated as if they were.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from madcal.metrics.base import register_metric, validate_inputs
from madcal.metrics.ece import DEFAULT_N_BINS, quantile_edges, statistics_from_edges


def _occupied_bins(
    confidences: ArrayLike, correct: ArrayLike, n_bins: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    conf, corr = validate_inputs(confidences, correct)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    stats = statistics_from_edges(conf, corr, quantile_edges(conf, n_bins))
    occupied = stats["count"] > 0
    counts = stats["count"][occupied].astype(np.float64)
    return counts, stats["accuracy"][occupied], stats["mean_confidence"][occupied]


def plugin_squared_calibration_error(
    confidences: ArrayLike, correct: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> float:
    """Uncorrected mean squared gap over quantile bins."""
    counts, accuracy, mean_confidence = _occupied_bins(confidences, correct, n_bins)
    weights = counts / counts.sum()
    return float(np.sum(weights * (accuracy - mean_confidence) ** 2))


def debiased_squared_calibration_error(
    confidences: ArrayLike, correct: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> float:
    """Debiased mean squared gap over quantile bins. Negative means indistinguishable."""
    counts, accuracy, mean_confidence = _occupied_bins(confidences, correct, n_bins)
    if (counts < 2).any():
        raise ValueError(
            f"every occupied bin needs at least 2 questions to estimate its sampling "
            f"variance; smallest bin holds {int(counts.min())}. Lower n_bins from {n_bins}"
        )

    gap_squared = (accuracy - mean_confidence) ** 2
    variance = accuracy * (1.0 - accuracy) / (counts - 1.0)
    weights = counts / counts.sum()
    return float(np.sum(weights * (gap_squared - variance)))


@register_metric("ece_debiased")
def ece_debiased(confidences: ArrayLike, correct: ArrayLike, n_bins: int = DEFAULT_N_BINS) -> float:
    """Root of the debiased mean squared gap, floored at zero."""
    debiased = debiased_squared_calibration_error(confidences, correct, n_bins=n_bins)
    return float(np.sqrt(max(0.0, debiased)))
