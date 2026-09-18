"""Expected Calibration Error (equal-width binning).

    ECE = sum_b (n_b / N) * |acc(b) - conf(b)|

This is the Guo et al. (2017) estimator, kept because prior work reports it and
comparability matters. It is biased, and the bias depends on the shape of the
confidence distribution -- which is precisely the thing alignment training
changes. Treat it as the comparability metric, not the ground truth; see the
open issue on making a debiased estimator (SmoothECE) the primary target.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from madcal.metrics.base import register_metric, validate_inputs

DEFAULT_N_BINS = 15


def statistics_from_edges(
    conf: NDArray[np.float64], corr: NDArray[np.bool_], edges: NDArray[np.float64]
) -> dict[str, np.ndarray]:
    """Per-bin counts, mean confidence, and accuracy for already-validated inputs."""
    n_bins = edges.size - 1
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)

    counts = np.bincount(idx, minlength=n_bins).astype(np.int64)
    conf_sum = np.bincount(idx, weights=conf, minlength=n_bins)
    corr_sum = np.bincount(idx, weights=corr.astype(np.float64), minlength=n_bins)

    occupied = counts > 0
    mean_conf = np.full(n_bins, np.nan)
    accuracy = np.full(n_bins, np.nan)
    mean_conf[occupied] = conf_sum[occupied] / counts[occupied]
    accuracy[occupied] = corr_sum[occupied] / counts[occupied]

    return {
        "lower": edges[:-1],
        "upper": edges[1:],
        "count": counts,
        "mean_confidence": mean_conf,
        "accuracy": accuracy,
    }


def bin_statistics(
    confidences: ArrayLike, correct: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> dict[str, np.ndarray]:
    """Per-bin counts, mean confidence, and accuracy. Empty bins report NaN.

    Bins are equal-width over [0, 1], half-open on the right, with 1.0 folded
    into the final bin so a fully-confident answer is not dropped.
    """
    conf, corr = validate_inputs(confidences, correct)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    return statistics_from_edges(conf, corr, np.linspace(0.0, 1.0, n_bins + 1))


def equal_mass_bin_statistics(
    confidences: ArrayLike, correct: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> dict[str, np.ndarray]:
    """Per-bin statistics over quantile bins. Equal confidence values never split."""
    conf, corr = validate_inputs(confidences, correct)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    return statistics_from_edges(conf, corr, quantile_edges(conf, n_bins))


def quantile_edges(conf: NDArray[np.float64], n_bins: int) -> NDArray[np.float64]:
    """Bin edges at the empirical quantiles of `conf`, deduplicated."""
    quantiles = np.quantile(conf, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.asarray(np.unique(quantiles), dtype=np.float64)
    if edges.size == 1:
        return np.repeat(edges, 2)
    return edges


def weighted_gap(stats: dict[str, np.ndarray]) -> float:
    """Count-weighted mean absolute gap between accuracy and confidence."""
    occupied = stats["count"] > 0
    gaps = np.abs(stats["accuracy"][occupied] - stats["mean_confidence"][occupied])
    weights = stats["count"][occupied] / stats["count"].sum()
    return float(np.sum(weights * gaps))


@register_metric("ece")
def expected_calibration_error(
    confidences: ArrayLike, correct: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> float:
    """Equal-width binned ECE. Empty bins contribute nothing."""
    return weighted_gap(bin_statistics(confidences, correct, n_bins=n_bins))


@register_metric("ece_equal_mass")
def ece_equal_mass(
    confidences: ArrayLike, correct: ArrayLike, n_bins: int = DEFAULT_N_BINS
) -> float:
    """Quantile-binned ECE, for confidence distributions that pile onto few values."""
    return weighted_gap(equal_mass_bin_statistics(confidences, correct, n_bins=n_bins))


@register_metric("confidence_accuracy_gap")
def confidence_accuracy_gap(confidences: ArrayLike, correct: ArrayLike) -> float:
    """Signed mean(confidence) - accuracy.

    A lower bound on ECE in absolute value, and the quantity mean-confidence
    monitoring can actually see. Logged per condition so the analysis can
    separate "the signal moved" from "accuracy moved".
    """
    conf, corr = validate_inputs(confidences, correct)
    return float(conf.mean() - corr.mean())
