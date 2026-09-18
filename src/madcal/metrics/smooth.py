"""SmoothECE: binning-free calibration error, after Blasiok and Nakkiran (2023)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from madcal.metrics.base import register_metric, validate_inputs

GRID_SIZE = 4096
KERNEL_RADII = 4.0
MIN_BANDWIDTH = KERNEL_RADII / GRID_SIZE
BISECTION_STEPS = 40


def convolve_valid(signal: NDArray[np.float64], kernel: NDArray[np.float64]) -> NDArray[np.float64]:
    """`np.convolve(signal, kernel, "valid")` evaluated through the frequency domain."""
    size = 1 << int(signal.size + kernel.size - 1).bit_length()
    product = np.fft.rfft(signal, size) * np.fft.rfft(kernel, size)
    full = np.fft.irfft(product, size)[: signal.size + kernel.size - 1]
    return full[kernel.size - 1 : signal.size]


def reflected_gaussian_smoothing(
    values: NDArray[np.float64], bandwidth: float
) -> NDArray[np.float64]:
    """Gaussian smoothing of a signal on [0, 1], reflected at both boundaries."""
    sigma_cells = bandwidth * values.size
    radius = math.ceil(KERNEL_RADII * sigma_cells)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_cells) ** 2)
    kernel /= kernel.sum()
    return convolve_valid(np.pad(values, radius, mode="symmetric"), kernel)


def smooth_ece_at(confidences: ArrayLike, correct: ArrayLike, bandwidth: float) -> float:
    """SmoothECE at a stated kernel bandwidth."""
    conf, corr = validate_inputs(confidences, correct)
    if not 0.0 < bandwidth <= 1.0:
        raise ValueError(f"bandwidth must lie in (0, 1], got {bandwidth}")

    residual = corr.astype(np.float64) - conf
    cell = np.clip((conf * GRID_SIZE).astype(np.int64), 0, GRID_SIZE - 1)
    signed_mass = np.asarray(
        np.bincount(cell, weights=residual, minlength=GRID_SIZE), dtype=np.float64
    )
    return float(np.abs(reflected_gaussian_smoothing(signed_mass, bandwidth)).sum() / conf.size)


@register_metric("smooth_ece")
def smooth_ece(confidences: ArrayLike, correct: ArrayLike) -> float:
    """SmoothECE at its self-consistent bandwidth, where the bandwidth equals the error."""
    conf, corr = validate_inputs(confidences, correct)
    low = MIN_BANDWIDTH
    if smooth_ece_at(conf, corr, low) <= low:
        return smooth_ece_at(conf, corr, low)

    high = 1.0
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if smooth_ece_at(conf, corr, middle) > middle:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)
