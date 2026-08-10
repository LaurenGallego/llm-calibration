"""Brier score: mean squared error between confidence and outcome.

    Brier = (1/N) * sum_i (conf_i - correct_i)^2

Unlike ECE it is a proper scoring rule and needs no binning, so it has no
bin-count sensitivity. It also conflates calibration with discrimination, which
is why it is reported alongside ECE rather than instead of it.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from caldrift.metrics.base import register_metric, validate_inputs


@register_metric("brier")
def brier_score(confidences: ArrayLike, correct: ArrayLike) -> float:
    conf, corr = validate_inputs(confidences, correct)
    return float(np.mean((conf - corr.astype(np.float64)) ** 2))
