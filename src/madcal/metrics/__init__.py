"""Calibration metrics -- the labeled track. Single source of truth for "true" calibration.

Importing this package registers every metric, so `metric_registry` is fully
populated after `import madcal.metrics`.
"""

from madcal.metrics.base import (
    CalibrationMetric,
    metric_registry,
    register_metric,
    validate_inputs,
)
from madcal.metrics.brier import brier_score
from madcal.metrics.debiased import (
    debiased_squared_calibration_error,
    ece_debiased,
    plugin_squared_calibration_error,
)
from madcal.metrics.discrimination import auroc, average_ranks
from madcal.metrics.ece import (
    bin_statistics,
    confidence_accuracy_gap,
    ece_equal_mass,
    equal_mass_bin_statistics,
    expected_calibration_error,
    quantile_edges,
    statistics_from_edges,
    weighted_gap,
)
from madcal.metrics.smooth import (
    reflected_gaussian_smoothing,
    smooth_ece,
    smooth_ece_at,
)

__all__ = [
    "CalibrationMetric",
    "auroc",
    "average_ranks",
    "bin_statistics",
    "brier_score",
    "confidence_accuracy_gap",
    "debiased_squared_calibration_error",
    "ece_debiased",
    "ece_equal_mass",
    "equal_mass_bin_statistics",
    "expected_calibration_error",
    "metric_registry",
    "plugin_squared_calibration_error",
    "quantile_edges",
    "reflected_gaussian_smoothing",
    "register_metric",
    "smooth_ece",
    "smooth_ece_at",
    "statistics_from_edges",
    "validate_inputs",
    "weighted_gap",
]
