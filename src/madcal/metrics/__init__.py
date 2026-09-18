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

__all__ = [
    "CalibrationMetric",
    "bin_statistics",
    "brier_score",
    "confidence_accuracy_gap",
    "ece_equal_mass",
    "equal_mass_bin_statistics",
    "expected_calibration_error",
    "metric_registry",
    "quantile_edges",
    "register_metric",
    "statistics_from_edges",
    "validate_inputs",
    "weighted_gap",
]
