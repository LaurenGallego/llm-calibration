"""Calibration metrics -- the labeled track. Single source of truth for "true" calibration.

Importing this package registers every metric, so `metric_registry` is fully
populated after `import caldrift.metrics`.
"""

from caldrift.metrics.base import (
    CalibrationMetric,
    metric_registry,
    register_metric,
    validate_inputs,
)
from caldrift.metrics.brier import brier_score
from caldrift.metrics.ece import (
    bin_statistics,
    confidence_accuracy_gap,
    expected_calibration_error,
)

__all__ = [
    "CalibrationMetric",
    "bin_statistics",
    "brier_score",
    "confidence_accuracy_gap",
    "expected_calibration_error",
    "metric_registry",
    "register_metric",
    "validate_inputs",
]
