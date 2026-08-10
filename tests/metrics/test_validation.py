"""Metric inputs are validated strictly: bad input raises, it is never repaired.

Silently coercing a NaN confidence or dropping an unparseable answer would change
the metric without leaving a trace in the results. See CLAUDE.md.
"""

import numpy as np
import pytest

from caldrift.metrics import expected_calibration_error, metric_registry, validate_inputs


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        validate_inputs([0.5, 0.6], [1])


def test_empty_input_raises():
    with pytest.raises(ValueError, match="zero questions"):
        validate_inputs([], [])


def test_nan_confidence_raises():
    with pytest.raises(ValueError, match="NaN or inf"):
        validate_inputs([0.5, np.nan], [1, 0])


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_out_of_range_confidence_raises(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_inputs([0.5, bad], [1, 0])


def test_non_binary_labels_raise():
    with pytest.raises(ValueError, match="boolean or contain only 0/1"):
        validate_inputs([0.5, 0.6], [1, 2])


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match="1-D"):
        validate_inputs([[0.5, 0.6]], [[1, 0]])


def test_boolean_and_integer_labels_agree():
    conf = [0.2, 0.8, 0.9]
    assert expected_calibration_error(conf, [True, False, True]) == pytest.approx(
        expected_calibration_error(conf, [1, 0, 1])
    )


def test_registry_exposes_every_metric():
    assert set(metric_registry.names()) == {"brier", "confidence_accuracy_gap", "ece"}


def test_registry_lookup_of_unknown_metric_lists_what_exists():
    with pytest.raises(KeyError, match="registered: brier"):
        metric_registry.get("smoothece")


def test_registry_rejects_duplicate_registration():
    with pytest.raises(ValueError, match="already registered"):
        metric_registry.register("ece")(lambda c, k: 0.0)
