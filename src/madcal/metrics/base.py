"""Calibration-metric protocol, registry, and shared input validation.

A metric consumes ground-truth labels. Nothing in `madcal.signals` may import
from this package -- see CLAUDE.md on the label-leakage boundary.

Inputs are always the same pair:

    confidences  float array in [0, 1] -- the model's confidence in the answer it gave
    correct      boolean array         -- whether that answer was right

Both are per-question and must line up. Validation is strict on purpose: a
NaN or an out-of-range confidence is a bug upstream, and silently dropping it
would change the metric without leaving a trace.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from madcal.registry import Registry


@runtime_checkable
class CalibrationMetric(Protocol):
    """Maps (confidences, correctness) to a single scalar."""

    def __call__(self, confidences: ArrayLike, correct: ArrayLike) -> float: ...


metric_registry: Registry[CalibrationMetric] = Registry("calibration metric")


def register_metric(name: str):
    return metric_registry.register(name)


def validate_inputs(
    confidences: ArrayLike, correct: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Coerce and check the standard metric inputs. Raises rather than repairing."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr_raw = np.asarray(correct)

    if conf.ndim != 1 or corr_raw.ndim != 1:
        raise ValueError(
            f"expected 1-D arrays, got confidences.ndim={conf.ndim}, correct.ndim={corr_raw.ndim}"
        )
    if conf.shape != corr_raw.shape:
        raise ValueError(
            f"length mismatch: {conf.shape[0]} confidences, {corr_raw.shape[0]} labels"
        )
    if conf.size == 0:
        raise ValueError("cannot compute a calibration metric on zero questions")
    if not np.isfinite(conf).all():
        raise ValueError("confidences contain NaN or inf")
    if conf.min() < 0.0 or conf.max() > 1.0:
        raise ValueError(f"confidences must lie in [0, 1], got [{conf.min()}, {conf.max()}]")

    if corr_raw.dtype == np.bool_:
        corr = corr_raw
    else:
        if not np.isin(corr_raw, (0, 1)).all():
            raise ValueError("correct must be boolean or contain only 0/1")
        corr = corr_raw.astype(np.bool_)

    return conf, corr
