"""Cell-level estimates whose intervals are resampled by question, never by row."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

DEFAULT_RESAMPLES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95

type Statistic = Callable[[NDArray[np.intp]], float]


class EstimateNull(StrEnum):
    NO_ROWS = "no_rows"
    UNDEFINED_ON_CELL = "undefined_on_cell"


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with its resampled interval, or a null with a reason."""

    value: float | None
    ci_low: float | None
    ci_high: float | None
    n_resamples_failed: int
    reason: EstimateNull | None

    def __post_init__(self) -> None:
        if (self.value is None) == (self.reason is None):
            raise ValueError("Estimate needs exactly one of value or reason")
        if (self.ci_low is None) != (self.ci_high is None):
            raise ValueError("an interval needs both bounds or neither")
        if self.value is None and self.ci_low is not None:
            raise ValueError("a null estimate carries no interval")
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError(f"interval is inverted: [{self.ci_low}, {self.ci_high}]")
        if self.n_resamples_failed < 0:
            raise ValueError(f"n_resamples_failed must be >= 0, got {self.n_resamples_failed}")


def bootstrap_by_question(
    statistic: Statistic,
    question_ids: Sequence[str],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> Estimate:
    """Estimate a statistic over rows, resampling whole questions to build its interval."""
    if n_resamples < 0:
        raise ValueError(f"n_resamples must be >= 0, got {n_resamples}")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must lie in (0, 1), got {confidence_level}")
    if len(question_ids) == 0:
        return Estimate(None, None, None, 0, EstimateNull.NO_ROWS)

    rows = np.arange(len(question_ids), dtype=np.intp)
    try:
        point = float(statistic(rows))
    except ValueError:
        return Estimate(None, None, None, 0, EstimateNull.UNDEFINED_ON_CELL)

    _, inverse = np.unique(np.asarray(question_ids), return_inverse=True)
    grouped = [np.flatnonzero(inverse == group) for group in range(int(inverse.max()) + 1)]

    generator = np.random.default_rng(seed)
    resampled: list[float] = []
    failed = 0
    for _ in range(n_resamples):
        picks = generator.integers(0, len(grouped), size=len(grouped))
        indices = np.concatenate([grouped[pick] for pick in picks])
        try:
            resampled.append(float(statistic(indices)))
        except ValueError:
            failed += 1

    if not resampled:
        return Estimate(point, None, None, failed, None)
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(resampled, [tail, 1.0 - tail])
    return Estimate(point, float(low), float(high), failed, None)
