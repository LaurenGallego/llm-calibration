"""AUROC: how well confidence ranks correct answers above incorrect ones."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from madcal.metrics.base import register_metric, validate_inputs


def average_ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """One-based ranks, with tied values sharing their mean rank."""
    order = np.argsort(values, kind="stable")
    _, start, counts = np.unique(values[order], return_index=True, return_counts=True)
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    return ranks


@register_metric("auroc")
def auroc(confidences: ArrayLike, correct: ArrayLike) -> float:
    """Probability that a correct answer outranks an incorrect one, ties counting a half."""
    conf, corr = validate_inputs(confidences, correct)
    n_correct = int(corr.sum())
    n_incorrect = corr.size - n_correct
    if n_correct == 0 or n_incorrect == 0:
        raise ValueError(
            f"AUROC needs both classes; got {n_correct} correct and {n_incorrect} incorrect. "
            f"A cell with one class has no discrimination to measure, and 0.5 would report "
            f"chance performance that was never observed"
        )

    rank_sum = float(average_ranks(conf)[corr].sum())
    return (rank_sum - n_correct * (n_correct + 1) / 2.0) / (n_correct * n_incorrect)
