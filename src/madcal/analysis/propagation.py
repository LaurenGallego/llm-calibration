"""Calibration from individual agents to the debate system, by level and paired contrast."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from madcal.analysis.estimate import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RESAMPLES,
    Estimate,
    bootstrap_by_question,
)
from madcal.analysis.load import AnalysisFrame
from madcal.analysis.report import ACCURACY, ANALYSIS_STATUS
from madcal.debate import choice_options, pooled_answer
from madcal.metrics import metric_registry

MEAN_CONFIDENCE = "mean_confidence"
INDIVIDUAL = "individual"
ENSEMBLE = "ensemble"
DEBATED_AGENTS = "debated_agents"
SYSTEM = "system"
LEVELS = (INDIVIDUAL, ENSEMBLE, DEBATED_AGENTS, SYSTEM)
CONTRASTS = (
    (ENSEMBLE, INDIVIDUAL),
    (DEBATED_AGENTS, INDIVIDUAL),
    (SYSTEM, ENSEMBLE),
    (SYSTEM, INDIVIDUAL),
)
CONDITION_KEYS = ("model_slug", "variant", "protocol", "benchmark")

type Labels = Mapping[str, tuple[str, int]]


@dataclass(frozen=True, slots=True)
class PropagationRow:
    """One metric on one level, or the paired difference between two levels."""

    model_slug: str
    variant: str
    protocol: str
    benchmark: str
    comparison: str
    metric: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    reason: str | None
    n_questions: int
    n_rows: int
    n_analysable: int
    n_resamples_failed: int
    analysis_status: str


@dataclass(frozen=True, slots=True)
class LevelRows:
    """Graded answers and stated confidences of one level, one row per agent turn or question."""

    question_ids: NDArray[np.str_]
    correct: NDArray[np.bool_]
    confidence: NDArray[np.float64]


def propagation_rows(
    frame: AnalysisFrame,
    labels: Labels,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> list[PropagationRow]:
    """Compute every metric on every level and contrast of every debate condition in the frame."""
    rows: list[PropagationRow] = []
    for condition, records in _conditions(frame).items():
        levels = propagation_levels(records, labels)
        metrics = confidence_metrics()
        for name in LEVELS:
            for metric, compute in metrics.items():
                estimate = bootstrap_by_question(
                    _statistic(levels[name], compute),
                    levels[name].question_ids.tolist(),
                    n_resamples=n_resamples,
                    seed=seed,
                    confidence_level=confidence_level,
                )
                rows.append(_row(condition, name, metric, estimate, [levels[name]]))
        for later, earlier in CONTRASTS:
            pair = [levels[earlier], levels[later]]
            for metric, compute in metrics.items():
                estimate = bootstrap_by_question(
                    _difference(pair, compute),
                    np.concatenate([level.question_ids for level in pair]).tolist(),
                    n_resamples=n_resamples,
                    seed=seed,
                    confidence_level=confidence_level,
                )
                rows.append(_row(condition, f"{later} - {earlier}", metric, estimate, pair))
    return rows


def propagation_levels(
    records: Sequence[Mapping[str, object]], labels: Labels
) -> dict[str, LevelRows]:
    """Return the individual, ensemble, debated-agent and system levels of one condition."""
    agents = [record for record in records if record["level"] == "agent"]
    if not agents:
        raise ValueError("a condition needs agent rows to trace propagation")
    final = max(int(record["round"]) for record in agents)  # pyright: ignore[reportArgumentType]
    if final == 0:
        raise ValueError("a condition with no debate round has nothing to propagate through")
    opening = [record for record in agents if record["round"] == 0]
    closing = [record for record in agents if record["round"] == final]
    return {
        INDIVIDUAL: _agent_level(opening),
        ENSEMBLE: _pooled_level(opening, labels),
        DEBATED_AGENTS: _agent_level(closing),
        SYSTEM: _pooled_level(closing, labels),
    }


def propagation_table(rows: list[PropagationRow]) -> pa.Table:
    """Return the propagation rows as an Arrow table."""
    return pa.Table.from_pylist([asdict(row) for row in rows])


def _conditions(frame: AnalysisFrame) -> dict[tuple[str, ...], list[dict[str, object]]]:
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for record in frame.rows.to_pylist():
        key = tuple(str(record[name]) for name in CONDITION_KEYS)
        grouped.setdefault(key, []).append(record)
    return grouped


def _agent_level(records: Sequence[Mapping[str, object]]) -> LevelRows:
    return LevelRows(
        question_ids=np.asarray([record["question_id"] for record in records], dtype=np.str_),
        correct=np.asarray([record["graded_correct"] for record in records], dtype=np.bool_),
        confidence=np.asarray(
            [
                np.nan if record["confidence"] is None else record["confidence"]
                for record in records
            ],
            dtype=np.float64,
        ),
    )


def _pooled_level(records: Sequence[Mapping[str, object]], labels: Labels) -> LevelRows:
    by_question: dict[str, list[tuple[str, float]]] = {}
    for record in records:
        votes = by_question.setdefault(str(record["question_id"]), [])
        if record["predicted"] is not None and record["confidence"] is not None:
            votes.append((str(record["predicted"]), float(record["confidence"])))  # pyright: ignore[reportArgumentType]
    question_ids, correct, confidence = [], [], []
    for question_id in sorted(by_question):
        if question_id not in labels:
            raise ValueError(f"no label for question {question_id!r}; grade against its benchmark")
        gold, n_choices = labels[question_id]
        answer = pooled_answer(by_question[question_id], choice_options(n_choices))
        question_ids.append(question_id)
        correct.append(answer.answer == gold)
        confidence.append(np.nan if answer.confidence is None else answer.confidence)
    return LevelRows(
        question_ids=np.asarray(question_ids, dtype=np.str_),
        correct=np.asarray(correct, dtype=np.bool_),
        confidence=np.asarray(confidence, dtype=np.float64),
    )


type Compute = Callable[[NDArray[np.bool_], NDArray[np.float64]], float]


def confidence_metrics() -> dict[str, Compute]:
    """Return accuracy, mean confidence and every registered calibration metric."""
    metrics: dict[str, Compute] = {
        ACCURACY: lambda correct, _: _mean(correct),
        MEAN_CONFIDENCE: lambda _, confidence: _mean(confidence[~np.isnan(confidence)]),
    }
    for name, metric in metric_registry:
        metrics[name] = _calibration(metric)
    return metrics


def _calibration(metric) -> Compute:
    def compute(correct: NDArray[np.bool_], confidence: NDArray[np.float64]) -> float:
        stated = ~np.isnan(confidence)
        return float(metric(confidence[stated], correct[stated]))

    return compute


def _mean(values: NDArray) -> float:
    if values.size == 0:
        raise ValueError("a mean needs at least one row")
    return float(np.mean(values))


def _statistic(level: LevelRows, compute: Compute) -> Callable[[NDArray[np.intp]], float]:
    return lambda indices: compute(level.correct[indices], level.confidence[indices])


def _difference(pair: Sequence[LevelRows], compute: Compute) -> Callable[[NDArray[np.intp]], float]:
    earlier, later = pair
    boundary = earlier.question_ids.size

    def statistic(indices: NDArray[np.intp]) -> float:
        first = indices[indices < boundary]
        second = indices[indices >= boundary] - boundary
        return compute(later.correct[second], later.confidence[second]) - compute(
            earlier.correct[first], earlier.confidence[first]
        )

    return statistic


def _row(
    condition: tuple[str, ...],
    comparison: str,
    metric: str,
    estimate: Estimate,
    levels: Sequence[LevelRows],
) -> PropagationRow:
    model_slug, variant, protocol, benchmark = condition
    return PropagationRow(
        model_slug=model_slug,
        variant=variant,
        protocol=protocol,
        benchmark=benchmark,
        comparison=comparison,
        metric=metric,
        estimate=estimate.value,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        reason=None if estimate.reason is None else str(estimate.reason),
        n_questions=len({q for level in levels for q in level.question_ids.tolist()}),
        n_rows=sum(int(level.question_ids.size) for level in levels),
        n_analysable=sum(int((~np.isnan(level.confidence)).sum()) for level in levels),
        n_resamples_failed=estimate.n_resamples_failed,
        analysis_status=ANALYSIS_STATUS,
    )
