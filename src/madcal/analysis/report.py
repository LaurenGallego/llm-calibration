"""The tidy report: one row per cell and metric, carrying its interval and its coverage."""

from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from madcal.analysis.estimate import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RESAMPLES,
    Estimate,
    bootstrap_by_question,
)
from madcal.analysis.load import (
    ANSWER_POLICY,
    CALIBRATION_POLICY,
    CELL_KEYS,
    AnalysisFrame,
    load_frame,
)
from madcal.metrics import metric_registry
from madcal.storage import ResultStore

ACCURACY = "accuracy"
ANALYSIS_STATUS = "exploratory"


@dataclass(frozen=True, slots=True)
class ReportRow:
    """One metric on one cell, with the interval and the row counts it was computed from."""

    model_slug: str
    variant: str
    protocol: str
    benchmark: str
    level: str
    agent_id: str | None
    round: int | None
    metric: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    reason: str | None
    n_questions: int
    n_rows: int
    n_analysable: int
    retention: float
    n_correct: int
    n_incorrect: int
    n_resamples_failed: int
    answer_policy: str
    calibration_policy: str
    analysis_status: str


@dataclass(frozen=True, slots=True)
class Cell:
    """One (protocol, variant, level, agent, round) group of the frame."""

    key: tuple[str | int | None, ...]
    question_ids: NDArray[np.str_]
    graded_correct: NDArray[np.bool_]
    confidence: NDArray[np.float64]
    analysable: NDArray[np.bool_]


def build_report(
    store: ResultStore,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    allow_stub: bool = False,
) -> list[ReportRow]:
    """Compute every metric on every cell of every readable run in the store."""
    frame = load_frame(store, allow_stub=allow_stub)
    return report_rows(
        frame,
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


def report_rows(
    frame: AnalysisFrame,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> list[ReportRow]:
    """Compute every metric on every cell of an already loaded frame."""
    rows: list[ReportRow] = []
    for cell in cells(frame):
        rows.append(
            _row(
                cell,
                ACCURACY,
                _accuracy_estimate(cell, n_resamples, seed, confidence_level),
            )
        )
        for name, metric in metric_registry:
            rows.append(
                _row(
                    cell,
                    name,
                    _metric_estimate(cell, metric, n_resamples, seed, confidence_level),
                )
            )
    return rows


def cells(frame: AnalysisFrame) -> Iterator[Cell]:
    """Split the frame into cells, ordered by cell key."""
    columns = frame.rows.to_pydict()
    n_rows = frame.rows.num_rows
    grouped: dict[tuple[str | int | None, ...], list[int]] = {}
    for index in range(n_rows):
        key = tuple(columns[name][index] for name in CELL_KEYS)
        grouped.setdefault(key, []).append(index)

    question_ids = np.asarray(columns["question_id"], dtype=np.str_)
    graded = np.asarray(columns["graded_correct"], dtype=np.bool_)
    confidence = np.asarray(
        [np.nan if value is None else value for value in columns["confidence"]],
        dtype=np.float64,
    )
    for key in sorted(grouped, key=_sort_key):
        index = np.asarray(grouped[key], dtype=np.intp)
        yield Cell(
            key=key,
            question_ids=question_ids[index],
            graded_correct=graded[index],
            confidence=confidence[index],
            analysable=~np.isnan(confidence[index]),
        )


def _sort_key(key: tuple[str | int | None, ...]) -> tuple[str, ...]:
    return tuple("" if part is None else str(part) for part in key)


def _accuracy_estimate(
    cell: Cell, n_resamples: int, seed: int, confidence_level: float
) -> Estimate:
    graded = cell.graded_correct

    def statistic(indices: NDArray[np.intp]) -> float:
        if indices.size == 0:
            raise ValueError("accuracy needs at least one row")
        return float(np.mean(graded[indices]))

    return bootstrap_by_question(
        statistic,
        cell.question_ids.tolist(),
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


def _metric_estimate(
    cell: Cell, metric, n_resamples: int, seed: int, confidence_level: float
) -> Estimate:
    confidence = cell.confidence[cell.analysable]
    graded = cell.graded_correct[cell.analysable]

    def statistic(indices: NDArray[np.intp]) -> float:
        return float(metric(confidence[indices], graded[indices]))

    return bootstrap_by_question(
        statistic,
        cell.question_ids[cell.analysable].tolist(),
        n_resamples=n_resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


def _row(cell: Cell, metric: str, estimate: Estimate) -> ReportRow:
    model_slug, variant, protocol, benchmark, level, agent_id, round_ = cell.key
    n_rows = int(cell.graded_correct.size)
    n_analysable = int(cell.analysable.sum())
    analysable_correct = cell.graded_correct[cell.analysable]
    return ReportRow(
        model_slug=str(model_slug),
        variant=str(variant),
        protocol=str(protocol),
        benchmark=str(benchmark),
        level=str(level),
        agent_id=None if agent_id is None else str(agent_id),
        round=None if round_ is None else int(round_),
        metric=metric,
        estimate=estimate.value,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        reason=None if estimate.reason is None else str(estimate.reason),
        n_questions=int(np.unique(cell.question_ids).size),
        n_rows=n_rows,
        n_analysable=n_analysable,
        retention=0.0 if n_rows == 0 else n_analysable / n_rows,
        n_correct=int(analysable_correct.sum()),
        n_incorrect=int((~analysable_correct).sum()),
        n_resamples_failed=estimate.n_resamples_failed,
        answer_policy=ANSWER_POLICY,
        calibration_policy=CALIBRATION_POLICY,
        analysis_status=ANALYSIS_STATUS,
    )


def to_arrow(rows: list[ReportRow]) -> pa.Table:
    """Return the report as a table, so a number in a plot and a number in a table agree."""
    columns = {
        field.name: [getattr(row, field.name) for row in rows] for field in fields(ReportRow)
    }
    return pa.table(columns, schema=REPORT_SCHEMA)


REPORT_SCHEMA = pa.schema(
    [
        pa.field("model_slug", pa.string(), nullable=False),
        pa.field("variant", pa.string(), nullable=False),
        pa.field("protocol", pa.string(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("level", pa.string(), nullable=False),
        pa.field("agent_id", pa.string(), nullable=True),
        pa.field("round", pa.int32(), nullable=True),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("estimate", pa.float64(), nullable=True),
        pa.field("ci_low", pa.float64(), nullable=True),
        pa.field("ci_high", pa.float64(), nullable=True),
        pa.field("reason", pa.string(), nullable=True),
        pa.field("n_questions", pa.int64(), nullable=False),
        pa.field("n_rows", pa.int64(), nullable=False),
        pa.field("n_analysable", pa.int64(), nullable=False),
        pa.field("retention", pa.float64(), nullable=False),
        pa.field("n_correct", pa.int64(), nullable=False),
        pa.field("n_incorrect", pa.int64(), nullable=False),
        pa.field("n_resamples_failed", pa.int64(), nullable=False),
        pa.field("answer_policy", pa.string(), nullable=False),
        pa.field("calibration_policy", pa.string(), nullable=False),
        pa.field("analysis_status", pa.string(), nullable=False),
    ]
)


def as_dicts(rows: list[ReportRow]) -> list[dict[str, object]]:
    """Return the report rows as plain dictionaries."""
    return [asdict(row) for row in rows]
