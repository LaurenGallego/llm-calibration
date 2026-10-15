"""Calibration of every semantic measure per arm, and its paired difference from a reference."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

from madcal.analysis.estimate import (
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_RESAMPLES,
    Estimate,
    Statistic,
    bootstrap_by_question,
)
from madcal.analysis.load import has_views
from madcal.analysis.propagation import Compute, confidence_metrics
from madcal.analysis.report import ACCURACY, ANALYSIS_STATUS
from madcal.storage import SEMANTIC_PROTOCOL, ResultStore, SignalKind, Variant

SIGNAL_PREFIX = "semantic_"
ANSWER = "answer"
ARM_QUERY = """
WITH readable AS (
    SELECT run_id, config_hash FROM runs
    WHERE status = 'complete' AND completed_at IS NOT NULL AND protocol = $protocol
)
SELECT q.model_slug, q.variant, q.benchmark, r.config_hash, q.question_id,
       coalesce(q.correct, false) AS correct, s.signal, s.value
FROM questions q
JOIN readable r ON q.run_id = r.run_id
JOIN signals s
    ON q.run_id = s.run_id AND q.question_id = s.question_id AND s.level = 'system'
WHERE q.level = 'system' AND s.kind = $kind
ORDER BY ALL
"""

type ArmKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Arm:
    """The graded reference answers of one semantic condition and every measure's confidence."""

    model_slug: str
    benchmark: str
    config_hash: str
    variant: str
    question_ids: tuple[str, ...]
    correct: NDArray[np.bool_]
    confidence: dict[str, NDArray[np.float64]]


@dataclass(frozen=True, slots=True)
class SemanticRow:
    """One metric of one measure for one arm, or its paired difference from the reference arm."""

    model_slug: str
    benchmark: str
    arm: str
    variant: str
    comparison: str
    measure: str
    metric: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    reason: str | None
    n_questions: int
    n_analysable: int
    n_resamples_failed: int
    analysis_status: str


def load_arms(store: ResultStore, allow_stub: bool = False) -> list[Arm]:
    """Read every complete semantic run as one arm per config hash; stub rows only if asked."""
    with store.connect() as connection:
        if not has_views(connection, "runs", "questions", "signals"):
            return []
        rows = connection.execute(
            ARM_QUERY, {"protocol": SEMANTIC_PROTOCOL, "kind": str(SignalKind.SIGNAL)}
        ).fetchall()
    answers: dict[tuple[str, str, str, str], dict[str, tuple[bool, dict[str, float]]]] = {}
    for model, variant, benchmark, config_hash, question_id, correct, signal, value in rows:
        if variant == Variant.STUB and not allow_stub:
            raise ValueError(f"semantic runs hold {Variant.STUB} rows, which are never a result")
        if not signal.startswith(SIGNAL_PREFIX):
            continue
        questions = answers.setdefault((model, benchmark, config_hash, variant), {})
        _, measures = questions.setdefault(question_id, (bool(correct), {}))
        measure = signal.removeprefix(SIGNAL_PREFIX)
        if measure in measures:
            raise ValueError(
                f"{question_id!r} has two {signal!r} rows under config {config_hash!r}; "
                f"a duplicate would count the question twice"
            )
        measures[measure] = math.nan if value is None else float(value)
    hashes = [key[2] for key in answers]
    if len(hashes) != len(set(hashes)):
        raise ValueError("one config hash appears under more than one model, benchmark or variant")
    return [_arm(key, questions) for key, questions in sorted(answers.items())]


def references(arms: Sequence[Arm], chosen: Sequence[str] = ()) -> dict[ArmKey, Arm | None]:
    """Return each model and benchmark's reference arm: the one chosen, else its only base arm."""
    by_hash = {arm.config_hash: arm for arm in arms}
    unknown = sorted(set(chosen) - by_hash.keys())
    if unknown:
        raise ValueError(f"no semantic arm has config hash {unknown}")
    result: dict[ArmKey, Arm | None] = {}
    for key in sorted({(arm.model_slug, arm.benchmark) for arm in arms}):
        picked = [
            by_hash[h] for h in chosen if (by_hash[h].model_slug, by_hash[h].benchmark) == key
        ]
        if len(picked) > 1:
            raise ValueError(f"more than one reference arm chosen for {key}")
        base = [a for a in arms if (a.model_slug, a.benchmark) == key and a.variant == Variant.BASE]
        result[key] = picked[0] if picked else base[0] if len(base) == 1 else None
    return result


def semantic_rows(
    arms: Sequence[Arm],
    reference_arms: Mapping[ArmKey, Arm | None],
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> list[SemanticRow]:
    """Compute every metric of every measure per arm, and against the reference arm."""
    metrics = confidence_metrics()
    accuracy = metrics.pop(ACCURACY)
    settings = {"n_resamples": n_resamples, "seed": seed, "confidence_level": confidence_level}
    rows: list[SemanticRow] = []
    for arm in arms:
        reference = reference_arms.get((arm.model_slug, arm.benchmark))
        contrast = reference is not None and reference is not arm
        own = _aligned(arm, None, arm.question_ids)
        estimate = bootstrap_by_question(_single(own, accuracy), arm.question_ids, **settings)
        rows.append(_row(arm, "arm", ANSWER, ACCURACY, estimate, own))
        if contrast:
            assert reference is not None
            pair = _pair(arm, reference, ANSWER)
            estimate = bootstrap_by_question(_paired(pair, accuracy), pair[0].ids, **settings)
            rows.append(_row(arm, _versus(reference), ANSWER, ACCURACY, estimate, pair[0]))
        for measure, confidence in arm.confidence.items():
            if np.isnan(confidence).all():
                continue
            single = _aligned(arm, measure, arm.question_ids)
            for metric, compute in metrics.items():
                estimate = bootstrap_by_question(
                    _single(single, compute), arm.question_ids, **settings
                )
                rows.append(_row(arm, "arm", measure, metric, estimate, single))
            if not contrast or reference is None or measure not in reference.confidence:
                continue
            pair = _pair(arm, reference, measure)
            for metric, compute in metrics.items():
                estimate = bootstrap_by_question(_paired(pair, compute), pair[0].ids, **settings)
                rows.append(_row(arm, _versus(reference), measure, metric, estimate, pair[0]))
    return rows


def semantic_table(rows: Sequence[SemanticRow]) -> pa.Table:
    """Return the semantic rows as an Arrow table."""
    return pa.Table.from_pylist([asdict(row) for row in rows])


@dataclass(frozen=True, slots=True)
class _Aligned:
    ids: tuple[str, ...]
    correct: NDArray[np.bool_]
    confidence: NDArray[np.float64]


def _arm(
    key: tuple[str, str, str, str], questions: dict[str, tuple[bool, dict[str, float]]]
) -> Arm:
    model, benchmark, config_hash, variant = key
    ids = tuple(sorted(questions))
    names = sorted({measure for _, measures in questions.values() for measure in measures})
    return Arm(
        model_slug=model,
        benchmark=benchmark,
        config_hash=config_hash,
        variant=variant,
        question_ids=ids,
        correct=np.array([questions[q][0] for q in ids], dtype=np.bool_),
        confidence={
            name: np.array([questions[q][1].get(name, math.nan) for q in ids], dtype=np.float64)
            for name in names
        },
    )


def _aligned(arm: Arm, measure: str | None, ids: Sequence[str]) -> _Aligned:
    index = {question: position for position, question in enumerate(arm.question_ids)}
    positions = np.array([index[q] for q in ids], dtype=np.intp)
    confidence = (
        np.full(len(ids), math.nan) if measure is None else arm.confidence[measure][positions]
    )
    return _Aligned(tuple(ids), arm.correct[positions], confidence)


def _pair(arm: Arm, reference: Arm, measure: str) -> tuple[_Aligned, _Aligned]:
    common = sorted(set(arm.question_ids) & set(reference.question_ids))
    if not common:
        raise ValueError(
            f"arms {arm.config_hash!r} and {reference.config_hash!r} share no question"
        )
    name = None if measure == ANSWER else measure
    return _aligned(arm, name, common), _aligned(reference, name, common)


def _single(rows: _Aligned, compute: Compute) -> Statistic:
    return lambda indices: compute(rows.correct[indices], rows.confidence[indices])


def _paired(pair: tuple[_Aligned, _Aligned], compute: Compute) -> Statistic:
    arm, reference = pair
    return lambda indices: (
        compute(arm.correct[indices], arm.confidence[indices])
        - compute(reference.correct[indices], reference.confidence[indices])
    )


def _versus(reference: Arm) -> str:
    return f"arm - {reference.config_hash}"


def _row(
    arm: Arm,
    comparison: str,
    measure: str,
    metric: str,
    estimate: Estimate,
    rows: _Aligned,
) -> SemanticRow:
    analysable = rows.correct.size if measure == ANSWER else int((~np.isnan(rows.confidence)).sum())
    return SemanticRow(
        model_slug=arm.model_slug,
        benchmark=arm.benchmark,
        arm=arm.config_hash,
        variant=arm.variant,
        comparison=comparison,
        measure=measure,
        metric=metric,
        estimate=estimate.value,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        reason=None if estimate.reason is None else str(estimate.reason),
        n_questions=len(rows.ids),
        n_analysable=analysable,
        n_resamples_failed=estimate.n_resamples_failed,
        analysis_status=ANALYSIS_STATUS,
    )
