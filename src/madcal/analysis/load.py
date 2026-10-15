"""Reading the results store into the analysable (confidence, correctness) frame."""

from dataclasses import dataclass

import pyarrow as pa

from madcal.storage import (
    AGENT_CONFIDENCE,
    SEMANTIC_PROTOCOL,
    SYSTEM_CONFIDENCE,
    ResultStore,
    Variant,
)

ANSWER_POLICY = "score_unparsed_incorrect"
CALIBRATION_POLICY = "drop_without_confidence"

CELL_KEYS = ("model_slug", "variant", "protocol", "benchmark", "level", "agent_id", "round")

FRAME_QUERY = """
WITH readable AS (
    SELECT run_id FROM runs
    WHERE status = 'complete' AND completed_at IS NOT NULL AND protocol <> $semantic
)
SELECT
    q.model_slug,
    q.variant,
    q.protocol,
    q.benchmark,
    q.level,
    q.agent_id,
    q.round,
    q.question_id,
    q.subject,
    q.predicted,
    q.correct,
    coalesce(q.correct, false) AS graded_correct,
    s.value AS confidence,
    s.signal IS NOT NULL AS has_confidence_row
FROM questions q
JOIN readable r ON q.run_id = r.run_id
LEFT JOIN signals s
    ON q.run_id = s.run_id
    AND q.question_id = s.question_id
    AND q.level = s.level
    AND q.agent_id IS NOT DISTINCT FROM s.agent_id
    AND q.round IS NOT DISTINCT FROM s.round
    AND s.signal = CASE q.level WHEN 'agent' THEN $agent ELSE $system END
ORDER BY q.model_slug, q.variant, q.protocol, q.benchmark,
         q.level, q.agent_id, q.round, q.question_id
"""

GRADED_COUNT_QUERY = """
SELECT count(*) FROM questions q
WHERE q.run_id IN (
    SELECT run_id FROM runs
    WHERE status = 'complete' AND completed_at IS NOT NULL AND protocol <> $semantic
)
"""


@dataclass(frozen=True, slots=True)
class AnalysisFrame:
    """Every graded row of every readable run, paired with the confidence stated for it."""

    rows: pa.Table
    excluded_runs: tuple[str, ...]


def load_frame(store: ResultStore, allow_stub: bool = False) -> AnalysisFrame:
    """Return the joined frame, refusing unfinished runs and, unless asked, stub rows."""
    excluded = tuple(store.incomplete_runs())
    with store.connect() as connection:
        if not has_views(connection, "runs", "questions", "signals"):
            return AnalysisFrame(rows=_empty_frame(), excluded_runs=excluded)
        rows = connection.execute(
            FRAME_QUERY,
            {"agent": AGENT_CONFIDENCE, "system": SYSTEM_CONFIDENCE, "semantic": SEMANTIC_PROTOCOL},
        ).to_arrow_table()
        graded = connection.execute(GRADED_COUNT_QUERY, {"semantic": SEMANTIC_PROTOCOL}).fetchone()

    _check_join_is_one_to_one(rows, 0 if graded is None else int(graded[0]))
    _check_every_graded_row_has_a_confidence_row(rows)
    if not allow_stub:
        _check_no_stub_rows(rows)
    return AnalysisFrame(rows=rows.drop_columns(["has_confidence_row"]), excluded_runs=excluded)


def has_views(connection, *names: str) -> bool:
    """Return whether the store exposes every named table."""
    present = {row[0] for row in connection.execute("SHOW TABLES").fetchall()}
    return all(name in present for name in names)


def _empty_frame() -> pa.Table:
    return pa.table(
        {
            "model_slug": pa.array([], pa.string()),
            "variant": pa.array([], pa.string()),
            "protocol": pa.array([], pa.string()),
            "benchmark": pa.array([], pa.string()),
            "level": pa.array([], pa.string()),
            "agent_id": pa.array([], pa.string()),
            "round": pa.array([], pa.int32()),
            "question_id": pa.array([], pa.string()),
            "subject": pa.array([], pa.string()),
            "predicted": pa.array([], pa.string()),
            "correct": pa.array([], pa.bool_()),
            "graded_correct": pa.array([], pa.bool_()),
            "confidence": pa.array([], pa.float64()),
        }
    )


def _check_join_is_one_to_one(rows: pa.Table, graded: int) -> None:
    if rows.num_rows != graded:
        raise ValueError(
            f"the confidence join changed the row count: {graded} graded rows became "
            f"{rows.num_rows}; a duplicate signal row would inflate every metric"
        )


def _check_every_graded_row_has_a_confidence_row(rows: pa.Table) -> None:
    present = rows.column("has_confidence_row").to_pylist()
    missing = [index for index, has_row in enumerate(present) if not has_row]
    if not missing:
        return
    first = {
        name: rows.column(name)[missing[0]].as_py() for name in ("question_id", "level", "agent_id")
    }
    raise ValueError(
        f"{len(missing)} graded rows carry no confidence row at all, the first being "
        f"{first}; a parse failure is stored as a null with a reason, so an absent row means "
        f"the store is incomplete rather than the confidence being unparseable"
    )


def _check_no_stub_rows(rows: pa.Table) -> None:
    variants = set(rows.column("variant").to_pylist())
    if Variant.STUB in variants:
        raise ValueError(
            f"the store holds {Variant.STUB} rows, which are fixture output and never a result; "
            f"pass allow_stub=True only to exercise the pipeline"
        )
