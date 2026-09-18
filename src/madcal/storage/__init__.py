"""Results store -- Parquet on disk, DuckDB for querying, no server process.

Importing this package gives the schemas and the store; nothing here runs a model or
touches a label beyond recording the grade it was handed.
"""

from madcal.storage.schema import (
    PARTITION_KEYS,
    QUESTIONS_SCHEMA,
    RESUME_KEY,
    RUNS_SCHEMA,
    SIGNALS_SCHEMA,
    TABLES,
    AnswerNull,
    Level,
    QuestionRow,
    RunStatus,
    SignalKind,
    SignalRow,
    Variant,
    check_grain,
    model_slug,
)
from madcal.storage.store import ResultStore, RunManifest

__all__ = [
    "PARTITION_KEYS",
    "QUESTIONS_SCHEMA",
    "RESUME_KEY",
    "RUNS_SCHEMA",
    "SIGNALS_SCHEMA",
    "TABLES",
    "AnswerNull",
    "Level",
    "QuestionRow",
    "ResultStore",
    "RunManifest",
    "RunStatus",
    "SignalKind",
    "SignalRow",
    "Variant",
    "check_grain",
    "model_slug",
]
