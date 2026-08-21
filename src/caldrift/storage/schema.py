"""Explicit Arrow schemas for the results store, and the row types that fill them.

Three tables, a star schema:

    runs        one row per job execution        (dimension)
    questions   one row per run x question       (the graded side)
    signals     one row per run x question x signal   (the unlabeled side)

    runs (run_id) --1:N--> questions (run_id, question_id)
                  --1:N--> signals   (run_id, question_id, signal)

Why the schemas are written out rather than inferred from the data: an inferred schema
is how a column becomes int64 in one job's file and double in another, which surfaces
months later as a coercion or a failed union across the dataset. Writing fails here if a
row does not match, which is the only point at which that is cheap to fix.

Two properties of this layout are load-bearing rather than stylistic:

- **`signals` contains no label.** The dataset can be copied, shared or globbed by
  accident and still cannot leak ground truth. That is the `import-linter` contract
  expressed in the file layout.
- **Provenance is repeated on every fact row** even though `runs` already holds it.
  CLAUDE.md requires a row to be traceable on its own, and Parquet dictionary-encodes a
  column with one distinct value per file down to a few bytes -- so a self-describing
  file costs almost nothing. `analysis/` should assert the two copies agree.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from string import ascii_letters, digits

import pyarrow as pa


class Stage(StrEnum):
    """Alignment stage. An experimental claim about a checkpoint, supplied by config --
    an adapter cannot verify which stage the weights it loaded came from."""

    BASE = "base"
    SFT = "sft"
    DPO = "dpo"
    RLVR = "rlvr"
    STUB = "stub"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class SignalKind(StrEnum):
    SIGNAL = "signal"
    DIAGNOSTIC = "diagnostic"


# Columns every fact row repeats. Dictionary-encoded to near-nothing in Parquet.
PROVENANCE_FIELDS = [
    pa.field("checkpoint_sha", pa.string(), nullable=False),
    pa.field("code_sha", pa.string(), nullable=False),
    pa.field("config_hash", pa.string(), nullable=False),
    pa.field("seed", pa.int64(), nullable=True),
    pa.field("dtype", pa.string(), nullable=False),
    pa.field("batch_size", pa.int32(), nullable=False),
    pa.field("backend", pa.string(), nullable=False),
    pa.field("backend_version", pa.string(), nullable=False),
    pa.field("created_at", pa.timestamp("us", tz="UTC"), nullable=False),
]

# Hive partition keys, in path order. Low cardinality, and every analysis query filters
# on them. `subject` is deliberately absent: 57 values would multiply the directory
# count into the thousands, and Parquet row-group statistics give pushdown anyway.
PARTITION_KEYS = ("model_slug", "stage", "benchmark")

# What makes a stored row reusable by a requeued job. `config_hash` is in here on
# purpose: without it, changing n_shots and requeueing leaves prior rows looking
# complete, so the job skips them and the condition silently blends two protocols.
RESUME_KEY = ("config_hash", "question_id")

RUNS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        # config_hash is not repeated here: PROVENANCE_FIELDS already carries it, and a
        # duplicate name makes the column unaddressable in SQL and unfillable from a dict.
        pa.field("status", pa.string(), nullable=False),
        pa.field("started_at", pa.timestamp("us", tz="UTC"), nullable=False),
        # NULL until the job finishes cleanly. This is the partial-run detector: a
        # walltime kill leaves a valid Parquet file holding a truncated, subject-ordered
        # prefix of the benchmark, which is indistinguishable from a smaller complete
        # run without this column.
        pa.field("completed_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("n_questions_planned", pa.int64(), nullable=False),
        pa.field("n_questions_written", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_slug", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("benchmark_revision", pa.string(), nullable=True),
        pa.field("subjects", pa.list_(pa.string()), nullable=True),
        pa.field("prompt_protocol", pa.string(), nullable=False),
        pa.field("n_shots", pa.int32(), nullable=False),
        pa.field("scoring_mode", pa.string(), nullable=False),
        pa.field("temperature", pa.float64(), nullable=False),
        pa.field("n_samples", pa.int32(), nullable=False),
        *PROVENANCE_FIELDS,
    ]
)

QUESTIONS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("question_id", pa.string(), nullable=False),
        pa.field("model_slug", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("subject", pa.string(), nullable=True),
        pa.field("task_format", pa.string(), nullable=False),
        pa.field("predicted", pa.string(), nullable=True),
        # NULL when parse_failed, never False. A parse failure is a different event
        # from a wrong answer, and `analysis/` has to state which policy it applies
        # rather than inheriting one from the storage layer.
        pa.field("correct", pa.bool_(), nullable=True),
        pa.field("parse_failed", pa.bool_(), nullable=False),
        *PROVENANCE_FIELDS,
    ]
)

SIGNALS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("question_id", pa.string(), nullable=False),
        pa.field("signal", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("reason", pa.string(), nullable=True),
        pa.field("model_slug", pa.string(), nullable=False),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        *PROVENANCE_FIELDS,
    ]
)

TABLES = {
    "runs": RUNS_SCHEMA,
    "questions": QUESTIONS_SCHEMA,
    "signals": SIGNALS_SCHEMA,
}


@dataclass(frozen=True, slots=True)
class QuestionRow:
    question_id: str
    subject: str | None
    task_format: str
    predicted: str | None
    correct: bool | None
    parse_failed: bool

    def __post_init__(self) -> None:
        # TODO: parse_failed and correct must agree -- parse_failed implies correct is
        # None, and a present `predicted` implies parse_failed is False. Inconsistency
        # here is a wrong accuracy, not a crash.
        if self.parse_failed and self.correct is not None:
            raise ValueError("parse_failed implies correct is None")

        if self.predicted is not None and self.parse_failed:
            raise ValueError("predicted implies parse_failed is False")


@dataclass(frozen=True, slots=True)
class SignalRow:
    question_id: str
    signal: str
    kind: SignalKind
    value: float | None
    reason: str | None

    def __post_init__(self) -> None:
        # TODO: exactly one of value/reason, mirroring SignalValue. Reject non-finite
        # values here too -- a NaN that reaches Parquet is a NaN in every aggregate
        # computed from it afterwards.
        if self.value is None and self.reason is None:
            raise ValueError("Exactly one of value or reason must be specified")
        if self.value is not None and self.reason is not None:
            raise ValueError("Only one of value or reason can be specified")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("value must be a finite number")


# What a Hive partition value can hold without being reinterpreted: no "/" to invent a
# directory level, no "=" to look like another partition key, no whitespace or shell
# metacharacters.
_SAFE_CHARACTERS = frozenset(ascii_letters + digits + "-._")


def model_slug(model_id: str) -> str:
    """`mistralai/Mistral-7B-v0.1` -> `mistralai__Mistral-7B-v0.1`.

    The organisation separator becomes a double underscore and anything else unsafe
    becomes a single one, so the slug still reads as the model it came from. Stable by
    construction: one `model_id` always maps to one slug.

    Not provably injective -- an id already containing `__` could in principle collide
    with one containing `/`. The exact `model_id` is kept as a column for that reason,
    and `analysis/` should assert each slug maps to exactly one `model_id` rather than
    trusting this function to make a collision impossible.
    """
    cleaned = model_id.strip()
    if not cleaned:
        raise ValueError("model_id cannot be empty")

    slug = "".join(
        character if character in _SAFE_CHARACTERS else "_"
        for character in cleaned.replace("/", "__")
    )

    # A slug of pure punctuation names nothing, and a ".."-style value is a path
    # traversal rather than a partition.
    if not any(character.isalnum() for character in slug):
        raise ValueError(f"model_id {model_id!r} leaves no readable slug: {slug!r}")
    if slug.startswith("."):
        raise ValueError(f"model_id {model_id!r} would produce a hidden directory: {slug!r}")

    return slug
