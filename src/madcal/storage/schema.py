"""Arrow schemas for the results store, and the row types that fill them."""

import math
from dataclasses import dataclass
from enum import StrEnum
from string import ascii_letters, digits

import pyarrow as pa


class Variant(StrEnum):
    BASE = "base"
    VAE = "vae"
    DROPOUT = "dropout"
    STUB = "stub"


class Level(StrEnum):
    AGENT = "agent"
    SYSTEM = "system"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class SignalKind(StrEnum):
    SIGNAL = "signal"
    DIAGNOSTIC = "diagnostic"
    AGGREGATE = "aggregate"


class AnswerNull(StrEnum):
    PARSE_FAILED = "parse_failed"
    TIE = "tie"
    NO_VALID_ANSWERS = "no_valid_answers"


AGENT_NULLS = frozenset({AnswerNull.PARSE_FAILED})
SYSTEM_NULLS = frozenset({AnswerNull.TIE, AnswerNull.NO_VALID_ANSWERS})

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

GRAIN_FIELDS = [
    pa.field("model_slug", pa.string(), nullable=False),
    pa.field("variant", pa.string(), nullable=False),
    pa.field("protocol", pa.string(), nullable=False),
    pa.field("benchmark", pa.string(), nullable=False),
    pa.field("level", pa.string(), nullable=False),
    pa.field("agent_id", pa.string(), nullable=True),
    pa.field("round", pa.int32(), nullable=True),
]

PARTITION_KEYS = ("model_slug", "variant", "protocol", "benchmark")

RESUME_KEY = ("config_hash", "question_id")

RUNS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("started_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("completed_at", pa.timestamp("us", tz="UTC"), nullable=True),
        pa.field("n_questions_planned", pa.int64(), nullable=False),
        pa.field("n_questions_written", pa.int64(), nullable=False),
        pa.field("model_id", pa.string(), nullable=False),
        pa.field("model_slug", pa.string(), nullable=False),
        pa.field("variant", pa.string(), nullable=False),
        pa.field("benchmark", pa.string(), nullable=False),
        pa.field("benchmark_revision", pa.string(), nullable=True),
        pa.field("prompt_protocol", pa.string(), nullable=False),
        pa.field("protocol", pa.string(), nullable=False),
        pa.field("n_agents", pa.int32(), nullable=False),
        pa.field("n_rounds", pa.int32(), nullable=False),
        pa.field("confidence_mode", pa.string(), nullable=False),
        pa.field("aggregation", pa.string(), nullable=False),
        pa.field("temperature", pa.float64(), nullable=False),
        pa.field("max_tokens", pa.int32(), nullable=False),
        *PROVENANCE_FIELDS,
    ]
)

QUESTIONS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("question_id", pa.string(), nullable=False),
        *GRAIN_FIELDS,
        pa.field("subject", pa.string(), nullable=True),
        pa.field("task_format", pa.string(), nullable=False),
        pa.field("predicted", pa.string(), nullable=True),
        pa.field("correct", pa.bool_(), nullable=True),
        pa.field("null_reason", pa.string(), nullable=True),
        pa.field("finish_reason", pa.string(), nullable=True),
        *PROVENANCE_FIELDS,
    ]
)

SIGNALS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("question_id", pa.string(), nullable=False),
        *GRAIN_FIELDS,
        pa.field("signal", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("value", pa.float64(), nullable=True),
        pa.field("reason", pa.string(), nullable=True),
        *PROVENANCE_FIELDS,
    ]
)

TABLES = {
    "runs": RUNS_SCHEMA,
    "questions": QUESTIONS_SCHEMA,
    "signals": SIGNALS_SCHEMA,
}


def check_grain(level: Level, agent_id: str | None, round_: int | None) -> None:
    """Raise unless agent id and round are present exactly when the level is agent."""
    if level is Level.SYSTEM and (agent_id is not None or round_ is not None):
        raise ValueError(f"system rows carry no agent or round, got {agent_id!r} and {round_!r}")
    if level is Level.AGENT and (agent_id is None or round_ is None):
        raise ValueError(f"agent rows need an agent and a round, got {agent_id!r} and {round_!r}")
    if round_ is not None and round_ < 0:
        raise ValueError(f"round must be >= 0, got {round_}")


@dataclass(frozen=True, slots=True)
class QuestionRow:
    """One graded answer: an agent's at one round, or the system's for the question."""

    question_id: str
    level: Level
    agent_id: str | None
    round: int | None
    subject: str | None
    task_format: str
    predicted: str | None
    correct: bool | None
    null_reason: AnswerNull | None
    finish_reason: str | None

    def __post_init__(self) -> None:
        check_grain(self.level, self.agent_id, self.round)
        if (self.predicted is None) != (self.null_reason is not None):
            raise ValueError("predicted is present exactly when null_reason is absent")
        if (self.correct is None) != (self.predicted is None):
            raise ValueError("correct is present exactly when predicted is")
        if self.null_reason is not None:
            allowed = AGENT_NULLS if self.level is Level.AGENT else SYSTEM_NULLS
            if self.null_reason not in allowed:
                raise ValueError(f"{self.null_reason} cannot occur at level {self.level}")
        if (self.finish_reason is None) != (self.level is Level.SYSTEM):
            raise ValueError("finish_reason is recorded for agent rows only")


@dataclass(frozen=True, slots=True)
class SignalRow:
    """One signal value for one agent turn, or for the system answer."""

    question_id: str
    level: Level
    agent_id: str | None
    round: int | None
    signal: str
    kind: SignalKind
    value: float | None
    reason: str | None

    def __post_init__(self) -> None:
        check_grain(self.level, self.agent_id, self.round)
        if self.value is None and self.reason is None:
            raise ValueError("Exactly one of value or reason must be specified")
        if self.value is not None and self.reason is not None:
            raise ValueError("Only one of value or reason can be specified")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("value must be a finite number")


_SAFE_CHARACTERS = frozenset(ascii_letters + digits + "-._")


def model_slug(model_id: str) -> str:
    """Return a Hive-safe partition value for a model id, one slug per id."""
    cleaned = model_id.strip()
    if not cleaned:
        raise ValueError("model_id cannot be empty")

    slug = "".join(
        character if character in _SAFE_CHARACTERS else "_"
        for character in cleaned.replace("/", "__")
    )

    if not any(character.isalnum() for character in slug):
        raise ValueError(f"model_id {model_id!r} leaves no readable slug: {slug!r}")
    if slug.startswith("."):
        raise ValueError(f"model_id {model_id!r} would produce a hidden directory: {slug!r}")

    return slug
