"""Reading and writing the results store."""

import os
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from madcal.storage.schema import (
    QUESTIONS_SCHEMA,
    RUNS_SCHEMA,
    SIGNALS_SCHEMA,
    TABLES,
    Level,
    QuestionRow,
    RunStatus,
    SignalRow,
    Variant,
    model_slug,
)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Everything that identifies one execution: its condition and its provenance."""

    run_id: str
    config_hash: str
    model_id: str
    variant: Variant
    benchmark: str
    protocol: str
    n_agents: int
    n_rounds: int
    confidence_mode: str
    aggregation: str
    temperature: float
    max_tokens: int
    prompt_protocol: str
    checkpoint_sha: str
    code_sha: str
    dtype: str
    batch_size: int
    backend: str
    backend_version: str
    n_questions_planned: int
    benchmark_revision: str | None = None
    seed: int | None = None

    @property
    def rows_per_question(self) -> int:
        """Number of question rows one complete debate produces."""
        return self.n_agents * (self.n_rounds + 1) + 1


class ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def begin_run(self, manifest: RunManifest) -> None:
        """Write the manifest that marks this run as started."""
        path = self._run_path(manifest.run_id)
        if path.exists():
            raise FileExistsError(
                f"run_id {manifest.run_id!r} already has a manifest at {path}; "
                f"reusing one makes two executions indistinguishable"
            )
        record = self._run_record(manifest, RunStatus.RUNNING, None, 0)
        self._write_atomic(pa.Table.from_pylist([record], schema=RUNS_SCHEMA), path)

    def write_chunk(
        self,
        manifest: RunManifest,
        questions: Sequence[QuestionRow],
        signals: Sequence[SignalRow],
        seq: int,
    ) -> None:
        """Write one chunk of complete debates, signals first so no question lacks them."""
        if seq < 0:
            raise ValueError(f"seq must be >= 0, got {seq}")
        if not questions:
            raise ValueError("write_chunk needs at least one question row")

        self._check_debates_are_complete(manifest, questions)

        graded = {question.question_id for question in questions}
        orphans = sorted({signal.question_id for signal in signals} - graded)
        if orphans:
            raise ValueError(
                f"signal rows reference questions that were never graded and could "
                f"never be joined: {orphans[:5]}"
            )

        shared = self._shared_columns(manifest)
        question_rows = [{**shared, **asdict(question)} for question in questions]
        signal_rows = [{**shared, **asdict(signal)} for signal in signals]

        self._write_atomic(
            pa.Table.from_pylist(signal_rows, schema=SIGNALS_SCHEMA),
            self._chunk_path("signals", manifest, seq),
        )
        self._write_atomic(
            pa.Table.from_pylist(question_rows, schema=QUESTIONS_SCHEMA),
            self._chunk_path("questions", manifest, seq),
        )

    def finish_run(self, run_id: str, status: RunStatus, n_written: int) -> None:
        """Close the manifest, timestamping only a clean finish."""
        path = self._run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(
                f"no manifest for run_id {run_id!r}; finish_run without begin_run means "
                f"the two calls disagree about which run this is"
            )
        record = pq.read_table(path).to_pylist()[0]
        record["status"] = str(status)
        record["completed_at"] = datetime.now(UTC) if status is RunStatus.COMPLETE else None
        record["n_questions_written"] = n_written
        self._write_atomic(pa.Table.from_pylist([record], schema=RUNS_SCHEMA), path)

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Open a DuckDB connection with one view per table."""
        connection = duckdb.connect()
        for name in TABLES:
            directory = self.root / name
            if not directory.exists():
                continue
            pattern = str(directory / "**" / "*.parquet").replace("'", "''")
            connection.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet("
                f"'{pattern}', hive_partitioning = true, union_by_name = true)"
            )
        return connection

    def completed_questions(self, config_hash: str) -> set[str]:
        """Questions already computed under this config, for a requeued job to skip."""
        if not self._has_tables("questions"):
            return set()
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT question_id FROM questions WHERE config_hash = ?",
                [config_hash],
            ).fetchall()
        return {row[0] for row in rows}

    def incomplete_runs(self) -> list[str]:
        """Run ids that never finished cleanly, which analysis must refuse."""
        if not self._has_tables("runs"):
            return []
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs "
                "WHERE completed_at IS NULL OR status <> 'complete' ORDER BY run_id"
            ).fetchall()
        return [row[0] for row in rows]

    def _check_debates_are_complete(
        self, manifest: RunManifest, questions: Sequence[QuestionRow]
    ) -> None:
        agent_turns = Counter(
            question.question_id for question in questions if question.level is Level.AGENT
        )
        system_answers = Counter(
            question.question_id for question in questions if question.level is Level.SYSTEM
        )
        expected_turns = manifest.n_agents * (manifest.n_rounds + 1)
        for question_id in sorted(agent_turns.keys() | system_answers.keys()):
            turns = agent_turns[question_id]
            systems = system_answers[question_id]
            if turns != expected_turns or systems != 1:
                raise ValueError(
                    f"debate for {question_id!r} is incomplete: {turns} agent rows and "
                    f"{systems} system rows, expected {expected_turns} and 1; a partial "
                    f"debate must never be written because resume would treat it as done"
                )

    def _has_tables(self, *names: str) -> bool:
        return all((self.root / name).exists() for name in names)

    def _run_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"part-{run_id}.parquet"

    def _chunk_path(self, table: str, manifest: RunManifest, seq: int) -> Path:
        return (
            self.root
            / table
            / f"model_slug={model_slug(manifest.model_id)}"
            / f"variant={manifest.variant}"
            / f"protocol={manifest.protocol}"
            / f"benchmark={manifest.benchmark}"
            / f"part-{manifest.run_id}-{seq:04d}.parquet"
        )

    def _shared_columns(self, manifest: RunManifest) -> dict[str, Any]:
        return {
            "run_id": manifest.run_id,
            "model_slug": model_slug(manifest.model_id),
            "variant": str(manifest.variant),
            "protocol": manifest.protocol,
            "benchmark": manifest.benchmark,
            "checkpoint_sha": manifest.checkpoint_sha,
            "code_sha": manifest.code_sha,
            "config_hash": manifest.config_hash,
            "seed": manifest.seed,
            "dtype": manifest.dtype,
            "batch_size": manifest.batch_size,
            "backend": manifest.backend,
            "backend_version": manifest.backend_version,
            "created_at": datetime.now(UTC),
        }

    def _run_record(
        self,
        manifest: RunManifest,
        status: RunStatus,
        completed_at: datetime | None,
        n_written: int,
    ) -> dict[str, Any]:
        record = {
            **self._shared_columns(manifest),
            "status": str(status),
            "started_at": datetime.now(UTC),
            "completed_at": completed_at,
            "n_questions_planned": manifest.n_questions_planned,
            "n_questions_written": n_written,
            "model_id": manifest.model_id,
            "benchmark_revision": manifest.benchmark_revision,
            "prompt_protocol": manifest.prompt_protocol,
            "n_agents": manifest.n_agents,
            "n_rounds": manifest.n_rounds,
            "confidence_mode": manifest.confidence_mode,
            "aggregation": manifest.aggregation,
            "temperature": manifest.temperature,
            "max_tokens": manifest.max_tokens,
        }
        missing = set(RUNS_SCHEMA.names) - record.keys()
        extra = record.keys() - set(RUNS_SCHEMA.names)
        if missing or extra:
            raise ValueError(f"run record does not match RUNS_SCHEMA: {missing=} {extra=}")
        return record

    def _write_atomic(self, table: pa.Table, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
