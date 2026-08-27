"""Reading and writing the results store.

Layout:

    root/
    |- runs/       part-{run_id}.parquet
    |- questions/  model_slug=../stage=../benchmark=../part-{run_id}-{seq:04d}.parquet
    |- signals/    model_slug=../stage=../benchmark=../part-{run_id}-{seq:04d}.parquet

Rules this module exists to enforce, all from CLAUDE.md:

- **No job ever appends to another job's file.** Each run owns files named by its own
  `run_id`, so a job array cannot collide however it is scheduled.
- **Nothing is overwritten or edited.** A wrong result is fixed by regenerating it under
  a new run, not by rewriting a file.
- **A partial run is detectable.** `runs.completed_at` stays NULL until a job finishes,
  because a walltime kill otherwise leaves a valid Parquet file holding a truncated,
  subject-ordered prefix of the benchmark that no reader can distinguish from a smaller
  complete run.

Note the chunked file names (`-{seq}`). CLAUDE.md says "one file per job"; the rule it is
protecting is "never concurrent-append to a shared file", which distinct per-chunk files
satisfy. Writing in chunks means a job killed at its walltime keeps the work it already
did, instead of restarting a benchmark it had nearly finished. Flagged in DEVLOG as a
clarification of that rule rather than a silent departure from it.
"""

import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from caldrift.storage.schema import (
    QUESTIONS_SCHEMA,
    RUNS_SCHEMA,
    SIGNALS_SCHEMA,
    TABLES,
    QuestionRow,
    RunStatus,
    SignalRow,
    Stage,
    model_slug,
)


@dataclass(frozen=True, slots=True)
class RunManifest:
    """Everything that identifies one execution: the condition plus its provenance."""

    run_id: str
    config_hash: str
    model_id: str
    stage: Stage
    benchmark: str
    prompt_protocol: str
    n_shots: int
    scoring_mode: str
    checkpoint_sha: str
    code_sha: str
    dtype: str
    batch_size: int
    backend: str
    backend_version: str
    n_questions_planned: int
    benchmark_revision: str | None = None
    subjects: tuple[str, ...] | None = None
    temperature: float = 0.0
    n_samples: int = 1
    seed: int | None = None


class ResultStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    # -- writing ---------------------------------------------------------------

    def begin_run(self, manifest: RunManifest) -> None:
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
        if seq < 0:
            raise ValueError(f"seq must be >= 0, got {seq}")
        if not questions:
            raise ValueError("write_chunk needs at least one question row")

        graded = {question.question_id for question in questions}
        orphans = sorted({signal.question_id for signal in signals} - graded)
        if orphans:
            raise ValueError(
                f"signal rows reference questions that were never graded and could "
                f"never be joined: {orphans[:5]}"
            )

        shared = self._shared_columns(manifest)
        question_rows = [{**shared, **asdict(question)} for question in questions]
        signal_rows = [{**shared, **asdict(signal), "kind": str(signal.kind)} for signal in signals]

        # Signals first, questions second, and the order is load-bearing. `resume` keys
        # on question rows, so a question row must never exist without its signals: a
        # crash between the two writes would otherwise leave a question that resume
        # skips, which then contributes to accuracy while contributing nothing to ECE.
        # The reverse crash leaves orphan signal rows, which are harmless -- they do not
        # join, and a later run writes its own under a different run_id.
        self._write_atomic(
            pa.Table.from_pylist(signal_rows, schema=SIGNALS_SCHEMA),
            self._chunk_path("signals", manifest, seq),
        )
        self._write_atomic(
            pa.Table.from_pylist(question_rows, schema=QUESTIONS_SCHEMA),
            self._chunk_path("questions", manifest, seq),
        )

    def finish_run(self, run_id: str, status: RunStatus, n_written: int) -> None:
        path = self._run_path(run_id)
        if not path.exists():
            raise FileNotFoundError(
                f"no manifest for run_id {run_id!r}; finish_run without begin_run means "
                f"the two calls disagree about which run this is"
            )
        record = pq.read_table(path).to_pylist()[0]
        record["status"] = str(status)
        # Only a clean finish gets a timestamp. `analysis/` reads completed_at, so a
        # failed run must not acquire one -- it is the difference between "this file
        # holds the whole condition" and "this file holds however far it got".
        record["completed_at"] = datetime.now(UTC) if status is RunStatus.COMPLETE else None
        record["n_questions_written"] = n_written
        self._write_atomic(pa.Table.from_pylist([record], schema=RUNS_SCHEMA), path)

    # -- reading ---------------------------------------------------------------

    def connect(self) -> duckdb.DuckDBPyConnection:
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
        """Questions already computed under this config, for a requeued job to skip.

        Deliberately does **not** filter on run status. Chunks are written by atomic
        rename, so a row that exists is a whole, correct result whether or not the job
        that produced it later hit its walltime -- and requiring `status = 'complete'`
        here would make a requeue recompute everything a killed job had finished, which
        is the entire reason chunked writes exist.

        Run completion answers a different question: is this *condition* whole? That
        guard belongs to `analysis/`, which refuses rows from runs without a
        `completed_at` (see `incomplete_runs`). Two guards, two purposes.
        """
        if not self._has_tables("questions"):
            return set()
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT DISTINCT question_id FROM questions WHERE config_hash = ?",
                [config_hash],
            ).fetchall()
        return {row[0] for row in rows}

    def incomplete_runs(self) -> list[str]:
        if not self._has_tables("runs"):
            return []
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM runs "
                "WHERE completed_at IS NULL OR status <> 'complete' ORDER BY run_id"
            ).fetchall()
        return [row[0] for row in rows]

    # -- internals -------------------------------------------------------------

    def _has_tables(self, *names: str) -> bool:
        return all((self.root / name).exists() for name in names)

    def _run_path(self, run_id: str) -> Path:
        return self.root / "runs" / f"part-{run_id}.parquet"

    def _chunk_path(self, table: str, manifest: RunManifest, seq: int) -> Path:
        return (
            self.root
            / table
            / f"model_slug={model_slug(manifest.model_id)}"
            / f"stage={manifest.stage}"
            / f"benchmark={manifest.benchmark}"
            / f"part-{manifest.run_id}-{seq:04d}.parquet"
        )

    def _shared_columns(self, manifest: RunManifest) -> dict[str, Any]:
        """Partition and provenance columns, derived once so both facts agree."""
        return {
            "run_id": manifest.run_id,
            "model_slug": model_slug(manifest.model_id),
            "stage": str(manifest.stage),
            "benchmark": manifest.benchmark,
            "checkpoint_sha": manifest.checkpoint_sha,
            "code_sha": manifest.code_sha,
            "config_hash": manifest.config_hash,
            "seed": manifest.seed,
            "dtype": manifest.dtype,
            "batch_size": manifest.batch_size,
            "backend": manifest.backend,
            "backend_version": manifest.backend_version,
            # Timezone-aware: the schema declares tz="UTC", and a naive datetime is
            # silently reinterpreted rather than rejected.
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
            "subjects": list(manifest.subjects) if manifest.subjects else None,
            "prompt_protocol": manifest.prompt_protocol,
            "n_shots": manifest.n_shots,
            "scoring_mode": manifest.scoring_mode,
            "temperature": manifest.temperature,
            "n_samples": manifest.n_samples,
        }
        # A typo here would produce a NULL column rather than an error, so the key set
        # is checked against the schema instead of trusted.
        missing = set(RUNS_SCHEMA.names) - record.keys()
        extra = record.keys() - set(RUNS_SCHEMA.names)
        if missing or extra:
            raise ValueError(f"run record does not match RUNS_SCHEMA: {missing=} {extra=}")
        return record

    def _write_atomic(self, table: pa.Table, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # with_name, not with_suffix: the latter would replace ".parquet" rather than
        # append, and a "*.parquet" glob would then pick up the half-written file.
        temporary = path.with_name(path.name + ".tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, path)
