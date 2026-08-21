"""ResultStore round-trips, and the guards that keep a wrong condition out of analysis.

The failures targeted here are all silent. A resume that skips a question nothing ever
finished, a signal row that cannot join, a partition path that disagrees with the row it
holds -- none of them raises at the time, and all of them change a published number.
"""

from pathlib import Path

import pytest

from caldrift.storage import (
    QuestionRow,
    ResultStore,
    RunManifest,
    RunStatus,
    SignalKind,
    SignalRow,
    Stage,
)

QUESTIONS = ("mmlu/anatomy/test/0", "mmlu/anatomy/test/1")


def make_manifest(run_id: str, config_hash: str = "cfg-a") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        config_hash=config_hash,
        model_id="mistralai/Mistral-7B-v0.1",
        stage=Stage.BASE,
        benchmark="mmlu",
        prompt_protocol="fewshot_completion",
        n_shots=5,
        scoring_mode="likelihood",
        checkpoint_sha="a" * 40,
        code_sha="b" * 40,
        dtype="bfloat16",
        batch_size=8,
        backend="transformers",
        backend_version="5.15.1",
        n_questions_planned=len(QUESTIONS),
        seed=0,
    )


def make_rows(question_ids):
    questions = [QuestionRow(q, "anatomy", "mcq", "B", True, False) for q in question_ids]
    signals = [SignalRow(q, "confidence", SignalKind.SIGNAL, 0.68, None) for q in question_ids]
    signals += [
        SignalRow(q, "option_mass", SignalKind.DIAGNOSTIC, 0.99, None) for q in question_ids
    ]
    return questions, signals


@pytest.fixture
def store(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / "results")


def complete_run(store: ResultStore, run_id: str, question_ids, config_hash="cfg-a") -> None:
    manifest = make_manifest(run_id, config_hash)
    store.begin_run(manifest)
    store.write_chunk(manifest, *make_rows(question_ids), seq=0)
    store.finish_run(run_id, RunStatus.COMPLETE, n_written=len(question_ids))


def test_round_trip_joins_signals_to_questions(store: ResultStore):
    complete_run(store, "run-1", QUESTIONS)
    with store.connect() as connection:
        rows = connection.execute(
            """
            SELECT q.question_id, q.correct, s.value
            FROM questions q JOIN signals s USING (run_id, question_id)
            WHERE s.signal = 'confidence' ORDER BY q.question_id
            """
        ).fetchall()
    assert rows == [(QUESTIONS[0], True, 0.68), (QUESTIONS[1], True, 0.68)]


def test_partition_path_agrees_with_the_row(store: ResultStore):
    # A row filed under a directory that contradicts its own columns would be grouped
    # into the wrong condition by every query that filters on the partition.
    complete_run(store, "run-1", QUESTIONS)
    written = next((store.root / "questions").rglob("*.parquet"))
    assert "model_slug=mistralai__Mistral-7B-v0.1" in written.parts
    with store.connect() as connection:
        assert connection.execute(
            "SELECT DISTINCT model_slug, stage, benchmark FROM questions"
        ).fetchall() == [("mistralai__Mistral-7B-v0.1", "base", "mmlu")]


def test_unfinished_run_does_not_count_as_completed(store: ResultStore):
    # The guard that matters. A job killed at its walltime leaves valid rows behind;
    # counting them as done makes the requeue skip questions nothing ever finished.
    manifest = make_manifest("run-killed")
    store.begin_run(manifest)
    store.write_chunk(manifest, *make_rows(QUESTIONS), seq=0)

    assert store.completed_questions("cfg-a") == set()
    assert store.incomplete_runs() == ["run-killed"]

    store.finish_run("run-killed", RunStatus.COMPLETE, n_written=len(QUESTIONS))
    assert store.completed_questions("cfg-a") == set(QUESTIONS)
    assert store.incomplete_runs() == []


def test_failed_run_never_counts_as_completed(store: ResultStore):
    manifest = make_manifest("run-failed")
    store.begin_run(manifest)
    store.write_chunk(manifest, *make_rows(QUESTIONS), seq=0)
    store.finish_run("run-failed", RunStatus.FAILED, n_written=len(QUESTIONS))
    assert store.completed_questions("cfg-a") == set()
    assert store.incomplete_runs() == ["run-failed"]


def test_resume_is_scoped_to_the_config(store: ResultStore):
    # Without config_hash in the key, changing n_shots and requeueing would skip these
    # rows and blend two protocols inside one condition.
    complete_run(store, "run-1", QUESTIONS, config_hash="cfg-a")
    assert store.completed_questions("cfg-a") == set(QUESTIONS)
    assert store.completed_questions("cfg-b") == set()


def test_orphan_signal_rows_are_rejected(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    questions, _ = make_rows(QUESTIONS)
    stray = [SignalRow("mmlu/anatomy/test/99", "confidence", SignalKind.SIGNAL, 0.5, None)]
    # An unjoinable signal row does not error in analysis -- it silently vanishes.
    with pytest.raises(ValueError, match="never graded"):
        store.write_chunk(manifest, questions, stray, seq=0)


def test_reusing_a_run_id_is_refused(store: ResultStore):
    store.begin_run(make_manifest("run-1"))
    with pytest.raises(FileExistsError):
        store.begin_run(make_manifest("run-1"))


def test_finishing_a_run_that_never_began_is_refused(store: ResultStore):
    with pytest.raises(FileNotFoundError):
        store.finish_run("ghost", RunStatus.COMPLETE, n_written=0)


def test_empty_store_is_empty_not_an_error(store: ResultStore):
    assert store.completed_questions("cfg-a") == set()
    assert store.incomplete_runs() == []


def test_chunks_accumulate_within_a_run(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    store.write_chunk(manifest, *make_rows([QUESTIONS[0]]), seq=0)
    store.write_chunk(manifest, *make_rows([QUESTIONS[1]]), seq=1)
    store.finish_run("run-1", RunStatus.COMPLETE, n_written=2)
    assert store.completed_questions("cfg-a") == set(QUESTIONS)
