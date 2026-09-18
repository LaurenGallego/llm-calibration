from dataclasses import replace
from pathlib import Path

import pytest

from madcal.storage import (
    AnswerNull,
    Level,
    QuestionRow,
    ResultStore,
    RunManifest,
    RunStatus,
    SignalKind,
    SignalRow,
    Variant,
)

QUESTIONS = ("mmlu/anatomy/test/0", "mmlu/anatomy/test/1")
N_AGENTS = 3
N_ROUNDS = 2


def make_manifest(run_id: str, config_hash: str = "cfg-a") -> RunManifest:
    return RunManifest(
        run_id=run_id,
        config_hash=config_hash,
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        variant=Variant.BASE,
        benchmark="mmlu",
        protocol="vanilla",
        n_agents=N_AGENTS,
        n_rounds=N_ROUNDS,
        confidence_mode="none",
        aggregation="majority",
        temperature=0.7,
        max_tokens=768,
        prompt_protocol="chat_template",
        checkpoint_sha="a" * 40,
        code_sha="b" * 40,
        dtype="bfloat16",
        batch_size=8,
        backend="transformers",
        backend_version="5.15.1",
        n_questions_planned=len(QUESTIONS),
        seed=0,
    )


def agent_row(question_id: str, agent: int, round_: int) -> QuestionRow:
    return QuestionRow(
        question_id=question_id,
        level=Level.AGENT,
        agent_id=f"agent_{agent}",
        round=round_,
        subject="anatomy",
        task_format="mcq",
        predicted="B",
        correct=True,
        null_reason=None,
        finish_reason="stop",
    )


def system_row(question_id: str) -> QuestionRow:
    return QuestionRow(
        question_id=question_id,
        level=Level.SYSTEM,
        agent_id=None,
        round=None,
        subject="anatomy",
        task_format="mcq",
        predicted="B",
        correct=True,
        null_reason=None,
        finish_reason=None,
    )


def debate_rows(question_id: str) -> list[QuestionRow]:
    turns = [
        agent_row(question_id, agent, round_)
        for round_ in range(N_ROUNDS + 1)
        for agent in range(N_AGENTS)
    ]
    return [*turns, system_row(question_id)]


def make_rows(question_ids):
    questions = [row for question_id in question_ids for row in debate_rows(question_id)]
    signals = [
        SignalRow(
            question_id=question.question_id,
            level=question.level,
            agent_id=question.agent_id,
            round=question.round,
            signal="verbalized_confidence",
            kind=SignalKind.SIGNAL if question.level is Level.AGENT else SignalKind.AGGREGATE,
            value=0.68,
            reason=None,
        )
        for question in questions
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


def test_round_trip_keeps_every_turn_and_the_system_answer(store: ResultStore):
    complete_run(store, "run-1", QUESTIONS)
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT level, count(*) FROM questions WHERE question_id = ? GROUP BY level "
            "ORDER BY level",
            [QUESTIONS[0]],
        ).fetchall()
        rounds = connection.execute(
            "SELECT DISTINCT round FROM questions WHERE level = 'agent' ORDER BY round"
        ).fetchall()
    assert rows == [("agent", N_AGENTS * (N_ROUNDS + 1)), ("system", 1)]
    assert [row[0] for row in rounds] == [0, 1, 2]


NULL_SAFE_JOIN = """
    SELECT count(*) FROM questions q JOIN signals s
    ON q.run_id = s.run_id
    AND q.question_id = s.question_id
    AND q.level = s.level
    AND q.agent_id IS NOT DISTINCT FROM s.agent_id
    AND q.round IS NOT DISTINCT FROM s.round
"""

EQUALITY_JOIN = NULL_SAFE_JOIN.replace("IS NOT DISTINCT FROM", "=")


def test_signals_join_to_questions_on_the_full_grain(store: ResultStore):
    complete_run(store, "run-1", QUESTIONS)
    with store.connect() as connection:
        joined = connection.execute(NULL_SAFE_JOIN).fetchone()
    assert joined is not None
    assert joined[0] == len(QUESTIONS) * (N_AGENTS * (N_ROUNDS + 1) + 1)


def test_an_equality_join_on_the_grain_drops_every_system_row(store: ResultStore):
    complete_run(store, "run-1", QUESTIONS)
    with store.connect() as connection:
        joined = connection.execute(EQUALITY_JOIN).fetchone()
    assert joined is not None
    assert joined[0] == len(QUESTIONS) * N_AGENTS * (N_ROUNDS + 1)


def test_partition_path_agrees_with_the_row(store: ResultStore):
    complete_run(store, "run-1", QUESTIONS)
    written = list((store.root / "questions").rglob("*.parquet"))
    assert len(written) == 1
    parts = written[0].relative_to(store.root / "questions").parts[:-1]
    assert parts == (
        "model_slug=Qwen__Qwen2.5-0.5B-Instruct",
        "variant=base",
        "protocol=vanilla",
        "benchmark=mmlu",
    )
    with store.connect() as connection:
        distinct = connection.execute(
            "SELECT DISTINCT model_slug, variant, protocol, benchmark FROM questions"
        ).fetchall()
    assert distinct == [("Qwen__Qwen2.5-0.5B-Instruct", "base", "vanilla", "mmlu")]


def test_a_partial_debate_is_never_written(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    questions, signals = make_rows(QUESTIONS)
    truncated = [
        row for row in questions if not (row.question_id == QUESTIONS[1] and row.round == 2)
    ]
    with pytest.raises(ValueError, match="incomplete"):
        store.write_chunk(manifest, truncated, signals, seq=0)
    assert not list((store.root / "questions").rglob("*.parquet"))


def test_a_debate_without_its_system_answer_is_refused(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    questions, signals = make_rows(QUESTIONS[:1])
    with pytest.raises(ValueError, match="incomplete"):
        store.write_chunk(manifest, questions[:-1], signals[:-1], seq=0)


def test_a_debate_with_an_extra_turn_is_refused(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    questions, signals = make_rows(QUESTIONS[:1])
    with pytest.raises(ValueError, match="incomplete"):
        store.write_chunk(manifest, [*questions, agent_row(QUESTIONS[0], 0, 3)], signals, seq=0)


def test_a_killed_run_keeps_the_work_it_finished(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    store.write_chunk(manifest, *make_rows(QUESTIONS[:1]), seq=0)
    store.finish_run("run-1", RunStatus.FAILED, n_written=1)

    assert store.completed_questions("cfg-a") == {QUESTIONS[0]}
    assert store.incomplete_runs() == ["run-1"]


def test_completion_is_tracked_separately_from_resume(store: ResultStore):
    complete_run(store, "run-1", QUESTIONS[:1])
    assert store.incomplete_runs() == []
    assert store.completed_questions("cfg-a") == {QUESTIONS[0]}


def test_resume_is_scoped_to_the_config(store: ResultStore):
    complete_run(store, "run-1", QUESTIONS, config_hash="cfg-a")
    assert store.completed_questions("cfg-b") == set()


def test_orphan_signal_rows_are_rejected(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    questions, _ = make_rows(QUESTIONS[:1])
    orphan = SignalRow(
        question_id="mmlu/anatomy/test/99",
        level=Level.SYSTEM,
        agent_id=None,
        round=None,
        signal="system_confidence",
        kind=SignalKind.AGGREGATE,
        value=0.5,
        reason=None,
    )
    with pytest.raises(ValueError, match="never graded"):
        store.write_chunk(manifest, questions, [orphan], seq=0)


def test_reusing_a_run_id_is_refused(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    with pytest.raises(FileExistsError):
        store.begin_run(make_manifest("run-1"))


def test_finishing_a_run_that_never_began_is_refused(store: ResultStore):
    with pytest.raises(FileNotFoundError):
        store.finish_run("run-1", RunStatus.COMPLETE, n_written=0)


def test_empty_store_is_empty_not_an_error(store: ResultStore):
    assert store.completed_questions("cfg-a") == set()
    assert store.incomplete_runs() == []


def test_chunks_accumulate_within_a_run(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    store.write_chunk(manifest, *make_rows(QUESTIONS[:1]), seq=0)
    store.write_chunk(manifest, *make_rows(QUESTIONS[1:]), seq=1)
    store.finish_run("run-1", RunStatus.COMPLETE, n_written=2)

    assert store.completed_questions("cfg-a") == set(QUESTIONS)
    assert len(list((store.root / "questions").rglob("*.parquet"))) == 2


def test_the_system_null_reason_survives_the_round_trip(store: ResultStore):
    manifest = make_manifest("run-1")
    store.begin_run(manifest)
    questions, signals = make_rows(QUESTIONS[:1])
    tied = QuestionRow(
        question_id=QUESTIONS[0],
        level=Level.SYSTEM,
        agent_id=None,
        round=None,
        subject="anatomy",
        task_format="mcq",
        predicted=None,
        correct=None,
        null_reason=AnswerNull.TIE,
        finish_reason=None,
    )
    store.write_chunk(manifest, [*questions[:-1], tied], signals, seq=0)
    with store.connect() as connection:
        row = connection.execute(
            "SELECT null_reason, predicted, correct FROM questions WHERE level = 'system'"
        ).fetchone()
    assert row == ("tie", None, None)


def test_a_stub_run_lands_in_its_own_partition(store: ResultStore):
    manifest = replace(
        make_manifest("run-stub"), model_id="stub", variant=Variant.STUB, protocol="best_of_n"
    )
    store.begin_run(manifest)
    store.write_chunk(manifest, *make_rows(QUESTIONS[:1]), seq=0)
    store.finish_run("run-stub", RunStatus.COMPLETE, n_written=1)

    written = list((store.root / "questions").rglob("*.parquet"))
    assert [path.relative_to(store.root / "questions").parts[:-1] for path in written] == [
        ("model_slug=stub", "variant=stub", "protocol=best_of_n", "benchmark=mmlu")
    ]
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT DISTINCT variant, protocol FROM questions WHERE model_slug = 'stub'"
        ).fetchall()
    assert rows == [("stub", "best_of_n")]
