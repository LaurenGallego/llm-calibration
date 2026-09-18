from pathlib import Path

import pytest

from madcal.config import RunConfig
from madcal.orchestration import run
from madcal.storage import ResultStore, RunStatus

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mmlu_sample.jsonl"


def make_config(tmp_path: Path, **overrides) -> RunConfig:
    payload = {
        "model": {
            "name": "stub",
            "response_template": "Answer: {letter}\nConfidence: {confidence}%",
        },
        "benchmark": {"name": "mmlu", "source": str(FIXTURE)},
        "prompt": {"name": "plain_chat"},
        "debate": {"protocol": "best_of_n", "temperature": 1.0, "n_agents": 3},
        "chunk_size": 2,
        "output_root": str(tmp_path / "results"),
    }
    return RunConfig.model_validate({**payload, **overrides})


@pytest.fixture
def store(tmp_path: Path) -> ResultStore:
    return ResultStore(tmp_path / "results")


def n_questions(config: RunConfig) -> int:
    from madcal.orchestration import build_benchmark

    return len(build_benchmark(config.benchmark).load())


def test_a_run_stores_every_question_and_reports_it(tmp_path, store):
    config = make_config(tmp_path)
    outcome = run(config, store)

    assert outcome.status is RunStatus.COMPLETE
    assert outcome.n_written == n_questions(config)
    assert outcome.n_skipped == 0
    assert store.completed_questions(outcome.config_hash)


def test_the_manifest_records_the_debate_configuration(tmp_path, store):
    config = make_config(tmp_path)
    outcome = run(config, store)
    with store.connect() as connection:
        row = connection.execute(
            "SELECT protocol, n_agents, n_rounds, confidence_mode, aggregation, temperature, "
            "variant, prompt_protocol, backend, checkpoint_sha, n_questions_planned "
            "FROM runs WHERE run_id = ?",
            [outcome.run_id],
        ).fetchone()
    assert row == (
        "best_of_n",
        3,
        0,
        "verbalized",
        "argmax_confidence",
        1.0,
        "stub",
        "plain_chat",
        "stub",
        "stub",
        n_questions(config),
    )


def test_a_rerun_skips_stored_questions_and_writes_nothing_new(tmp_path, store):
    config = make_config(tmp_path)
    first = run(config, store)
    second = run(config, store)

    assert second.config_hash == first.config_hash
    assert second.n_written == 0
    assert second.n_skipped == n_questions(config)
    assert second.status is RunStatus.COMPLETE
    assert store.incomplete_runs() == []
    with store.connect() as connection:
        rows = connection.execute("SELECT count(DISTINCT run_id) FROM questions").fetchone()
    assert rows == (1,)


def test_changing_the_protocol_changes_the_condition_and_recomputes(tmp_path, store):
    first = run(make_config(tmp_path), store)
    other = make_config(tmp_path, debate={"protocol": "vanilla", "temperature": 1.0})
    second = run(other, store)

    assert second.config_hash != first.config_hash
    assert second.n_skipped == 0
    assert second.n_written == n_questions(other)


def test_planned_counts_the_whole_condition_not_the_remainder(tmp_path, store):
    config = make_config(tmp_path)
    run(config, store)
    outcome = run(config, store)
    with store.connect() as connection:
        planned = connection.execute(
            "SELECT n_questions_planned, n_questions_written FROM runs WHERE run_id = ?",
            [outcome.run_id],
        ).fetchone()
    assert planned == (n_questions(config), 0)


def test_a_failure_marks_the_run_failed_and_keeps_written_chunks(tmp_path, store, monkeypatch):
    import madcal.orchestration.runner as runner

    calls = {"n": 0}
    original = runner._process_chunk

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("model died")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_process_chunk", failing)
    config = make_config(tmp_path)
    with pytest.raises(RuntimeError, match="model died"):
        run(config, store)

    assert store.incomplete_runs()
    survivors = store.completed_questions(config.config_hash("stub"))
    assert len(survivors) == config.chunk_size
    with store.connect() as connection:
        status = connection.execute("SELECT status, completed_at FROM runs").fetchone()
    assert status is not None
    assert status[0] == "failed"
    assert status[1] is None


def test_a_requeue_after_a_failure_finishes_the_condition(tmp_path, store, monkeypatch):
    import madcal.orchestration.runner as runner

    calls = {"n": 0}
    original = runner._process_chunk

    def failing(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("model died")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_process_chunk", failing)
    config = make_config(tmp_path)
    with pytest.raises(RuntimeError):
        run(config, store)

    monkeypatch.setattr(runner, "_process_chunk", original)
    outcome = run(config, store)
    assert outcome.status is RunStatus.COMPLETE
    assert outcome.n_skipped == config.chunk_size
    assert outcome.n_written == n_questions(config) - config.chunk_size
    assert store.completed_questions(outcome.config_hash) == {
        question.id for question in _questions(config)
    }


def _questions(config: RunConfig):
    from madcal.orchestration import build_benchmark

    return build_benchmark(config.benchmark).load()


def predictions(store: ResultStore, config_hash: str) -> list[str | None]:
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT predicted FROM questions WHERE config_hash = ? AND level = 'agent' "
            "ORDER BY question_id, agent_id, round",
            [config_hash],
        ).fetchall()
    return [row[0] for row in rows]


def test_the_configured_seed_reaches_the_debate(tmp_path, store):
    first = run(make_config(tmp_path, seed=0), store)
    second = run(make_config(tmp_path, seed=1), store)

    assert first.config_hash != second.config_hash
    answers = predictions(store, first.config_hash)
    assert answers != predictions(store, second.config_hash)


def test_the_same_seed_reproduces_the_same_answers(tmp_path, store):
    outcome = run(make_config(tmp_path, seed=0), store)
    other = ResultStore(tmp_path / "second")
    again = run(make_config(tmp_path, seed=0), other)

    assert again.config_hash == outcome.config_hash
    assert predictions(store, outcome.config_hash) == predictions(other, again.config_hash)
