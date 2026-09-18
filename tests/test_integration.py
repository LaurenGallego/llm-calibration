from pathlib import Path

import pytest

from madcal.config import RunConfig
from madcal.metrics import brier_score, expected_calibration_error
from madcal.orchestration import STATED_CONFIDENCE, SYSTEM_CONFIDENCE, build_benchmark, run
from madcal.storage import ResultStore, RunStatus

FIXTURE = Path(__file__).parent / "fixtures" / "mmlu_sample.jsonl"
TEMPLATE = "Answer: {letter}\nConfidence: {confidence}%"
N_AGENTS = 3
CHUNK = 2

SYSTEM_ROWS = """
    SELECT q.question_id, q.correct, s.value
    FROM questions q JOIN signals s
    ON q.run_id = s.run_id
    AND q.question_id = s.question_id
    AND q.level = s.level
    AND q.agent_id IS NOT DISTINCT FROM s.agent_id
    AND q.round IS NOT DISTINCT FROM s.round
    WHERE q.level = 'system' AND s.signal = ?
    ORDER BY q.question_id
"""


def make_config(tmp_path: Path) -> RunConfig:
    return RunConfig.model_validate(
        {
            "model": {"name": "stub", "response_template": TEMPLATE},
            "benchmark": {"name": "mmlu", "source": str(FIXTURE)},
            "prompt": {"name": "plain_chat"},
            "debate": {"protocol": "best_of_n", "temperature": 1.0, "n_agents": N_AGENTS},
            "chunk_size": CHUNK,
            "output_root": str(tmp_path / "results"),
        }
    )


@pytest.fixture
def stored(tmp_path: Path):
    config = make_config(tmp_path)
    store = ResultStore(config.output_root)
    benchmark = build_benchmark(config.benchmark)
    questions = list(benchmark.load())
    outcome = run(config, store)
    assert outcome.status is RunStatus.COMPLETE
    assert outcome.n_written == len(questions)
    return store, questions, config.debate.resolve(), outcome


def test_every_question_is_stored_once_with_all_its_turns(stored):
    store, questions, config, _ = stored
    with store.connect() as connection:
        counts = connection.execute(
            "SELECT level, count(*) FROM questions GROUP BY level ORDER BY level"
        ).fetchall()
        distinct = connection.execute(
            "SELECT count(DISTINCT question_id) FROM questions"
        ).fetchone()
    assert distinct is not None
    assert distinct[0] == len(questions)
    assert counts == [
        ("agent", len(questions) * config.n_agents * (config.n_rounds + 1)),
        ("system", len(questions)),
    ]


def test_stored_grades_match_the_benchmark(stored):
    store, questions, _, _ = stored
    gold = {question.id: question.answer for question in questions}
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT question_id, predicted, correct FROM questions WHERE predicted IS NOT NULL"
        ).fetchall()
    assert rows
    assert all(
        correct == (predicted == gold[question_id]) for question_id, predicted, correct in rows
    )


def test_metrics_can_be_computed_from_the_stored_rows(stored):
    store, questions, _, _ = stored
    with store.connect() as connection:
        rows = connection.execute(SYSTEM_ROWS, [SYSTEM_CONFIDENCE]).fetchall()

    assert len(rows) == len(questions)
    pairs = [(value, correct) for _, correct, value in rows if correct is not None]
    confidences = [value for value, _ in pairs]
    correct = [bool(flag) for _, flag in pairs]
    ece = expected_calibration_error(confidences, correct)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= brier_score(confidences, correct) <= 1.0


def test_the_run_is_complete_and_contained_to_the_stub_partition(stored):
    store, questions, _, outcome = stored
    assert store.incomplete_runs() == []
    assert store.completed_questions(outcome.config_hash) == {q.id for q in questions}
    written = list((store.root / "questions").rglob("*.parquet"))
    assert len(written) == (len(questions) + CHUNK - 1) // CHUNK
    assert all("model_slug=stub" in str(path) and "variant=stub" in str(path) for path in written)


def test_no_stored_signal_row_carries_a_label(stored):
    store, _, _, _ = stored
    with store.connect() as connection:
        columns = {row[0] for row in connection.execute("DESCRIBE signals").fetchall()}
    assert "correct" not in columns
    assert "predicted" not in columns
    assert "answer" not in columns


def test_a_rerun_skips_every_question_it_already_stored(stored, tmp_path):
    store, questions, _, _ = stored
    again = run(make_config(tmp_path), store)
    assert again.n_written == 0
    assert again.n_skipped == len(questions)


def test_every_registered_signal_is_stored_at_the_agent_grain(stored):
    store, questions, config, _ = stored
    with store.connect() as connection:
        rows = connection.execute(
            "SELECT signal, kind, count(*), count(value) FROM signals "
            "WHERE level = 'agent' GROUP BY signal, kind ORDER BY signal"
        ).fetchall()

    turns = len(questions) * config.n_agents * (config.n_rounds + 1)
    assert rows == [
        ("length_normalised_likelihood", "signal", turns, turns),
        (STATED_CONFIDENCE, "signal", turns, turns),
    ]


def test_the_stored_signals_are_distinct_measurements_not_one_column_twice(stored):
    store, _, _, _ = stored
    with store.connect() as connection:
        pairs = connection.execute(
            "SELECT a.value, b.value FROM signals a JOIN signals b "
            "ON a.run_id = b.run_id AND a.question_id = b.question_id "
            "AND a.agent_id IS NOT DISTINCT FROM b.agent_id "
            "AND a.round IS NOT DISTINCT FROM b.round "
            "WHERE a.signal = ? AND b.signal = 'length_normalised_likelihood'",
            [STATED_CONFIDENCE],
        ).fetchall()

    assert pairs
    assert any(stated != likelihood for stated, likelihood in pairs)


def test_no_signal_row_names_the_confidence_mode_signal_as_well_as_its_column(stored):
    store, _, config, _ = stored
    assert config.confidence_mode == "verbalized"
    with store.connect() as connection:
        duplicated = connection.execute(
            "SELECT count(*) FROM signals WHERE signal = 'verbalized_confidence'"
        ).fetchone()
    assert duplicated is not None
    assert duplicated[0] == 0
