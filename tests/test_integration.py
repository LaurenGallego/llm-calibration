from collections.abc import Sequence
from pathlib import Path

import pytest

from madcal.benchmarks import MMLU
from madcal.debate import Message, preset, run_debate
from madcal.metrics import brier_score, expected_calibration_error
from madcal.models import StubAdapter
from madcal.orchestration import (
    SYSTEM_CONFIDENCE,
    code_sha,
    debate_question,
    new_run_id,
    transcript_rows,
)
from madcal.storage import ResultStore, RunManifest, RunStatus, Variant

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


def render(messages: Sequence[Message]) -> str:
    return "\n".join(f"[{message.role}] {message.content}" for message in messages)


def make_manifest(run_id: str, config_hash: str, n_planned: int, config) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        config_hash=config_hash,
        model_id="stub",
        variant=Variant.STUB,
        benchmark="mmlu",
        protocol="best_of_n",
        n_agents=config.n_agents,
        n_rounds=config.n_rounds,
        confidence_mode=config.confidence_mode,
        aggregation=config.aggregation,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        prompt_protocol="test_renderer",
        checkpoint_sha="stub",
        code_sha=code_sha(),
        dtype="float32",
        batch_size=8,
        backend="stub",
        backend_version="0.0.1",
        n_questions_planned=n_planned,
        seed=0,
    )


@pytest.fixture
def stored(tmp_path: Path):
    benchmark = MMLU(source=FIXTURE)
    questions = list(benchmark.load())
    config = preset("best_of_n", n_agents=N_AGENTS, temperature=1.0)
    adapter = StubAdapter(response_template=TEMPLATE)
    store = ResultStore(tmp_path / "results")
    run_id = new_run_id()
    manifest = make_manifest(run_id, "cfg-integration", len(questions), config)

    store.begin_run(manifest)
    written = 0
    for seq, start in enumerate(range(0, len(questions), CHUNK)):
        batch = questions[start : start + CHUNK]
        transcripts = run_debate(
            [debate_question(question, benchmark) for question in batch],
            config,
            adapter,
            render,
            benchmark.extract_answer,
            seed=0,
        )
        question_rows = []
        signal_rows = []
        for transcript, question in zip(transcripts, batch, strict=True):
            rows, signals = transcript_rows(transcript, question, benchmark)
            question_rows.extend(rows)
            signal_rows.extend(signals)
        store.write_chunk(manifest, question_rows, signal_rows, seq=seq)
        written += len(batch)
    store.finish_run(run_id, RunStatus.COMPLETE, n_written=written)
    return store, questions, config


def test_every_question_is_stored_once_with_all_its_turns(stored):
    store, questions, config = stored
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
    store, questions, _ = stored
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
    store, questions, _ = stored
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
    store, questions, _ = stored
    assert store.incomplete_runs() == []
    assert store.completed_questions("cfg-integration") == {q.id for q in questions}
    written = list((store.root / "questions").rglob("*.parquet"))
    assert len(written) == (len(questions) + CHUNK - 1) // CHUNK
    assert all("model_slug=stub" in str(path) and "variant=stub" in str(path) for path in written)


def test_no_stored_signal_row_carries_a_label(stored):
    store, _, _ = stored
    with store.connect() as connection:
        columns = {row[0] for row in connection.execute("DESCRIBE signals").fetchall()}
    assert "correct" not in columns
    assert "predicted" not in columns
    assert "answer" not in columns


def test_a_rerun_skips_every_question_it_already_stored(stored):
    store, questions, _ = stored
    already = store.completed_questions("cfg-integration")
    assert [question for question in questions if question.id not in already] == []
