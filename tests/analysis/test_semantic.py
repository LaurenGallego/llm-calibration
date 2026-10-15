import math
from collections.abc import Mapping

import pyarrow.parquet as pq
import pytest

from madcal.analysis import load_arms, load_frame, references, semantic_rows
from madcal.cli import main
from madcal.storage import (
    SEMANTIC_PROTOCOL,
    Level,
    QuestionRow,
    ResultStore,
    RunStatus,
    SignalKind,
    SignalRow,
    Variant,
)
from tests.analysis.conftest import make_manifest, single_agent_debates, write_run

type Outcomes = Mapping[str, tuple[bool, Mapping[str, float | None]]]

BASE: Outcomes = {
    "q0": (True, {"e_sc": 0.8, "l_sc": 0.7, "token": None}),
    "q1": (False, {"e_sc": 0.6, "l_sc": 0.5, "token": None}),
    "q2": (True, {"e_sc": 0.9, "l_sc": 0.9, "token": None}),
    "q3": (False, {"e_sc": 0.3, "l_sc": None, "token": None}),
}
VAE: Outcomes = {
    "q0": (True, {"e_sc": 0.7, "l_sc": 0.6, "token": None}),
    "q1": (False, {"e_sc": 0.5, "l_sc": 0.5, "token": None}),
    "q2": (True, {"e_sc": 0.6, "l_sc": 0.8, "token": None}),
    "q3": (True, {"e_sc": 0.4, "l_sc": 0.4, "token": None}),
}


def write_semantic(
    store: ResultStore,
    run_id: str,
    config_hash: str,
    variant: Variant,
    outcomes: Outcomes,
    status: RunStatus = RunStatus.COMPLETE,
) -> None:
    manifest = make_manifest(
        run_id,
        n_agents=1,
        variant=variant,
        protocol=SEMANTIC_PROTOCOL,
        config_hash=config_hash,
        n_questions=len(outcomes),
    )
    questions: list[QuestionRow] = []
    signals: list[SignalRow] = []
    for question_id, (correct, measures) in outcomes.items():
        for level, agent, round_, finish in (
            (Level.AGENT, "agent_0", 0, "stop"),
            (Level.SYSTEM, None, None, None),
        ):
            questions.append(
                QuestionRow(
                    question_id=question_id,
                    level=level,
                    agent_id=agent,
                    round=round_,
                    subject=None,
                    task_format="open",
                    predicted="Paris",
                    correct=correct,
                    null_reason=None,
                    finish_reason=finish,
                )
            )
        for measure, value in {**measures, "n_clusters": 2.0}.items():
            signals.append(
                SignalRow(
                    question_id=question_id,
                    level=Level.SYSTEM,
                    agent_id=None,
                    round=None,
                    signal=f"semantic_{measure}",
                    kind=SignalKind.DIAGNOSTIC if measure == "n_clusters" else SignalKind.SIGNAL,
                    value=value,
                    reason=None if value is not None else "not_applicable",
                )
            )
    store.begin_run(manifest)
    store.write_chunk(manifest, questions, signals, seq=0)
    if status is not RunStatus.RUNNING:
        store.finish_run(run_id, status, n_written=len(outcomes))


@pytest.fixture
def two_arms(store):
    write_semantic(store, "run-base", "cfg-base", Variant.BASE, BASE)
    write_semantic(store, "run-vae", "cfg-vae", Variant.VAE, VAE)
    return store


def table(store, chosen=()):
    arms = load_arms(store)
    rows = semantic_rows(arms, references(arms, chosen), n_resamples=50)
    return {(row.arm, row.comparison, row.measure, row.metric): row for row in rows}


def test_arms_hold_one_graded_answer_and_every_measure_per_question(two_arms):
    base, vae = load_arms(two_arms)
    assert (base.config_hash, base.variant, vae.config_hash) == ("cfg-base", "base", "cfg-vae")
    assert base.question_ids == ("q0", "q1", "q2", "q3")
    assert base.correct.tolist() == [True, False, True, False]
    assert base.confidence["e_sc"].tolist() == [0.8, 0.6, 0.9, 0.3]
    assert math.isnan(base.confidence["l_sc"][3])
    assert "n_clusters" not in base.confidence


def test_each_arm_has_its_hand_computed_metrics(two_arms):
    rows = table(two_arms)
    assert rows[("cfg-base", "arm", "answer", "accuracy")].estimate == pytest.approx(0.5)
    assert rows[("cfg-base", "arm", "e_sc", "mean_confidence")].estimate == pytest.approx(0.65)
    assert rows[("cfg-base", "arm", "e_sc", "brier")].estimate == pytest.approx(0.125)
    assert rows[("cfg-base", "arm", "e_sc", "auroc")].estimate == pytest.approx(1.0)
    assert rows[("cfg-vae", "arm", "e_sc", "brier")].estimate == pytest.approx(0.215)
    assert rows[("cfg-vae", "arm", "e_sc", "auroc")].estimate == pytest.approx(2 / 3)


def test_a_null_confidence_is_dropped_from_calibration_but_not_from_accuracy(two_arms):
    rows = table(two_arms)
    assert rows[("cfg-base", "arm", "l_sc", "mean_confidence")].estimate == pytest.approx(0.7)
    assert rows[("cfg-base", "arm", "l_sc", "brier")].n_analysable == 3
    assert rows[("cfg-base", "arm", "answer", "accuracy")].n_analysable == 4


def test_a_measure_that_is_null_everywhere_gives_no_rows(two_arms):
    assert not [key for key in table(two_arms) if key[2] == "token"]


def test_each_arm_is_compared_with_the_base_arm_on_the_same_questions(two_arms):
    rows = table(two_arms)
    versus = "arm - cfg-base"
    assert rows[("cfg-vae", versus, "answer", "accuracy")].estimate == pytest.approx(0.25)
    assert rows[("cfg-vae", versus, "e_sc", "mean_confidence")].estimate == pytest.approx(-0.1)
    assert rows[("cfg-vae", versus, "e_sc", "brier")].estimate == pytest.approx(0.09)
    assert rows[("cfg-vae", versus, "e_sc", "brier")].n_questions == 4
    assert not [key for key in rows if key[0] == "cfg-base" and key[1] != "arm"]
    assert {row.analysis_status for row in rows.values()} == {"exploratory"}


def test_a_chosen_reference_replaces_the_base_arm(two_arms):
    rows = table(two_arms, chosen=["cfg-vae"])
    assert rows[("cfg-base", "arm - cfg-vae", "answer", "accuracy")].estimate == pytest.approx(
        -0.25
    )


def test_without_a_single_base_arm_there_is_no_reference(store):
    write_semantic(store, "run-a", "cfg-a", Variant.BASE, BASE)
    write_semantic(store, "run-b", "cfg-b", Variant.BASE, VAE)
    arms = load_arms(store)
    assert set(references(arms).values()) == {None}
    with pytest.raises(ValueError, match="no semantic arm"):
        references(arms, ["cfg-z"])


def test_an_unfinished_semantic_run_is_not_read(store):
    write_semantic(store, "run-base", "cfg-base", Variant.BASE, BASE)
    write_semantic(store, "run-vae", "cfg-vae", Variant.VAE, VAE, status=RunStatus.RUNNING)
    assert [arm.config_hash for arm in load_arms(store)] == ["cfg-base"]


def test_a_question_stored_twice_under_one_config_is_refused(store):
    write_semantic(store, "run-1", "cfg-base", Variant.BASE, BASE)
    write_semantic(store, "run-2", "cfg-base", Variant.BASE, {"q0": BASE["q0"]})
    with pytest.raises(ValueError, match="two"):
        load_arms(store)


def test_stub_rows_are_refused_unless_allowed(store):
    write_semantic(store, "run-stub", "cfg-stub", Variant.STUB, BASE)
    with pytest.raises(ValueError, match="stub"):
        load_arms(store)
    assert len(load_arms(store, allow_stub=True)) == 1


def test_debate_analysis_ignores_semantic_runs_in_the_same_store(store):
    write_semantic(store, "run-sem", "cfg-sem", Variant.BASE, BASE)
    write_run(store, "run-debate", single_agent_debates([(True, 0.9), (False, 0.4)]))
    frame = load_frame(store)
    assert set(frame.rows.column("protocol").to_pylist()) == {"vanilla"}
    assert frame.rows.num_rows == 4


def test_the_semantic_command_writes_its_table(two_arms):
    main(["semantic", str(two_arms.root)])
    written = pq.read_table(two_arms.root / "semantic.parquet")
    assert written.num_rows > 0
    assert set(written.column("arm").to_pylist()) == {"cfg-base", "cfg-vae"}


def test_an_empty_store_has_no_arms(store):
    assert load_arms(store) == []
