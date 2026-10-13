import pyarrow as pa
import pytest

from madcal.analysis import AnalysisFrame, propagation_levels, propagation_rows

LABELS = {"q1": ("A", 4), "q2": ("B", 4)}
TURNS = [
    ("q1", 0, "agent_0", "A", 0.9),
    ("q1", 0, "agent_1", "B", 0.6),
    ("q1", 0, "agent_2", "A", 0.5),
    ("q2", 0, "agent_0", "B", 0.8),
    ("q2", 0, "agent_1", None, None),
    ("q2", 0, "agent_2", "C", 0.7),
    ("q1", 1, "agent_0", "A", 0.9),
    ("q1", 1, "agent_1", "A", 0.9),
    ("q1", 1, "agent_2", "A", 0.9),
    ("q2", 1, "agent_0", "B", 1.0),
    ("q2", 1, "agent_1", "B", 1.0),
    ("q2", 1, "agent_2", "C", 0.4),
]


def records(turns=TURNS):
    return [
        {
            "model_slug": "m",
            "variant": "base",
            "protocol": "p",
            "benchmark": "mmlu",
            "level": "agent",
            "agent_id": agent,
            "round": round_,
            "question_id": question,
            "predicted": predicted,
            "graded_correct": predicted == LABELS[question][0],
            "confidence": confidence,
        }
        for question, round_, agent, predicted, confidence in turns
    ]


def test_levels_pool_the_opening_and_closing_rounds_separately():
    levels = propagation_levels(records(), LABELS)
    assert levels["individual"].correct.tolist() == [True, False, True, True, False, False]
    assert levels["ensemble"].confidence.tolist() == pytest.approx([23 / 45, 0.45])
    assert levels["ensemble"].correct.tolist() == [True, True]
    assert levels["debated_agents"].correct.tolist() == [True, True, True, True, True, False]
    assert levels["system"].confidence.tolist() == pytest.approx([0.9, 11 / 15])
    assert levels["system"].correct.tolist() == [True, True]


def test_contrasts_are_differences_between_levels_on_the_same_questions():
    frame = AnalysisFrame(rows=pa.Table.from_pylist(records()), excluded_runs=())
    rows = {(row.comparison, row.metric): row for row in propagation_rows(frame, LABELS, 0)}
    assert rows[("individual", "accuracy")].estimate == pytest.approx(0.5)
    assert rows[("ensemble - individual", "accuracy")].estimate == pytest.approx(0.5)
    assert rows[("individual", "mean_confidence")].estimate == pytest.approx(0.7)
    assert rows[("system", "mean_confidence")].estimate == pytest.approx((0.9 + 11 / 15) / 2)
    assert rows[("ensemble - individual", "accuracy")].n_questions == 2
    assert {row.analysis_status for row in rows.values()} == {"exploratory"}


def test_a_question_without_a_label_is_refused():
    with pytest.raises(ValueError, match="no label"):
        propagation_levels(records(), {"q1": ("A", 4)})


def test_a_condition_without_a_debate_round_is_refused():
    opening = [turn for turn in TURNS if turn[1] == 0]
    with pytest.raises(ValueError, match="no debate round"):
        propagation_levels(records(opening), LABELS)
