import pytest
from pydantic import ValidationError

from madcal.debate import (
    AgentTurn,
    DebateConfig,
    DebateQuestion,
    DebateTranscript,
    SystemAnswer,
    SystemNull,
    agent_id,
)
from madcal.signals import SignalNull, SignalValue

NO_CONFIDENCE = SignalValue(None, SignalNull.NOT_APPLICABLE)


def turn(agent: int, round_: int) -> AgentTurn:
    return AgentTurn(agent_id(agent), round_, "Answer: A", "A", NO_CONFIDENCE, "stop")


def config(**overrides) -> DebateConfig:
    settings = {
        "n_agents": 3,
        "n_rounds": 1,
        "confidence_mode": "none",
        "aggregation": "majority",
        "temperature": 1.0,
    }
    return DebateConfig.model_validate({**settings, **overrides})


def test_transcript_accepts_one_turn_per_agent_per_round_including_round_zero():
    turns = tuple(turn(agent, round_) for round_ in range(3) for agent in range(3))
    transcript = DebateTranscript("q", 3, 2, turns, SystemAnswer("A", 1.0))
    assert len(transcript.turns) == 9
    assert [t.agent_id for t in transcript.round_turns(2)] == ["agent_0", "agent_1", "agent_2"]


def test_transcript_with_zero_rounds_holds_only_round_zero():
    turns = tuple(turn(agent, 0) for agent in range(5))
    assert len(DebateTranscript("q", 5, 0, turns, SystemAnswer("A", 1.0)).turns) == 5


@pytest.mark.parametrize(
    "turns",
    [
        (turn(0, 0), turn(1, 0), turn(0, 1)),
        (turn(0, 0), turn(1, 0), turn(0, 1), turn(1, 1), turn(1, 1)),
        (turn(1, 0), turn(0, 0), turn(0, 1), turn(1, 1)),
    ],
)
def test_truncated_duplicated_or_misordered_transcript_is_refused(turns):
    with pytest.raises(ValueError, match="one turn per"):
        DebateTranscript("q", 2, 1, turns, SystemAnswer("A", 1.0))


def test_turn_confidence_outside_unit_interval_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        AgentTurn("agent_0", 0, "text", "A", SignalValue(1.5), "stop")


def test_negative_round_is_refused():
    with pytest.raises(ValueError, match="round"):
        AgentTurn("agent_0", -1, "text", "A", NO_CONFIDENCE, "stop")


def test_unknown_finish_reason_is_refused():
    with pytest.raises(ValueError, match="finish_reason"):
        AgentTurn("agent_0", 0, "text", "A", NO_CONFIDENCE, "truncated")


@pytest.mark.parametrize(
    ("answer", "confidence", "reason"),
    [
        ("A", 0.5, SystemNull.TIE),
        (None, None, None),
        ("A", None, None),
        (None, 0.5, SystemNull.TIE),
        ("A", 1.2, None),
    ],
)
def test_system_answer_invariants(answer, confidence, reason):
    with pytest.raises(ValueError):
        SystemAnswer(answer, confidence, reason)


def test_empty_question_text_is_refused():
    with pytest.raises(ValueError, match="empty text"):
        DebateQuestion("q", "   ", None)


def test_question_with_fewer_than_two_choices_is_refused():
    with pytest.raises(ValueError, match="at least 2 choices"):
        DebateQuestion("q", "Which?", 1)


def test_majority_with_even_agents_is_refused():
    with pytest.raises(ValidationError, match="odd n_agents"):
        config(n_agents=4)


def test_argmax_confidence_without_stated_confidence_is_refused():
    with pytest.raises(ValidationError, match="states confidence"):
        config(aggregation="argmax_confidence", confidence_mode="none")


def test_greedy_decoding_with_several_agents_is_refused():
    with pytest.raises(ValidationError, match="identical"):
        config(temperature=0.0)


def test_debate_rounds_with_a_single_agent_are_refused():
    with pytest.raises(ValidationError, match="at least two agents"):
        config(n_agents=1)


@pytest.mark.parametrize("field", ["confidence_mode", "aggregation"])
def test_unknown_switch_is_refused(field):
    with pytest.raises(ValidationError, match=f"unknown {field}"):
        config(**{field: "nonexistent"})


def test_unknown_config_key_is_refused():
    with pytest.raises(ValidationError):
        config(n_agent=3)
