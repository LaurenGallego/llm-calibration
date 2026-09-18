import pytest

from madcal.debate import AgentTurn, SystemAnswer, SystemNull, argmax_confidence, majority_vote
from madcal.signals import SignalNull, SignalValue


def turn(answer: str | None, confidence: float | None = None) -> AgentTurn:
    value = (
        SignalValue(confidence)
        if confidence is not None
        else SignalValue(None, SignalNull.PARSE_FAILED)
    )
    return AgentTurn("agent_0", 0, "text", answer, value, "stop")


def test_majority_reports_the_share_of_all_agents_including_unparsed():
    result = majority_vote([turn("A"), turn("A"), turn(None)])
    assert result.answer == "A"
    assert result.confidence == pytest.approx(2 / 3)


def test_majority_of_five():
    assert majority_vote([turn(x) for x in "ABABA"]) == SystemAnswer("A", 0.6)


def test_three_different_answers_tie_even_with_odd_agents():
    assert majority_vote([turn("A"), turn("B"), turn("C")]).reason is SystemNull.TIE


def test_unparsed_agent_can_leave_an_even_split():
    assert majority_vote([turn("A"), turn("B"), turn(None)]).reason is SystemNull.TIE


def test_majority_with_no_parsed_answers():
    assert majority_vote([turn(None)] * 3).reason is SystemNull.NO_VALID_ANSWERS


def test_argmax_takes_the_most_confident_parsed_answer():
    result = argmax_confidence([turn("A", 0.6), turn("B", 0.9), turn("C", 0.7)])
    assert result == SystemAnswer("B", 0.9)


def test_argmax_ignores_agents_without_a_parsed_answer_or_confidence():
    result = argmax_confidence([turn(None, 0.99), turn("A"), turn("C", 0.4)])
    assert result == SystemAnswer("C", 0.4)


def test_argmax_tie_between_different_answers():
    result = argmax_confidence([turn("A", 0.8), turn("B", 0.8)])
    assert result.reason is SystemNull.TIE


def test_argmax_tie_on_the_same_answer_is_not_a_tie():
    assert argmax_confidence([turn("A", 0.8), turn("A", 0.8)]) == SystemAnswer("A", 0.8)


def test_argmax_with_nothing_rated():
    assert argmax_confidence([turn("A"), turn(None)]).reason is SystemNull.NO_VALID_ANSWERS
