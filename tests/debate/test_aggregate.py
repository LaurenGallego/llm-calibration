import math
from dataclasses import replace

import pytest

from madcal.debate import (
    AgentTurn,
    SystemAnswer,
    SystemNull,
    argmax_confidence,
    majority_probability,
    majority_vote,
    mean_distribution,
    pooled_distribution,
)
from madcal.models import ChoiceScores, Generation
from madcal.signals import SignalNull, SignalValue


def turn(answer: str | None, confidence: float | None = None) -> AgentTurn:
    value = (
        SignalValue(confidence)
        if confidence is not None
        else SignalValue(None, SignalNull.PARSE_FAILED)
    )
    generation = Generation(text="text", token_logprobs=None, tokens=None, finish_reason="stop")
    return AgentTurn("agent_0", 0, generation, answer, value)


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


def scored(answer: str | None, probabilities: tuple[float, ...] = ()) -> AgentTurn:
    if answer is None:
        return turn(None)
    scores = ChoiceScores(choices=("A", "B", "C"), logprobs=tuple(map(math.log, probabilities)))
    return replace(turn(answer, probabilities["ABC".index(answer)]), scores=scores)


def test_majority_probability_averages_every_scored_agent_probability_on_the_majority():
    result = majority_probability(
        [scored("A", (0.6, 0.3, 0.1)), scored("A", (0.8, 0.1, 0.1)), scored("B", (0.2, 0.7, 0.1))]
    )
    assert result.answer == "A"
    assert result.confidence == pytest.approx((0.6 + 0.8 + 0.2) / 3)


def test_majority_probability_leaves_unparsed_agents_out_of_the_mean():
    result = majority_probability(
        [scored("A", (0.5, 0.25, 0.25)), scored("A", (0.9, 0.05, 0.05)), scored(None)]
    )
    assert result.confidence == pytest.approx(0.7)


def test_majority_probability_keeps_the_vote_nulls():
    split = [scored("A", (0.4, 0.3, 0.3)), scored("B", (0.3, 0.4, 0.3)), scored(None)]
    assert majority_probability(split).reason is SystemNull.TIE


def test_majority_probability_without_scores_is_refused():
    with pytest.raises(ValueError, match="scored turns"):
        majority_probability([turn("A", 0.5), turn("A", 0.5), turn("B", 0.5)])


def test_pooled_distribution_spreads_each_voters_remaining_mass_evenly():
    pooled = pooled_distribution([("A", 0.7), ("A", 0.9), ("B", 0.4)], "ABCD")
    assert pooled == pytest.approx({"A": 0.6, "B": 8 / 45, "C": 1 / 9, "D": 1 / 9})


def test_one_agent_keeps_its_own_answer_and_stated_confidence():
    assert mean_distribution([turn("C", 0.8)], 4) == SystemAnswer("C", 0.8)


def test_dissent_lowers_the_pooled_confidence():
    result = mean_distribution([turn("A", 0.7), turn("A", 0.9), turn("B", 0.4)], 4)
    assert result.answer == "A"
    assert result.confidence == pytest.approx(0.6)


def test_a_confident_minority_can_outweigh_a_hesitant_majority():
    result = mean_distribution([turn("A", 0.3), turn("A", 0.3), turn("B", 1.0)], 4)
    assert result.answer == "B"
    assert result.confidence == pytest.approx(22 / 45)


def test_pooling_ignores_agents_without_an_answer_or_confidence():
    result = mean_distribution([turn("A", 0.6), turn(None, 0.9), turn("B")], 4)
    assert result == SystemAnswer("A", 0.6)


def test_equal_pooled_mass_is_a_tie():
    assert mean_distribution([turn("A", 0.8), turn("B", 0.8)], 4).reason is SystemNull.TIE


def test_pooling_without_votes_has_no_valid_answers():
    assert mean_distribution([turn(None)] * 3, 4).reason is SystemNull.NO_VALID_ANSWERS


def test_pooling_needs_a_fixed_option_set():
    with pytest.raises(ValueError, match="options"):
        mean_distribution([turn("A", 0.8)], None)


def test_votes_outside_the_options_are_refused():
    with pytest.raises(ValueError, match="outside"):
        pooled_distribution([("E", 0.8)], "ABCD")
