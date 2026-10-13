"""Aggregation rules."""

from collections import Counter
from collections.abc import Sequence

from madcal.debate.base import (
    AgentTurn,
    AggregationRule,
    SystemAnswer,
    SystemNull,
    aggregation_registry,
)


def majority_vote(turns: Sequence[AgentTurn]) -> SystemAnswer:
    """Return the most common parsed answer, with the share of all agents that gave it."""
    votes = Counter(turn.answer for turn in turns if turn.answer is not None)
    if not votes:
        return SystemAnswer(None, None, SystemNull.NO_VALID_ANSWERS)
    ranked = votes.most_common()
    answer, count = ranked[0]
    if len(ranked) > 1 and ranked[1][1] == count:
        return SystemAnswer(None, None, SystemNull.TIE)
    return SystemAnswer(answer, count / len(turns))


def majority_probability(turns: Sequence[AgentTurn]) -> SystemAnswer:
    """Return the majority answer, with the mean probability the scored agents put on it."""
    vote = majority_vote(turns)
    if vote.answer is None:
        return vote
    probabilities = [
        turn.scores.probabilities()[turn.scores.choices.index(vote.answer)]
        for turn in turns
        if turn.scores is not None
    ]
    if not probabilities:
        raise ValueError("majority_probability needs scored turns; none carry scores")
    return SystemAnswer(vote.answer, sum(probabilities) / len(probabilities))


def argmax_confidence(turns: Sequence[AgentTurn]) -> SystemAnswer:
    """Return the answer of the most confident agent among those with a parsed answer."""
    rated = [
        (turn.confidence.value, turn.answer)
        for turn in turns
        if turn.answer is not None and turn.confidence.value is not None
    ]
    if not rated:
        return SystemAnswer(None, None, SystemNull.NO_VALID_ANSWERS)
    best = max(confidence for confidence, _ in rated)
    answers = {answer for confidence, answer in rated if confidence == best}
    if len(answers) > 1:
        return SystemAnswer(None, None, SystemNull.TIE)
    return SystemAnswer(answers.pop(), best)


MAJORITY = aggregation_registry.register("majority")(
    AggregationRule(
        name="majority",
        requires_confidence=False,
        requires_odd_agents=True,
        aggregate=majority_vote,
    )
)

ARGMAX_CONFIDENCE = aggregation_registry.register("argmax_confidence")(
    AggregationRule(
        name="argmax_confidence",
        requires_confidence=True,
        requires_odd_agents=False,
        aggregate=argmax_confidence,
    )
)

MAJORITY_PROBABILITY = aggregation_registry.register("majority_probability")(
    AggregationRule(
        name="majority_probability",
        requires_confidence=True,
        requires_odd_agents=True,
        aggregate=majority_probability,
        requires_scores=True,
    )
)
