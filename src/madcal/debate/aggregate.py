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
