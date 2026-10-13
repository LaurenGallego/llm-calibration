"""Aggregation rules."""

import math
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


def pooled_distribution(
    votes: Sequence[tuple[str, float]], options: Sequence[str]
) -> dict[str, float]:
    """Return the mean of the voters' distributions: confidence on the answer, the rest even."""
    if not votes:
        raise ValueError("pooling needs at least one vote")
    if len(options) < 2:
        raise ValueError(f"pooling needs at least two options, got {len(options)}")
    unknown = {answer for answer, _ in votes} - set(options)
    if unknown:
        raise ValueError(f"votes name answers outside the options: {sorted(unknown)}")
    spread = len(options) - 1
    return {
        option: sum(
            confidence if answer == option else (1.0 - confidence) / spread
            for answer, confidence in votes
        )
        / len(votes)
        for option in options
    }


def pooled_answer(votes: Sequence[tuple[str, float]], options: Sequence[str]) -> SystemAnswer:
    """Return the option with the most pooled probability, with that probability."""
    if not votes:
        return SystemAnswer(None, None, SystemNull.NO_VALID_ANSWERS)
    pooled = pooled_distribution(votes, options)
    ranked = sorted(pooled.items(), key=lambda item: item[1], reverse=True)
    if math.isclose(ranked[0][1], ranked[1][1]):
        return SystemAnswer(None, None, SystemNull.TIE)
    return SystemAnswer(ranked[0][0], ranked[0][1])


def mean_distribution(turns: Sequence[AgentTurn], n_choices: int | None) -> SystemAnswer:
    """Return the top option of the agents' pooled stated-confidence distribution."""
    if n_choices is None:
        raise ValueError("mean_distribution needs a fixed set of answer options")
    votes = [
        (turn.answer, turn.confidence.value)
        for turn in turns
        if turn.answer is not None and turn.confidence.value is not None
    ]
    return pooled_answer(votes, choice_options(n_choices))


def choice_options(n_choices: int) -> tuple[str, ...]:
    """Return the option letters of a question with `n_choices` choices."""
    return tuple(chr(ord("A") + index) for index in range(n_choices))


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
        aggregate=lambda turns, _: majority_vote(turns),
    )
)

ARGMAX_CONFIDENCE = aggregation_registry.register("argmax_confidence")(
    AggregationRule(
        name="argmax_confidence",
        requires_confidence=True,
        requires_odd_agents=False,
        aggregate=lambda turns, _: argmax_confidence(turns),
    )
)

MAJORITY_PROBABILITY = aggregation_registry.register("majority_probability")(
    AggregationRule(
        name="majority_probability",
        requires_confidence=True,
        requires_odd_agents=True,
        aggregate=lambda turns, _: majority_probability(turns),
        requires_scores=True,
    )
)

MEAN_DISTRIBUTION = aggregation_registry.register("mean_distribution")(
    AggregationRule(
        name="mean_distribution",
        requires_confidence=True,
        requires_odd_agents=False,
        aggregate=mean_distribution,
    )
)
