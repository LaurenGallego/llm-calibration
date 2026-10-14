"""Semantic clusters of sampled answers and the confidence each measure gives a reference answer."""

import math
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass

type Entails = Callable[[Sequence[tuple[str, str]]], Sequence[bool]]

MEASURES = ("e_sc", "l_sc", "l_sc_norm", "ml_sc", "b_sc")

_PUNCTUATION = str.maketrans("", "", string.punctuation)


@dataclass(frozen=True, slots=True)
class SampledAnswer:
    """One sampled response: the answer read from it, its token log-probabilities and tokens."""

    answer: str
    token_logprobs: tuple[float, ...]
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError("a sampled answer cannot be empty")
        if not self.token_logprobs:
            raise ValueError("a sampled answer needs the log-probabilities of its tokens")
        if len(self.tokens) != len(self.token_logprobs):
            raise ValueError("tokens and token_logprobs must align")

    @property
    def logprob(self) -> float:
        """Log-probability of the whole sampled sequence."""
        return math.fsum(self.token_logprobs)

    @property
    def normalised_logprob(self) -> float:
        """Log-probability of the sequence divided by its length in tokens."""
        return self.logprob / len(self.token_logprobs)


@dataclass(frozen=True, slots=True)
class SemanticConfidence:
    """Confidence of the reference answer under every measure, and how the samples clustered."""

    confidences: dict[str, float]
    n_clusters: int
    matched: bool


def canonical(answer: str) -> str:
    """Return `answer` lowercased with punctuation removed and whitespace collapsed."""
    return " ".join(answer.casefold().translate(_PUNCTUATION).split())


def cluster_answers(question: str, answers: Sequence[str], entails: Entails | None) -> list[int]:
    """Return a cluster index per answer, joining an answer to the first equivalent cluster."""
    representatives: list[str] = []
    labels: list[int] = []
    for answer in answers:
        match = first_equivalent(question, answer, representatives, entails)
        if match is None:
            representatives.append(answer)
            match = len(representatives) - 1
        labels.append(match)
    return labels


def first_equivalent(
    question: str, answer: str, candidates: Sequence[str], entails: Entails | None
) -> int | None:
    """Return the index of the first candidate equivalent to `answer`, or None."""
    for index, candidate in enumerate(candidates):
        if canonical(candidate) == canonical(answer):
            return index
    if entails is None or not candidates:
        return None
    pairs = []
    for candidate in candidates:
        first, second = f"{question} {answer}", f"{question} {candidate}"
        pairs += [(first, second), (second, first)]
    verdicts = list(entails(pairs))
    for index in range(len(candidates)):
        if verdicts[2 * index] and verdicts[2 * index + 1]:
            return index
    return None


def cluster_distributions(
    labels: Sequence[int], samples: Sequence[SampledAnswer]
) -> dict[str, list[float]]:
    """Return each measure's probability distribution over the clusters."""
    if len(labels) != len(samples) or not samples:
        raise ValueError("every sample needs exactly one cluster label")
    n_clusters = max(labels) + 1
    members = [
        [s for s, label in zip(samples, labels, strict=True) if label == c]
        for c in range(n_clusters)
    ]
    total = len(samples)
    distinct = [{s.tokens: s.logprob for s in cluster} for cluster in members]
    normalised = [_logsumexp([s.normalised_logprob for s in cluster]) for cluster in members]
    return {
        "e_sc": [len(cluster) / total for cluster in members],
        "l_sc": _softmax([_logsumexp(list(sequences.values())) for sequences in distinct]),
        "l_sc_norm": _softmax(normalised),
        "ml_sc": _softmax(
            [
                score - math.log(len(cluster))
                for score, cluster in zip(normalised, members, strict=True)
            ]
        ),
        "b_sc": _softmax(
            [
                math.fsum(s.normalised_logprob for s in cluster) + math.log(len(cluster) / total)
                for cluster in members
            ]
        ),
    }


def semantic_confidence(
    question: str,
    reference: str,
    samples: Sequence[SampledAnswer],
    entails: Entails | None,
) -> SemanticConfidence:
    """Return the probability each measure puts on the cluster of the reference answer."""
    labels = cluster_answers(question, [sample.answer for sample in samples], entails)
    distributions = cluster_distributions(labels, samples)
    representatives = [samples[labels.index(c)].answer for c in range(max(labels) + 1)]
    match = first_equivalent(question, reference, representatives, entails)
    confidences = {
        measure: 0.0 if match is None else distribution[match]
        for measure, distribution in distributions.items()
    }
    return SemanticConfidence(confidences, len(representatives), match is not None)


def _logsumexp(values: Sequence[float]) -> float:
    peak = max(values)
    return peak + math.log(math.fsum(math.exp(value - peak) for value in values))


def _softmax(scores: Sequence[float]) -> list[float]:
    total = _logsumexp(scores)
    return [math.exp(score - total) for score in scores]
