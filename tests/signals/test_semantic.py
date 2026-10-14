import math

import pytest

from madcal.signals import (
    SampledAnswer,
    cluster_answers,
    cluster_distributions,
    semantic_confidence,
)

QUESTION = "What is the capital of France?"


def sample(answer: str, *logprobs: float) -> SampledAnswer:
    return SampledAnswer(answer, logprobs, tuple(f"{answer}:{i}" for i in range(len(logprobs))))


PARIS = sample("Paris", -0.5, -0.5)
PARIS_DOT = sample("paris.", -1.0, -1.0, -1.0)
LYON = sample("Lyon", -1.0, -1.0)
E = math.exp


def table(*equivalent: tuple[str, str]):
    calls: list[tuple[str, str]] = []
    holds = {(f"{QUESTION} {a}", f"{QUESTION} {b}") for a, b in equivalent}

    def entails(pairs):
        calls.extend(pairs)
        return [pair in holds for pair in pairs]

    return entails, calls


def test_answers_with_the_same_canonical_form_cluster_without_asking_the_judge():
    entails, calls = table()
    assert cluster_answers(QUESTION, ["Paris", "paris.", " PARIS "], entails) == [0, 0, 0]
    assert calls == []


def test_answers_cluster_only_on_entailment_in_both_directions():
    entails, _ = table(
        ("Shakespeare", "William Shakespeare"),
        ("William Shakespeare", "Shakespeare"),
        ("Shakespeare", "a playwright"),
    )
    answers = ["Shakespeare", "William Shakespeare", "a playwright", "Marlowe"]
    assert cluster_answers(QUESTION, answers, entails) == [0, 0, 1, 2]


def test_without_a_judge_only_canonical_forms_cluster():
    assert cluster_answers(QUESTION, ["Shakespeare", "William Shakespeare"], None) == [0, 1]


def test_each_measure_matches_its_hand_computed_distribution():
    distributions = cluster_distributions([0, 0, 1], [PARIS, PARIS_DOT, LYON])
    assert distributions["e_sc"] == pytest.approx([2 / 3, 1 / 3])
    paris_l = E(-1.0) + E(-3.0)
    assert distributions["l_sc"] == pytest.approx(
        [paris_l / (paris_l + E(-2.0)), E(-2.0) / (paris_l + E(-2.0))]
    )
    paris_norm = E(-0.5) + E(-1.0)
    assert distributions["l_sc_norm"] == pytest.approx(
        [paris_norm / (paris_norm + E(-1.0)), E(-1.0) / (paris_norm + E(-1.0))]
    )
    assert distributions["ml_sc"] == pytest.approx(
        [(paris_norm / 2) / (paris_norm / 2 + E(-1.0)), E(-1.0) / (paris_norm / 2 + E(-1.0))]
    )
    paris_b = E(-1.5) * 2 / 3
    lyon_b = E(-1.0) / 3
    assert distributions["b_sc"] == pytest.approx(
        [paris_b / (paris_b + lyon_b), lyon_b / (paris_b + lyon_b)]
    )


def test_the_summed_likelihood_counts_a_repeated_sequence_once():
    distributions = cluster_distributions([0, 0, 1], [PARIS, PARIS, LYON])
    assert distributions["l_sc"][0] == pytest.approx(E(-1.0) / (E(-1.0) + E(-2.0)))
    assert distributions["e_sc"][0] == pytest.approx(2 / 3)
    assert distributions["l_sc_norm"][0] == pytest.approx(2 * E(-0.5) / (2 * E(-0.5) + E(-1.0)))


def test_the_reference_takes_the_probability_of_its_cluster():
    result = semantic_confidence(QUESTION, "PARIS", [PARIS, PARIS_DOT, LYON], None)
    assert result.matched
    assert result.n_clusters == 2
    assert result.confidences["e_sc"] == pytest.approx(2 / 3)
    paris_l = E(-1.0) + E(-3.0)
    assert result.confidences["l_sc"] == pytest.approx(paris_l / (paris_l + E(-2.0)))


def test_a_reference_outside_every_cluster_has_zero_confidence():
    result = semantic_confidence(QUESTION, "Marseille", [PARIS, PARIS_DOT, LYON], None)
    assert not result.matched
    assert set(result.confidences.values()) == {0.0}


def test_every_distribution_sums_to_one():
    distributions = cluster_distributions([0, 1, 2], [PARIS, PARIS_DOT, LYON])
    for values in distributions.values():
        assert math.fsum(values) == pytest.approx(1.0)


def test_an_empty_answer_or_a_sample_without_logprobs_is_refused():
    with pytest.raises(ValueError, match="empty"):
        SampledAnswer(" ", (-1.0,), ("a",))
    with pytest.raises(ValueError, match="log-probabilities"):
        SampledAnswer("Paris", (), ())
