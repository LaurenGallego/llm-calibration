import pytest
import torch

from madcal.signals.semantic import SampledAnswer, cluster_distributions
from madcal.storage.states import AnswerStates
from madcal.training import answer_changed, answer_log_confidence, brier_loss
from madcal.training.objective import l_sc_norm, sequence_log_probs


def open_answers() -> AnswerStates:
    return AnswerStates(
        question_ids=("a", "b"),
        roles=("train", "train"),
        correct=torch.tensor([True, False]),
        states=torch.zeros(3, 2),
        offsets=torch.tensor([0, 2, 3]),
        targets=torch.tensor([0, 2, 1]),
    )


OPEN_PROBS = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8], [0.25, 0.25, 0.5]])


def test_an_open_answer_is_the_product_of_its_token_probabilities():
    confidence = answer_log_confidence(OPEN_PROBS.log(), open_answers()).exp()
    assert confidence.tolist() == pytest.approx([0.4, 0.25])


def test_a_multiple_choice_answer_is_renormalised_over_its_options():
    answers = AnswerStates(
        question_ids=("q",),
        roles=("train",),
        correct=torch.tensor([True]),
        states=torch.zeros(1, 2),
        offsets=torch.tensor([0, 1]),
        targets=torch.tensor([3]),
        options=torch.tensor([[1, 3]]),
        chosen=torch.tensor([1]),
    )
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    assert float(answer_log_confidence(probs.log(), answers).exp()) == pytest.approx(2 / 3)


def test_an_answer_changes_where_another_token_is_most_probable():
    assert answer_changed(OPEN_PROBS.log(), open_answers()).tolist() == [False, False, True]


def test_brier_is_the_mean_squared_gap_to_correctness():
    loss = brier_loss(torch.tensor([0.4, 0.25]), torch.tensor([True, False]))
    assert float(loss) == pytest.approx(0.21125)


def test_l_sc_norm_shares_length_normalised_mass_between_a_question_s_clusters():
    confidence = l_sc_norm(
        sample_log_probs=torch.tensor([-1.0, -2.0, -4.0, -3.0]),
        sample_lengths=torch.tensor([1, 2, 2, 3]),
        sample_cluster=torch.tensor([0, 0, 1, 2]),
        cluster_question=torch.tensor([0, 0, 1]),
        reference_cluster=torch.tensor([0, -1]),
    )
    assert confidence.tolist() == pytest.approx([0.8446375965, 0.0])


def test_l_sc_norm_matches_the_semantic_signal(random_samples):
    log_probs, lengths, labels = random_samples
    samples = [
        SampledAnswer(f"answer {label}", tuple([lp / n] * n), tuple(["t"] * n))
        for lp, n, label in zip(log_probs.tolist(), lengths.tolist(), labels.tolist(), strict=True)
    ]
    expected = cluster_distributions(labels.tolist(), samples)["l_sc_norm"]
    n_clusters = len(expected)
    confidence = l_sc_norm(
        log_probs,
        lengths,
        labels,
        torch.zeros(n_clusters, dtype=torch.long),
        torch.tensor([n_clusters - 1]),
    )
    assert float(confidence) == pytest.approx(expected[-1])


@pytest.fixture
def random_samples() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    lengths = torch.randint(1, 6, (10,), generator=generator)
    log_probs = -torch.rand(10, generator=generator, dtype=torch.float64) * lengths
    labels = torch.tensor([0, 1, 0, 2, 1, 0, 3, 2, 0, 1])
    return log_probs, lengths, labels


def test_a_single_cluster_gives_confidence_one_and_no_gradient():
    log_probs = torch.tensor([-1.0, -3.0], requires_grad=True)
    confidence = l_sc_norm(
        log_probs,
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        torch.tensor([0]),
        torch.tensor([0]),
    )
    confidence.sum().backward()
    assert float(confidence) == pytest.approx(1.0)
    assert log_probs.grad is not None and log_probs.grad.abs().max() < 1e-7


def test_sequence_log_probs_sums_each_sequence_s_positions():
    totals = sequence_log_probs(OPEN_PROBS.log(), torch.tensor([0, 2, 1]), torch.tensor([0, 2, 3]))
    assert totals.exp().tolist() == pytest.approx([0.4, 0.25])
