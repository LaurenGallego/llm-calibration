import pytest
import torch

from madcal.storage.states import AnswerStates
from madcal.training import answer_changed, answer_log_confidence, brier_loss


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
