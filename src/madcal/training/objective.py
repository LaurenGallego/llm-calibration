"""Answer-level confidence from per-position log-probabilities, and the Brier objective."""

import torch

from madcal.storage.states import AnswerStates


def answer_log_confidence(log_probs: torch.Tensor, answers: AnswerStates) -> torch.Tensor:
    """Log-probability of each question's own answer: the token sequence, or the chosen option."""
    if answers.options is None or answers.chosen is None:
        picked = log_probs.gather(1, answers.targets[:, None]).squeeze(1)
        total = torch.zeros(len(answers), dtype=picked.dtype, device=picked.device)
        return total.index_add(0, answers.positions(), picked)
    options = log_probs.gather(1, answers.options).log_softmax(dim=-1)
    return options.gather(1, answers.chosen[:, None]).squeeze(1)


def answer_changed(log_probs: torch.Tensor, answers: AnswerStates) -> torch.Tensor:
    """Whether each position's most probable token (or option) differs from the answer given."""
    if answers.options is None or answers.chosen is None:
        return log_probs.argmax(dim=-1) != answers.targets
    return log_probs.gather(1, answers.options).argmax(dim=-1) != answers.chosen


def brier_loss(confidence: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Mean squared gap between confidence and correctness."""
    return ((confidence - correct.to(confidence.dtype)) ** 2).mean()
