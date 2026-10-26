"""Answer-level confidence from per-position log-probabilities, and the training objectives."""

import torch

from madcal.storage.states import AnswerStates

LOG_CONFIDENCE_CEILING = -1e-12


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


def log_loss_terms(log_confidence: torch.Tensor, correct: torch.Tensor) -> torch.Tensor:
    """Binary log loss of each answer's confidence against its correctness."""
    log_confidence = log_confidence.clamp(max=LOG_CONFIDENCE_CEILING)
    target = correct.to(log_confidence.dtype)
    log_doubt = torch.log(-torch.expm1(log_confidence))
    return -(target * log_confidence + (1.0 - target) * log_doubt)


def target_log_probs(log_probs: torch.Tensor, answers: AnswerStates) -> torch.Tensor:
    """Log-probability of the stored target token at each answer position."""
    return log_probs.gather(1, answers.targets[:, None]).squeeze(1)


def kl_to_base(log_probs: torch.Tensor, base_log_probs: torch.Tensor) -> torch.Tensor:
    """KL divergence of each position's distribution from the base head's."""
    return (log_probs.exp() * (log_probs - base_log_probs)).sum(dim=-1)


def sequence_log_probs(
    log_probs: torch.Tensor, targets: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Summed log-probability of each sequence, whose positions `offsets` delimits."""
    picked = log_probs.gather(1, targets[:, None]).squeeze(1)
    owner = torch.repeat_interleave(
        torch.arange(len(offsets) - 1, device=picked.device), offsets.diff()
    )
    total = torch.zeros(len(offsets) - 1, dtype=picked.dtype, device=picked.device)
    return total.index_add(0, owner, picked)


def l_sc_norm(
    sample_log_probs: torch.Tensor,
    sample_lengths: torch.Tensor,
    sample_cluster: torch.Tensor,
    cluster_question: torch.Tensor,
    reference_cluster: torch.Tensor,
) -> torch.Tensor:
    """L-SC-norm of each question's reference cluster; 0 where the reference matched no cluster."""
    normalised = sample_log_probs / sample_lengths.to(sample_log_probs.dtype)
    scores = _grouped_logsumexp(normalised, sample_cluster, len(cluster_question))
    totals = _grouped_logsumexp(scores, cluster_question, len(reference_cluster))
    shares = (scores - totals[cluster_question]).exp()
    matched = reference_cluster >= 0
    picked = shares[reference_cluster.clamp(min=0)]
    return torch.where(matched, picked, torch.zeros_like(picked))


def _grouped_logsumexp(values: torch.Tensor, groups: torch.Tensor, n_groups: int) -> torch.Tensor:
    peaks = torch.full((n_groups,), -torch.inf, dtype=values.dtype, device=values.device)
    peaks = peaks.scatter_reduce(0, groups, values.detach(), reduce="amax").detach()
    sums = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    sums = sums.index_add(0, groups, (values - peaks[groups]).exp())
    return peaks + sums.log()
