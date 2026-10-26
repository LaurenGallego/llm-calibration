"""Fit an arm on training answers, selecting on the whole held-out selection split."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from madcal.storage.states import AnswerStates
from madcal.training.arms import Arm, Loss, batched_loss, loss_terms


@dataclass(frozen=True, slots=True)
class TrainResult:
    best_step: int
    selection_loss: float
    curve: tuple[tuple[float, float], ...]


def in_selection(question_ids: Sequence[str], fraction: float, seed: int) -> list[bool]:
    """Assign each question to the selection split by a seeded hash of its id."""
    if not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must lie in (0, 1), got {fraction}")
    return [
        int.from_bytes(hashlib.sha256(f"{seed}:{qid}".encode()).digest()[:8]) / 2**64 < fraction
        for qid in question_ids
    ]


def split(answers: AnswerStates, fraction: float, seed: int) -> tuple[AnswerStates, AnswerStates]:
    """Return the fit and selection parts of the training answers."""
    chosen = in_selection(answers.question_ids, fraction, seed)
    fit = answers.select([i for i, picked in enumerate(chosen) if not picked])
    selection = answers.select([i for i, picked in enumerate(chosen) if picked])
    if len(fit) == 0 or len(selection) == 0:
        raise ValueError("the fit/selection split left one side empty")
    return fit, selection


def train(
    arm: Arm,
    fit: AnswerStates,
    selection: AnswerStates,
    lr: float,
    steps: int,
    batch_questions: int,
    eval_every: int,
    seed: int,
    loss: Loss = "brier",
) -> TrainResult:
    """Fit `arm` on `loss` and leave it at the step with the lowest selection loss."""
    direct = arm.fit_directly(fit, batch_questions, loss)
    if direct is not None:
        return TrainResult(0, batched_loss(arm, selection, batch_questions, loss), tuple(direct))

    parameters = [p for p in arm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    best = (0, batched_loss(arm, selection, batch_questions, loss), arm.checkpoint())
    curve = [(0.0, best[1])]
    for step in range(1, steps + 1):
        picks = torch.randperm(len(fit), generator=generator)[:batch_questions]
        batch = fit.select(picks)
        objective = loss_terms(arm, batch, loss).mean() + arm.penalty(batch)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        if step % eval_every == 0 or step == steps:
            value = batched_loss(arm, selection, batch_questions, loss)
            curve.append((float(step), value))
            if value < best[1]:
                best = (step, value, arm.checkpoint())
    arm.load_checkpoint(best[2])
    return TrainResult(best[0], best[1], tuple(curve))
