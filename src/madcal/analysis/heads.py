"""Answer-level calibration of every fitted arm on a cache's evaluation questions."""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
from numpy.typing import NDArray

from madcal.analysis.estimate import bootstrap_by_question
from madcal.analysis.propagation import Compute, confidence_metrics
from madcal.analysis.report import ANALYSIS_STATUS
from madcal.models.calibrated import file_sha256
from madcal.storage.states import StateCache
from madcal.training import Arm, OutputHead, answer_changed, batched_confidence, build_arm

BASE = "base"
TEMPERATURE = "temperature"
ANSWER_CHANGED = "answer_changed"


@dataclass(frozen=True, slots=True)
class ScoredArm:
    """One arm's confidence in every evaluation answer."""

    label: str
    arm: str
    confidence: NDArray[np.float64]
    changed: float | None


@dataclass(frozen=True, slots=True)
class HeadRow:
    """One metric of one arm, or its paired difference from a reference arm."""

    benchmark: str
    arm: str
    comparison: str
    metric: str
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    reason: str | None
    n_questions: int
    n_resamples_failed: int
    analysis_status: str


def load_fitted_arm(fit_dir: Path, cache: StateCache, head: OutputHead) -> tuple[str, Arm]:
    """Rebuild the arm a fit directory holds, refusing one fitted on a different cache."""
    record = json.loads((fit_dir / "fit.json").read_text())
    if record["cache_hash"] != cache.manifest["config_hash"]:
        raise ValueError(f"{fit_dir} was fitted on cache {record['cache_hash']}, not this one")
    if file_sha256(fit_dir / "checkpoint.pt") != record["checkpoint_sha256"]:
        raise ValueError(f"{fit_dir / 'checkpoint.pt'} does not match its recorded sha256")
    settings = {k: v for k, v in record["config"]["arm"].items() if k != "name"}
    arm = build_arm(record["arm"], head, cache.answers.hidden_size, int(record["seed"]), **settings)
    arm.load_checkpoint(torch.load(fit_dir / "checkpoint.pt", weights_only=True))
    loss = record.get("loss", "brier")
    name = record["arm"] if loss == "brier" else f"{record['arm']}-{loss}"
    return f"{name}/lr{record['config']['lr']:g}/seed{record['seed']}", arm


def score_arms(
    cache: StateCache, fit_dirs: Sequence[Path], device: str, batch_questions: int = 64
) -> tuple[tuple[str, ...], NDArray[np.bool_], list[ScoredArm]]:
    """Return the evaluation question ids, their grades and every arm's confidence in them."""
    head = OutputHead(cache.weight, cache.bias, device)
    answers = cache.answers.with_role("evaluation").to(device)
    arms = [(BASE, build_arm(BASE, head, answers.hidden_size, 0))]
    arms += [load_fitted_arm(path, cache, head) for path in fit_dirs]
    labels = [label for label, _ in arms]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two fits share an arm, loss, lr and seed: {sorted(labels)}")
    scored = []
    for label, arm in arms:
        arm.eval()
        confidence = batched_confidence(arm, answers, batch_questions).cpu().double().numpy()
        scored.append(
            ScoredArm(
                label, label.split("/")[0], confidence, _changed(arm, answers, batch_questions)
            )
        )
    return answers.question_ids, answers.correct.cpu().numpy(), scored


def head_rows(
    benchmark: str,
    question_ids: Sequence[str],
    correct: NDArray[np.bool_],
    scored: Sequence[ScoredArm],
    n_resamples: int = 2000,
    seed: int = 0,
) -> list[HeadRow]:
    """Every metric per arm, and its paired difference from the base and from the first global T."""
    metrics = confidence_metrics()
    references = [
        next(s for s in scored if s.arm == arm)
        for arm in (BASE, TEMPERATURE)
        if any(s.arm == arm for s in scored)
    ]
    rows: list[HeadRow] = []
    for arm in scored:
        if arm.changed is not None:
            rows.append(_fixed(benchmark, arm.label, ANSWER_CHANGED, arm.changed, len(correct)))
        for metric, compute in metrics.items():
            estimate = bootstrap_by_question(
                _single(correct, arm.confidence, compute), question_ids, n_resamples, seed
            )
            rows.append(_row(benchmark, arm.label, "arm", metric, estimate, len(correct)))
            for reference in references:
                if reference.label == arm.label:
                    continue
                estimate = bootstrap_by_question(
                    _paired(correct, arm.confidence, reference.confidence, compute),
                    question_ids,
                    n_resamples,
                    seed,
                )
                comparison = f"{arm.label} - {reference.label}"
                rows.append(_row(benchmark, arm.label, comparison, metric, estimate, len(correct)))
    return rows


def heads_table(rows: Sequence[HeadRow]) -> pa.Table:
    """The rows as an Arrow table."""
    return pa.Table.from_pylist([asdict(row) for row in rows])


def _changed(arm: Arm, answers, batch_questions: int) -> float | None:
    changed: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(answers), batch_questions):
            batch = answers.select(range(start, min(start + batch_questions, len(answers))))
            log_probs = arm.log_probs(batch)
            if log_probs is None:
                return None
            changed.append(answer_changed(log_probs, batch).cpu())
    return float(torch.cat(changed).float().mean())


def _single(correct, confidence, compute: Compute):
    return lambda indices: compute(correct[indices], confidence[indices])


def _paired(correct, confidence, reference, compute: Compute):
    return lambda indices: (
        compute(correct[indices], confidence[indices])
        - compute(correct[indices], reference[indices])
    )


def _row(benchmark, label, comparison, metric, estimate, n) -> HeadRow:
    return HeadRow(
        benchmark=benchmark,
        arm=label,
        comparison=comparison,
        metric=metric,
        estimate=estimate.value,
        ci_low=estimate.ci_low,
        ci_high=estimate.ci_high,
        reason=None if estimate.reason is None else str(estimate.reason),
        n_questions=n,
        n_resamples_failed=estimate.n_resamples_failed,
        analysis_status=ANALYSIS_STATUS,
    )


def _fixed(benchmark, label, metric, value, n) -> HeadRow:
    return HeadRow(benchmark, label, "arm", metric, value, None, None, None, n, 0, ANALYSIS_STATUS)
