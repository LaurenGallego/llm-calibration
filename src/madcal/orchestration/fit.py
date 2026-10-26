"""Fit one arm on a harvested cache and write its checkpoint and fit record."""

import io
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from madcal.config import FitConfig
from madcal.models.calibrated import file_sha256
from madcal.orchestration.provenance import code_sha
from madcal.storage.states import AnswerStates, StateCache, StateStore
from madcal.training import OutputHead, build_arm, split, train

CHECKPOINT = "checkpoint.pt"
RECORD = "fit.json"


@dataclass(frozen=True, slots=True)
class FitOutcome:
    fit_hash: str
    root: Path
    best_step: int
    selection_loss: float


def resolve_device(device: str) -> str:
    """Return `device`, or for `auto` CUDA when present and CPU otherwise."""
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit(config: FitConfig) -> FitOutcome:
    """Fit the configured arm and write `checkpoint.pt` and `fit.json` under its own hash."""
    if config.cache is None:
        raise ValueError("a fit needs a cache directory: set `cache` or pass --cache")
    cache = StateStore(Path(config.cache)).load()
    code = code_sha()
    fit_hash = config.config_hash(cache.manifest["config_hash"], code)
    root = Path(config.output_root) / fit_hash
    if (root / RECORD).exists():
        raise FileExistsError(f"{root} already holds this fit; results are never overwritten")
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)

    head = OutputHead(cache.weight, cache.bias, device)
    answers = cache.answers.with_role("train")
    target = None
    if config.target_cache is not None:
        target = StateStore(Path(config.target_cache)).load()
        answers = _target_answers(cache, target)
    fit_part, selection = split(answers, config.selection_fraction, config.split_seed)
    settings = config.arm.model_dump(exclude={"name"})
    arm = build_arm(config.arm.name, head, answers.hidden_size, config.seed, **settings)
    result = train(
        arm,
        fit_part.to(device),
        selection.to(device),
        config.lr,
        config.steps,
        config.batch_questions,
        config.eval_every,
        config.seed,
        config.loss,
    )

    root.mkdir(parents=True, exist_ok=True)
    _write_bytes(_saved(arm.checkpoint()), root / CHECKPOINT)
    record: dict[str, Any] = {
        "fit_hash": fit_hash,
        "config": config.model_dump(mode="json"),
        "arm": config.arm.name,
        "seed": config.seed,
        "cache": str(config.cache),
        "cache_hash": cache.manifest["config_hash"],
        "target_cache_hash": None if target is None else target.manifest["config_hash"],
        "loss": config.loss,
        "model_id": cache.manifest["model_id"],
        "model_checkpoint_sha": cache.manifest["checkpoint_sha"],
        "benchmark": cache.manifest["benchmark"],
        "code_sha": code,
        "device": device,
        "n_fit": len(fit_part),
        "n_selection": len(selection),
        "best_step": result.best_step,
        "selection_loss": result.selection_loss,
        "curve": [list(point) for point in result.curve],
        "diagnostics": arm.diagnostics(selection.to(device)),
        "checkpoint_sha256": file_sha256(root / CHECKPOINT),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_bytes(json.dumps(record, indent=2, sort_keys=True).encode(), root / RECORD)
    return FitOutcome(fit_hash, root, result.best_step, result.selection_loss)


def _target_answers(cache: StateCache, target: StateCache) -> AnswerStates:
    for key in ("checkpoint_sha", "benchmark"):
        if target.manifest[key] != cache.manifest[key]:
            raise ValueError(f"the target cache differs from the cache in {key}")
    answers = cache.answers.with_role("train")
    targets = target.answers.with_role("train")
    if targets.question_ids != answers.question_ids or not torch.equal(
        targets.correct.cpu(), answers.correct.cpu()
    ):
        raise ValueError("the target cache holds other training questions or grades")
    return targets


def _saved(state: dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _write_bytes(data: bytes, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)
