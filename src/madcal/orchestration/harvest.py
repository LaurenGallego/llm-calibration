"""Harvest: greedy answers, their grades, and the output head's inputs at the answer positions."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from madcal.benchmarks import Benchmark, Question, normalize_answer
from madcal.benchmarks.base import sample_questions
from madcal.config import BenchmarkConfig, HarvestConfig
from madcal.debate import Message, Renderer, Role
from madcal.models import TransformersAdapter
from madcal.orchestration.build import build_adapter, build_benchmark, build_renderer
from madcal.orchestration.debate_inputs import debate_question
from madcal.orchestration.provenance import code_sha, new_run_id
from madcal.storage.states import AnswerStates, NullAnswer, StateChunk, StateStore

PARSE_FAILED = "parse_failed"


@dataclass(frozen=True, slots=True)
class HarvestItem:
    question: Question
    role: str
    benchmark: Benchmark


@dataclass(frozen=True, slots=True)
class HarvestOutcome:
    config_hash: str
    root: Path
    n_written: int
    n_skipped: int


def harvest_items(
    train: Benchmark,
    train_config: BenchmarkConfig,
    evaluation: Benchmark,
    evaluation_config: BenchmarkConfig,
) -> tuple[list[HarvestItem], int]:
    """Return the evaluation questions and the training questions that share no text with them."""
    held_out = list(
        evaluation.load(limit=evaluation_config.n_questions, seed=evaluation_config.sample_seed)
    )
    held_ids = {question.id for question in held_out}
    held_bodies = {normalize_answer(question.body) for question in held_out}
    pool = list(train.load())
    kept = [
        question
        for question in pool
        if question.id not in held_ids and normalize_answer(question.body) not in held_bodies
    ]
    chosen = sample_questions(kept, train_config.n_questions, train_config.sample_seed)
    items = [HarvestItem(question, "train", train) for question in chosen]
    items += [HarvestItem(question, "evaluation", evaluation) for question in held_out]
    return items, len(pool) - len(kept)


def harvest_chunk(
    items: Sequence[HarvestItem],
    adapter: TransformersAdapter,
    render: Renderer,
    max_tokens: int,
) -> StateChunk:
    """Answer, grade and locate the answer positions of one chunk of questions."""
    prompts = [
        render([Message(Role.USER, debate_question(item.question, item.benchmark).text)])
        for item in items
    ]
    references = adapter.generate(prompts, 1, 0.0, max_tokens)
    if len(references) != len(items) or any(len(r) != 1 for r in references):
        raise RuntimeError("adapter returned a different number of answers than prompts")

    graded: list[tuple[HarvestItem, bool]] = []
    sequences: list[tuple[list[int], int]] = []
    targets: list[list[int]] = []
    options: list[list[int]] = []
    chosen: list[int] = []
    nulls: list[NullAnswer] = []
    for item, prompt, (reference,) in zip(items, prompts, references, strict=True):
        question = item.question
        n_choices = len(question.choices) if question.choices is not None else None
        predicted = item.benchmark.extract_answer(reference.text, n_choices)
        if predicted is None:
            nulls.append(NullAnswer(question.id, item.role, PARSE_FAILED))
            continue
        if n_choices is None:
            full, ids = adapter.generated_ids(prompt, reference)
            sequences.append((full, len(ids)))
            targets.append(ids)
        else:
            slot = item.benchmark.answer_slot(reference.text, n_choices)
            if slot is None:
                nulls.append(NullAnswer(question.id, item.role, PARSE_FAILED))
                continue
            context, lead, labels = slot
            located = [adapter.continuation_ids(prompt + context, lead + label) for label in labels]
            if any(len(continuation) != 1 for _, continuation in located):
                raise ValueError(f"an option letter of {question.id!r} is not a single token")
            index = labels.index(predicted)
            sequences.append((located[index][0], 1))
            options.append([continuation[0] for _, continuation in located])
            chosen.append(index)
            targets.append([options[-1][index]])
        graded.append((item, item.benchmark.grade(question, predicted)))

    if not graded:
        return StateChunk(None, tuple(nulls))
    states = adapter.head_inputs(sequences)
    lengths = torch.tensor([len(t) for t in targets])
    answers = AnswerStates(
        question_ids=tuple(item.question.id for item, _ in graded),
        roles=tuple(item.role for item, _ in graded),
        correct=torch.tensor([correct for _, correct in graded]),
        states=torch.cat(states),
        offsets=torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)]),
        targets=torch.tensor([token for t in targets for token in t]),
        options=torch.tensor(options) if options else None,
        chosen=torch.tensor(chosen) if options else None,
    )
    return StateChunk(answers, tuple(nulls))


def harvest(config: HarvestConfig) -> HarvestOutcome:
    """Fill the cache this config names, resuming any questions already stored."""
    adapter = build_adapter(config.model)
    if not isinstance(adapter, TransformersAdapter) or adapter.revision is None:
        raise ValueError("harvesting needs a transformers model at a resolved revision")
    code = code_sha()
    config_hash = config.config_hash(adapter.revision, code)
    store = StateStore(Path(config.output_root) / config_hash)
    items, n_excluded = harvest_items(
        build_benchmark(config.train),
        config.train,
        build_benchmark(config.evaluation),
        config.evaluation,
    )
    weight, bias = adapter.output_head()
    run_id = new_run_id()
    store.begin(
        {
            "config_hash": config_hash,
            "config": config.model_dump(mode="json"),
            "model_id": adapter.model_id,
            "checkpoint_sha": adapter.revision,
            "code_sha": code,
            "dtype": adapter.dtype,
            "backend": adapter.name,
            "backend_version": adapter.backend_version,
            "benchmark": config.train.name,
            "hidden_size": int(weight.shape[1]),
            "n_train": sum(item.role == "train" for item in items),
            "n_evaluation": sum(item.role == "evaluation" for item in items),
            "n_excluded": n_excluded,
            "created_at": datetime.now(UTC).isoformat(),
        },
        run_id,
        weight,
        bias,
    )
    stored = store.stored_ids()
    pending = [item for item in items if item.question.id not in stored]
    render = build_renderer(config.prompt, adapter)
    for seq, start in enumerate(range(0, len(pending), config.chunk_size)):
        batch = pending[start : start + config.chunk_size]
        store.write_chunk(run_id, seq, harvest_chunk(batch, adapter, render, config.max_tokens))
    store.finish({})
    return HarvestOutcome(config_hash, store.root, len(pending), len(items) - len(pending))
