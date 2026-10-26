"""On-policy training: the live head answers and samples, and is fitted on L-SC-norm."""

import json
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from madcal.benchmarks import Benchmark, Question
from madcal.config import OnPolicyConfig
from madcal.debate import Message, Renderer, Role, round_seed
from madcal.models import CalibratedTransformersAdapter
from madcal.models.calibrated import file_sha256
from madcal.orchestration.build import build_adapter, build_benchmark, build_renderer
from madcal.orchestration.debate_inputs import debate_question
from madcal.orchestration.fit import _saved, _write_bytes
from madcal.orchestration.harvest import harvest_items
from madcal.orchestration.provenance import code_sha
from madcal.orchestration.semantic import SAMPLING_TEMPERATURE, build_judge
from madcal.signals.semantic import Entails, cluster_answers, first_equivalent
from madcal.storage.states import AnswerStates
from madcal.training import Arm, OutputHead, brier_loss, build_arm
from madcal.training.objective import kl_to_base, l_sc_norm, sequence_log_probs

CHECKPOINT = "checkpoint.pt"
RECORD = "on_policy.json"
NO_ANSWER = "no_answer"
NO_PARSED_SAMPLES = "no_parsed_samples"
SINGLE_CLUSTER = "single_cluster"
UNMATCHED = "unmatched"


@dataclass(slots=True)
class Rollout:
    """Greedy answers and clustered samples of a batch of questions, in the loss's index layout."""

    greedy: list[str] = field(default_factory=list)
    graded: list[bool | None] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    correct: list[bool] = field(default_factory=list)
    states: list[torch.Tensor] = field(default_factory=list)
    targets: list[int] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    sample_cluster: list[int] = field(default_factory=list)
    cluster_question: list[int] = field(default_factory=list)
    reference: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.correct)


@dataclass(frozen=True, slots=True)
class OnPolicyOutcome:
    run_hash: str
    root: Path
    best_step: int
    selection_brier: float


def rollout(
    questions: Sequence[Question],
    benchmark: Benchmark,
    adapter: CalibratedTransformersAdapter,
    render: Renderer,
    entails: Entails | None,
    n_samples: int,
    max_tokens: int,
    seed: int,
    micro_batch: int,
) -> list[Rollout]:
    """Answer greedily, sample, grade and cluster with the installed head, in micro-batches."""
    prompts = [
        render([Message(Role.USER, debate_question(question, benchmark).text)])
        for question in questions
    ]
    references = adapter.generate(prompts, 1, 0.0, max_tokens, seed=round_seed(seed, 0))
    samples = adapter.generate(
        prompts, n_samples, SAMPLING_TEMPERATURE, max_tokens, seed=round_seed(seed, 1)
    )
    if len(references) != len(questions) or len(samples) != len(questions):
        raise RuntimeError("adapter returned a different number of results than prompts")

    batches: list[Rollout] = []
    sequences: list[tuple[list[int], int]] = []
    owners: list[int] = []
    for index, (question, prompt, (reference,), sampled) in enumerate(
        zip(questions, prompts, references, samples, strict=True)
    ):
        if index % micro_batch == 0:
            batches.append(Rollout())
        out = batches[-1]
        if len(sampled) != n_samples:
            raise RuntimeError(f"adapter returned the wrong number of samples for {question.id!r}")
        n_choices = len(question.choices) if question.choices is not None else None
        out.greedy.append(reference.text)
        predicted = benchmark.extract_answer(reference.text, n_choices)
        if predicted is None:
            out.graded.append(None)
            out.counts[NO_ANSWER] += 1
            continue
        correct = benchmark.grade(question, predicted)
        out.graded.append(correct)
        parsed = [
            (answer, sample)
            for sample in sampled
            if (answer := benchmark.extract_answer(sample.text, n_choices))
        ]
        if not parsed:
            out.counts[NO_PARSED_SAMPLES] += 1
            continue
        labels = cluster_answers(question.body, [answer for answer, _ in parsed], entails)
        n_clusters = max(labels) + 1
        representatives = [parsed[labels.index(c)][0] for c in range(n_clusters)]
        match = first_equivalent(question.body, predicted, representatives, entails)
        out.counts[SINGLE_CLUSTER] += n_clusters == 1
        out.counts[UNMATCHED] += match is None
        first_cluster = len(out.cluster_question)
        out.cluster_question += [len(out.correct)] * n_clusters
        out.reference.append(-1 if match is None else first_cluster + match)
        out.correct.append(correct)
        for label, (_, sample) in zip(labels, parsed, strict=True):
            full, ids = adapter.generated_ids(prompt, sample)
            sequences.append((full, len(ids)))
            owners.append(len(batches) - 1)
            out.targets += ids
            out.lengths.append(len(ids))
            out.sample_cluster.append(first_cluster + label)
    for owner, state in zip(
        owners, adapter.head_inputs(sequences) if sequences else [], strict=True
    ):
        batches[owner].states.append(state)
    return batches


def rollout_confidence(
    arm: Arm, batch: Rollout, device: str, kl_grad: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """L-SC-norm of each scored question's greedy answer, and each position's KL from the base."""
    lengths = torch.tensor(batch.lengths, device=device)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), lengths.cumsum(0)])
    states = torch.cat(batch.states).to(device)
    log_probs = arm.position_log_probs(states)
    if log_probs is None:
        raise TypeError(f"{type(arm).__name__} defines no output distribution to train on-policy")
    with torch.no_grad():
        base = arm.head.logits(states).log_softmax(dim=-1)
    with torch.set_grad_enabled(kl_grad and torch.is_grad_enabled()):
        kl = kl_to_base(log_probs, base)
    del base
    sequences = sequence_log_probs(log_probs, torch.tensor(batch.targets, device=device), offsets)
    confidence = l_sc_norm(
        sequences,
        lengths,
        torch.tensor(batch.sample_cluster, device=device),
        torch.tensor(batch.cluster_question, device=device),
        torch.tensor(batch.reference, device=device),
    )
    return confidence, kl


def accuracy(batches: Sequence[Rollout]) -> float | None:
    """Share of parsed greedy answers that are correct."""
    graded = [value for batch in batches for value in batch.graded if value is not None]
    return sum(graded) / len(graded) if graded else None


def sample_states(batches: Sequence[Rollout]) -> AnswerStates:
    """The rollouts' sampled sequences as answer states, one per sample, for arm diagnostics."""
    lengths = torch.tensor([n for batch in batches for n in batch.lengths])
    return AnswerStates(
        question_ids=tuple(f"sample/{index}" for index in range(len(lengths))),
        roles=("train",) * len(lengths),
        correct=torch.zeros(len(lengths), dtype=torch.bool),
        states=torch.cat([state for batch in batches for state in batch.states]),
        offsets=torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)]),
        targets=torch.tensor([target for batch in batches for target in batch.targets]),
    )


def on_policy(config: OnPolicyConfig) -> OnPolicyOutcome:
    """Train the configured arm on-policy and write `checkpoint.pt` and `on_policy.json`."""
    adapter = build_adapter(config.model)
    if not isinstance(adapter, CalibratedTransformersAdapter) or adapter.revision is None:
        raise ValueError("on-policy training needs a calibrated transformers model at a revision")
    code = code_sha()
    run_hash = config.config_hash(adapter.revision, code)
    root = Path(config.output_root) / run_hash
    if (root / RECORD).exists():
        raise FileExistsError(f"{root} already holds this run; results are never overwritten")

    benchmark = build_benchmark(config.train)
    items, n_excluded = harvest_items(
        benchmark, config.train, build_benchmark(config.evaluation), config.evaluation
    )
    pool = [item.question for item in items if item.role == "train"]
    if len(pool) <= config.n_selection:
        raise ValueError(f"{len(pool)} training questions leave none after the selection subset")
    selection, fit_pool = pool[: config.n_selection], pool[config.n_selection :]

    torch.manual_seed(config.seed)
    device = adapter.model_device
    weight, bias = adapter.output_head()
    head = OutputHead(weight, bias, device)
    settings = config.arm.model_dump(exclude={"name"})
    arm = build_arm(config.arm.name, head, int(weight.shape[1]), config.seed, **settings)
    adapter.install_head(arm, arm.reseed)
    render = build_renderer(config.prompt, adapter)
    entails = build_judge(config.semantic)
    n_samples, max_tokens = config.semantic.n_samples, config.semantic.max_tokens

    def play(questions: Sequence[Question], seed: int) -> list[Rollout]:
        return rollout(
            questions,
            benchmark,
            adapter,
            render,
            entails,
            n_samples,
            max_tokens,
            seed,
            config.micro_batch_questions,
        )

    def evaluate(step: int, base: list[Rollout] | None) -> tuple[dict[str, Any], list[Rollout]]:
        played = play(selection, _seed(config.seed, 0))
        scored = [batch for batch in played if len(batch)]
        if not scored:
            raise RuntimeError(f"no selection question could be scored at step {step}")
        with torch.no_grad():
            pairs = [rollout_confidence(arm, batch, device) for batch in scored]
        confidence = torch.cat([pair[0] for pair in pairs])
        kl = torch.cat([pair[1] for pair in pairs])
        correct = torch.tensor([c for batch in scored for c in batch.correct], device=device)
        reference = base if base is not None else played
        greedy = [text for batch in played for text in batch.greedy]
        base_greedy = [text for batch in reference for text in batch.greedy]
        changed = sum(a != b for a, b in zip(greedy, base_greedy, strict=True))
        played_accuracy, base_accuracy = accuracy(played), accuracy(reference)
        entry: dict[str, Any] = {
            "step": step,
            "brier": float(brier_loss(confidence, correct)),
            "kl_to_base": float(kl.mean()),
            "accuracy": played_accuracy,
            "answer_changed": changed / len(selection),
            "n_scored": len(correct),
            "counts": dict(sum((batch.counts for batch in played), Counter())),
            "diagnostics": arm.diagnostics(sample_states(scored).to(device)),
        }
        entry["passes_guardrail"] = (
            played_accuracy is not None
            and base_accuracy is not None
            and abs(played_accuracy - base_accuracy) <= config.max_accuracy_change
            and entry["answer_changed"] <= config.max_answer_changed
        )
        print(json.dumps({"evaluation": entry}), flush=True)
        return entry, played

    first, base = evaluate(0, None)
    evaluations = [first]
    best = (0, first["brier"], arm.checkpoint())
    parameters = [p for p in arm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(config.seed)
    steps: list[dict[str, Any]] = []
    for step in range(1, config.steps + 1):
        started = time.monotonic()
        picks = torch.randperm(len(fit_pool), generator=generator)[: config.batch_questions]
        batches = play([fit_pool[int(i)] for i in picks], _seed(config.seed, step))
        entry: dict[str, Any] = {"step": step, "accuracy": accuracy(batches)}
        entry["counts"] = dict(sum((batch.counts for batch in batches), Counter()))
        n_scored = sum(len(batch) for batch in batches)
        n_positions = sum(len(batch.targets) for batch in batches)
        if n_scored:
            optimizer.zero_grad()
            brier, divergence = 0.0, 0.0
            for batch in batches:
                if not len(batch):
                    continue
                confidence, kl = rollout_confidence(arm, batch, device, config.kl_to_base > 0)
                correct = torch.tensor(batch.correct, device=device)
                batch_brier = brier_loss(confidence, correct) * (len(batch) / n_scored)
                batch_kl = kl.sum() / n_positions
                (batch_brier + config.kl_to_base * batch_kl).backward()
                brier += batch_brier.item()
                divergence += batch_kl.item()
            optimizer.step()
            entry["brier"] = brier
            entry["kl_to_base"] = divergence
            entry["loss"] = brier + config.kl_to_base * divergence
        entry["seconds"] = time.monotonic() - started
        steps.append(entry)
        print(json.dumps({"train": entry}), flush=True)
        if step % config.eval_every == 0 or step == config.steps:
            evaluation, _ = evaluate(step, base)
            evaluations.append(evaluation)
            if evaluation["passes_guardrail"] and evaluation["brier"] < best[1]:
                best = (step, evaluation["brier"], arm.checkpoint())
    arm.load_checkpoint(best[2])

    root.mkdir(parents=True, exist_ok=True)
    _write_bytes(_saved(arm.checkpoint()), root / CHECKPOINT)
    record: dict[str, Any] = {
        "run_hash": run_hash,
        "config": config.model_dump(mode="json"),
        "arm": config.arm.name,
        "seed": config.seed,
        "model_id": adapter.model_id,
        "model_checkpoint_sha": adapter.revision,
        "benchmark": config.train.name,
        "code_sha": code,
        "device": device,
        "n_fit": len(fit_pool),
        "n_selection": len(selection),
        "n_excluded": n_excluded,
        "best_step": best[0],
        "selection_brier": best[1],
        "evaluations": evaluations,
        "steps": steps,
        "peak_memory_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
        "checkpoint_sha256": file_sha256(root / CHECKPOINT),
        "completed_at": datetime.now(UTC).isoformat(),
    }
    _write_bytes(json.dumps(record, indent=2, sort_keys=True).encode(), root / RECORD)
    return OnPolicyOutcome(run_hash, root, best[0], best[1])


def _seed(seed: int, step: int) -> int:
    derived = round_seed(seed, step)
    if derived is None:
        raise ValueError("a seed is required")
    return derived
