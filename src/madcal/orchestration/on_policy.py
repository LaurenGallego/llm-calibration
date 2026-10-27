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
from madcal.training.objective import (
    kl_to_base,
    l_sc_norm,
    policy_gradient_terms,
    sequence_log_probs,
    smoothed_l_sc_norm,
)

CHECKPOINT = "checkpoint.pt"
RECORD = "on_policy.json"
NO_ANSWER = "no_answer"
NO_PARSED_SAMPLES = "no_parsed_samples"
SINGLE_CLUSTER = "single_cluster"
UNMATCHED = "unmatched"
SINGLE_CLUSTER_WRONG = "single_cluster_wrong"
UNMATCHED_CORRECT = "unmatched_correct"
BASELINE_DECAY = 0.9


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
    greedy_states: list[torch.Tensor] = field(default_factory=list)
    greedy_targets: list[int] = field(default_factory=list)
    greedy_lengths: list[int] = field(default_factory=list)

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
    with_greedy: bool = False,
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
    greedy_sequences: list[tuple[list[int], int]] = []
    greedy_owners: list[int] = []
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
        out.counts[SINGLE_CLUSTER_WRONG] += n_clusters == 1 and match is not None and not correct
        out.counts[UNMATCHED_CORRECT] += match is None and correct
        first_cluster = len(out.cluster_question)
        out.cluster_question += [len(out.correct)] * n_clusters
        out.reference.append(-1 if match is None else first_cluster + match)
        out.correct.append(correct)
        if with_greedy:
            full, ids = adapter.generated_ids(prompt, reference)
            greedy_sequences.append((full, len(ids)))
            greedy_owners.append(len(batches) - 1)
            out.greedy_targets += ids
            out.greedy_lengths.append(len(ids))
        for label, (_, sample) in zip(labels, parsed, strict=True):
            full, ids = adapter.generated_ids(prompt, sample)
            sequences.append((full, len(ids)))
            owners.append(len(batches) - 1)
            out.targets += ids
            out.lengths.append(len(ids))
            out.sample_cluster.append(first_cluster + label)
    inputs = adapter.head_inputs(sequences + greedy_sequences) if sequences else []
    for owner, state in zip(owners, inputs[: len(sequences)], strict=True):
        batches[owner].states.append(state)
    for owner, state in zip(greedy_owners, inputs[len(sequences) :], strict=True):
        batches[owner].greedy_states.append(state)
    return batches


@dataclass(frozen=True, slots=True)
class RolloutConfidence:
    """A batch's evaluated and trained confidences, KL per position and sample log-probabilities."""

    evaluated: torch.Tensor
    trained: torch.Tensor
    kl: torch.Tensor
    sample_log_probs: torch.Tensor


def rollout_confidence(
    arm: Arm,
    reference_arm: Arm,
    batch: Rollout,
    device: str,
    kl_grad: bool = False,
    smoothing: float | None = None,
) -> RolloutConfidence:
    """L-SC-norm of each scored question's greedy answer, and each position's KL to a reference."""
    lengths = torch.tensor(batch.lengths, device=device)
    offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), lengths.cumsum(0)])
    states = torch.cat(batch.states).to(device)
    log_probs = arm.position_log_probs(states)
    if log_probs is None:
        raise TypeError(f"{type(arm).__name__} defines no output distribution to train on-policy")
    with torch.no_grad():
        base = reference_arm.position_log_probs(states)
    if base is None:
        raise TypeError(f"{type(reference_arm).__name__} defines no output distribution")
    with torch.set_grad_enabled(kl_grad and torch.is_grad_enabled()):
        kl = kl_to_base(log_probs, base)
    del base
    sequences = sequence_log_probs(log_probs, torch.tensor(batch.targets, device=device), offsets)
    clusters = (
        sequences,
        lengths,
        torch.tensor(batch.sample_cluster, device=device),
        torch.tensor(batch.cluster_question, device=device),
        torch.tensor(batch.reference, device=device),
    )
    evaluated = l_sc_norm(*clusters)
    trained = evaluated
    if smoothing is not None:
        greedy_lengths = torch.tensor(batch.greedy_lengths, device=device)
        greedy_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), greedy_lengths.cumsum(0)]
        )
        greedy_log_probs = arm.position_log_probs(torch.cat(batch.greedy_states).to(device))
        if greedy_log_probs is None:
            raise TypeError(f"{type(arm).__name__} defines no output distribution")
        greedy = sequence_log_probs(
            greedy_log_probs, torch.tensor(batch.greedy_targets, device=device), greedy_offsets
        )
        trained = smoothed_l_sc_norm(*clusters, greedy, greedy_lengths, smoothing)
    return RolloutConfidence(evaluated, trained, kl, sequences)


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
    hidden_size = int(weight.shape[1])
    arm = build_arm(config.arm.name, head, hidden_size, config.seed, **settings)
    reference_arm = build_arm("base", head, hidden_size, config.seed)
    init_sha = None
    if config.init_checkpoint is not None:
        init_sha = file_sha256(config.init_checkpoint)
        start = torch.load(config.init_checkpoint, weights_only=True)
        arm.load_checkpoint(start)
        reference_arm = build_arm(config.arm.name, head, hidden_size, config.seed, **settings)
        reference_arm.load_checkpoint(start)
        reference_arm.requires_grad_(False)
    render = build_renderer(config.prompt, adapter)
    entails = build_judge(config.semantic)
    n_samples, max_tokens = config.semantic.n_samples, config.semantic.max_tokens
    smoothing = config.smoothing

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
            smoothing is not None,
        )

    base_played = None
    if config.init_checkpoint is not None:
        base_arm = build_arm("base", head, hidden_size, config.seed)
        adapter.install_head(base_arm, base_arm.reseed)
        base_played = play(selection, _seed(config.seed, 0))
    adapter.install_head(arm, arm.reseed)

    def evaluate(step: int, start: list[Rollout] | None) -> tuple[dict[str, Any], list[Rollout]]:
        played = play(selection, _seed(config.seed, 0))
        scored = [batch for batch in played if len(batch)]
        if not scored:
            raise RuntimeError(f"no selection question could be scored at step {step}")
        with torch.no_grad():
            outputs = [
                rollout_confidence(arm, reference_arm, batch, device, smoothing=smoothing)
                for batch in scored
            ]
        confidence = torch.cat([output.evaluated for output in outputs])
        kl = torch.cat([output.kl for output in outputs])
        correct = torch.tensor([c for batch in scored for c in batch.correct], device=device)
        start = start if start is not None else played
        base = base_played if base_played is not None else start
        played_accuracy, base_accuracy = accuracy(played), accuracy(base)
        entry: dict[str, Any] = {
            "step": step,
            "brier": float(brier_loss(confidence, correct)),
            "kl_to_base": float(kl.mean()),
            "accuracy": played_accuracy,
            "base_accuracy": base_accuracy,
            "answer_changed": _changed(played, base),
            "answer_changed_from_start": _changed(played, start),
            "n_scored": len(correct),
            "counts": dict(sum((batch.counts for batch in played), Counter())),
            "diagnostics": arm.diagnostics(sample_states(scored).to(device)),
        }
        if smoothing is not None:
            trained = torch.cat([output.trained for output in outputs])
            entry["brier_trained"] = float(brier_loss(trained, correct))
        entry["passes_guardrail"] = (
            played_accuracy is not None
            and base_accuracy is not None
            and abs(played_accuracy - base_accuracy) <= config.max_accuracy_change
            and entry["answer_changed"] <= config.max_answer_changed
        )
        print(json.dumps({"evaluation": entry}), flush=True)
        return entry, played

    first, start = evaluate(0, None)
    evaluations = [first]
    baseline = first["brier"]
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
            brier, divergence, advantage_abs = 0.0, 0.0, 0.0
            for batch in batches:
                if not len(batch):
                    continue
                output = rollout_confidence(
                    arm, reference_arm, batch, device, config.kl_to_base > 0, smoothing
                )
                correct = torch.tensor(batch.correct, device=device)
                terms = (output.trained - correct.to(output.trained.dtype)) ** 2
                batch_brier = terms.sum() / n_scored
                batch_kl = output.kl.sum() / n_positions
                loss = batch_brier + config.kl_to_base * batch_kl
                if config.policy_gradient:
                    advantage = terms.detach() - baseline
                    question = torch.tensor(batch.cluster_question, device=device)[
                        torch.tensor(batch.sample_cluster, device=device)
                    ]
                    score = policy_gradient_terms(output.sample_log_probs, question, advantage)
                    loss = loss + score.sum() / n_scored
                    advantage_abs += float(advantage.abs().sum()) / n_scored
                loss.backward()
                brier += batch_brier.item()
                divergence += batch_kl.item()
            optimizer.step()
            entry["brier"] = brier
            entry["kl_to_base"] = divergence
            entry["loss"] = brier + config.kl_to_base * divergence
            if config.policy_gradient:
                entry["baseline"] = baseline
                entry["advantage_abs"] = advantage_abs
                baseline = BASELINE_DECAY * baseline + (1.0 - BASELINE_DECAY) * brier
        entry["seconds"] = time.monotonic() - started
        steps.append(entry)
        print(json.dumps({"train": entry}), flush=True)
        if step % config.eval_every == 0 or step == config.steps:
            evaluation, _ = evaluate(step, start)
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
        "init_checkpoint_sha256": init_sha,
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


def _changed(played: Sequence[Rollout], reference: Sequence[Rollout]) -> float:
    greedy = [text for batch in played for text in batch.greedy]
    other = [text for batch in reference for text in batch.greedy]
    return sum(a != b for a, b in zip(greedy, other, strict=True)) / len(greedy)


def _seed(seed: int, step: int) -> int:
    derived = round_seed(seed, step)
    if derived is None:
        raise ValueError("a seed is required")
    return derived
