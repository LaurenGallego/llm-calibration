"""The experiment loop: one config, one condition, rows on disk."""

from collections.abc import Sequence
from dataclasses import dataclass

from madcal.benchmarks import Benchmark, Question
from madcal.config import RunConfig
from madcal.debate import DebateConfig, Renderer, run_debate
from madcal.models import ModelAdapter
from madcal.orchestration.build import build_adapter, build_benchmark, build_renderer
from madcal.orchestration.debate_inputs import debate_question
from madcal.orchestration.provenance import code_sha, new_run_id
from madcal.orchestration.rows import transcript_rows
from madcal.storage import QuestionRow, ResultStore, RunManifest, RunStatus, SignalRow


@dataclass(frozen=True, slots=True)
class RunOutcome:
    run_id: str
    config_hash: str
    n_written: int
    n_skipped: int
    status: RunStatus


def run(config: RunConfig, store: ResultStore | None = None) -> RunOutcome:
    """Execute one condition, writing complete debates in chunks as they finish."""
    store = store or ResultStore(config.output_root)
    adapter = build_adapter(config.model)
    if adapter.revision is None:
        raise ValueError(
            f"adapter for {adapter.model_id!r} resolved no checkpoint revision; "
            f"rows that cannot be traced to a checkpoint are not usable"
        )

    config_hash = config.config_hash(adapter.revision)
    benchmark = build_benchmark(config.benchmark)
    renderer = build_renderer(config.prompt, adapter)
    debate_config = config.debate.resolve()

    questions = list(benchmark.load())
    already = store.completed_questions(config_hash)
    pending = [question for question in questions if question.id not in already]

    run_id = new_run_id()
    manifest = _manifest(config, adapter, debate_config, len(questions), run_id, config_hash)
    store.begin_run(manifest)

    written = 0
    try:
        for seq, start in enumerate(range(0, len(pending), config.chunk_size)):
            batch = pending[start : start + config.chunk_size]
            questions_rows, signal_rows = _process_chunk(
                batch, config, debate_config, adapter, benchmark, renderer
            )
            store.write_chunk(manifest, questions_rows, signal_rows, seq=seq)
            written += len(batch)
    except Exception:
        store.finish_run(run_id, RunStatus.FAILED, written)
        raise

    store.finish_run(run_id, RunStatus.COMPLETE, written)
    return RunOutcome(
        run_id=run_id,
        config_hash=config_hash,
        n_written=written,
        n_skipped=len(questions) - len(pending),
        status=RunStatus.COMPLETE,
    )


def _process_chunk(
    batch: Sequence[Question],
    config: RunConfig,
    debate_config: DebateConfig,
    adapter: ModelAdapter,
    benchmark: Benchmark,
    renderer: Renderer,
) -> tuple[list[QuestionRow], list[SignalRow]]:
    transcripts = run_debate(
        [debate_question(question, benchmark) for question in batch],
        debate_config,
        adapter,
        renderer,
        benchmark.extract_answer,
        seed=config.seed,
    )
    question_rows: list[QuestionRow] = []
    signal_rows: list[SignalRow] = []
    for transcript, question in zip(transcripts, batch, strict=True):
        rows, signals = transcript_rows(transcript, question, benchmark)
        question_rows.extend(rows)
        signal_rows.extend(signals)
    return question_rows, signal_rows


def _manifest(
    config: RunConfig,
    adapter: ModelAdapter,
    debate_config: DebateConfig,
    n_planned: int,
    run_id: str,
    config_hash: str,
) -> RunManifest:
    if adapter.revision is None:
        raise ValueError("a manifest needs a resolved checkpoint revision")
    return RunManifest(
        run_id=run_id,
        config_hash=config_hash,
        model_id=adapter.model_id,
        variant=config.model.variant,
        benchmark=config.benchmark.name,
        protocol=config.debate.protocol,
        n_agents=debate_config.n_agents,
        n_rounds=debate_config.n_rounds,
        confidence_mode=debate_config.confidence_mode,
        aggregation=debate_config.aggregation,
        temperature=debate_config.temperature,
        max_tokens=debate_config.max_tokens,
        prompt_protocol=config.prompt.name,
        checkpoint_sha=adapter.revision,
        code_sha=code_sha(),
        dtype=adapter.dtype,
        batch_size=adapter.batch_size,
        backend=adapter.name,
        backend_version=adapter.backend_version,
        n_questions_planned=n_planned,
        benchmark_revision=config.benchmark.revision,
        seed=config.seed,
    )
