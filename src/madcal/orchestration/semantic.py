"""Semantic sampling: per question a greedy reference answer and sampled answers, clustered."""

from collections.abc import Sequence

from madcal.benchmarks import Benchmark, Question
from madcal.config import RunConfig, SemanticSettings
from madcal.debate import Message, Renderer, Role, agent_id, round_seed
from madcal.models import Generation, ModelAdapter, NLIJudge
from madcal.orchestration.debate_inputs import debate_question
from madcal.signals import (
    MEASURES,
    Entails,
    SampledAnswer,
    SignalNull,
    SignalValue,
    semantic_confidence,
)
from madcal.storage import AnswerNull, Level, QuestionRow, SignalKind, SignalRow

PROTOCOL = "semantic"
AGGREGATION = "greedy_reference"
CONFIDENCE_MODE = "none"
SAMPLING_TEMPERATURE = 1.0
DIAGNOSTICS = ("n_clusters", "matched", "n_parsed", "reference_truncated")


def signal_name(measure: str) -> str:
    """Return the stored signal name of a semantic measure or diagnostic."""
    return f"semantic_{measure}"


def build_judge(settings: SemanticSettings) -> NLIJudge | None:
    """Construct the NLI judge the settings name, or None to cluster by exact match only."""
    if settings.nli_model is None:
        return None
    return NLIJudge(settings.nli_model, settings.nli_revision)


def semantic_chunk(
    batch: Sequence[Question],
    config: RunConfig,
    settings: SemanticSettings,
    adapter: ModelAdapter,
    benchmark: Benchmark,
    render: Renderer,
    entails: Entails | None,
) -> tuple[list[QuestionRow], list[SignalRow]]:
    """Generate, grade and cluster one chunk of questions."""
    prompts = [
        render([Message(Role.USER, debate_question(question, benchmark).text)])
        for question in batch
    ]
    references = adapter.generate(
        prompts, 1, 0.0, settings.max_tokens, seed=round_seed(config.seed, 0)
    )
    samples = adapter.generate(
        prompts,
        settings.n_samples,
        SAMPLING_TEMPERATURE,
        settings.max_tokens,
        seed=round_seed(config.seed, 1),
    )
    if len(references) != len(batch) or len(samples) != len(batch):
        raise RuntimeError("adapter returned a different number of results than prompts")
    question_rows: list[QuestionRow] = []
    signal_rows: list[SignalRow] = []
    for question, reference, sampled in zip(batch, references, samples, strict=True):
        if len(reference) != 1 or len(sampled) != settings.n_samples:
            raise RuntimeError(f"adapter returned the wrong number of samples for {question.id!r}")
        rows, values = semantic_rows(question, benchmark, reference[0], sampled, entails)
        question_rows.extend(rows)
        signal_rows.extend(values)
    return question_rows, signal_rows


def semantic_rows(
    question: Question,
    benchmark: Benchmark,
    reference: Generation,
    samples: Sequence[Generation],
    entails: Entails | None,
) -> tuple[list[QuestionRow], list[SignalRow]]:
    """Grade the reference and the samples, and compute every measure for the reference."""
    n_choices = len(question.choices) if question.choices is not None else None
    subject = question.metadata.get("subject")
    answers = [benchmark.extract_answer(sample.text, n_choices) for sample in samples]
    rows = [
        QuestionRow(
            question_id=question.id,
            level=Level.AGENT,
            agent_id=agent_id(index),
            round=0,
            subject=subject,
            task_format=str(question.task_format),
            predicted=answer,
            correct=None if answer is None else benchmark.grade(question, answer),
            null_reason=None if answer is not None else AnswerNull.PARSE_FAILED,
            finish_reason=sample.finish_reason,
        )
        for index, (answer, sample) in enumerate(zip(answers, samples, strict=True))
    ]
    predicted = benchmark.extract_answer(reference.text, n_choices)
    rows.append(
        QuestionRow(
            question_id=question.id,
            level=Level.SYSTEM,
            agent_id=None,
            round=None,
            subject=subject,
            task_format=str(question.task_format),
            predicted=predicted,
            correct=None if predicted is None else benchmark.grade(question, predicted),
            null_reason=None if predicted is not None else AnswerNull.NO_VALID_ANSWERS,
            finish_reason=None,
        )
    )
    values = _measures(question.body, predicted, answers, samples, entails)
    values["reference_truncated"] = SignalValue(float(reference.finish_reason == "length"))
    signals = [
        SignalRow(
            question_id=question.id,
            level=Level.SYSTEM,
            agent_id=None,
            round=None,
            signal=signal_name(name),
            kind=SignalKind.DIAGNOSTIC if name in DIAGNOSTICS else SignalKind.SIGNAL,
            value=value.value,
            reason=None if value.reason is None else str(value.reason),
        )
        for name, value in values.items()
    ]
    return rows, signals


def _measures(
    question_text: str,
    reference: str | None,
    answers: Sequence[str | None],
    samples: Sequence[Generation],
    entails: Entails | None,
) -> dict[str, SignalValue]:
    parsed = [(answer, sample) for answer, sample in zip(answers, samples, strict=True) if answer]
    values: dict[str, SignalValue] = {"n_parsed": SignalValue(float(len(parsed)))}
    if any(sample.token_logprobs is None or sample.tokens is None for _, sample in parsed):
        return values | _nulls(SignalNull.NO_LOGPROBS)
    if reference is None or not parsed:
        return values | _nulls(SignalNull.PARSE_FAILED)
    sampled = [
        SampledAnswer(answer, sample.token_logprobs or (), sample.tokens or ())
        for answer, sample in parsed
    ]
    result = semantic_confidence(question_text, reference, sampled, entails)
    values |= {measure: SignalValue(result.confidences[measure]) for measure in MEASURES}
    values["n_clusters"] = SignalValue(float(result.n_clusters))
    values["matched"] = SignalValue(float(result.matched))
    return values


def _nulls(reason: SignalNull) -> dict[str, SignalValue]:
    return {name: SignalValue(None, reason) for name in (*MEASURES, "n_clusters", "matched")}
