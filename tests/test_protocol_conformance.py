"""Every registered implementation satisfies its protocol.

`Registry.register` is generic in a free type variable so decorating does not
erase the concrete class, which means it can no longer check conformance at the
decoration site. These module-level annotated assignments restore that check in
one place: a type checker verifies each assignment, and adding an implementation
without adding a line here is the only way to lose the guarantee.

There is nothing to run -- the value is entirely in `uv run pyright`. The test
below exists so the file is not mistaken for dead code.
"""

from madcal.benchmarks import MMLU, Benchmark
from madcal.metrics import CalibrationMetric, brier_score, expected_calibration_error
from madcal.models import ModelAdapter, StubAdapter, TransformersAdapter
from madcal.prompts import FewShotCompletion, PromptProtocol
from madcal.signals import Confidence, ConfidenceEntropy, Signal

BENCHMARK_MMLU: type[Benchmark] = MMLU
MODEL_STUB: type[ModelAdapter] = StubAdapter
MODEL_TRANSFORMERS: type[ModelAdapter] = TransformersAdapter
PROMPT_FEWSHOT: type[PromptProtocol] = FewShotCompletion
SIGNAL_CONFIDENCE: type[Signal] = Confidence
SIGNAL_ENTROPY: type[Signal] = ConfidenceEntropy
METRIC_ECE: CalibrationMetric = expected_calibration_error
METRIC_BRIER: CalibrationMetric = brier_score


def test_implementations_are_bound_to_their_protocols():
    assert BENCHMARK_MMLU is MMLU
    assert MODEL_STUB is StubAdapter
    assert MODEL_TRANSFORMERS is TransformersAdapter
    assert PROMPT_FEWSHOT is FewShotCompletion
    assert SIGNAL_CONFIDENCE is Confidence
    assert SIGNAL_ENTROPY is ConfidenceEntropy
