"""Every registered implementation satisfies its protocol.

`Registry.register` is generic in a free type variable so decorating does not
erase the concrete class, which means it can no longer check conformance at the
decoration site. These module-level annotated assignments restore that check in
one place: a type checker verifies each assignment, and adding an implementation
without adding a line here is the only way to lose the guarantee.

There is nothing to run -- the value is entirely in `uv run pyright`. The test
below exists so the file is not mistaken for dead code.
"""

from caldrift.benchmarks import MMLU, Benchmark
from caldrift.metrics import CalibrationMetric, brier_score, expected_calibration_error
from caldrift.models import ModelAdapter, StubAdapter
from caldrift.prompts import FewShotCompletion, PromptProtocol

BENCHMARK_MMLU: type[Benchmark] = MMLU
MODEL_STUB: type[ModelAdapter] = StubAdapter
PROMPT_FEWSHOT: type[PromptProtocol] = FewShotCompletion
METRIC_ECE: CalibrationMetric = expected_calibration_error
METRIC_BRIER: CalibrationMetric = brier_score


def test_implementations_are_bound_to_their_protocols():
    assert BENCHMARK_MMLU is MMLU
    assert MODEL_STUB is StubAdapter
    assert PROMPT_FEWSHOT is FewShotCompletion
