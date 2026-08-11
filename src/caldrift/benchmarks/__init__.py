"""Benchmarks -- evaluation sets, answer extraction, and grading.

Importing this package registers every benchmark, so `benchmark_registry` is
fully populated after `import caldrift.benchmarks`. Each new implementation must
be imported here or its decorator never runs.
"""

from caldrift.benchmarks.base import (
    LETTERS,
    Benchmark,
    Question,
    TaskFormat,
    answer_letter,
    benchmark_registry,
    register_benchmark,
)
from caldrift.benchmarks.mmlu import MMLU

__all__ = [
    "LETTERS",
    "MMLU",
    "Benchmark",
    "Question",
    "TaskFormat",
    "answer_letter",
    "benchmark_registry",
    "register_benchmark",
]
