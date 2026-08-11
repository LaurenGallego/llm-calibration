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

__all__ = [
    "LETTERS",
    "Benchmark",
    "Question",
    "TaskFormat",
    "answer_letter",
    "benchmark_registry",
    "register_benchmark",
]
