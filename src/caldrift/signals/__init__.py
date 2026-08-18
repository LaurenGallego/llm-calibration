"""Signals -- the unlabeled track. Computed from model output, never from a label.

Importing this package registers every signal, so `signal_registry` is fully populated
after `import caldrift.signals`. Each new implementation must be imported here or its
decorator never runs.

This package may import `caldrift.models` and `caldrift.registry`, and nothing else from
the project. See the label contract in `base.py` and the `import-linter` contract that
enforces it.
"""

from caldrift.signals.base import (
    Evidence,
    Requirements,
    Signal,
    SignalNull,
    SignalValue,
    applicable,
    register_signal,
    signal_registry,
)
from caldrift.signals.confidence import Confidence
from caldrift.signals.entropy import (
    ConfidenceEntropy,
    residual_distribution,
    shannon_entropy,
)

__all__ = [
    "Confidence",
    "ConfidenceEntropy",
    "Evidence",
    "Requirements",
    "Signal",
    "SignalNull",
    "SignalValue",
    "applicable",
    "register_signal",
    "residual_distribution",
    "shannon_entropy",
    "signal_registry",
]
