"""Signals -- the unlabeled track. Computed from model output, never from a label.

Importing this package registers every signal, so `signal_registry` is fully populated
after `import madcal.signals`. Each new implementation must be imported here or its
decorator never runs.

This package may import `madcal.models` and `madcal.registry`, and nothing else from
the project. See the label contract in `base.py` and the `import-linter` contract that
enforces it.
"""

from madcal.signals.base import (
    Evidence,
    Requirements,
    Signal,
    SignalNull,
    SignalValue,
    applicable,
    register_signal,
    signal_registry,
    sole_generation,
)
from madcal.signals.confidence import Confidence
from madcal.signals.entropy import (
    ConfidenceEntropy,
    residual_distribution,
    shannon_entropy,
)
from madcal.signals.likelihood import LengthNormalisedLikelihood
from madcal.signals.semantic import (
    MEASURES,
    SampledAnswer,
    SemanticConfidence,
    cluster_answers,
    cluster_distributions,
    semantic_confidence,
)
from madcal.signals.verbalized import VerbalizedConfidence, parse_verbalized_confidence

__all__ = [
    "MEASURES",
    "Confidence",
    "ConfidenceEntropy",
    "Evidence",
    "LengthNormalisedLikelihood",
    "Requirements",
    "SampledAnswer",
    "SemanticConfidence",
    "Signal",
    "SignalNull",
    "SignalValue",
    "VerbalizedConfidence",
    "applicable",
    "cluster_answers",
    "cluster_distributions",
    "parse_verbalized_confidence",
    "register_signal",
    "residual_distribution",
    "semantic_confidence",
    "shannon_entropy",
    "signal_registry",
    "sole_generation",
]
