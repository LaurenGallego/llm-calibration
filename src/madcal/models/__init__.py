"""Model adapters -- the only package that runs a neural network.

Importing this package registers every adapter, so `model_adapter_registry` is fully
populated after `import madcal.models`. Each new implementation must be imported
here or its decorator never runs.
"""

from madcal.models.base import (
    PROBABILITY_TOLERANCE,
    ChoiceScores,
    Generation,
    ModelAdapter,
    model_adapter_registry,
    register_model_adapter,
)
from madcal.models.stub import StubAdapter
from madcal.models.transformers import TransformersAdapter

__all__ = [
    "PROBABILITY_TOLERANCE",
    "ChoiceScores",
    "Generation",
    "ModelAdapter",
    "StubAdapter",
    "TransformersAdapter",
    "model_adapter_registry",
    "register_model_adapter",
]
