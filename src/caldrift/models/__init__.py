"""Model adapters -- the only package that runs a neural network.

Importing this package registers every adapter, so `model_registry` is fully
populated after `import caldrift.models`. Each new implementation must be imported
here or its decorator never runs.
"""

from caldrift.models.base import (
    PROBABILITY_TOLERANCE,
    ChoiceScores,
    Generation,
    ModelAdapter,
    model_registry,
    register_model,
)
from caldrift.models.stub import StubAdapter

__all__ = [
    "PROBABILITY_TOLERANCE",
    "ChoiceScores",
    "Generation",
    "ModelAdapter",
    "StubAdapter",
    "model_registry",
    "register_model",
]
