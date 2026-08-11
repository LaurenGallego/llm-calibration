"""Model adapters -- the only package that runs a neural network.

Importing this package registers every adapter, so `model_registry` is fully
populated after `import caldrift.models`. Each new implementation must be imported
here or its decorator never runs.
"""

from caldrift.models.base import (
    ChoiceScores,
    Generation,
    ModelAdapter,
    model_registry,
    register_model,
)

__all__ = [
    "ChoiceScores",
    "Generation",
    "ModelAdapter",
    "model_registry",
    "register_model",
]
