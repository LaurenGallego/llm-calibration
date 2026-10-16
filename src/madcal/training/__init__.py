"""Answer-level calibration heads fitted on the hidden-state cache."""

from madcal.training.arms import (
    Arm,
    OutputHead,
    arm_registry,
    batched_brier,
    batched_confidence,
    build_arm,
    noise_only_vae,
    register_arm,
)
from madcal.training.loop import TrainResult, in_selection, split, train
from madcal.training.objective import answer_changed, answer_log_confidence, brier_loss

__all__ = [
    "Arm",
    "OutputHead",
    "TrainResult",
    "answer_changed",
    "answer_log_confidence",
    "arm_registry",
    "batched_brier",
    "batched_confidence",
    "brier_loss",
    "build_arm",
    "in_selection",
    "noise_only_vae",
    "register_arm",
    "split",
    "train",
]
