"""Multi-agent debate engine."""

from madcal.debate.aggregate import argmax_confidence, majority_vote
from madcal.debate.base import (
    AgentTurn,
    AggregationRule,
    AnswerExtractor,
    ConfidenceMode,
    DebateConfig,
    DebateQuestion,
    DebateTranscript,
    Message,
    Renderer,
    Role,
    SystemAnswer,
    SystemNull,
    agent_id,
    aggregation_registry,
    confidence_mode_registry,
)
from madcal.debate.confidence import VERBALIZED_INSTRUCTION
from madcal.debate.engine import opening_request, round_seed, run_debate, update_request
from madcal.debate.protocols import PRESETS, preset

__all__ = [
    "PRESETS",
    "VERBALIZED_INSTRUCTION",
    "AgentTurn",
    "AggregationRule",
    "AnswerExtractor",
    "ConfidenceMode",
    "DebateConfig",
    "DebateQuestion",
    "DebateTranscript",
    "Message",
    "Renderer",
    "Role",
    "SystemAnswer",
    "SystemNull",
    "agent_id",
    "aggregation_registry",
    "argmax_confidence",
    "confidence_mode_registry",
    "majority_vote",
    "opening_request",
    "preset",
    "round_seed",
    "run_debate",
    "update_request",
]
