"""Debate data types, configuration and switch registries."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from madcal.models.base import FINISHING_REASONS
from madcal.registry import Registry
from madcal.signals import SignalValue


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


type Renderer = Callable[[Sequence[Message]], str]
type AnswerExtractor = Callable[[str, int | None], str | None]


def agent_id(index: int) -> str:
    """Return the stable identifier of the agent at `index`."""
    return f"agent_{index}"


@dataclass(frozen=True, slots=True)
class DebateQuestion:
    """An answer-free question: its id, the text every agent is shown, and its option count."""

    question_id: str
    text: str
    n_choices: int | None

    def __post_init__(self) -> None:
        if not self.question_id:
            raise ValueError("question_id must be non-empty")
        if not self.text.strip():
            raise ValueError(f"question {self.question_id!r} has empty text")
        if self.n_choices is not None and self.n_choices < 2:
            raise ValueError(f"question {self.question_id!r} needs at least 2 choices")


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """One agent's response at one round."""

    agent_id: str
    round: int
    text: str
    answer: str | None
    confidence: SignalValue
    finish_reason: str

    def __post_init__(self) -> None:
        if self.round < 0:
            raise ValueError(f"round must be >= 0, got {self.round}")
        if self.finish_reason not in FINISHING_REASONS:
            raise ValueError(f"finish_reason {self.finish_reason!r} not in {FINISHING_REASONS}")
        value = self.confidence.value
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {value}")


class SystemNull(StrEnum):
    NO_VALID_ANSWERS = "no_valid_answers"
    TIE = "tie"


@dataclass(frozen=True, slots=True)
class SystemAnswer:
    """The aggregated final answer, or a null with a reason."""

    answer: str | None
    confidence: float | None
    reason: SystemNull | None = None

    def __post_init__(self) -> None:
        if (self.answer is None) == (self.reason is None):
            raise ValueError("SystemAnswer needs exactly one of answer or reason")
        if (self.answer is None) != (self.confidence is None):
            raise ValueError("SystemAnswer confidence must be present iff answer is")
        if self.confidence is not None and not (
            math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError(f"system confidence must be in [0, 1], got {self.confidence}")


@dataclass(frozen=True, slots=True)
class DebateTranscript:
    """Every agent turn of one question's debate, and the system answer."""

    question_id: str
    n_agents: int
    n_rounds: int
    turns: tuple[AgentTurn, ...]
    system: SystemAnswer

    def __post_init__(self) -> None:
        expected = [
            (agent_id(agent), round_)
            for round_ in range(self.n_rounds + 1)
            for agent in range(self.n_agents)
        ]
        actual = [(turn.agent_id, turn.round) for turn in self.turns]
        if actual != expected:
            raise ValueError(
                f"transcript for {self.question_id!r} must hold one turn per (agent, round) "
                f"ordered by round then agent: expected {len(expected)} turns, got {actual}"
            )

    def round_turns(self, round_: int) -> tuple[AgentTurn, ...]:
        """Return the turns of one round, ordered by agent."""
        if not 0 <= round_ <= self.n_rounds:
            raise ValueError(f"round {round_} outside 0..{self.n_rounds}")
        return tuple(turn for turn in self.turns if turn.round == round_)


@dataclass(frozen=True, slots=True)
class ConfidenceMode:
    """How an agent is asked for its confidence, and how it is read back."""

    name: str
    instruction: str | None
    parse: Callable[[str], SignalValue]


@dataclass(frozen=True, slots=True)
class AggregationRule:
    """How the final round's turns become the system answer."""

    name: str
    requires_confidence: bool
    requires_odd_agents: bool
    aggregate: Callable[[Sequence[AgentTurn]], SystemAnswer]


confidence_mode_registry: Registry[ConfidenceMode] = Registry("confidence mode")
aggregation_registry: Registry[AggregationRule] = Registry("aggregation")


class DebateConfig(BaseModel):
    """Structure, switches and sampling settings of one debate protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_agents: int = Field(ge=1)
    n_rounds: int = Field(ge=0)
    confidence_mode: str
    aggregation: str
    temperature: float = Field(ge=0.0)
    max_tokens: int = Field(default=512, ge=1)

    @model_validator(mode="after")
    def _switches_are_coherent(self) -> Self:
        if self.confidence_mode not in confidence_mode_registry:
            known = confidence_mode_registry.names()
            raise ValueError(f"unknown confidence_mode {self.confidence_mode!r}; known: {known}")
        if self.aggregation not in aggregation_registry:
            known = aggregation_registry.names()
            raise ValueError(f"unknown aggregation {self.aggregation!r}; known: {known}")
        mode = confidence_mode_registry.get(self.confidence_mode)
        rule = aggregation_registry.get(self.aggregation)
        if rule.requires_odd_agents and self.n_agents % 2 == 0:
            raise ValueError(f"aggregation {rule.name!r} requires an odd n_agents")
        if rule.requires_confidence and mode.instruction is None:
            raise ValueError(
                f"aggregation {rule.name!r} requires a confidence_mode that states confidence"
            )
        if self.n_rounds > 0 and self.n_agents < 2:
            raise ValueError("debate rounds require at least two agents")
        if self.temperature == 0.0 and self.n_agents > 1:
            raise ValueError("temperature=0 makes every agent's round-0 answer identical")
        return self
