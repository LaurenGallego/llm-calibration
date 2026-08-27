"""Model adapter protocol, registry, and the objects a model hands back.

This package is the only one that knows a neural network exists. It takes rendered
prompt strings and returns evidence about what the model would answer and how sure
it was. It never receives a gold answer, never builds a prompt, and never computes
a signal -- it produces the raw material `signals/` and `metrics/` consume.

"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from caldrift.registry import Registry

FINISHING_REASONS = ("stop", "length")

# Slack allowed when checking that probabilities sum to what they should. Shared with
# `signals/` so the two cannot drift apart and disagree about what "sums to 1" means.
PROBABILITY_TOLERANCE = 1e-9


@dataclass(frozen=True, slots=True)
class Generation:
    text: str
    token_logprobs: tuple[float, ...] | None
    tokens: tuple[str, ...] | None
    finish_reason: str

    def __post_init__(self) -> None:
        if self.finish_reason not in FINISHING_REASONS:
            raise ValueError(f"finish_reason {self.finish_reason} not in {FINISHING_REASONS}")

        if (
            self.token_logprobs is not None
            and self.tokens is not None
            and len(self.token_logprobs) != len(self.tokens)
        ):
            raise ValueError(
                f"token_logprobs length {len(self.token_logprobs)} does not match "
                f"tokens length {len(self.tokens)}"
            )


@dataclass(frozen=True, slots=True)
class ChoiceScores:
    choices: tuple[str, ...]
    logprobs: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.choices) != len(self.logprobs):
            raise ValueError(
                f"choices length {len(self.choices)} does not match "
                f"logprobs length {len(self.logprobs)}"
            )
        if len(self.choices) == 0:
            raise ValueError("choices must contain at least one candidate")
        if any(not math.isfinite(lp) for lp in self.logprobs):
            raise ValueError(f"logprobs must all be finite, got {self.logprobs}")
        if any(lp > 0 for lp in self.logprobs):
            raise ValueError("logprobs must be <= 0; found a positive value")
        # The candidates are mutually exclusive continuations, so their probabilities
        # cannot sum past 1. A mass above 1 means the adapter scored events that overlap
        # -- a bug in the adapter, not a quirk to be clamped away downstream.
        mass = self.option_mass()
        if mass > 1.0 + PROBABILITY_TOLERANCE:
            raise ValueError(f"option mass must be <= 1, got {mass}")

    def probabilities(self) -> tuple[float, ...]:
        """Normalised distribution over the candidates, aligned to `choices`.

        Conditional on the answer being one of `choices`: any mass the model put
        elsewhere in the vocabulary is discarded here. See `option_mass` for how
        much that was.
        """
        shift = max(self.logprobs)
        weights = [math.exp(logprob - shift) for logprob in self.logprobs]
        total = sum(weights)
        return tuple(weight / total for weight in weights)

    def option_mass(self) -> float:
        """Total probability on the candidates, before renormalising."""
        shift = max(self.logprobs)
        return math.exp(shift) * sum(math.exp(logprob - shift) for logprob in self.logprobs)


class ModelAdapter(Protocol):
    name: str
    model_id: str
    revision: str | None
    # The version of the library that actually produced the numbers. Not derivable from
    # anything else on the row: vLLM 0.6.1 and 0.6.3 can return different logprobs for
    # one checkpoint, and so can two `transformers` minors. Recorded per row so a
    # condition run either side of an upgrade is detectable rather than invisible.
    backend_version: str
    supports_scoring: bool
    supports_logprobs: bool

    def score_choices(
        self,
        prompts: Sequence[str],
        choices: Sequence[Sequence[str]],
    ) -> Sequence[ChoiceScores]:
        """Score candidate continuations without generating.

        `choices[i]` are the candidates for `prompts[i]` -- per-prompt, because
        different questions can have different option sets. Returns one
        ChoiceScores per prompt, aligned by position.
        """
        ...

    def generate(
        self,
        prompts: Sequence[str],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Sequence[Sequence[Generation]]:
        """Generate text.

        Returns `result[i][j]`: the j-th of `n` samples for `prompts[i]`. The
        nesting exists for signal 3, which needs N resamples per question; n=1
        with temperature=0.0 covers the ordinary greedy answer and signal 4.
        """
        ...


model_adapter_registry: Registry[type[ModelAdapter]] = Registry("model adapter")


def register_model_adapter(name: str):
    return model_adapter_registry.register(name)
