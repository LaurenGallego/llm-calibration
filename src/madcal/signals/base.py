"""Signal protocol, registry, and the answer-free evidence a signal is computed from.

A signal is an unlabeled estimate of how sure the model was about the answer it just
gave. It is the production-available half of the experiment: whatever a practitioner
could watch without paying for fresh labels.

Contracts implementations must honour, since none is expressible in a signature:

- **A signal never sees a label.** The label is `Question.answer`, so this package must
  not import `madcal.benchmarks` any more than it may import `madcal.metrics`. That
  is checked by an `import-linter` contract run from the test suite. What the contract
  cannot check is a label arriving as an argument, which is why `compute` takes
  `Evidence` and not a `Question`: there is no parameter for a label to arrive through.

- **Values are per-question.** The one-number-per-condition of `purpose.md` 6.3 is an
  aggregate, and choosing the aggregator is `analysis/`'s job. A signal returns one
  scalar for one question and does not know what condition it belongs to.

- **A signal that cannot be computed returns a null with a reason, never a number.**
  `SignalValue(None, SignalNull.NO_CHOICE_SCORES)`, not 0.0 and not a skipped row. A
  sentinel confidence averages into a condition-level number as though it were a
  measurement, which is the failure mode this whole codebase is arranged against.

- **Definitions are frozen before data exists.** Changing what a signal means after
  seeing its relationship to ECE makes every run from that point exploratory. See the
  entropy decision in `DEVLOG.md`.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from madcal.models import ChoiceScores, ModelAdapter
from madcal.registry import Registry


class SignalNull(StrEnum):
    NO_CHOICE_SCORES = "no_choice_scores"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class SignalValue:
    value: float | None
    reason: SignalNull | None = None

    def __post_init__(self) -> None:
        if self.value is not None and self.reason is not None:
            raise ValueError("SignalValue must have either a value or a reason, not both.")
        if self.value is None and self.reason is None:
            raise ValueError("SignalValue must have either a value or a reason, not neither.")
        if self.value is not None and not isinstance(self.value, float):
            raise TypeError("SignalValue.value must be a float or None.")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"SignalValue.value must be finite, got {self.value}.")


@dataclass(frozen=True, slots=True)
class Requirements:
    # What a model must be able to do for this signal to mean anything. One field for
    # now: needs_resampling and needs_instruction_following have nothing to check
    # against until alignment stage is a recorded field (DEVLOG open item 10).
    needs_choice_scores: bool = False


@dataclass(frozen=True, slots=True)
class Evidence:
    # Everything a signal is allowed to see. Only types from `models/` appear here, and
    # that is the point -- see the label contract in the module docstring. `generations`
    # arrives with signals 3 and 4, not before.
    scores: ChoiceScores | None = None


class Signal(Protocol):
    name: str
    requires: Requirements

    def compute(self, evidence: Evidence) -> SignalValue: ...


signal_registry: Registry[type[Signal]] = Registry("signal")


def register_signal(name: str):
    return signal_registry.register(name)


def applicable(signal: type[Signal], adapter: ModelAdapter) -> bool:
    return not (signal.requires.needs_choice_scores and not adapter.supports_scoring)
