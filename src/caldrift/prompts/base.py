"""Prompt protocol, registry, and the rendered prompt a model receives.

A prompt protocol turns a `Question` into the exact string a model sees. That
string is an experimental variable, not an implementation detail -- protocol
choice has been measured to move ECE more than switching benchmarks does -- so it
is named, registered, and recorded on every result row, and can be held fixed
while alignment stage varies.

Conventions this package fixes, following lm-evaluation-harness:

- `text` ends at the answer cue with no trailing space:  "...\nAnswer:"
- `continuations` carry the leading space instead:       " A", " B", ...

Those two together decide how the answer tokenizes, and therefore every logprob
in the project. Moving the space from one side to the other changes results
without changing anything a human would notice.

Contracts implementations must honour:

- Rendering is per-question, not batched. It is cheap CPU string work, unlike
  `models/`, where batching is the whole source of throughput.
- Few-shot exemplars arrive already chosen. Which questions serve as exemplars is
  orchestration's concern, so protocols stay stateless and testable with no
  dataset access.
- The target question's own answer must never appear in `text`. Exemplars show
  their answers; the target shows only its cue. A leak here raises accuracy
  toward 100% and raises nothing else.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from caldrift.benchmarks import Question
from caldrift.registry import Registry

# The space between the answer cue and the answer token. It lives on the
# continuation, never at the end of `text` -- see the convention note above.
CONTINUATION_PREFIX = " "


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    # Exactly what the model receives. For the likelihood path this stops at the
    # answer cue; for the generative path it is the whole prompt. Same string
    # either way, so a likelihood run and a generative run stay comparable.
    text: str

    # The candidates to score, in the order they should be returned. None for
    # free-form benchmarks (GSM8K, TriviaQA), where there is nothing enumerable
    # to rank and only `generate` applies.
    continuations: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("RenderedPrompt.text must be non-empty")
        if self.text[-1].isspace():
            raise ValueError("RenderedPrompt.text must not end in whitespace")
        if self.continuations is not None:
            if len(self.continuations) < 2:
                raise ValueError("RenderedPrompt.continuations must have at least 2 candidates")
            if len(set(self.continuations)) != len(self.continuations):
                raise ValueError("RenderedPrompt.continuations must not contain duplicates")
            for c in self.continuations:
                if not c.startswith(CONTINUATION_PREFIX):
                    raise ValueError(
                        f"RenderedPrompt.continuations must start with {CONTINUATION_PREFIX!r}"
                    )


class PromptProtocol(Protocol):
    # Registry key, written into every result row so a stored number can be traced
    # back to the exact wording the model saw.
    name: str

    # Constructor config, not a per-call argument: a 0-shot and a 5-shot rendering
    # are different experimental conditions and must be distinguishable in results.
    n_shots: int

    def render(
        self,
        question: Question,
        examples: Sequence[Question] = (),
    ) -> RenderedPrompt:
        """Render one question, optionally preceded by few-shot exemplars.

        `examples` are already selected and ordered by the caller; an
        implementation formats them and does not choose them. Raise if the count
        disagrees with `n_shots` rather than silently rendering a different
        condition than the one the config asked for.
        """
        ...


prompt_registry: Registry[type[PromptProtocol]] = Registry("prompt protocol")


def register_prompt(name: str):
    return prompt_registry.register(name)
