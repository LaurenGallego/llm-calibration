"""A deterministic fake model.

Not a GPU substitute -- a *determinism* substitute. It exists so the pipeline can
be exercised end to end with hand-computable expected values, so responses a real
checkpoint cannot be made to produce on demand are reachable (an unparseable
answer is just a different `response_template`), and so a real experiment config
can be dry-run in seconds before it costs cluster time.

It is a registered adapter like any other, selectable from config as
`model: {name: stub}`, and it reports `model_id = "stub"` honestly so `storage/`
can partition its output away from real runs and `analysis/` can refuse to read
it -- see the stub-run containment decision in DEVLOG.md.
"""

import hashlib
import math
import random
from collections.abc import Sequence

from madcal import __version__
from madcal.models.base import ChoiceScores, Generation, register_model_adapter


@register_model_adapter("stub")
class StubAdapter:
    name = "stub"
    model_id = "stub"
    revision: str | None = None
    # The stub has no inference library; its behaviour is defined entirely by this
    # codebase, so madcal's own version is the honest answer.
    backend_version = __version__
    supports_scoring = True
    supports_logprobs = True

    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(
        self,
        seed: int = 0,
        n_choices: int = 4,
        response_template: str = "Answer: {letter}",
    ) -> None:
        self.seed = seed
        self.n_choices = n_choices
        self.response_template = response_template

    def score_choices(
        self,
        prompts: Sequence[str],
        choices: Sequence[Sequence[str]],
    ) -> Sequence[ChoiceScores]:
        if len(prompts) != len(choices):
            raise ValueError(
                f"prompts length {len(prompts)} does not match choices length {len(choices)}"
            )
        scores = []
        for prompt, choice_set in zip(prompts, choices, strict=True):
            probs = self._distribution(prompt, len(choice_set))
            logprobs = tuple(math.log(p) for p in probs)
            scores.append(ChoiceScores(choices=tuple(choice_set), logprobs=logprobs))
        return scores

    def generate(
        self,
        prompts: Sequence[str],
        n: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
    ) -> Sequence[Sequence[Generation]]:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")

        results = []
        for prompt in prompts:
            probs = self._distribution(prompt, self.n_choices)

            # A different rng from the one _distribution used, so drawing samples
            # never perturbs what this fake model "believes".
            rng = self._rng(f"generate:{seed}:{prompt}")

            samples = []
            for _ in range(n):
                if temperature == 0.0:
                    index = max(range(len(probs)), key=probs.__getitem__)
                else:
                    index = rng.choices(range(len(probs)), weights=probs)[0]

                letter = self.LETTERS[index]
                samples.append(
                    Generation(
                        text=self.response_template.format(
                            letter=letter,
                            confidence=round(100 * probs[index] / sum(probs)),
                        ),
                        token_logprobs=(math.log(probs[index]),),
                        tokens=(letter,),
                        finish_reason="stop",
                    )
                )
            results.append(samples)
        return results

    def _distribution(self, prompt: str, n_choices: int) -> tuple[float, ...]:
        if n_choices < 1 or n_choices > len(self.LETTERS):
            raise ValueError(f"n_choices must be in 1..{len(self.LETTERS)}, got {n_choices}")

        rng = self._rng(prompt)

        # uniform(0.05, 1.0), not rng.random(): random() can return exactly 0.0,
        # which would make a probability 0 and math.log() raise in score_choices.
        weights = [rng.uniform(0.05, 1.0) for _ in range(n_choices)]

        # How much probability lands on the options at all. The remainder is the
        # mass a real model would spread over the rest of its vocabulary, and it
        # is what ChoiceScores.option_mass() reports.
        mass = rng.uniform(0.3, 0.99)

        total = sum(weights)
        return tuple(w / total * mass for w in weights)

    def _rng(self, prompt: str) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{prompt}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))
