"""HuggingFace `transformers` adapter -- the primary scoring backend.

This is the path the paper's numbers come from. It runs the model in-process; nothing
here starts a server, and no HTTP is involved.

`score_choices` carries most of the silent-failure risk in the project, concentrated in
four places that all return plausible numbers when wrong:

1. **The tokenisation boundary.** `tok(prompt + " A")` is not reliably
   `tok(prompt) + tok(" A")`. Encode the whole string, encode the context, and slice at
   `len(context_ids)`. The prompt convention in `prompts/base.py` -- no trailing space
   on `text`, the space carried on the continuation -- exists to make that slice
   well-defined.
2. **The off-by-one.** `logits[i]` predicts token `i + 1`, so the logprob of the
   continuation token at position `j` is read from `logits[j - 1]`.
3. **BOS.** Mistral and Llama tokenisers prepend it. It belongs on the context, once,
   and must not reappear inside the continuation slice.
4. **Padding.** A padded position leaking into the sum changes a logprob without
   changing anything visible.

Decisions this file implements, all recorded in `DEVLOG.md`:

- **bf16**, the precision the checkpoints were released in. Recorded per row: dtype
  moves logprobs in the third decimal and ECE is a function of those logprobs.
- **Continuation score is the sum of its token logprobs, never length-normalised.**
  Summed logprobs are the log-probabilities of mutually exclusive continuations, so
  renormalising them gives a real conditional probability. Per-token averages give a
  ratio of geometric means that looks identical and is the probability of nothing.
- **`local_files_only`.** Cluster nodes may have no internet; a missing artefact must
  fail loudly rather than hang on a download that cannot happen.
- **`revision` is resolved to a concrete commit SHA** at load time. A row recording
  `"main"` does not trace to a checkpoint.
"""

from collections.abc import Sequence

from caldrift.models.base import ChoiceScores, Generation, register_model_adapter

DEFAULT_DTYPE = "bfloat16"


@register_model_adapter("transformers")
class TransformersAdapter:
    name = "transformers"
    supports_scoring = True
    supports_logprobs = True

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        dtype: str = DEFAULT_DTYPE,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        # Imported here rather than at module scope so `import caldrift.models` stays
        # fast for the stub path. Follows mmlu.py's precedent with `datasets`.
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        self.model_id = model_id
        self.dtype = dtype
        self.device = device
        self.batch_size = batch_size
        self._torch = torch
        # Read from the library that is actually loaded, never pinned in config: a
        # hardcoded string would keep reporting the version someone typed once.
        self.backend_version: str = transformers.__version__

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, local_files_only=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, dtype),
            device_map=device,
            local_files_only=True,
        ).eval()

        # Mistral and Llama ship without a pad token; batching raises without one.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # A row recording "main" does not trace to a checkpoint. `_commit_hash` is what
        # transformers records for a hub load, and is available offline from the cache.
        resolved = getattr(self._model.config, "_commit_hash", None)
        if resolved is None:
            raise ValueError(
                f"could not resolve a commit SHA for {model_id!r} at revision "
                f"{revision!r}. A result that cannot be traced to an exact checkpoint "
                f"is not usable; load from a hub cache rather than a bare local path."
            )
        self.revision: str | None = resolved

    def score_choices(
        self,
        prompts: Sequence[str],
        choices: Sequence[Sequence[str]],
    ) -> Sequence[ChoiceScores]:
        if len(prompts) != len(choices):
            raise ValueError(
                f"prompts length {len(prompts)} does not match choices length {len(choices)}"
            )

        rows: list[tuple[list[int], list[int]]] = []
        for prompt, choice_set in zip(prompts, choices, strict=True):
            if len(choice_set) == 0:
                raise ValueError("every prompt needs at least one candidate continuation")
            context_ids = self._encode(prompt)
            if not context_ids:
                raise ValueError("prompt tokenised to nothing; cannot score a continuation")
            for continuation in choice_set:
                rows.append(self._split(context_ids, prompt, continuation))

        flat: list[float] = []
        for start in range(0, len(rows), self.batch_size):
            flat.extend(self._score_batch(rows[start : start + self.batch_size]))

        results: list[ChoiceScores] = []
        cursor = 0
        for choice_set in choices:
            width = len(choice_set)
            results.append(
                ChoiceScores(
                    choices=tuple(choice_set),
                    logprobs=tuple(flat[cursor : cursor + width]),
                )
            )
            cursor += width
        return results

    def _encode(self, text: str) -> list[int]:
        return list(self._tokenizer(text)["input_ids"])

    def _split(
        self, context_ids: list[int], prompt: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """Tokenise prompt+continuation once and slice, rather than tokenising apart."""
        full_ids = self._encode(prompt + continuation)
        if full_ids[: len(context_ids)] != context_ids:
            raise ValueError(
                f"tokeniser merged across the prompt/continuation boundary for "
                f"{continuation!r}: the context is not a prefix of the full sequence, "
                f"so the continuation cannot be sliced off unambiguously"
            )
        continuation_ids = full_ids[len(context_ids) :]
        if not continuation_ids:
            raise ValueError(f"continuation {continuation!r} added no tokens to the prompt")
        return full_ids, continuation_ids

    def _score_batch(self, batch: Sequence[tuple[list[int], list[int]]]) -> list[float]:
        torch = self._torch
        pad_id = self._tokenizer.pad_token_id
        width = max(len(full) for full, _ in batch)

        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (full, _) in enumerate(batch):
            input_ids[row, : len(full)] = torch.tensor(full, dtype=torch.long)
            attention[row, : len(full)] = 1

        device = self._model.device
        with torch.no_grad():
            logits = self._model(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
            ).logits

        # logits: (len(batch), width, vocab_size) -- one distribution over the whole
        # vocabulary per input position, still in the model's dtype. `width` is the
        # padded length, so rows shorter than it carry trailing positions that predict
        # from pad tokens; the slice below never reaches them.

        scores: list[float] = []
        for row, (full, continuation_ids) in enumerate(batch):
            start = len(full) - len(continuation_ids)
            # logits[j - 1] predicts token j, so the window is shifted back by one.
            # Sliced before log_softmax: the full-sequence softmax in float32 would be
            # batch x positions x vocab, which is gigabytes on a few-shot prompt.
            window = logits[row, start - 1 : len(full) - 1, :].float()
            window = torch.log_softmax(window, dim=-1)
            targets = torch.tensor(continuation_ids, dtype=torch.long, device=window.device)
            picked = window.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            scores.append(float(picked.sum()))
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
        if temperature == 0.0 and n > 1:
            # Greedy decoding is deterministic, so n samples would be n copies of one
            # answer. Returning them would hand signal 3 perfect agreement that the
            # model never demonstrated -- fabricated cluster tightness, not a
            # measurement. Resampling requires temperature > 0.
            raise ValueError(
                f"temperature=0 is greedy and cannot produce {n} distinct samples; "
                f"raise temperature or set n=1"
            )

        results: list[list[Generation]] = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = list(prompts[start : start + self.batch_size])
            results.extend(self._generate_batch(chunk, n, temperature, max_tokens, seed, start))
        return results

    def _generate_batch(
        self,
        prompts: list[str],
        n: int,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        offset: int,
    ) -> list[list[Generation]]:
        torch = self._torch

        # LEFT padding, the opposite of score_choices: generation continues from the
        # final position, so right-padding would have the model continue from a pad
        # token and return fluent nonsense rather than fail.
        encoded = self._tokenizer(
            prompts, return_tensors="pt", padding=True, padding_side="left"
        ).to(self._model.device)
        prompt_width = int(encoded["input_ids"].shape[1])

        if seed is not None:
            # generate() takes no seed. Derived per chunk so a rerun with the same
            # config reproduces; note that changing batch_size changes the chunking and
            # therefore the samples, which is why batch_size belongs in provenance.
            torch.manual_seed(seed + offset)

        options: dict[str, object] = {
            "max_new_tokens": max_tokens,
            "num_return_sequences": n,
            "return_dict_in_generate": True,
            "output_scores": True,
            "do_sample": temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            options["temperature"] = temperature

        # The two ignores below are a transformers 5.x stub gap: the class returned by
        # AutoModelForCausalLM does not satisfy its own GenerativePreTrainedModel
        # protocol. Both methods exist and are exercised by the tests.
        with torch.no_grad():
            out = self._model.generate(**encoded, **options)  # pyright: ignore[reportAttributeAccessIssue]

        # `scores` are the distributions the sampler actually drew from, so above
        # temperature 0 these logprobs are under the sampling distribution rather than
        # the model's own. Signal 3 uses the text, not these values; anything reading
        # them as model confidence must account for the temperature.
        transition = self._model.compute_transition_scores(  # pyright: ignore[reportAttributeAccessIssue]
            out.sequences, out.scores, normalize_logits=True
        )

        eos_id = self._tokenizer.eos_token_id
        new_ids = out.sequences[:, prompt_width:]

        # generate returns prompts * n sequences, prompt-major.
        grouped: list[list[Generation]] = []
        for prompt_index in range(len(prompts)):
            samples: list[Generation] = []
            for sample_index in range(n):
                row = prompt_index * n + sample_index
                ids = [int(token) for token in new_ids[row]]
                if eos_id is not None and eos_id in ids:
                    length = ids.index(eos_id) + 1
                    finish_reason = "stop"
                else:
                    length = len(ids)
                    finish_reason = "length"

                kept = ids[:length]
                samples.append(
                    Generation(
                        text=self._tokenizer.decode(kept, skip_special_tokens=True),
                        token_logprobs=tuple(float(value) for value in transition[row, :length]),
                        tokens=tuple(self._tokenizer.convert_ids_to_tokens(kept)),
                        finish_reason=finish_reason,
                    )
                )
            grouped.append(samples)
        return grouped
