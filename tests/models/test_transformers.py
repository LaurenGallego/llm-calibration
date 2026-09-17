"""TransformersAdapter against a real (tiny) checkpoint, on CPU.

Every test here targets a failure that returns a plausible number rather than raising:
a mis-sliced continuation, a softmax on the wrong axis, padding leaking into a score,
a truncated generation recorded as a completed one. None of them is caught by the
adapter running without error, which is the only thing a mock could establish.

The model is 2MB of random weights -- quality is irrelevant, arithmetic is the point.
It is never downloaded from inside a test: cluster nodes have no internet, so these
skip when the checkpoint is absent from the local cache rather than reaching out.
"""

import math

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from madcal.models import TransformersAdapter

TINY_MODEL = "sshleifer/tiny-gpt2"
PROMPT = "What is 2+2?\nAnswer:"
CONTINUATIONS = (" A", " B", " C", " D")


def _is_cached() -> bool:
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(TINY_MODEL, local_files_only=True)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _is_cached(),
    reason=(
        f"{TINY_MODEL} is not in the local HF cache. Fetch it once with: "
        f'uv run python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download('{TINY_MODEL}')\""
    ),
)


@pytest.fixture(scope="module")
def adapter() -> TransformersAdapter:
    # float32 so a mismatch against the reference means a logic error, not bf16 noise.
    return TransformersAdapter(model_id=TINY_MODEL, dtype="float32", batch_size=4)


@pytest.fixture(scope="module")
def reference():
    """A second, independently loaded copy -- the adapter's internals are not reused."""
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32).eval()
    return tokenizer, model


def test_revision_resolves_to_a_commit_sha(adapter: TransformersAdapter):
    # A row recording "main" does not trace to a checkpoint (CLAUDE.md provenance).
    assert adapter.revision is not None
    assert len(adapter.revision) == 40
    assert all(character in "0123456789abcdef" for character in adapter.revision)


def test_scores_match_an_independent_forward_pass(adapter: TransformersAdapter, reference):
    tokenizer, model = reference
    prompt_ids = tokenizer(PROMPT)["input_ids"]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt_ids])).logits
    final = torch.log_softmax(logits[0, -1, :], dim=-1)

    # The softmax is over the vocabulary axis, so it must sum to 1 there.
    assert float(final.exp().sum()) == pytest.approx(1.0, abs=1e-5)

    scored = adapter.score_choices([PROMPT], [CONTINUATIONS])[0]
    for continuation, produced in zip(CONTINUATIONS, scored.logprobs, strict=True):
        continuation_ids = tokenizer(PROMPT + continuation)["input_ids"][len(prompt_ids) :]
        assert len(continuation_ids) == 1
        expected = float(final[continuation_ids[0]])
        assert produced == pytest.approx(expected, abs=1e-5)


def test_reading_the_wrong_position_would_be_visible(adapter: TransformersAdapter, reference):
    # Guards the guard: if the off-by-one produced the same number, the test above
    # would pass with the offset removed and prove nothing.
    tokenizer, model = reference
    prompt_ids = tokenizer(PROMPT)["input_ids"]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt_ids])).logits
    target = tokenizer(PROMPT + " A")["input_ids"][len(prompt_ids) :][0]

    correct = float(torch.log_softmax(logits[0, -1, :], dim=-1)[target])
    shifted = float(torch.log_softmax(logits[0, -2, :], dim=-1)[target])
    assert not math.isclose(correct, shifted, abs_tol=1e-4)


def test_padding_does_not_change_a_score(adapter: TransformersAdapter):
    # A ragged batch pads the short rows; batch_size=1 pads nothing at all.
    short, long = PROMPT, "The following are questions about arithmetic.\n\n" + PROMPT
    prompts = [short, long, short]
    choices = [CONTINUATIONS] * 3

    batched = adapter.score_choices(prompts, choices)
    solo = TransformersAdapter(model_id=TINY_MODEL, dtype="float32", batch_size=1).score_choices(
        prompts, choices
    )

    for from_batch, from_solo in zip(batched, solo, strict=True):
        assert from_batch.logprobs == pytest.approx(from_solo.logprobs, abs=1e-5)
    assert batched[0].logprobs != batched[1].logprobs
    assert batched[0].logprobs == batched[2].logprobs


def test_multi_token_continuation_is_summed_not_averaged(adapter: TransformersAdapter, reference):
    """The sum-vs-normalise decision, which single-token candidates cannot exercise.

    Every other scoring test uses " A".." D", one token each, where the sum and the
    mean are the same number -- so none of them can tell the two apart.
    """
    tokenizer, model = reference
    long_continuation = " four hundred"
    prompt_ids = tokenizer(PROMPT)["input_ids"]
    full_ids = tokenizer(PROMPT + long_continuation)["input_ids"]
    continuation_ids = full_ids[len(prompt_ids) :]
    assert len(continuation_ids) > 1, "this test is pointless on a single token"

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([full_ids])).logits
    window = torch.log_softmax(logits[0, len(prompt_ids) - 1 : len(full_ids) - 1, :], dim=-1)
    per_token = [float(window[step, token]) for step, token in enumerate(continuation_ids)]

    scored = adapter.score_choices([PROMPT], [(long_continuation,)])[0]
    assert scored.logprobs[0] == pytest.approx(sum(per_token), abs=1e-5)
    # The mean is what length-normalisation would give: same sign, plausible size,
    # different number. Asserting it is *not* that is the point of the test.
    assert scored.logprobs[0] != pytest.approx(sum(per_token) / len(per_token), abs=1e-3)


def test_generate_padding_does_not_change_output(adapter: TransformersAdapter):
    """Left-padding alignment. A single-prompt call pads nothing and proves nothing."""
    short, long = PROMPT, "The following are questions about arithmetic.\n\n" + PROMPT
    batched = adapter.generate([short, long, short], n=1, temperature=0.0, max_tokens=4)
    alone = [
        adapter.generate([short], n=1, temperature=0.0, max_tokens=4)[0][0].text,
        adapter.generate([long], n=1, temperature=0.0, max_tokens=4)[0][0].text,
    ]
    assert batched[0][0].text == alone[0]
    assert batched[1][0].text == alone[1]
    assert batched[2][0].text == alone[0]


def test_merged_boundary_raises(adapter: TransformersAdapter):
    # "ab" + "c" tokenises as one piece, so the continuation cannot be sliced off.
    # Silently mis-slicing here would shift every logprob in the project.
    with pytest.raises(ValueError, match="merged across"):
        adapter.score_choices(["ab"], [("c", " A")])


def test_mismatched_lengths_raise(adapter: TransformersAdapter):
    with pytest.raises(ValueError, match="does not match"):
        adapter.score_choices([PROMPT, PROMPT], [CONTINUATIONS])


def test_greedy_is_deterministic(adapter: TransformersAdapter):
    first = adapter.generate([PROMPT], n=1, temperature=0.0, max_tokens=4)
    second = adapter.generate([PROMPT], n=1, temperature=0.0, max_tokens=4)
    assert first[0][0].text == second[0][0].text


def test_seeded_sampling_reproduces_and_varies(adapter: TransformersAdapter):
    def run(seed: int) -> list[str]:
        return [g.text for g in adapter.generate([PROMPT], 4, 1.0, 4, seed)[0]]

    assert run(7) == run(7)
    assert run(7) != run(8)
    # If the samples were identical, cluster tightness would read as perfect agreement.
    assert len(set(run(7))) > 1


def test_greedy_cannot_be_resampled(adapter: TransformersAdapter):
    # n identical greedy answers would be fabricated agreement, not a measurement.
    with pytest.raises(ValueError, match="cannot produce"):
        adapter.generate([PROMPT], n=5, temperature=0.0)


def test_truncation_is_not_recorded_as_a_stop(adapter: TransformersAdapter):
    truncated = adapter.generate([PROMPT], n=1, temperature=0.0, max_tokens=2)[0][0]
    assert truncated.finish_reason == "length"
    assert truncated.tokens is not None
    assert len(truncated.tokens) == 2


def test_token_logprobs_align_with_tokens(adapter: TransformersAdapter):
    # Generation's own validation covers the lengths; this covers the values being
    # real logprobs rather than raw scores or padding.
    generated = adapter.generate([PROMPT], n=1, temperature=0.0, max_tokens=4)[0][0]
    assert generated.token_logprobs is not None
    assert all(math.isfinite(value) and value <= 0.0 for value in generated.token_logprobs)
