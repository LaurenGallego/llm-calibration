import pytest
import torch

from madcal.models import TransformersAdapter
from tests.models.test_transformers import CONTINUATIONS, PROMPT, TINY_MODEL, _is_cached

pytestmark = pytest.mark.skipif(not _is_cached(), reason=f"{TINY_MODEL} is not cached")

LONG_PROMPT = "The following are questions about arithmetic.\n\n" + PROMPT


@pytest.fixture(scope="module")
def adapter() -> TransformersAdapter:
    return TransformersAdapter(model_id=TINY_MODEL, dtype="float32", device="cpu", batch_size=4)


def rescored(adapter: TransformersAdapter, states: torch.Tensor, targets: list[int]) -> list[float]:
    weight, bias = adapter.output_head()
    log_probs = torch.log_softmax(torch.nn.functional.linear(states, weight, bias), dim=-1)
    return [float(log_probs[i, token]) for i, token in enumerate(targets)]


def test_head_inputs_rescore_to_the_choice_scores_in_a_ragged_batch(adapter):
    prompts = [PROMPT, LONG_PROMPT, PROMPT]
    sequences = [adapter.continuation_ids(prompt, " A") for prompt in prompts]
    states = adapter.head_inputs([(full, len(cont)) for full, cont in sequences])
    expected = adapter.score_choices(prompts, [(" A",)] * 3)
    for (_, cont), state, scores in zip(sequences, states, expected, strict=True):
        assert state.shape[0] == len(cont) == 1
        assert rescored(adapter, state, cont) == pytest.approx(list(scores.logprobs), abs=1e-5)


def test_head_inputs_cover_every_token_of_a_multi_token_continuation(adapter):
    full, cont = adapter.continuation_ids(PROMPT, " four hundred")
    assert len(cont) > 1
    (state,) = adapter.head_inputs([(full, len(cont))])
    expected = adapter.score_choices([PROMPT], [(" four hundred",)])[0].logprobs[0]
    assert sum(rescored(adapter, state, cont)) == pytest.approx(expected, abs=1e-5)


def test_head_inputs_rescore_to_the_greedy_generation(adapter):
    (generation,) = adapter.generate([PROMPT], 1, 0.0, 6)[0]
    full, ids = adapter.generated_ids(PROMPT, generation)
    assert generation.tokens is not None
    assert adapter._tokenizer.convert_ids_to_tokens(ids) == list(generation.tokens)
    (state,) = adapter.head_inputs([(full, len(ids))])
    assert generation.token_logprobs is not None
    assert rescored(adapter, state, ids) == pytest.approx(list(generation.token_logprobs), abs=1e-5)


def test_a_shifted_position_would_rescore_differently(adapter):
    full, cont = adapter.continuation_ids(PROMPT, " B")
    (state,) = adapter.head_inputs([(full, 1)])
    (shifted,) = adapter.head_inputs([(full[:-1], 1)])
    assert rescored(adapter, state, cont) != pytest.approx(
        rescored(adapter, shifted, cont), abs=1e-4
    )


def test_head_inputs_refuse_a_continuation_as_long_as_the_sequence(adapter):
    full, _ = adapter.continuation_ids(PROMPT, " A")
    with pytest.raises(ValueError, match="continuation positions"):
        adapter.head_inputs([(full, len(full))])


def test_every_option_letter_is_one_token_after_the_prompt(adapter):
    for continuation in CONTINUATIONS:
        _, cont = adapter.continuation_ids(PROMPT, continuation)
        assert len(cont) == 1
