"""What config_hash is and is not sensitive to.

Both directions are silent when wrong. A hash blind to `n_shots` lets a requeued job
blend two protocols inside one condition; a hash that moves with `batch_size` makes an
ordinary throughput tweak discard every row already computed.
"""

import pytest
from pydantic import ValidationError

from madcal.config import RunConfig

BASE = {
    "model": {"name": "transformers", "model_id": "mistralai/Mistral-7B-v0.1", "stage": "base"},
    "output_root": "results",
}
SHA = "a" * 40


def build(**overrides) -> RunConfig:
    payload = {**BASE, **overrides}
    return RunConfig.model_validate(payload)


def with_model(**overrides) -> RunConfig:
    return build(model={**BASE["model"], **overrides})


def test_hash_is_stable():
    assert build().config_hash(SHA) == build().config_hash(SHA)


def test_environment_fields_do_not_move_the_hash():
    reference = build().config_hash(SHA)
    assert with_model(batch_size=64).config_hash(SHA) == reference
    assert with_model(device="cpu").config_hash(SHA) == reference
    assert build(output_root="/somewhere/else").config_hash(SHA) == reference
    assert build(chunk_size=1).config_hash(SHA) == reference


@pytest.mark.parametrize(
    "override",
    [
        {"prompt": {"n_shots": 0}},
        {"scoring_mode": "generative"},
        {"seed": 99},
        {"benchmark": {"name": "mmlu", "revision": "abc123"}},
    ],
)
def test_experimental_fields_move_the_hash(override):
    assert build(**override).config_hash(SHA) != build().config_hash(SHA)


def test_dtype_moves_the_hash():
    # bf16 vs fp32 shifts logprobs in the third decimal, and ECE is a function of those.
    assert with_model(dtype="float32").config_hash(SHA) != build().config_hash(SHA)


def test_resolved_checkpoint_moves_the_hash():
    # A run pinned to a branch name must not reuse rows from before the branch moved.
    assert build().config_hash("sha-a") != build().config_hash("sha-b")


def test_a_benchmark_block_must_name_its_benchmark():
    # A discriminated union cannot infer the tag from a default, which is the point:
    # every config states which benchmark it runs rather than inheriting one.
    with pytest.raises(ValidationError, match="discriminator"):
        build(benchmark={"revision": "abc123"})


def test_a_mistyped_key_is_an_error_not_a_default():
    with pytest.raises(ValidationError, match="Extra inputs"):
        with_model(dtyp="float32")


def test_settings_that_do_not_apply_to_an_adapter_are_refused():
    with pytest.raises(ValidationError, match="Extra inputs"):
        RunConfig.model_validate(
            {"model": {"name": "stub", "dtype": "float16"}, "output_root": "r"}
        )


def test_greedy_resampling_is_refused_at_config_time():
    with pytest.raises(ValidationError, match="cannot produce"):
        build(n_samples=4)


def test_a_run_cannot_be_narrowed_to_a_subset():
    # A run is the full benchmark. Topic shards are cut from the stored `subject`
    # column by analysis, not by narrowing what was executed -- otherwise a condition's
    # ECE is over whichever questions someone chose that day.
    with pytest.raises(ValidationError, match="Extra inputs"):
        build(benchmark={"name": "mmlu", "limit": 100})
    with pytest.raises(ValidationError, match="Extra inputs"):
        build(benchmark={"name": "mmlu", "subjects": ["anatomy"]})
