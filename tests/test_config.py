"""What config_hash is and is not sensitive to.

Both directions are silent when wrong. A hash blind to `n_shots` lets a requeued job
blend two protocols inside one condition; a hash that moves with `batch_size` makes an
ordinary throughput tweak discard every row already computed.
"""

import pytest
from pydantic import ValidationError

from madcal.config import FitConfig, HarvestConfig, OnPolicyConfig, RunConfig

BASE = {
    "model": {"name": "transformers", "model_id": "mistralai/Mistral-7B-v0.1", "variant": "base"},
    "debate": {"protocol": "vanilla", "temperature": 1.0},
    "output_root": "results",
}
SHA = "a" * 40
CODE = "c" * 40


def build(**overrides) -> RunConfig:
    payload = {**BASE, **overrides}
    return RunConfig.model_validate(payload)


def with_model(**overrides) -> RunConfig:
    return build(model={**BASE["model"], **overrides})


def test_hash_is_stable():
    assert build().config_hash(SHA, CODE) == build().config_hash(SHA, CODE)


def test_environment_fields_do_not_move_the_hash():
    reference = build().config_hash(SHA, CODE)
    assert with_model(batch_size=64).config_hash(SHA, CODE) == reference
    assert with_model(device="cpu").config_hash(SHA, CODE) == reference
    assert build(output_root="/somewhere/else").config_hash(SHA, CODE) == reference
    assert build(chunk_size=1).config_hash(SHA, CODE) == reference


@pytest.mark.parametrize(
    "override",
    [
        {"prompt": {"name": "plain_chat"}},
        {"seed": 99},
        {"benchmark": {"name": "mmlu", "revision": "abc123"}},
        {"benchmark": {"name": "mmlu", "n_questions": 500}},
        {"benchmark": {"name": "mmlu", "n_questions": 500, "sample_seed": 1}},
        {"benchmark": {"name": "mmlu", "reasoning": False}},
        {"benchmark": {"name": "trivia_qa"}},
        {"debate": {"protocol": "vanilla", "temperature": 0.7}},
        {"debate": {"protocol": "best_of_n", "temperature": 1.0, "n_agents": 3}},
        {"debate": {"protocol": "vanilla", "temperature": 1.0, "max_tokens": 256}},
    ],
)
def test_experimental_fields_move_the_hash(override):
    assert build(**override).config_hash(SHA, CODE) != build().config_hash(SHA, CODE)


def test_dtype_moves_the_hash():
    # bf16 vs fp32 shifts logprobs in the third decimal, and ECE is a function of those.
    assert with_model(dtype="float32").config_hash(SHA, CODE) != build().config_hash(SHA, CODE)


def test_resolved_checkpoint_moves_the_hash():
    # A run pinned to a branch name must not reuse rows from before the branch moved.
    assert build().config_hash("sha-a", CODE) != build().config_hash("sha-b", CODE)


def test_code_moves_the_hash():
    assert build().config_hash(SHA, "code-a") != build().config_hash(SHA, "code-b")


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
        RunConfig.model_validate({**BASE, "model": {"name": "stub", "dtype": "float16"}})


def test_greedy_debate_is_refused_at_config_time():
    with pytest.raises(ValidationError, match="identical"):
        build(debate={"protocol": "vanilla", "temperature": 0.0})


def test_an_unknown_protocol_is_refused_at_config_time():
    with pytest.raises(ValidationError, match="unknown protocol"):
        build(debate={"protocol": "nonexistent", "temperature": 1.0})


def test_a_config_must_name_its_protocol():
    with pytest.raises(ValidationError, match="debate"):
        RunConfig.model_validate({k: v for k, v in BASE.items() if k != "debate"})


def test_a_run_cannot_be_narrowed_to_a_subset():
    # A run is the full benchmark. Topic shards are cut from the stored `subject`
    # column by analysis, not by narrowing what was executed -- otherwise a condition's
    # ECE is over whichever questions someone chose that day.
    with pytest.raises(ValidationError, match="Extra inputs"):
        build(benchmark={"name": "mmlu", "limit": 100})
    with pytest.raises(ValidationError, match="Extra inputs"):
        build(benchmark={"name": "mmlu", "subjects": ["anatomy"]})


def test_a_subsample_differs_from_the_same_size_subsample_under_another_seed():
    first = build(benchmark={"name": "mmlu", "n_questions": 500})
    second = build(benchmark={"name": "mmlu", "n_questions": 500, "sample_seed": 1})
    assert first.config_hash(SHA, CODE) != second.config_hash(SHA, CODE)


def test_a_subsample_must_hold_at_least_one_question():
    with pytest.raises(ValidationError, match="n_questions"):
        build(benchmark={"name": "mmlu", "n_questions": 0})


CALIBRATED = {"name": "calibrated_transformers", "model_id": "Qwen/Qwen2.5-7B-Instruct"}
DIGEST = "ab" * 32


def test_a_calibrated_head_without_a_vae_is_the_base_variant():
    assert build(model={**CALIBRATED, "variant": "base"}).model.variant == "base"
    with pytest.raises(ValidationError, match="variant"):
        build(model={**CALIBRATED, "variant": "vae"})


def test_a_vae_checkpoint_needs_its_sha256_and_the_vae_variant():
    vae = {**CALIBRATED, "variant": "vae", "vae_checkpoint": "/models/vae.pth"}
    assert build(model={**vae, "vae_sha256": DIGEST}).model.variant == "vae"
    with pytest.raises(ValidationError, match="together"):
        build(model=vae)
    with pytest.raises(ValidationError, match="variant"):
        build(model={**vae, "vae_sha256": DIGEST, "variant": "base"})


def test_the_vae_is_identified_by_its_sha256_not_its_path():
    vae = {**CALIBRATED, "variant": "vae", "vae_sha256": DIGEST}
    here = build(model={**vae, "vae_checkpoint": "/a/vae.pth"}).config_hash(SHA, CODE)
    there = build(model={**vae, "vae_checkpoint": "/b/vae.pth"}).config_hash(SHA, CODE)
    other = build(
        model={**vae, "vae_checkpoint": "/a/vae.pth", "vae_sha256": "cd" * 32}
    ).config_hash(SHA, CODE)
    assert here == there
    assert here != other


def test_the_temperature_moves_the_hash():
    base = {**CALIBRATED, "variant": "base"}
    assert build(model=base).config_hash(SHA, CODE) != build(
        model={**base, "temperature": 1.11}
    ).config_hash(SHA, CODE)


def test_the_nli_judge_checkpoint_moves_the_hash():
    semantic = build(debate=None, semantic={"n_samples": 10})
    first = semantic.config_hash(SHA, CODE, {"nli_sha": "1" * 40})
    second = semantic.config_hash(SHA, CODE, {"nli_sha": "2" * 40})
    assert first != second
    assert first != semantic.config_hash(SHA, CODE)


def test_the_sample_count_moves_the_hash():
    ten = build(debate=None, semantic={"n_samples": 10}).config_hash(SHA, CODE)
    twenty = build(debate=None, semantic={"n_samples": 20}).config_hash(SHA, CODE)
    assert ten != twenty


HARVEST = {
    "model": {"name": "transformers", "model_id": "Qwen/Qwen2.5-7B-Instruct", "variant": "base"},
    "train": {"name": "trivia_qa", "split": "train"},
    "evaluation": {"name": "trivia_qa", "n_questions": 2000},
    "output_root": "results/harvest",
}


def test_a_harvest_trains_and_evaluates_on_one_benchmark():
    with pytest.raises(ValidationError, match="same benchmark"):
        HarvestConfig.model_validate({**HARVEST, "evaluation": {"name": "mmlu"}})


def test_the_harvest_hash_moves_with_the_training_split_but_not_the_chunk_size():
    config = HarvestConfig.model_validate(HARVEST)
    other_split = HarvestConfig.model_validate(
        {**HARVEST, "train": {"name": "trivia_qa", "split": "validation"}}
    )
    rechunked = HarvestConfig.model_validate({**HARVEST, "chunk_size": 7})
    assert config.config_hash(SHA, CODE) != other_split.config_hash(SHA, CODE)
    assert config.config_hash(SHA, CODE) == rechunked.config_hash(SHA, CODE)


def test_the_fit_hash_ignores_where_the_cache_lives_but_not_the_seed():
    fit = {"cache": "a", "arm": {"name": "vae"}, "output_root": "fits"}
    moved = FitConfig.model_validate({**fit, "cache": "b", "device": "cpu"})
    reseeded = FitConfig.model_validate({**fit, "seed": 1})
    base = FitConfig.model_validate(fit).config_hash("cache", CODE)
    assert base == moved.config_hash("cache", CODE)
    assert base != reseeded.config_hash("cache", CODE)


def test_an_arm_refuses_a_setting_of_another_arm():
    with pytest.raises(ValidationError):
        FitConfig.model_validate(
            {"cache": "a", "arm": {"name": "probe", "latent_dim": 4}, "output_root": "fits"}
        )


ON_POLICY = {
    "model": {**CALIBRATED, "variant": "base"},
    "train": {"name": "trivia_qa", "split": "train"},
    "evaluation": {"name": "trivia_qa"},
    "arm": {"name": "vae"},
    "output_root": "results/on_policy",
}


def test_on_policy_training_starts_from_the_base_head():
    assert OnPolicyConfig.model_validate(ON_POLICY).arm.name == "vae"
    tempered = {**ON_POLICY, "model": {**ON_POLICY["model"], "temperature": 1.3}}
    with pytest.raises(ValidationError, match="base model"):
        OnPolicyConfig.model_validate(tempered)


@pytest.mark.parametrize("arm", [{"name": "platt"}, {"name": "probe"}])
def test_on_policy_training_refuses_an_arm_that_cannot_generate(arm):
    with pytest.raises(ValidationError):
        OnPolicyConfig.model_validate({**ON_POLICY, "arm": arm})


def test_on_policy_training_has_no_kl_term():
    with pytest.raises(ValidationError, match="kl_weight"):
        OnPolicyConfig.model_validate({**ON_POLICY, "arm": {"name": "vae", "kl_weight": 0.1}})


def test_the_on_policy_hash_moves_with_the_guardrail_but_not_the_batch_size():
    config = OnPolicyConfig.model_validate(ON_POLICY)
    looser = config.model_copy(update={"max_answer_changed": 0.3})
    batched = OnPolicyConfig.model_validate(
        {**ON_POLICY, "model": {**ON_POLICY["model"], "batch_size": 2}}
    )
    assert config.config_hash(SHA, CODE) != looser.config_hash(SHA, CODE)
    assert config.config_hash(SHA, CODE) == batched.config_hash(SHA, CODE)
