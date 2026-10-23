import math
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from madcal.models import CalibratedTransformersAdapter, TransformersAdapter
from madcal.models.calibrated import file_sha256
from madcal.training import Arm, OutputHead, build_arm
from tests.models.test_transformers import CONTINUATIONS, PROMPT, TINY_MODEL, _is_cached

pytestmark = pytest.mark.skipif(not _is_cached(), reason=f"{TINY_MODEL} is not cached")


def calibrated(**settings) -> CalibratedTransformersAdapter:
    return CalibratedTransformersAdapter(
        model_id=TINY_MODEL, dtype="float32", device="cpu", batch_size=4, **settings
    )


def vae_checkpoint(path: Path, decoder_scale: float) -> tuple[Path, str]:
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32)
    hidden = model.config.n_embd
    generator = torch.Generator().manual_seed(0)
    state = {
        "enc.0.0.weight": torch.zeros(4, hidden),
        "enc.0.0.bias": torch.zeros(4),
        "dec.0.0.weight": decoder_scale * torch.randn(hidden, 2, generator=generator),
        "dec.0.0.bias": torch.zeros(hidden),
        "model.lm_head.weight": model.lm_head.weight.detach().clone(),
    }
    torch.save(state, path)
    return path, file_sha256(path)


def letter_logprobs(adapter: TransformersAdapter) -> list[float]:
    return list(adapter.score_choices([PROMPT], [CONTINUATIONS])[0].logprobs)


def test_an_identity_head_scores_like_the_plain_adapter():
    plain = TransformersAdapter(model_id=TINY_MODEL, dtype="float32", device="cpu")
    assert letter_logprobs(calibrated()) == pytest.approx(letter_logprobs(plain), abs=1e-5)


def test_the_temperature_divides_the_model_logits():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32).eval()
    prompt_ids = tokenizer(PROMPT)["input_ids"]
    with torch.no_grad():
        logits = model(torch.tensor([prompt_ids])).logits[0, -1].float()
    tempered = torch.log_softmax(logits / 2.0, dim=-1)
    expected = [
        float(tempered[tokenizer(PROMPT + letter)["input_ids"][-1]]) for letter in CONTINUATIONS
    ]
    assert letter_logprobs(calibrated(temperature=2.0)) == pytest.approx(expected, abs=1e-5)


def test_a_vae_with_a_zero_decoder_scores_like_no_vae(tmp_path):
    path, digest = vae_checkpoint(tmp_path / "vae.pth", decoder_scale=0.0)
    with_vae = calibrated(vae_checkpoint=path, vae_sha256=digest, mc_samples=3)
    assert letter_logprobs(with_vae) == pytest.approx(letter_logprobs(calibrated()), abs=1e-5)


def test_a_vae_that_moves_the_hidden_state_changes_the_scores(tmp_path):
    path, digest = vae_checkpoint(tmp_path / "vae.pth", decoder_scale=5.0)
    with_vae = calibrated(vae_checkpoint=path, vae_sha256=digest, mc_samples=3)
    moved = letter_logprobs(with_vae)
    assert all(math.isfinite(value) and value <= 0.0 for value in moved)
    assert moved != pytest.approx(letter_logprobs(calibrated()), abs=1e-3)


def test_seeded_generation_with_a_vae_reproduces(tmp_path):
    path, digest = vae_checkpoint(tmp_path / "vae.pth", decoder_scale=5.0)
    with_vae = calibrated(vae_checkpoint=path, vae_sha256=digest, mc_samples=2)

    def run() -> list[str]:
        return [sample.text for sample in with_vae.generate([PROMPT], 3, 1.0, 4, seed=7)[0]]

    assert run() == run()


def test_a_checkpoint_that_does_not_match_its_sha256_is_refused(tmp_path):
    path, _ = vae_checkpoint(tmp_path / "vae.pth", decoder_scale=1.0)
    with pytest.raises(ValueError, match="does not match"):
        calibrated(vae_checkpoint=path, vae_sha256="0" * 64)


def test_a_checkpoint_without_its_sha256_is_refused(tmp_path):
    path, _ = vae_checkpoint(tmp_path / "vae.pth", decoder_scale=1.0)
    with pytest.raises(ValueError, match="together"):
        calibrated(vae_checkpoint=path)


def test_generation_samples_from_the_calibrated_head():
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, dtype=torch.float32).eval()
    prompt_ids = tokenizer(PROMPT)["input_ids"]
    for sample in calibrated(temperature=2.0).generate([PROMPT], 2, 1.0, 4, seed=0)[0]:
        assert sample.tokens is not None and sample.token_logprobs is not None
        ids = prompt_ids + tokenizer.convert_tokens_to_ids(list(sample.tokens))
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0].float()
        tempered = torch.log_softmax(logits / 2.0, dim=-1)
        expected = [
            float(tempered[len(prompt_ids) + index - 1, token])
            for index, token in enumerate(ids[len(prompt_ids) :])
        ]
        assert sample.token_logprobs == pytest.approx(expected, abs=1e-4)


def trained_arm(adapter: CalibratedTransformersAdapter, name: str, **settings) -> Arm:
    weight, bias = adapter.output_head()
    arm = build_arm(name, OutputHead(weight, bias, "cpu"), int(weight.shape[1]), 0, **settings)
    generator = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for parameter in arm.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator))
    return arm


def sampled(adapter: CalibratedTransformersAdapter) -> list[tuple[str, tuple[float, ...] | None]]:
    return [(s.text, s.token_logprobs) for s in adapter.generate([PROMPT], 3, 1.0, 4, seed=7)[0]]


@pytest.mark.parametrize(
    ("name", "settings", "option"),
    [
        ("vae", {"latent_dim": 2, "mc_samples": 2}, "vae"),
        ("hidden_temperature", {}, "hidden_temperature"),
    ],
)
def test_an_installed_arm_generates_as_its_saved_checkpoint_does(tmp_path, name, settings, option):
    live = calibrated(mc_samples=2)
    arm = trained_arm(live, name, **settings)
    live.install_head(arm, arm.reseed)
    path = tmp_path / "head.pt"
    torch.save(arm.checkpoint(), path)
    saved = calibrated(
        mc_samples=2, **{f"{option}_checkpoint": path, f"{option}_sha256": file_sha256(path)}
    )
    for (text, logprobs), (saved_text, saved_logprobs) in zip(
        sampled(live), sampled(saved), strict=True
    ):
        assert text == saved_text
        assert logprobs == pytest.approx(saved_logprobs, abs=1e-5)
    assert sampled(live) != sampled(calibrated())
