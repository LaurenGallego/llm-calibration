import math
from pathlib import Path

import pytest
import torch
from torch import nn

from madcal.models.vae_head import CalibratedHead, ResidualVAE, load_vae

WEIGHT = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
HIDDEN = torch.tensor([[[math.log(2.0), 0.0]]])


def vae(mean: float, log_var: float, shift: tuple[float, float]) -> ResidualVAE:
    encoder, decoder = nn.Linear(2, 2), nn.Linear(1, 2)
    with torch.no_grad():
        encoder.weight.zero_()
        encoder.bias.copy_(torch.tensor([mean, log_var]))
        decoder.weight.copy_(torch.tensor([[shift[0]], [shift[1]]]))
        decoder.bias.zero_()
    return ResidualVAE(nn.Sequential(nn.Sequential(encoder)), nn.Sequential(nn.Sequential(decoder)))


def probabilities(head: CalibratedHead) -> list[float]:
    return head(HIDDEN).exp().flatten().tolist()


def test_without_a_vae_the_head_is_the_tempered_softmax():
    assert probabilities(CalibratedHead(WEIGHT, None, 1.0, None, 5, 0)) == pytest.approx(
        [0.4, 0.2, 0.4]
    )
    root = math.sqrt(2.0)
    assert probabilities(CalibratedHead(WEIGHT, None, 2.0, None, 5, 0)) == pytest.approx(
        [root / (1 + 2 * root), 1 / (1 + 2 * root), root / (1 + 2 * root)]
    )


def test_a_noiseless_vae_shifts_the_hidden_state_by_its_decoded_mean():
    head = CalibratedHead(WEIGHT, None, 1.0, vae(1.0, -200.0, (1.0, 0.0)), 3, 0)
    e = math.e
    assert probabilities(head) == pytest.approx(
        [2 * e / (1 + 4 * e), 1 / (1 + 4 * e), 2 * e / (1 + 4 * e)]
    )


def test_a_zero_decoder_leaves_the_head_unchanged():
    head = CalibratedHead(WEIGHT, None, 1.0, vae(0.0, 0.0, (0.0, 0.0)), 4, 0)
    assert probabilities(head) == pytest.approx([0.4, 0.2, 0.4])


def test_draws_are_averaged_in_probability_space():
    head = CalibratedHead(WEIGHT, None, 1.0, vae(0.0, 0.0, (1.0, 0.0)), 2, 0)
    replay = torch.Generator().manual_seed(0)
    draws = [torch.randn((1, 1, 1), generator=replay).item() for _ in range(2)]
    shifted = [HIDDEN + torch.tensor([draw, 0.0]) for draw in draws]
    per_draw = [torch.softmax(nn.functional.linear(h, WEIGHT), -1) for h in shifted]
    expected = ((per_draw[0] + per_draw[1]) / 2).flatten().tolist()
    logits = torch.stack([nn.functional.linear(h, WEIGHT) for h in shifted])
    logit_mean = torch.softmax(logits.mean(dim=0), -1).flatten()
    result = probabilities(head)
    assert result == pytest.approx(expected, abs=1e-6)
    assert result != pytest.approx(logit_mean.tolist(), abs=1e-4)


def test_reseeding_reproduces_the_noise_and_continuing_does_not():
    head = CalibratedHead(WEIGHT, None, 1.0, vae(0.0, 0.0, (1.0, 0.0)), 1, 0)
    head.reseed(3)
    first = probabilities(head)
    second = probabilities(head)
    head.reseed(3)
    assert probabilities(head) == pytest.approx(first)
    assert second != pytest.approx(first)


def test_a_non_positive_temperature_is_refused():
    with pytest.raises(ValueError, match="temperature"):
        CalibratedHead(WEIGHT, None, 0.0, None, 1, 0)


def valm_state(hidden: int, latent: int, layers: int) -> dict[str, torch.Tensor]:
    sizes = [hidden] + [hidden] * (layers - 1) + [latent]
    state: dict[str, torch.Tensor] = {}
    for index in range(layers):
        last = index == layers - 1
        out = 2 * sizes[index + 1] if last else sizes[index + 1]
        state[f"enc.{index}.0.weight"] = torch.randn(out, sizes[index])
        state[f"enc.{index}.0.bias"] = torch.randn(out)
        state[f"dec.{index}.0.weight"] = torch.randn(sizes[-index - 2], sizes[-index - 1])
        state[f"dec.{index}.0.bias"] = torch.randn(sizes[-index - 2])
        if not last:
            state[f"enc.{index}.1.weight"] = torch.ones(sizes[index + 1])
            state[f"enc.{index}.1.bias"] = torch.zeros(sizes[index + 1])
            state[f"dec.{index}.1.weight"] = torch.ones(sizes[-index - 2])
            state[f"dec.{index}.1.bias"] = torch.zeros(sizes[-index - 2])
    state["log_T"] = torch.tensor(math.log(1.11))
    return state


def save(path: Path, state: dict[str, torch.Tensor]) -> Path:
    torch.save(state, path)
    return path


@pytest.mark.parametrize("layers", [1, 2])
def test_a_calibration_repo_checkpoint_loads_with_its_weights(tmp_path, layers):
    state = valm_state(hidden=4, latent=3, layers=layers)
    state["model.lm_head.weight"] = torch.ones(5, 4)
    loaded = load_vae(save(tmp_path / "vae.pth", state), torch.ones(5, 4))
    assert loaded.latent_dim == 3
    assert loaded.hidden_dim == 4
    for key, value in loaded.state_dict().items():
        assert torch.equal(value, state[key])
    mean, scale = loaded.moments(torch.randn(2, 4))
    assert mean.shape == (2, 3) and scale.shape == (2, 3)


def test_a_checkpoint_with_a_trained_head_is_refused(tmp_path):
    state = valm_state(hidden=4, latent=3, layers=1)
    state["model.lm_head.weight"] = torch.full((5, 4), 2.0)
    with pytest.raises(ValueError, match="differs"):
        load_vae(save(tmp_path / "vae.pth", state), torch.ones(5, 4))


def test_a_checkpoint_with_unknown_keys_is_refused(tmp_path):
    state = valm_state(hidden=4, latent=3, layers=1)
    state["model.lm_head.bias"] = torch.zeros(5)
    with pytest.raises(ValueError, match="does not know"):
        load_vae(save(tmp_path / "vae.pth", state), torch.ones(5, 4))


def test_a_vae_for_another_hidden_size_is_refused(tmp_path):
    loaded = load_vae(save(tmp_path / "vae.pth", valm_state(4, 3, 1)), torch.ones(5, 4))
    with pytest.raises(ValueError, match="hidden size"):
        CalibratedHead(torch.ones(5, 6), None, 1.0, loaded, 1, 0)
