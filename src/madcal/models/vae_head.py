"""Output head applying a temperature and an optional residual VAE before the vocabulary."""

import math
from pathlib import Path

import torch
from torch import nn

HEAD_KEY = "model.lm_head.weight"
TEMPERATURE_KEY = "log_T"
MIN_TEMPERATURE = 0.05


class ResidualVAE(nn.Module):
    """Encoder and decoder of a calibration-repo VAE acting on the final hidden state."""

    def __init__(self, enc: nn.Sequential, dec: nn.Sequential) -> None:
        super().__init__()
        self.enc = enc
        self.dec = dec
        self.latent_dim = _linears(dec)[0].in_features
        self.hidden_dim = _linears(enc)[0].in_features
        if _linears(enc)[-1].out_features != 2 * self.latent_dim:
            raise ValueError("the encoder must output a mean and a log-variance per latent unit")
        if _linears(dec)[-1].out_features != self.hidden_dim:
            raise ValueError("the decoder must map back to the hidden size the encoder reads")

    def moments(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the posterior mean and standard deviation of the latent for each position."""
        stats = self.enc(hidden)
        mean, log_var = stats[..., : self.latent_dim], stats[..., self.latent_dim :]
        return mean, torch.exp(0.5 * log_var)


class CalibratedHead(nn.Module):
    """Replacement for `lm_head` returning log-probabilities averaged over VAE draws."""

    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        temperature: float,
        vae: ResidualVAE | None,
        mc_samples: int,
        noise_seed: int,
    ) -> None:
        super().__init__()
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        if mc_samples < 1:
            raise ValueError(f"mc_samples must be >= 1, got {mc_samples}")
        weight = weight.detach().float()
        self.register_buffer("weight", weight)
        self.register_buffer("bias", None if bias is None else bias.detach().float())
        if vae is not None and vae.hidden_dim != weight.shape[1]:
            raise ValueError(
                f"the VAE reads hidden size {vae.hidden_dim}, but the head reads {weight.shape[1]}"
            )
        self.temperature = temperature
        self.vae = None if vae is None else vae.float().to(weight.device).eval()
        self.mc_samples = mc_samples
        self.generator = torch.Generator(device=weight.device)
        self.generator.manual_seed(noise_seed)

    def reseed(self, seed: int) -> None:
        """Restart the VAE noise stream at `seed`."""
        self.generator.manual_seed(seed)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        if self.vae is None:
            return self._log_probs(hidden)
        mean, scale = self.vae.moments(hidden)
        total: torch.Tensor | None = None
        for _ in range(self.mc_samples):
            noise = torch.randn(
                mean.shape, generator=self.generator, device=mean.device, dtype=mean.dtype
            )
            log_probs = self._log_probs(hidden + self.vae.dec(mean + scale * noise))
            total = log_probs if total is None else torch.logaddexp(total, log_probs)
        assert total is not None
        return total - math.log(self.mc_samples)

    def _log_probs(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = nn.functional.linear(hidden, self.weight, self.bias)
        return torch.log_softmax(logits / self.temperature, dim=-1)


class HiddenTemperatureHead(nn.Module):
    """Output head dividing the logits at each position by a temperature read from its input."""

    weight: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None) -> None:
        super().__init__()
        self.register_buffer("weight", weight.detach().float(), persistent=False)
        self.register_buffer(
            "bias", None if bias is None else bias.detach().float(), persistent=False
        )
        self.temperature = nn.Linear(weight.shape[1], 1, device=weight.device)
        nn.init.zeros_(self.temperature.weight)
        nn.init.constant_(self.temperature.bias, math.log(math.expm1(1.0 - MIN_TEMPERATURE)))

    def temperatures(self, hidden: torch.Tensor) -> torch.Tensor:
        """Temperature at each position."""
        return nn.functional.softplus(self.temperature(hidden.float())) + MIN_TEMPERATURE

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = nn.functional.linear(hidden.float(), self.weight, self.bias)
        return torch.log_softmax(logits / self.temperatures(hidden), dim=-1)


def load_hidden_temperature(
    path: Path, weight: torch.Tensor, bias: torch.Tensor | None
) -> HiddenTemperatureHead:
    """Rebuild a fitted hidden-temperature head on the output head `weight` and `bias`."""
    head = HiddenTemperatureHead(weight, bias)
    head.load_state_dict(torch.load(path, map_location=weight.device, weights_only=True))
    return head.eval()


def load_vae(path: Path, head_weight: torch.Tensor) -> ResidualVAE:
    """Rebuild a calibration-repo VAE from its saved state dict, refusing a trained head."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    unknown = sorted(
        key
        for key in state
        if not key.startswith(("enc.", "dec.")) and key not in (HEAD_KEY, TEMPERATURE_KEY)
    )
    if unknown:
        raise ValueError(f"VAE checkpoint {path} holds keys this adapter does not know: {unknown}")
    if HEAD_KEY in state and not torch.allclose(
        state[HEAD_KEY].float(), head_weight.detach().float().cpu(), rtol=1e-2, atol=1e-3
    ):
        raise ValueError(
            f"VAE checkpoint {path} carries an lm_head that differs from the model's; "
            f"a VAE trained together with its head is not supported"
        )
    vae = ResidualVAE(_blocks(state, "enc"), _blocks(state, "dec"))
    vae.load_state_dict(
        {key: value for key, value in state.items() if key.startswith(("enc.", "dec."))},
        strict=True,
    )
    return vae.float().eval()


def _blocks(state: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    blocks = []
    index = 0
    while f"{prefix}.{index}.0.weight" in state:
        weight = state[f"{prefix}.{index}.0.weight"]
        layers: list[nn.Module] = [
            nn.Linear(weight.shape[1], weight.shape[0], bias=f"{prefix}.{index}.0.bias" in state)
        ]
        if f"{prefix}.{index}.1.weight" in state:
            layers += [nn.LayerNorm(weight.shape[0]), nn.LeakyReLU()]
        blocks.append(nn.Sequential(*layers))
        index += 1
    if not blocks:
        raise ValueError(f"VAE checkpoint has no {prefix!r} layers")
    return nn.Sequential(*blocks)


def _linears(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
