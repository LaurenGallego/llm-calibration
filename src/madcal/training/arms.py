"""Answer-level calibration arms: each maps cached answer positions to a confidence in (0, 1)."""

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import nn

from madcal.models.vae_head import CalibratedHead, HiddenTemperatureHead, ResidualVAE
from madcal.registry import Registry
from madcal.storage.states import AnswerStates
from madcal.training.objective import answer_log_confidence, brier_loss

LOGIT_EPS = 1e-6


class OutputHead:
    """The frozen output head of the model the cache was harvested from."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, device: str) -> None:
        self.weight = weight.detach().to(device=device, dtype=torch.float32)
        self.bias = None if bias is None else bias.detach().to(device=device, dtype=torch.float32)

    def logits(self, states: torch.Tensor) -> torch.Tensor:
        """Logits of the head at each position."""
        return nn.functional.linear(states.float(), self.weight, self.bias)

    def base_log_confidence(self, answers: AnswerStates) -> torch.Tensor:
        """Log-confidence of the unmodified model in each answer."""
        return answer_log_confidence(self.logits(answers.states).log_softmax(dim=-1), answers)


class Arm(nn.Module):
    """A fitted map from answer positions to confidence."""

    head: OutputHead

    def confidence(self, answers: AnswerStates) -> torch.Tensor:
        """Confidence in each question's own answer."""
        raise NotImplementedError

    def log_probs(self, answers: AnswerStates) -> torch.Tensor | None:
        """Per-position log-probabilities over the vocabulary, for arms that change them."""
        return self.position_log_probs(answers.states)

    def position_log_probs(self, states: torch.Tensor) -> torch.Tensor | None:
        """Log-probabilities over the vocabulary at each head input, for arms that define them."""
        return None

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        log_probs = self.position_log_probs(states)
        if log_probs is None:
            raise TypeError(
                f"{type(self).__name__} defines no output distribution to generate from"
            )
        return log_probs

    def reseed(self, seed: int) -> None:
        """Restart the arm's noise stream at `seed`, for arms that draw noise."""

    def penalty(self, answers: AnswerStates) -> torch.Tensor:
        """Regulariser added to the Brier loss."""
        return torch.zeros((), device=self.head.weight.device)

    def fit_directly(
        self, answers: AnswerStates, batch_questions: int
    ) -> list[tuple[float, float]] | None:
        """Fit without gradients and return the search curve, or None for gradient arms."""
        return None

    def checkpoint(self) -> dict[str, torch.Tensor]:
        """Return the fitted parameters."""
        return {key: value.detach().cpu().clone() for key, value in self.state_dict().items()}

    def load_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        """Restore fitted parameters."""
        self.load_state_dict(state)

    def diagnostics(self, answers: AnswerStates) -> dict[str, float | None]:
        """Arm-specific quantities worth recording with a fit."""
        return {}


type ArmFactory = Callable[..., Arm]
arm_registry: Registry[ArmFactory] = Registry("arm")


def register_arm(name: str):
    return arm_registry.register(name)


def build_arm(name: str, head: OutputHead, hidden_size: int, seed: int, **settings: Any) -> Arm:
    """Construct the registered arm `name` on `head`'s device."""
    arm = arm_registry.get(name)(head=head, hidden_size=hidden_size, seed=seed, **settings)
    return arm.to(head.weight.device)


@register_arm("base")
class BaseArm(Arm):
    def __init__(self, head: OutputHead, hidden_size: int, seed: int) -> None:
        super().__init__()
        self.head = head

    def position_log_probs(self, states: torch.Tensor) -> torch.Tensor:
        return self.head.logits(states).log_softmax(dim=-1)

    def confidence(self, answers: AnswerStates) -> torch.Tensor:
        return answer_log_confidence(self.position_log_probs(answers.states), answers).exp()

    def fit_directly(
        self, answers: AnswerStates, batch_questions: int
    ) -> list[tuple[float, float]] | None:
        return []


@register_arm("temperature")
class TemperatureArm(Arm):
    log_temperature: torch.Tensor

    def __init__(
        self,
        head: OutputHead,
        hidden_size: int,
        seed: int,
        min_temperature: float = 0.25,
        max_temperature: float = 32.0,
        grid_points: int = 141,
    ) -> None:
        super().__init__()
        if not 0.0 < min_temperature < max_temperature:
            raise ValueError("the temperature grid needs 0 < min_temperature < max_temperature")
        self.head = head
        self.log_temperature = nn.Parameter(torch.zeros(()))
        self.grid = torch.linspace(
            math.log(min_temperature), math.log(max_temperature), grid_points
        )

    def position_log_probs(self, states: torch.Tensor) -> torch.Tensor:
        return (self.head.logits(states) / self.log_temperature.exp()).log_softmax(-1)

    def confidence(self, answers: AnswerStates) -> torch.Tensor:
        return answer_log_confidence(self.position_log_probs(answers.states), answers).exp()

    def fit_directly(
        self, answers: AnswerStates, batch_questions: int
    ) -> list[tuple[float, float]] | None:
        squared = torch.zeros(len(self.grid), device=self.head.weight.device)
        with torch.no_grad():
            for batch in _batches(answers, batch_questions):
                logits = self.head.logits(batch.states)
                correct = batch.correct.float()
                for index, log_t in enumerate(self.grid.tolist()):
                    log_probs = (logits / math.exp(log_t)).log_softmax(dim=-1)
                    confidence = answer_log_confidence(log_probs, batch).exp()
                    squared[index] += ((confidence - correct) ** 2).sum()
        brier = (squared / len(answers)).tolist()
        best = min(range(len(brier)), key=brier.__getitem__)
        if best in (0, len(brier) - 1):
            raise ValueError(
                f"the best temperature {math.exp(float(self.grid[best])):.3g} is at the edge "
                f"of the grid; widen min_temperature or max_temperature"
            )
        with torch.no_grad():
            self.log_temperature.fill_(float(self.grid[best]))
        return [
            (math.exp(log_t), value) for log_t, value in zip(self.grid.tolist(), brier, strict=True)
        ]

    def diagnostics(self, answers: AnswerStates) -> dict[str, float | None]:
        return {"temperature": float(self.log_temperature.exp())}


@register_arm("platt")
class PlattArm(Arm):
    def __init__(self, head: OutputHead, hidden_size: int, seed: int) -> None:
        super().__init__()
        self.head = head
        self.scale = nn.Parameter(torch.ones(()))
        self.shift = nn.Parameter(torch.zeros(()))

    def confidence(self, answers: AnswerStates) -> torch.Tensor:
        with torch.no_grad():
            base = torch.logit(self.head.base_log_confidence(answers).exp(), eps=LOGIT_EPS)
        return torch.sigmoid(self.scale * base + self.shift)

    def diagnostics(self, answers: AnswerStates) -> dict[str, float | None]:
        return {"scale": float(self.scale), "shift": float(self.shift)}


@register_arm("hidden_temperature")
class HiddenTemperatureArm(Arm):
    def __init__(self, head: OutputHead, hidden_size: int, seed: int) -> None:
        super().__init__()
        self.head = head
        self.calibrated = HiddenTemperatureHead(head.weight, head.bias)

    def temperatures(self, states: torch.Tensor) -> torch.Tensor:
        """Temperature at each position."""
        return self.calibrated.temperatures(states)

    def position_log_probs(self, states: torch.Tensor) -> torch.Tensor:
        return self.calibrated(states)

    def confidence(self, answers: AnswerStates) -> torch.Tensor:
        return answer_log_confidence(self.position_log_probs(answers.states), answers).exp()

    def checkpoint(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.calibrated.state_dict().items()}

    def load_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        self.calibrated.load_state_dict(state)

    def diagnostics(self, answers: AnswerStates) -> dict[str, float | None]:
        with torch.no_grad():
            temperatures = self.temperatures(answers.states)
        return {
            "temperature_mean": float(temperatures.mean()),
            "temperature_std": float(temperatures.std()) if len(temperatures) > 1 else 0.0,
        }


@register_arm("probe")
class ProbeArm(Arm):
    def __init__(self, head: OutputHead, hidden_size: int, seed: int) -> None:
        super().__init__()
        self.head = head
        self.probe = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.probe.weight)
        nn.init.zeros_(self.probe.bias)

    def confidence(self, answers: AnswerStates) -> torch.Tensor:
        last = answers.states[answers.last_positions()].float()
        return torch.sigmoid(self.probe(last)).squeeze(-1)


@register_arm("vae")
class VAEArm(Arm):
    def __init__(
        self,
        head: OutputHead,
        hidden_size: int,
        seed: int,
        latent_dim: int = 1024,
        n_layers: int = 1,
        mc_samples: int = 5,
        kl_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = head
        self.kl_weight = kl_weight
        self.vae = noise_only_vae(hidden_size, latent_dim, n_layers)
        self.calibrated = CalibratedHead(head.weight, head.bias, 1.0, self.vae, mc_samples, seed)

    def position_log_probs(self, states: torch.Tensor) -> torch.Tensor:
        return self.calibrated(states)

    def reseed(self, seed: int) -> None:
        self.calibrated.reseed(seed)

    def confidence(self, answers: AnswerStates) -> torch.Tensor:
        return answer_log_confidence(self.position_log_probs(answers.states), answers).exp()

    def penalty(self, answers: AnswerStates) -> torch.Tensor:
        if self.kl_weight == 0.0:
            return super().penalty(answers)
        _, scale = self.vae.moments(answers.states.float())
        variance = scale**2
        return self.kl_weight * 0.5 * (variance - 1.0 - variance.log()).sum(-1).mean()

    def checkpoint(self) -> dict[str, torch.Tensor]:
        return {key: value.detach().cpu().clone() for key, value in self.vae.state_dict().items()}

    def load_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        self.vae.load_state_dict(state)

    def diagnostics(self, answers: AnswerStates) -> dict[str, float | None]:
        with torch.no_grad():
            hidden = answers.states.float()
            mean, scale = self.vae.moments(hidden)
            shift = self.vae.dec(mean + scale * torch.randn_like(scale))
        ratio = shift.norm(dim=-1) / hidden.norm(dim=-1)
        return {
            "sigma_mean": float(scale.mean()),
            "sigma_std": float(scale.mean(-1).std()) if len(scale) > 1 else 0.0,
            "mu_abs_max": float(mean.abs().max()),
            "perturbation_ratio": float(ratio.mean()),
        }


def noise_only_vae(hidden_size: int, latent_dim: int, n_layers: int) -> ResidualVAE:
    """A VAE in the layout `load_vae` reads, with its latent mean and output bias held at zero."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    enc = _stack(hidden_size, [hidden_size] * (n_layers - 1), 2 * latent_dim)
    dec = _stack(latent_dim, [hidden_size] * (n_layers - 1), hidden_size)
    vae = ResidualVAE(enc, dec)
    encoder_out = _last_linear(enc)
    decoder_out = _last_linear(dec)
    for layer in (encoder_out, decoder_out):
        nn.init.zeros_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    mean_rows = torch.zeros(2 * latent_dim, 1)
    mean_rows[latent_dim:] = 1.0
    encoder_out.weight.register_hook(lambda grad: grad * mean_rows.to(grad.device))
    if encoder_out.bias is not None:
        encoder_out.bias.register_hook(lambda grad: grad * mean_rows[:, 0].to(grad.device))
    if decoder_out.bias is not None:
        decoder_out.bias.register_hook(torch.zeros_like)
    return vae


def _stack(width_in: int, hidden: Sequence[int], width_out: int) -> nn.Sequential:
    blocks: list[nn.Module] = []
    for width in hidden:
        blocks.append(
            nn.Sequential(nn.Linear(width_in, width), nn.LayerNorm(width), nn.LeakyReLU())
        )
        width_in = width
    blocks.append(nn.Sequential(nn.Linear(width_in, width_out)))
    return nn.Sequential(*blocks)


def _last_linear(module: nn.Module) -> nn.Linear:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)][-1]


def _batches(answers: AnswerStates, batch_questions: int):
    for start in range(0, len(answers), batch_questions):
        yield answers.select(range(start, min(start + batch_questions, len(answers))))


def batched_confidence(arm: Arm, answers: AnswerStates, batch_questions: int) -> torch.Tensor:
    """Confidence of every question, computed in batches without gradients."""
    with torch.no_grad():
        return torch.cat([arm.confidence(batch) for batch in _batches(answers, batch_questions)])


def batched_brier(arm: Arm, answers: AnswerStates, batch_questions: int) -> float:
    """Brier score of the arm over all of `answers`."""
    confidence = batched_confidence(arm, answers, batch_questions)
    return float(brier_loss(confidence, answers.correct))
