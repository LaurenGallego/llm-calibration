"""Transformers adapter whose output head applies a temperature and, optionally, a trained VAE."""

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from madcal.models.base import Generation, register_model_adapter
from madcal.models.transformers import DEFAULT_DTYPE, TransformersAdapter


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of the file at `path`."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


@register_model_adapter("calibrated_transformers")
class CalibratedTransformersAdapter(TransformersAdapter):
    name = "calibrated_transformers"

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        dtype: str = DEFAULT_DTYPE,
        device: str = "auto",
        batch_size: int = 8,
        temperature: float = 1.0,
        vae_checkpoint: Path | None = None,
        vae_sha256: str | None = None,
        mc_samples: int = 5,
        noise_seed: int = 0,
        hidden_temperature_checkpoint: Path | None = None,
        hidden_temperature_sha256: str | None = None,
    ) -> None:
        _verify(vae_checkpoint, vae_sha256, "VAE")
        _verify(hidden_temperature_checkpoint, hidden_temperature_sha256, "hidden temperature")
        if vae_checkpoint is not None and hidden_temperature_checkpoint is not None:
            raise ValueError("a VAE and a hidden temperature cannot be applied together")
        super().__init__(model_id, revision, dtype, device, batch_size)

        from madcal.models.vae_head import CalibratedHead, load_hidden_temperature, load_vae

        head = self._model.get_output_embeddings()
        if head is None:
            raise ValueError(f"{model_id!r} exposes no output head to calibrate")
        weight, bias = head.weight, getattr(head, "bias", None)
        vae = None if vae_checkpoint is None else load_vae(Path(vae_checkpoint), weight)
        self.temperature = temperature
        self.mc_samples = mc_samples
        self._head = CalibratedHead(weight, bias, temperature, vae, mc_samples, noise_seed)
        self._model.set_output_embeddings(self._head)
        self._reseed: Callable[[int], None] = self._head.reseed
        if hidden_temperature_checkpoint is not None:
            if temperature != 1.0:
                raise ValueError("a hidden temperature replaces the global temperature")
            self.install_head(
                load_hidden_temperature(Path(hidden_temperature_checkpoint), weight, bias)
            )

    def install_head(self, head: Any, reseed: Callable[[int], None] | None = None) -> None:
        """Generate and score through `head`, a module mapping head inputs to log-probabilities."""
        self._model.set_output_embeddings(head)
        self._reseed = reseed if reseed is not None else _no_reseed

    def _generate_batch(
        self,
        prompts: list[str],
        n: int,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        offset: int,
    ) -> list[list[Generation]]:
        if seed is not None:
            self._reseed(seed + offset)
        return super()._generate_batch(prompts, n, temperature, max_tokens, seed, offset)


def _verify(checkpoint: Path | None, sha256: str | None, kind: str) -> None:
    if (checkpoint is None) != (sha256 is None):
        raise ValueError(f"a {kind} checkpoint and its sha256 must be given together")
    if checkpoint is not None and file_sha256(Path(checkpoint)) != sha256:
        raise ValueError(f"{kind} checkpoint {checkpoint} does not match sha256 {sha256}")


def _no_reseed(seed: int) -> None:
    return None
