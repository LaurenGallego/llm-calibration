"""Transformers adapter whose output head applies a temperature and, optionally, a trained VAE."""

import hashlib
from pathlib import Path

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
    ) -> None:
        if (vae_checkpoint is None) != (vae_sha256 is None):
            raise ValueError("a VAE checkpoint and its sha256 must be given together")
        if vae_checkpoint is not None and file_sha256(Path(vae_checkpoint)) != vae_sha256:
            raise ValueError(f"VAE checkpoint {vae_checkpoint} does not match sha256 {vae_sha256}")
        super().__init__(model_id, revision, dtype, device, batch_size)

        from madcal.models.vae_head import CalibratedHead, load_vae

        head = self._model.get_output_embeddings()
        if head is None:
            raise ValueError(f"{model_id!r} exposes no output head to calibrate")
        weight, bias = head.weight, getattr(head, "bias", None)
        vae = None if vae_checkpoint is None else load_vae(Path(vae_checkpoint), weight)
        self.temperature = temperature
        self.mc_samples = mc_samples
        self._head = CalibratedHead(weight, bias, temperature, vae, mc_samples, noise_seed)
        self._model.set_output_embeddings(self._head)

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
            self._head.reseed(seed + offset)
        return super()._generate_batch(prompts, n, temperature, max_tokens, seed, offset)
