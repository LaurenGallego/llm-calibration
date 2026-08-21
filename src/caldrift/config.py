"""Experiment configuration: YAML in, validated objects out, one hash for the resume key.

Nothing downstream reads a raw dict. A YAML file is parsed into these models at the
boundary, and every value used later comes from a typed field -- so a misspelled key is
a validation error at submit time rather than a default silently taking effect on a
cluster three hours in.

`config_hash` follows the rule recorded in `DEVLOG.md`: **hash what defines the
experiment, record and verify what defines the environment.** Model identity, checkpoint
SHA, benchmark, dataset revision, subjects, protocol, shots, scoring mode, dtype,
sampling parameters and seed all enter the hash. `batch_size`, `device` and
`output_root` do not -- they are recorded per row instead, and `analysis/` asserts each
condition has one distinct value of each. Hashing them would make an ordinary throughput
tweak invalidate every stored row; ignoring them entirely would let a change go
unnoticed.
"""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from caldrift.storage import Stage

# Fields recorded per row and verified by analysis rather than hashed into the identity
# of the condition.
ENVIRONMENT_FIELDS = ("batch_size", "device", "output_root")


class _Strict(BaseModel):
    # extra="forbid" is the point of using pydantic here: a typo in a YAML key must
    # fail, not fall through to a default.
    # protected_namespaces=() because pydantic otherwise refuses a field named
    # `model_id`, treating `model_` as reserved.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class StubModelConfig(_Strict):
    name: Literal["stub"] = "stub"
    model_id: Literal["stub"] = "stub"
    stage: Stage = Stage.STUB
    seed: int = 0
    n_choices: int = Field(default=4, ge=1)
    response_template: str = "Answer: {letter}"


class TransformersModelConfig(_Strict):
    name: Literal["transformers"] = "transformers"
    model_id: str
    stage: Stage
    revision: str | None = None
    dtype: str = "bfloat16"
    device: str = "auto"
    batch_size: int = Field(default=8, ge=1)


# A discriminated union rather than one class with a passthrough dict: the adapters take
# genuinely different constructor arguments, and `extra="forbid"` on each arm is what
# makes `dtype: float16` on the stub an error instead of a silently ignored setting.
# This is the one place adding an adapter means editing a shared file -- acceptable
# because adapters are few, unlike signals, which stay purely registry-driven.
ModelConfig = Annotated[
    StubModelConfig | TransformersModelConfig,
    Field(discriminator="name"),
]


class BenchmarkConfig(_Strict):
    name: str = "mmlu"
    subjects: tuple[str, ...] | None = None
    split: str = "test"
    revision: str | None = None
    limit: int | None = Field(default=None, ge=1)
    sample_seed: int | None = None

    @model_validator(mode="after")
    def _sampling_is_reproducible(self) -> Self:
        # Without a seed, `limit` takes the first N in id order -- and MMLU ids sort by
        # subject, so an unseeded subset is alphabetically-early subjects rather than a
        # sample of the benchmark.
        if self.limit is not None and self.sample_seed is None:
            raise ValueError("benchmark.limit requires sample_seed; an unseeded subset is biased")
        return self


class PromptConfig(_Strict):
    name: str = "fewshot_completion"
    n_shots: int = Field(default=5, ge=0)


class RunConfig(_Strict):
    model: ModelConfig
    benchmark: BenchmarkConfig = BenchmarkConfig()
    prompt: PromptConfig = PromptConfig()
    scoring_mode: Literal["likelihood", "generative"] = "likelihood"
    temperature: float = Field(default=0.0, ge=0.0)
    n_samples: int = Field(default=1, ge=1)
    max_tokens: int = Field(default=512, ge=1)
    seed: int = 0
    # None means every registered signal. A list pins the set, so a run cannot silently
    # gain a signal because one was added to the registry between submissions.
    signals: tuple[str, ...] | None = None
    output_root: Path

    @model_validator(mode="after")
    def _sampling_is_coherent(self) -> Self:
        if self.temperature == 0.0 and self.n_samples > 1:
            raise ValueError(
                "temperature=0 is greedy and cannot produce n_samples>1 distinct samples; "
                "n identical answers would read as perfect agreement to a resampling signal"
            )
        if self.scoring_mode == "likelihood" and self.n_samples > 1:
            raise ValueError("n_samples>1 has no meaning on the likelihood path")
        return self

    def config_hash(self, checkpoint_sha: str) -> str:
        """Identity of the experimental condition, for the storage resume key.

        Takes the *resolved* checkpoint SHA rather than reading `revision` off the
        config: a run pinned to a branch name must not reuse rows produced before that
        branch moved. This is why the hash is computed after the adapter loads, not at
        parse time.
        """
        payload = self.model_dump(mode="json")
        payload.pop("output_root", None)
        for field in ENVIRONMENT_FIELDS:
            payload["model"].pop(field, None)
        payload["checkpoint_sha"] = checkpoint_sha
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        import yaml

        with path.open() as handle:
            return cls.model_validate(yaml.safe_load(handle))
