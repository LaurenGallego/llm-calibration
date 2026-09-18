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

from madcal.debate import PRESETS, DebateConfig, preset
from madcal.storage import Variant

# Fields recorded per row and verified by analysis rather than hashed into the identity
# of the condition.
ENVIRONMENT_FIELDS = ("batch_size", "chunk_size", "device", "output_root")


class _Strict(BaseModel):
    # extra="forbid" is the point of using pydantic here: a typo in a YAML key must
    # fail, not fall through to a default.
    # protected_namespaces=() because pydantic otherwise refuses a field named
    # `model_id`, treating `model_` as reserved.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class StubModelConfig(_Strict):
    name: Literal["stub"] = "stub"
    model_id: Literal["stub"] = "stub"
    variant: Variant = Variant.STUB
    seed: int = 0
    n_choices: int = Field(default=4, ge=1)
    response_template: str = "Answer: {letter}"


class TransformersModelConfig(_Strict):
    name: Literal["transformers"] = "transformers"
    model_id: str
    variant: Variant
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


class MMLUBenchmarkConfig(_Strict):
    name: Literal["mmlu"] = "mmlu"
    revision: str | None = None
    # Local copies, for an offline node or a test fixture. Absent both, the hub is read.
    source: Path | None = None
    exemplar_source: Path | None = None


# One arm today; adding a benchmark adds an arm rather than widening a shared class.
# Everything specific to running a benchmark as published -- which split is canonical,
# where its few-shot pool comes from, how many shots it supports -- lives on the
# benchmark class, not here. There is deliberately no `limit` or subject filter: a run
# is the full benchmark, and topic shards are cut from the stored `subject` column by
# `analysis/`, not by narrowing what was executed.
BenchmarkConfig = Annotated[MMLUBenchmarkConfig, Field(discriminator="name")]


class PromptConfig(_Strict):
    name: str = "chat_template"
    n_shots: int = Field(default=0, ge=0)


class DebateSettings(_Strict):
    protocol: str
    temperature: float = Field(ge=0.0)
    n_agents: int | None = Field(default=None, ge=1)
    max_tokens: int = Field(default=512, ge=1)

    def resolve(self) -> DebateConfig:
        """Return the debate configuration this block names."""
        if self.protocol not in PRESETS:
            raise ValueError(f"unknown protocol {self.protocol!r}; known: {sorted(PRESETS)}")
        settings: dict[str, object] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.n_agents is not None:
            settings["n_agents"] = self.n_agents
        return preset(self.protocol, **settings)


class RunConfig(_Strict):
    model: ModelConfig
    benchmark: BenchmarkConfig = MMLUBenchmarkConfig()
    prompt: PromptConfig = PromptConfig()
    debate: DebateSettings
    seed: int = 0
    # Questions per stored Parquet chunk. Smaller means a walltime kill loses less;
    # larger means fewer files. It changes nothing about the numbers, so it is an
    # environment field and does not enter config_hash.
    chunk_size: int = Field(default=200, ge=1)
    output_root: Path

    @model_validator(mode="after")
    def _debate_is_buildable(self) -> Self:
        self.debate.resolve()
        return self

    def config_hash(self, checkpoint_sha: str) -> str:
        """Identity of the experimental condition, for the storage resume key.

        Takes the *resolved* checkpoint SHA rather than reading `revision` off the
        config: a run pinned to a branch name must not reuse rows produced before that
        branch moved. This is why the hash is computed after the adapter loads, not at
        parse time.
        """
        payload = self.model_dump(mode="json")
        for field in ENVIRONMENT_FIELDS:
            payload.pop(field, None)
            payload["model"].pop(field, None)
        payload["checkpoint_sha"] = checkpoint_sha
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    @classmethod
    def from_yaml(cls, path: Path) -> "RunConfig":
        import yaml

        with path.open() as handle:
            return cls.model_validate(yaml.safe_load(handle))
