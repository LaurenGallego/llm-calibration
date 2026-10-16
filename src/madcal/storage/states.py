"""Hidden-state cache: the output head's inputs at answer positions, with each answer's grade."""

import hashlib
import io
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

ROLES = ("train", "evaluation")
MANIFEST = "manifest.json"
HEAD = "head.pt"


@dataclass(frozen=True)
class AnswerStates:
    """Graded answers: head inputs and target tokens per position, and for MCQ the options."""

    question_ids: tuple[str, ...]
    roles: tuple[str, ...]
    correct: torch.Tensor
    states: torch.Tensor
    offsets: torch.Tensor
    targets: torch.Tensor
    options: torch.Tensor | None = None
    chosen: torch.Tensor | None = None

    def __post_init__(self) -> None:
        n = len(self.question_ids)
        if len(self.roles) != n or self.correct.shape != (n,) or self.offsets.shape != (n + 1,):
            raise ValueError("question ids, roles, correct and offsets must describe the same n")
        if any(role not in ROLES for role in self.roles):
            raise ValueError(f"roles must be in {ROLES}")
        if len(set(self.question_ids)) != n:
            raise ValueError("a question appears more than once")
        lengths = self.offsets.diff()
        if int(self.offsets[0]) != 0 or bool((lengths < 1).any()):
            raise ValueError("offsets must start at 0 and give every question a position")
        if int(self.offsets[-1]) != self.states.shape[0] or self.targets.shape != (
            self.states.shape[0],
        ):
            raise ValueError("offsets, states and targets disagree on the number of positions")
        if (self.options is None) != (self.chosen is None):
            raise ValueError("options and chosen must be given together")
        if self.options is not None and self.chosen is not None:
            if bool((lengths != 1).any()):
                raise ValueError("a multiple-choice answer has exactly one position")
            picked = self.options.gather(1, self.chosen[:, None]).squeeze(1)
            if not torch.equal(picked, self.targets):
                raise ValueError("each target must be the option at the chosen index")

    def __len__(self) -> int:
        return len(self.question_ids)

    @property
    def hidden_size(self) -> int:
        """Width of the cached head inputs."""
        return self.states.shape[1]

    def positions(self) -> torch.Tensor:
        """Index of the question each position belongs to."""
        return torch.repeat_interleave(
            torch.arange(len(self), device=self.offsets.device), self.offsets.diff()
        )

    def last_positions(self) -> torch.Tensor:
        """Index of each question's last answer position."""
        return self.offsets[1:] - 1

    def select(self, indices: Sequence[int] | torch.Tensor) -> "AnswerStates":
        """Return the questions at `indices`, in that order."""
        index = torch.as_tensor(indices, dtype=torch.long)
        starts, ends = self.offsets[:-1][index], self.offsets[1:][index]
        rows = (
            torch.cat([torch.arange(int(s), int(e)) for s, e in zip(starts, ends, strict=True)])
            if len(index)
            else torch.zeros(0, dtype=torch.long)
        )
        lengths = ends - starts
        return AnswerStates(
            question_ids=tuple(self.question_ids[i] for i in index.tolist()),
            roles=tuple(self.roles[i] for i in index.tolist()),
            correct=self.correct[index],
            states=self.states[rows],
            offsets=torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)]),
            targets=self.targets[rows],
            options=None if self.options is None else self.options[index],
            chosen=None if self.chosen is None else self.chosen[index],
        )

    def with_role(self, role: str) -> "AnswerStates":
        """Return the questions of one role."""
        return self.select([i for i, r in enumerate(self.roles) if r == role])

    def to(self, device: str | torch.device) -> "AnswerStates":
        """Return a copy whose tensors live on `device`."""
        return AnswerStates(
            question_ids=self.question_ids,
            roles=self.roles,
            correct=self.correct.to(device),
            states=self.states.to(device),
            offsets=self.offsets.to(device),
            targets=self.targets.to(device),
            options=None if self.options is None else self.options.to(device),
            chosen=None if self.chosen is None else self.chosen.to(device),
        )

    @staticmethod
    def concat(parts: Sequence["AnswerStates"]) -> "AnswerStates":
        """Join chunks of the same benchmark into one."""
        if not parts:
            raise ValueError("nothing to concatenate")
        if len({part.options is None for part in parts}) != 1:
            raise ValueError("cannot mix multiple-choice and open answers")
        lengths = torch.cat([part.offsets.diff() for part in parts])
        options = [part.options for part in parts if part.options is not None]
        chosen = [part.chosen for part in parts if part.chosen is not None]
        return AnswerStates(
            question_ids=tuple(q for part in parts for q in part.question_ids),
            roles=tuple(r for part in parts for r in part.roles),
            correct=torch.cat([part.correct for part in parts]),
            states=torch.cat([part.states for part in parts]),
            offsets=torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)]),
            targets=torch.cat([part.targets for part in parts]),
            options=torch.cat(options) if options else None,
            chosen=torch.cat(chosen) if chosen else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the fields as plain tensors and lists, for saving."""
        return {
            "question_ids": list(self.question_ids),
            "roles": list(self.roles),
            "correct": self.correct,
            "states": self.states,
            "offsets": self.offsets,
            "targets": self.targets,
            "options": self.options,
            "chosen": self.chosen,
        }

    @staticmethod
    def from_dict(record: dict[str, Any]) -> "AnswerStates":
        """Rebuild from `to_dict`'s output."""
        return AnswerStates(
            question_ids=tuple(record["question_ids"]),
            roles=tuple(record["roles"]),
            correct=record["correct"],
            states=record["states"],
            offsets=record["offsets"],
            targets=record["targets"],
            options=record["options"],
            chosen=record["chosen"],
        )


@dataclass(frozen=True, slots=True)
class NullAnswer:
    """A question whose answer could not be graded or located, with the reason."""

    question_id: str
    role: str
    reason: str


@dataclass(frozen=True)
class StateChunk:
    """One chunk of harvested questions: the graded answers and the nulls."""

    answers: AnswerStates | None
    nulls: tuple[NullAnswer, ...]

    def question_ids(self) -> set[str]:
        """Every question this chunk accounts for."""
        answered = set() if self.answers is None else set(self.answers.question_ids)
        return answered | {null.question_id for null in self.nulls}


@dataclass(frozen=True)
class StateCache:
    """A complete cache: its manifest, the output head and every chunk joined."""

    manifest: dict[str, Any]
    weight: torch.Tensor
    bias: torch.Tensor | None
    answers: AnswerStates
    nulls: tuple[NullAnswer, ...]


class StateStore:
    """Directory holding one cache: manifest, output head and per-run chunk files."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def begin(
        self,
        manifest: dict[str, Any],
        run_id: str,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> None:
        """Record that `run_id` writes into this cache, creating the cache on first use."""
        path = self.root / MANIFEST
        if path.exists():
            stored = self.manifest()
            if stored["config_hash"] != manifest["config_hash"]:
                raise ValueError(f"{self.root} holds cache {stored['config_hash']!r}, not ours")
            stored["runs"] = [*stored["runs"], run_id]
            stored["completed_at"] = None
            self._write_json(stored, path)
            return
        self.root.mkdir(parents=True, exist_ok=True)
        head = {"weight": weight.detach().cpu(), "bias": None if bias is None else bias.cpu()}
        sha = self._write_torch(head, self.root / HEAD)
        self._write_json(
            {**manifest, "head_sha256": sha, "runs": [run_id], "completed_at": None}, path
        )

    def manifest(self) -> dict[str, Any]:
        """Return the stored manifest."""
        with (self.root / MANIFEST).open() as handle:
            return json.load(handle)

    def stored_ids(self) -> set[str]:
        """Return every question already written, by any run."""
        stored: set[str] = set()
        for chunk in self._chunks():
            stored |= chunk.question_ids()
        return stored

    def write_chunk(self, run_id: str, seq: int, chunk: StateChunk) -> None:
        """Write one chunk under this run's name."""
        if seq < 0:
            raise ValueError(f"seq must be >= 0, got {seq}")
        if run_id not in self.manifest()["runs"]:
            raise ValueError(f"run {run_id!r} never began on this cache")
        record = {
            "answers": None if chunk.answers is None else chunk.answers.to_dict(),
            "nulls": [[n.question_id, n.role, n.reason] for n in chunk.nulls],
        }
        self._write_torch(record, self.root / f"{run_id}-{seq:04d}.pt")

    def finish(self, counts: dict[str, int]) -> None:
        """Stamp the cache as complete."""
        stored = self.manifest()
        stored.update(counts)
        stored["completed_at"] = datetime.now(UTC).isoformat()
        self._write_json(stored, self.root / MANIFEST)

    def load(self) -> StateCache:
        """Return the whole cache, refusing one that never finished."""
        if not (self.root / MANIFEST).exists():
            raise FileNotFoundError(f"no cache at {self.root}")
        manifest = self.manifest()
        if manifest["completed_at"] is None:
            raise ValueError(f"cache {self.root} is incomplete: no completed_at")
        head_bytes = (self.root / HEAD).read_bytes()
        if hashlib.sha256(head_bytes).hexdigest() != manifest["head_sha256"]:
            raise ValueError(f"{self.root / HEAD} does not match the manifest's sha256")
        head = torch.load(io.BytesIO(head_bytes), weights_only=True)
        chunks = self._chunks()
        parts = [chunk.answers for chunk in chunks if chunk.answers is not None]
        if not parts:
            raise ValueError(f"cache {self.root} holds no graded answers")
        return StateCache(
            manifest=manifest,
            weight=head["weight"],
            bias=head["bias"],
            answers=AnswerStates.concat(parts),
            nulls=tuple(null for chunk in chunks for null in chunk.nulls),
        )

    def _chunks(self) -> list[StateChunk]:
        chunks = []
        for path in sorted(self.root.glob("*-[0-9][0-9][0-9][0-9].pt")):
            record = torch.load(path, weights_only=True)
            answers = record["answers"]
            chunks.append(
                StateChunk(
                    answers=None if answers is None else AnswerStates.from_dict(answers),
                    nulls=tuple(NullAnswer(*null) for null in record["nulls"]),
                )
            )
        return chunks

    def _write_torch(self, value: Any, path: Path) -> str:
        buffer = io.BytesIO()
        torch.save(value, buffer)
        data = buffer.getvalue()
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(data)
        os.replace(temporary, path)
        return hashlib.sha256(data).hexdigest()

    def _write_json(self, value: dict[str, Any], path: Path) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
        os.replace(temporary, path)
