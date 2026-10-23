import json

import pytest
import torch

from madcal.config import OnPolicyConfig
from madcal.models import CalibratedTransformersAdapter
from madcal.models.calibrated import file_sha256
from madcal.orchestration.on_policy import CHECKPOINT, RECORD, on_policy
from tests.models.test_transformers import PROMPT, TINY_MODEL, _is_cached
from tests.test_training_integration import EVALUATION, train_file

pytestmark = pytest.mark.skipif(not _is_cached(), reason=f"{TINY_MODEL} is not cached")


def config(tmp_path, arm: dict) -> OnPolicyConfig:
    return OnPolicyConfig.model_validate(
        {
            "model": {
                "name": "calibrated_transformers",
                "model_id": TINY_MODEL,
                "variant": "base",
                "dtype": "float32",
                "device": "cpu",
                "batch_size": 4,
            },
            "train": {"name": "trivia_qa", "split": "train", "source": str(train_file(tmp_path))},
            "evaluation": {"name": "trivia_qa", "source": str(EVALUATION)},
            "prompt": {"name": "plain_chat"},
            "semantic": {"n_samples": 3, "max_tokens": 4, "nli_model": None},
            "arm": arm,
            "n_selection": 6,
            "lr": 0.05,
            "steps": 3,
            "batch_questions": 4,
            "eval_every": 2,
            "output_root": str(tmp_path / "on_policy"),
        }
    )


@pytest.mark.parametrize(
    "arm", [{"name": "vae", "latent_dim": 2, "mc_samples": 2}, {"name": "temperature"}]
)
def test_on_policy_training_runs_end_to_end_and_its_head_loads_for_evaluation(tmp_path, arm):
    training = config(tmp_path, arm)
    outcome = on_policy(training)
    record = json.loads((outcome.root / RECORD).read_text())
    assert record["completed_at"] and record["n_selection"] == 6
    assert [entry["step"] for entry in record["evaluations"]] == [0, 2, 3]
    assert record["evaluations"][0]["passes_guardrail"]
    assert record["evaluations"][0]["answer_changed"] == 0.0
    assert len(record["steps"]) == 3
    assert record["checkpoint_sha256"] == file_sha256(outcome.root / CHECKPOINT)
    with pytest.raises(FileExistsError):
        on_policy(training)
    if arm["name"] == "vae":
        path = outcome.root / CHECKPOINT
        adapter = CalibratedTransformersAdapter(
            model_id=TINY_MODEL,
            dtype="float32",
            device="cpu",
            vae_checkpoint=path,
            vae_sha256=file_sha256(path),
            mc_samples=2,
        )
        assert adapter.generate([PROMPT], 1, 0.0, 2)[0][0].text is not None
    else:
        state = torch.load(outcome.root / CHECKPOINT, weights_only=True)
        assert sorted(state) == ["log_temperature"]


def same_length_parity(pairs):
    return [len(first) % 2 == len(second) % 2 for first, second in pairs]


def test_on_policy_training_moves_the_head_when_answers_share_clusters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "madcal.orchestration.on_policy.build_judge", lambda settings: same_length_parity
    )
    outcome = on_policy(config(tmp_path, {"name": "temperature"}))
    record = json.loads((outcome.root / RECORD).read_text())
    assert any(entry["loss"] > 0.0 for entry in record["steps"])
    briers = [entry["brier"] for entry in record["evaluations"]]
    assert len(set(briers)) > 1
    assert all(entry["answer_changed"] == 0.0 for entry in record["evaluations"])
    assert record["selection_brier"] == min(briers)
