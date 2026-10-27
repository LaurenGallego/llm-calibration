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
    for entry in [*record["evaluations"], *record["steps"]]:
        counts = entry["counts"]
        assert counts.get("single_cluster_wrong", 0) <= counts.get("single_cluster", 0)
        assert counts.get("unmatched_correct", 0) <= counts.get("unmatched", 0)
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


def test_micro_batches_train_the_head_as_one_batch_does(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "madcal.orchestration.on_policy.build_judge", lambda settings: same_length_parity
    )
    records = []
    for size in (1, 4):
        root = tmp_path / str(size)
        root.mkdir()
        training = config(root, {"name": "temperature"})
        update = {"micro_batch_questions": size, "kl_to_base": 0.5}
        outcome = on_policy(training.model_copy(update=update))
        records.append(json.loads((outcome.root / RECORD).read_text()))
    one, whole = records
    assert any(entry["kl_to_base"] > 0.0 for entry in whole["steps"][1:])
    for key, field in (("steps", "loss"), ("steps", "kl_to_base"), ("evaluations", "brier")):
        assert [entry.get(field) for entry in one[key]] == pytest.approx(
            [entry.get(field) for entry in whole[key]], abs=1e-6
        )


def test_a_warm_started_run_trains_with_both_fixes_and_records_where_it_started(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "madcal.orchestration.on_policy.build_judge", lambda settings: same_length_parity
    )
    first = on_policy(config(tmp_path, {"name": "temperature"}))
    path = first.root / CHECKPOINT
    update = {
        "init_checkpoint": path,
        "smoothing": 1.0,
        "policy_gradient": True,
        "kl_to_base": 0.5,
        "steps": 2,
    }
    outcome = on_policy(config(tmp_path, {"name": "temperature"}).model_copy(update=update))
    record = json.loads((outcome.root / RECORD).read_text())
    assert record["init_checkpoint_sha256"] == file_sha256(path)
    start = record["evaluations"][0]
    assert start["answer_changed_from_start"] == 0.0
    assert start["kl_to_base"] == pytest.approx(0.0, abs=1e-6)
    assert "brier_trained" in start and "answer_changed" in start
    assert all("baseline" in entry for entry in record["steps"])
    assert all(entry["loss"] == entry["loss"] for entry in record["steps"])


def test_an_evaluation_only_run_scores_step_zero(tmp_path):
    training = config(tmp_path, {"name": "temperature"}).model_copy(update={"steps": 0})
    record = json.loads((on_policy(training).root / RECORD).read_text())
    assert [entry["step"] for entry in record["evaluations"]] == [0]
    assert record["steps"] == [] and record["best_step"] == 0
