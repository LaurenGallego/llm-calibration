import json
from pathlib import Path

import pytest
import torch

from madcal.analysis.heads import ANSWER_CHANGED, head_rows, score_arms
from madcal.config import FitConfig, HarvestConfig
from madcal.orchestration.fit import fit
from madcal.orchestration.harvest import harvest
from madcal.storage.states import StateStore
from tests.models.test_transformers import TINY_MODEL, _is_cached

pytestmark = pytest.mark.skipif(not _is_cached(), reason=f"{TINY_MODEL} is not cached")

EVALUATION = Path(__file__).parent / "fixtures" / "trivia_qa_sample.jsonl"
ARMS = [
    {"name": "platt"},
    {"name": "hidden_temperature"},
    {"name": "probe"},
    {"name": "vae", "latent_dim": 2, "mc_samples": 2},
]


def train_file(tmp_path: Path) -> Path:
    rows = [
        {
            "question_id": f"tr_{i}",
            "question": f"What is {i} plus {i}?",
            "answer": {"value": str(2 * i), "aliases": [str(2 * i)]},
        }
        for i in range(24)
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def test_harvest_fit_and_compare_run_end_to_end_on_a_real_model(tmp_path):
    config = HarvestConfig.model_validate(
        {
            "model": {
                "name": "transformers",
                "model_id": TINY_MODEL,
                "variant": "base",
                "dtype": "float32",
                "device": "cpu",
                "batch_size": 4,
            },
            "train": {"name": "trivia_qa", "split": "train", "source": str(train_file(tmp_path))},
            "evaluation": {"name": "trivia_qa", "source": str(EVALUATION)},
            "prompt": {"name": "plain_chat"},
            "max_tokens": 4,
            "chunk_size": 10,
            "output_root": str(tmp_path / "caches"),
        }
    )
    outcome = harvest(config)
    assert outcome.n_written == 29 and outcome.n_skipped == 0
    assert harvest(config).n_written == 0
    cache = StateStore(outcome.root).load()
    assert len(cache.answers) + len(cache.nulls) == 29
    assert cache.manifest["n_evaluation"] == 5

    fit_configs = [
        FitConfig.model_validate(
            {
                "cache": str(outcome.root),
                "arm": arm,
                "selection_fraction": 0.5,
                "steps": 4,
                "eval_every": 2,
                "batch_questions": 4,
                "device": "cpu",
                "output_root": str(tmp_path / "fits"),
            }
        )
        for arm in ARMS
    ]
    assert not cache.answers.with_role("train").correct.any()
    never_correct = FitConfig.model_validate(
        {**fit_configs[0].model_dump(), "arm": {"name": "temperature"}}
    )
    with pytest.raises(ValueError, match="edge of the grid"):
        fit(never_correct)
    fit_dirs = []
    for fit_config in fit_configs:
        result = fit(fit_config)
        record = json.loads((result.root / "fit.json").read_text())
        assert record["cache_hash"] == cache.manifest["config_hash"]
        fit_dirs.append(result.root)
    with pytest.raises(FileExistsError):
        fit(fit_configs[0])

    gold = harvest(config.model_copy(update={"gold_targets": True}))
    gold_cache = StateStore(gold.root).load()
    greedy_train = cache.answers.with_role("train")
    gold_train = gold_cache.answers.with_role("train")
    assert gold_train.question_ids == greedy_train.question_ids
    assert not torch.equal(gold_train.targets, greedy_train.targets)
    assert torch.equal(
        gold_cache.answers.with_role("evaluation").targets,
        cache.answers.with_role("evaluation").targets,
    )
    token_nll = FitConfig.model_validate(
        {**fit_configs[3].model_dump(), "loss": "token_nll", "target_cache": str(gold.root)}
    )
    result = fit(token_nll)
    record = json.loads((result.root / "fit.json").read_text())
    assert record["cache_hash"] == cache.manifest["config_hash"]
    assert record["target_cache_hash"] == gold_cache.manifest["config_hash"]
    fit_dirs.append(result.root)

    ids, correct, scored = score_arms(cache, fit_dirs, "cpu")
    rows = head_rows("trivia_qa", ids, correct, scored, n_resamples=20)
    changed = {row.arm: row.estimate for row in rows if row.metric == ANSWER_CHANGED}
    assert changed["base"] == 0.0
    assert changed["hidden_temperature/lr0.001/seed0"] == 0.0
    assert {s.label for s in scored} == {
        "base",
        "vae-token_nll/lr0.001/seed0",
        *(f"{a['name']}/lr0.001/seed0" for a in ARMS),
    }
    assert any(row.comparison == "vae/lr0.001/seed0 - base" for row in rows)


def test_target_cache_is_only_for_token_nll(tmp_path):
    with pytest.raises(ValueError, match="target_cache"):
        FitConfig.model_validate(
            {"target_cache": str(tmp_path), "arm": {"name": "vae"}, "output_root": str(tmp_path)}
        )


def test_a_fit_needs_a_complete_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        fit(
            FitConfig.model_validate(
                {"cache": str(tmp_path), "arm": {"name": "probe"}, "output_root": str(tmp_path)}
            )
        )
