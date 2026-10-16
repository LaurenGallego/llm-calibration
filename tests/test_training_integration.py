import json
from pathlib import Path

import pytest

from madcal.analysis.heads import ANSWER_CHANGED, head_rows, score_arms
from madcal.config import FitConfig, HarvestConfig
from madcal.orchestration.fit import fit
from madcal.orchestration.harvest import harvest
from madcal.storage.states import StateStore
from tests.models.test_transformers import TINY_MODEL, _is_cached

pytestmark = pytest.mark.skipif(not _is_cached(), reason=f"{TINY_MODEL} is not cached")

EVALUATION = Path(__file__).parent / "fixtures" / "trivia_qa_sample.jsonl"
ARMS = [
    {"name": "temperature", "grid_points": 5},
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
    fit_dirs = []
    for fit_config in fit_configs:
        result = fit(fit_config)
        record = json.loads((result.root / "fit.json").read_text())
        assert record["cache_hash"] == cache.manifest["config_hash"]
        fit_dirs.append(result.root)
    with pytest.raises(FileExistsError):
        fit(fit_configs[0])

    ids, correct, scored = score_arms(cache, fit_dirs, "cpu")
    rows = head_rows("trivia_qa", ids, correct, scored, n_resamples=20)
    changed = {row.arm: row.estimate for row in rows if row.metric == ANSWER_CHANGED}
    assert changed["base"] == 0.0
    assert changed["temperature/seed0"] == 0.0
    assert changed["hidden_temperature/seed0"] == 0.0
    assert {s.label for s in scored} == {"base", *(f"{a['name']}/seed0" for a in ARMS)}
    assert any(row.comparison == "vae/seed0 - temperature/seed0" for row in rows)


def test_a_fit_needs_a_complete_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        fit(
            FitConfig.model_validate(
                {"cache": str(tmp_path), "arm": {"name": "probe"}, "output_root": str(tmp_path)}
            )
        )
