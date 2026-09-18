from pathlib import Path

import pytest
from pydantic import ValidationError

from madcal.benchmarks import MMLU
from madcal.config import RunConfig
from madcal.models import StubAdapter
from madcal.orchestration import build_adapter, build_benchmark, build_renderer
from madcal.prompts import PlainChat

FIXTURE = Path(__file__).parent.parent / "fixtures" / "mmlu_sample.jsonl"


def make_config(**overrides) -> RunConfig:
    payload = {
        "model": {"name": "stub", "response_template": "Answer: {letter}"},
        "benchmark": {"name": "mmlu", "source": str(FIXTURE)},
        "prompt": {"name": "plain_chat"},
        "debate": {"protocol": "vanilla", "temperature": 1.0},
        "output_root": "results",
    }
    return RunConfig.model_validate({**payload, **overrides})


def test_adapter_is_built_from_its_own_config_arm():
    adapter = build_adapter(make_config().model)
    assert isinstance(adapter, StubAdapter)
    assert adapter.model_id == "stub"
    assert adapter.response_template == "Answer: {letter}"


def test_benchmark_is_built_with_its_source():
    benchmark = build_benchmark(make_config().benchmark)
    assert isinstance(benchmark, MMLU)
    assert benchmark.load()[0].id == "mmlu/anatomy/test/0"


def test_renderer_is_built_by_name():
    config = make_config()
    renderer = build_renderer(config.prompt, build_adapter(config.model))
    assert isinstance(renderer, PlainChat)


def test_unknown_renderer_is_refused():
    config = make_config(prompt={"name": "nonexistent"})
    with pytest.raises(KeyError, match="unknown renderer"):
        build_renderer(config.prompt, build_adapter(config.model))


def test_a_setting_that_does_not_apply_to_the_adapter_is_refused():
    with pytest.raises(ValidationError):
        make_config(model={"name": "stub", "dtype": "float16"})


def test_debate_block_is_resolved_at_parse_time():
    with pytest.raises(ValidationError, match="identical"):
        make_config(debate={"protocol": "best_of_n", "temperature": 0.0, "n_agents": 3})


def test_a_preset_structural_field_cannot_be_overridden_from_config():
    with pytest.raises(ValidationError, match="fixes"):
        make_config(debate={"protocol": "vanilla", "temperature": 1.0, "n_agents": 5})
