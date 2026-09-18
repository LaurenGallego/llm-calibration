"""Config in, live components out."""

from madcal.benchmarks import Benchmark, benchmark_registry
from madcal.config import BenchmarkConfig, ModelConfig, PromptConfig
from madcal.debate import Renderer
from madcal.models import ModelAdapter, model_adapter_registry
from madcal.prompts import renderer_registry


def build_adapter(config: ModelConfig) -> ModelAdapter:
    """Construct the model adapter this config names."""
    adapter = model_adapter_registry.get(config.name)
    return adapter(**config.model_dump(exclude={"name", "variant"}))


def build_benchmark(config: BenchmarkConfig) -> Benchmark:
    """Construct the benchmark this config names."""
    benchmark = benchmark_registry.get(config.name)
    return benchmark(**config.model_dump(exclude={"name"}))


def build_renderer(config: PromptConfig, adapter: ModelAdapter) -> Renderer:
    """Construct the renderer this config names, bound to the adapter's checkpoint."""
    factory = renderer_registry.get(config.name)
    return factory(adapter.model_id, adapter.revision)
