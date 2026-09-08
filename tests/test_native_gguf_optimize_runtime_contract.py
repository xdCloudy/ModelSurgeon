"""Contract tests for native GGUF first-party runtime selection."""

from modelsurgeon.adapters import ModelFamily, ModelFormat
from modelsurgeon.config import CalibrationConfig, ModelConfig, RuntimeConfig, Settings
from modelsurgeon.first_party_optimize_runtime import build_first_party_optimize_runtime
from modelsurgeon.native_gguf_optimize_runtime import NativeGGUFOptimizeRuntime
from modelsurgeon.optimization import build_optimize_plan


def test_first_party_builder_selects_native_gguf_runtime() -> None:
    settings = Settings(
        model=ModelConfig(
            path="model.gguf",
            revision="sha256:model",
            format=ModelFormat.GGUF,
            family=ModelFamily.LLAMA,
        ),
        calibration=CalibrationConfig(dataset="calibration.txt", dataset_revision="dataset-v1"),
        runtime=RuntimeConfig(
            llama_cli="llama-cli",
            llama_perplexity="llama-perplexity",
            llama_bench="llama-bench",
            expected_revision="de8656bd9",
        ),
    )
    plan = build_optimize_plan(settings)
    runtime = build_first_party_optimize_runtime(plan)

    assert isinstance(runtime, NativeGGUFOptimizeRuntime)
