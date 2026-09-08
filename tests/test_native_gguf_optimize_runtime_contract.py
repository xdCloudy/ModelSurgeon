"""Contract tests for native GGUF first-party runtime selection."""

from types import SimpleNamespace

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


def test_native_runtime_enforces_declared_quality_ratio_and_unknown_resources() -> None:
    runtime = object.__new__(NativeGGUFOptimizeRuntime)
    runtime.plan = SimpleNamespace(
        resolved_config={
            "constraints": {
                "min_quality_retention_ratio": 0.99,
                "max_perplexity_delta": None,
                "max_ram_bytes": 1024,
                "max_vram_bytes": None,
                "max_disk_bytes": None,
                "min_latency_gain_ratio": None,
            }
        },
        quality_profile=SimpleNamespace(max_perplexity_delta=10.0),
    )

    gate = runtime._quality_gate(100.0, 101.1)
    assert gate["accepted"] is False
    assert runtime._constraints_pass(
        {
            "baseline_perplexity": 100.0,
            "candidate_perplexity": 101.1,
            "quality_gate": gate,
            "candidate_peak_ram_bytes": None,
            "candidate_peak_vram_bytes": None,
            "candidate_disk_bytes": 100,
            "baseline_median_seconds": 1.0,
            "candidate_median_seconds": 1.0,
        }
    ) is False
    good_gate = runtime._quality_gate(100.0, 99.0)
    assert runtime._constraints_pass(
        {
            "baseline_perplexity": 100.0,
            "candidate_perplexity": 99.0,
            "quality_gate": good_gate,
            "candidate_peak_ram_bytes": None,
            "candidate_peak_vram_bytes": None,
            "candidate_disk_bytes": 100,
            "baseline_median_seconds": 1.0,
            "candidate_median_seconds": 1.0,
        }
    ) is False
