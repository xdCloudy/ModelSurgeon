"""Contract tests for native GGUF first-party runtime selection."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from modelsurgeon.adapters import ModelFamily, ModelFormat
from modelsurgeon.adapters.gguf.container import GGML_TYPE_IDS, GGUFValueType
from modelsurgeon.adapters.gguf.disk import GGUFDiskEstimate, preflight_gguf_disk
from modelsurgeon.adapters.gguf.quantization import GGMLQuantizationType
from modelsurgeon.adapters.gguf.writer import (
    GGUFWriteMetadata,
    GGUFWriteTensor,
    plan_gguf_output,
    write_gguf_transactionally,
)
from modelsurgeon.config import (
    CalibrationConfig,
    ModelConfig,
    QuantizationConfig,
    RuntimeConfig,
    Settings,
)
from modelsurgeon.first_party_optimize_runtime import build_first_party_optimize_runtime
from modelsurgeon.native_gguf_optimize_runtime import NativeGGUFOptimizeRuntime
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.optimization_orchestrator import (
    OptimizeStage,
    StageContext,
    _new_run,
)


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
    assert (
        runtime._constraints_pass(
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
        )
        is False
    )


def _write_q8_gguf(path) -> None:
    metadata = (
        GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "llama"),
        GGUFWriteMetadata("general.quantization_version", GGUFValueType.UINT32, 2),
        GGUFWriteMetadata("general.alignment", GGUFValueType.UINT32, 32),
        GGUFWriteMetadata("general.file_type", GGUFValueType.UINT32, 7),
    )
    q8_type_id = next(
        key for key, value in GGML_TYPE_IDS.items() if value is GGMLQuantizationType.Q8_0
    )
    tensor = GGUFWriteTensor("test.weight", (256, 2), q8_type_id, (bytes(544),))
    layout = plan_gguf_output(metadata, (tensor,), alignment=32)
    disk_plan = preflight_gguf_disk(path, path.parent, GGUFDiskEstimate(layout.total_bytes, 0))
    write_gguf_transactionally(path, metadata, (tensor,), disk_plan, alignment=32)


def _write_activation_moments(path, source) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "calibration": {
                    "dataset": "fixture/dataset",
                    "dataset_revision": "fixture-r1",
                    "split": "train",
                    "text_sha256": "a" * 64,
                    "tokenizer": "fixture/model",
                    "tokenizer_revision": "fixture-r1",
                },
                "tensors": {
                    "test.weight": {
                        "input_axis": 0,
                        "sample_count": 4,
                        "second_moments": [1.0] * 256,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("candidate_perplexity", "expected_status"),
    ((100.0, "accepted"), (125.0, "rejected")),
)
def test_gguf_byte_budget_stage_uses_quality_gate_and_keeps_rejected_artifact(
    tmp_path, monkeypatch, candidate_perplexity: float, expected_status: str
) -> None:
    from modelsurgeon.adapters.gguf import open_gguf

    source = tmp_path / "source.gguf"
    _write_q8_gguf(source)
    settings = Settings(
        model=ModelConfig(
            path=str(source),
            revision="source-r1",
            format=ModelFormat.GGUF,
            family=ModelFamily.LLAMA,
        ),
        calibration=CalibrationConfig(dataset="unused.txt", dataset_revision="dataset-r1"),
        runtime=RuntimeConfig(
            llama_cli="unused-cli",
            llama_perplexity="unused-perplexity",
            llama_bench="unused-bench",
            expected_revision="runtime-r1",
        ),
        quantization=QuantizationConfig(
            method="gguf_byte_budget",
            target_file_bytes=source.stat().st_size - 1,
            target_type="Q3_K",
        ),
        artifact_dir=tmp_path / "artifacts",
    )
    plan = build_optimize_plan(settings, dry_run=False)
    runtime = NativeGGUFOptimizeRuntime(plan)
    original_digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    runtime.source_path = source
    runtime.discovery = SimpleNamespace(family=ModelFamily.LLAMA)
    runtime.artifact = source
    runtime.artifact_digest = original_digest
    runtime.selected = SimpleNamespace(
        candidate_id="candidate_fixture",
        mutation_id="mutation_fixture",
    )
    quality = {
        "baseline": {"perplexity": 100.0, "uncertainty": 0.0},
        "candidate": {"perplexity": candidate_perplexity, "uncertainty": 0.0},
    }
    monkeypatch.setattr(runtime, "_validate", lambda path: {"successful": True})
    monkeypatch.setattr(runtime, "_quality", lambda path: quality)
    monkeypatch.setattr(runtime, "_publish_evidence", lambda **kwargs: {"evidence_id": "e"})
    context = StageContext(
        run=_new_run(plan),
        plan=plan,
        stage=OptimizeStage.QUANTIZATION,
        completed_results={},
        overrides={},
    )

    result = runtime.run_stage(context)

    detail = json.loads(result.detail)
    assert detail["status"] == expected_status
    assert detail["quantization"]["candidate_bytes"] < source.stat().st_size
    quantized = list((tmp_path / "artifacts" / "optimize").rglob("*.gguf"))
    assert len(quantized) == 1
    with open_gguf(quantized[0]) as mapped:
        assert mapped.container.file_size == quantized[0].stat().st_size
        assert mapped.container.tensors[0].quant_type is GGMLQuantizationType.Q3_K
    if expected_status == "accepted":
        assert runtime.artifact == quantized[0]
        assert result.artifact_digest != original_digest
    else:
        assert runtime.artifact == source
        assert result.artifact_digest == original_digest


def test_gguf_byte_budget_stage_supports_source_bound_activation_weighting(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.gguf"
    _write_q8_gguf(source)
    activation_moments = tmp_path / "activation-moments.json"
    _write_activation_moments(activation_moments, source)
    settings = Settings(
        model=ModelConfig(
            path=str(source),
            revision="source-r1",
            format=ModelFormat.GGUF,
            family=ModelFamily.LLAMA,
        ),
        calibration=CalibrationConfig(dataset="unused.txt", dataset_revision="dataset-r1"),
        runtime=RuntimeConfig(
            llama_cli="unused-cli",
            llama_perplexity="unused-perplexity",
            llama_bench="unused-bench",
            expected_revision="runtime-r1",
        ),
        quantization=QuantizationConfig(
            method="gguf_byte_budget",
            target_file_bytes=source.stat().st_size - 1,
            target_type="Q3_K",
            loss_metric="activation_weighted_mse",
            activation_moments_path=activation_moments,
        ),
        artifact_dir=tmp_path / "artifacts",
    )
    plan = build_optimize_plan(settings, dry_run=False)
    runtime = NativeGGUFOptimizeRuntime(plan)
    source_digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    runtime.source_path = source
    runtime.discovery = SimpleNamespace(family=ModelFamily.LLAMA)
    runtime.artifact = source
    runtime.artifact_digest = source_digest
    runtime.selected = SimpleNamespace(
        candidate_id="candidate_activation_fixture",
        mutation_id="mutation_activation_fixture",
    )
    quality = {
        "baseline": {"perplexity": 100.0, "uncertainty": 0.0},
        "candidate": {"perplexity": 100.0, "uncertainty": 0.0},
    }
    monkeypatch.setattr(runtime, "_validate", lambda path: {"successful": True})
    monkeypatch.setattr(runtime, "_quality", lambda path: quality)
    monkeypatch.setattr(runtime, "_publish_evidence", lambda **kwargs: {"evidence_id": "e"})

    record = runtime._quantize_selected()

    assert record["status"] == "accepted"
    assert record["loss_metric"] == "activation_weighted_mse"
    assert record["loss_estimates"]["count"] == 1
    assert record["activation_moments"]["source_sha256"] == source_digest.removeprefix("sha256:")
    assert record["activation_moments"]["calibration"]["dataset"] == "fixture/dataset"
