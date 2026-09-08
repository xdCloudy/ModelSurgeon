"""Real-model coverage for the default optimize executor."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.config import CalibrationConfig, ModelConfig, Settings
from modelsurgeon.first_party_optimize_runtime import build_first_party_optimize_runtime
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.optimization_orchestrator import (
    OptimizeOrchestrator,
    WorkflowOutcome,
)

torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")


def _write_tiny_llama(path: Path) -> None:
    config = transformers.LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(path, safe_serialization=True)
    vocabulary = {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
    ).save_pretrained(path)


def test_default_optimize_runtime_publishes_reloadable_child(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_llama(model_path)
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("a b c d e a b c d e a b c d e", encoding="utf-8")
    settings = Settings(
        artifact_dir=tmp_path / "artifacts",
        model=ModelConfig(
            path=str(model_path),
            revision="test-revision",
            dtype="fp32",
        ),
        calibration=CalibrationConfig(
            dataset=str(calibration),
            samples=2,
            max_sequence_length=8,
            seed=7,
        ),
    )
    plan = build_optimize_plan(settings, preset="fast", quality_profile="fast", dry_run=False)
    runtime = build_first_party_optimize_runtime(plan)
    run = OptimizeOrchestrator(plan, tmp_path / "run.json").run(
        runtime,
        approvals=tuple(item.code for item in plan.approvals if item.required),
    )
    assert run.outcome in {WorkflowOutcome.SUPPORTED, WorkflowOutcome.UNSUPPORTED}
    assert run.stages[0].result is not None
    assert run.stages[2].result is not None
    assert run.stages[2].result.measured
    if run.outcome is WorkflowOutcome.SUPPORTED:
        assert run.accepted_artifact_digest is not None
        assert run.accepted_artifact_digest != run.source_artifact_digest
        assert (tmp_path / "artifacts" / "optimize" / run.run_id).is_dir()
