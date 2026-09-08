"""Real-model coverage for the default optimize executor."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.config import CalibrationConfig, ConstraintConfig, ModelConfig, Settings
from modelsurgeon.first_party_optimize_runtime import build_first_party_optimize_runtime
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.optimization_orchestrator import (
    OptimizeInterrupted,
    OptimizeOrchestrator,
    OptimizeStage,
    WorkflowOutcome,
    WorkflowStatus,
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
        eos_token="[UNK]",
        pad_token="[UNK]",
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
        constraints=ConstraintConfig(min_quality_retention_ratio=0.95),
    )
    plan = build_optimize_plan(settings, preset="fast", quality_profile="fast", dry_run=False)
    plan = replace(
        plan,
        budget=replace(plan.budget, evaluations=2),
        quality_profile=replace(plan.quality_profile, max_perplexity_delta=1_000_000.0),
    )
    runtime = build_first_party_optimize_runtime(plan)
    run = OptimizeOrchestrator(plan, tmp_path / "run.json").run(
        runtime,
        approvals=tuple(item.code for item in plan.approvals if item.required),
    )
    assert run.outcome is WorkflowOutcome.SUPPORTED
    assert run.stages[0].result is not None
    assert run.stages[2].result is not None
    assert run.stages[2].result.measured
    assert run.accepted_artifact_digest is not None
    assert run.accepted_artifact_digest != run.source_artifact_digest
    assert (tmp_path / "artifacts" / "optimize" / run.run_id).is_dir()
    pareto = run.stages[-2].result
    assert pareto is not None
    assert "decode_tokens_per_second" in pareto.detail


def test_first_party_runtime_rehydrates_published_sequence_on_resume(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_llama(model_path)
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("a b c d e a b c d e a b c d e", encoding="utf-8")
    settings = Settings(
        artifact_dir=tmp_path / "artifacts",
        model=ModelConfig(path=str(model_path), revision="test-revision", dtype="fp32"),
        calibration=CalibrationConfig(
            dataset=str(calibration), samples=2, max_sequence_length=8, seed=7
        ),
        constraints=ConstraintConfig(min_quality_retention_ratio=0.95),
    )
    plan = build_optimize_plan(settings, preset="fast", quality_profile="fast", dry_run=False)
    plan = replace(
        plan,
        budget=replace(plan.budget, evaluations=2),
        quality_profile=replace(plan.quality_profile, max_perplexity_delta=1_000_000.0),
    )
    state = tmp_path / "run.json"
    first_party = build_first_party_optimize_runtime(plan)

    class PauseBeforeDeployment:
        def run_stage(self, context: object) -> object:
            if getattr(context, "stage", None) is OptimizeStage.DEPLOYMENT_BENCHMARK:
                raise OptimizeInterrupted()
            return first_party.run_stage(context)  # type: ignore[arg-type]

    approvals = tuple(item.code for item in plan.approvals if item.required)
    paused = OptimizeOrchestrator(plan, state).run(PauseBeforeDeployment(), approvals=approvals)
    assert paused.status is WorkflowStatus.PAUSED
    surgery = paused.stages[5].result
    assert surgery is not None and surgery.artifact_digest is not None
    assert len(json.loads(surgery.detail)["stages"]) == 2

    resumed = OptimizeOrchestrator(plan, state).run(
        build_first_party_optimize_runtime(plan), resume=True, approvals=approvals
    )
    assert resumed.outcome is WorkflowOutcome.SUPPORTED
    assert resumed.accepted_artifact_digest == surgery.artifact_digest
