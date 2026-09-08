"""Real-model coverage for the default optimize executor."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.config import (
    CalibrationConfig,
    ConstraintConfig,
    ModelConfig,
    RepairConfig,
    SearchConfig,
    Settings,
    SurgeonConfig,
)
from modelsurgeon.experiments.hardware import (
    CPUInventory,
    CUDAInventory,
    DiskInventory,
    HardwareInventory,
    MemoryInventory,
    SoftwareInventory,
)
from modelsurgeon.first_party_optimize_runtime import (
    FirstPartyOptimizeRuntimeError,
    build_first_party_optimize_runtime,
)
from modelsurgeon.optimization import build_optimize_plan
from modelsurgeon.optimization_orchestrator import (
    OptimizeInterrupted,
    OptimizeOrchestrator,
    OptimizeStage,
    WorkflowOutcome,
    WorkflowStatus,
)
from modelsurgeon.surgeon import (
    DEFAULT_TARGET_SCHEMA,
    LinearConfig,
    LinearSurgeonModel,
    PretrainedSurgeonCard,
    PretrainedSurgeonRegistry,
    TrainingModelIdentity,
)
from modelsurgeon.surgeon.matrix import NumericPreprocessor, SurgeonPreprocessor

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
    vocabulary = {
        "[UNK]": 0,
        "a": 1,
        "b": 2,
        "c": 3,
        "d": 4,
        "e": 5,
        "f": 6,
        "g": 7,
        "h": 8,
        "i": 9,
        "j": 10,
        "k": 11,
        "l": 12,
        "m": 13,
        "n": 14,
        "o": 15,
    }
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
    profile_detail = json.loads(run.stages[0].result.detail)
    assert profile_detail["hardware_inventory"]["memory"]["total_bytes"] is not None
    repair_detail = json.loads(run.stages[6].result.detail)
    quantization_detail = json.loads(run.stages[7].result.detail)
    assert repair_detail["status"] == "not_requested"
    assert repair_detail["method"] == "none"
    assert quantization_detail["status"] == "unsupported"
    assert Path(quantization_detail["evidence"]["path"]).is_file()
    assert run.accepted_artifact_digest is not None
    assert run.accepted_artifact_digest != run.source_artifact_digest
    assert (tmp_path / "artifacts" / "optimize" / run.run_id).is_dir()
    pareto = run.stages[-2].result
    assert pareto is not None
    assert "decode_tokens_per_second" in pareto.detail
    pareto_detail = json.loads(pareto.detail)
    assert pareto_detail["measured_frontier"]["status"] == "measured"
    active_search = run.stages[4].result
    assert active_search is not None
    detail = json.loads(active_search.detail)
    frontier = detail["measured_frontier"]
    assert frontier["status"] == "measured"
    assert Path(frontier["archive_path"]).is_file()
    assert frontier["preferred_candidate_id"] == detail["candidate_id"]
    resource_preflight = detail["resource_preflight"]
    assert resource_preflight["status"] == "ready"
    assert resource_preflight["available_ram_bytes"] > 0
    assert resource_preflight["available_disk_bytes"] > 0
    assert frontier["frontier_candidate_ids"]
    assert detail["feature_evidence"]
    assert all(Path(item["cache"]["path"]).is_file() for item in detail["feature_evidence"])
    assert len(list(Path(detail["feature_cache_root"]).glob("*.json"))) >= len(
        detail["feature_evidence"]
    )
    assert len(detail["evidence_observations"]) == len(detail["feature_evidence"])
    assert all(
        Path(item["path"]).is_file()
        for item in detail["evidence_observations"]
    )
    measurements = [
        json.loads(Path(item["path"]).read_text(encoding="utf-8"))["measurement"]
        for item in detail["evidence_observations"]
    ]
    assert all(measurements)
    assert all(
        item["measurement_authority"] == "physical_model_evaluation"
        for item in measurements
    )
    assert all(item["parameter_delta"] < 0 for item in measurements)


@pytest.mark.parametrize(
    ("available_ram", "free_disk", "message"),
    (
        (None, 1 << 40, "available system memory is unknown"),
        (1 << 40, 1, "insufficient disk headroom"),
    ),
)
def test_first_party_resource_preflight_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    available_ram: int | None,
    free_disk: int,
    message: str,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_llama(model_path)
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("a b c d e a b c d e", encoding="utf-8")
    settings = Settings(
        artifact_dir=tmp_path / "artifacts",
        model=ModelConfig(path=str(model_path), revision="test-revision", dtype="fp32"),
        calibration=CalibrationConfig(
            dataset=str(calibration), samples=2, max_sequence_length=8, seed=7
        ),
        constraints=ConstraintConfig(min_quality_retention_ratio=0.95),
    )
    plan = build_optimize_plan(settings, preset="fast", quality_profile="fast", dry_run=False)
    runtime = build_first_party_optimize_runtime(plan)
    proof = runtime._ensure_loaded()  # type: ignore[attr-defined]
    inventory = HardwareInventory(
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 4),
        MemoryInventory(1 << 40, available_ram),
        DiskInventory(str(tmp_path), 1 << 40, free_disk),
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12", "CPython", "test", "test"),
    )
    monkeypatch.setattr(runtime, "_collect_runtime_hardware", lambda _path: inventory)
    with pytest.raises(FirstPartyOptimizeRuntimeError, match=message):
        runtime._preflight_resources(proof.model, tmp_path / "artifacts", phase="test")  # type: ignore[attr-defined]


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
    surgery_detail = json.loads(surgery.detail)
    assert len(surgery_detail["stages"]) == 2
    assert all(item["evaluation"]["accepted"] for item in surgery_detail["stages"])
    assert surgery_detail["failed_index"] is None
    assert len(surgery_detail["evidence_observations"]) == len(surgery_detail["stages"])
    assert all(
        Path(item["path"]).is_file()
        for item in surgery_detail["evidence_observations"]
    )
    assert all(
        json.loads(Path(item["path"]).read_text(encoding="utf-8"))["outcome"]
        == "accepted"
        for item in surgery_detail["evidence_observations"]
    )
    assert len(surgery_detail["state_updates"]) == 2
    assert surgery_detail["stages"][1]["mutation_id"].startswith("state-mlp-channel-")
    assert (
        surgery_detail["state_updates"][0]["next_edit"]["selection_authority"]
        == "reloaded_child_physical_evaluation"
    )
    assert surgery_detail["state_updates"][0]["candidate_measurements"]
    assert all(
        item["state_authority"] == "reloaded_child_runtime"
        for item in surgery_detail["state_updates"]
    )
    assert all(item["candidate_count"] > 0 for item in surgery_detail["state_updates"])
    assert {item["path"] for item in surgery_detail["artifact_manifest"]} >= {
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    }

    resumed = OptimizeOrchestrator(plan, state).run(
        build_first_party_optimize_runtime(plan), resume=True, approvals=approvals
    )
    assert resumed.outcome is WorkflowOutcome.SUPPORTED
    assert resumed.accepted_artifact_digest == surgery.artifact_digest


def test_first_party_runtime_executes_real_lora_repair_when_requested(tmp_path: Path) -> None:
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
        repair=RepairConfig(
            method="lora",
            target_modules=("model.layers.0.mlp.down_proj",),
            max_steps=1,
        ),
        constraints=ConstraintConfig(min_quality_retention_ratio=0.95),
    )
    plan = build_optimize_plan(settings, preset="fast", quality_profile="fast", dry_run=False)
    plan = replace(
        plan,
        budget=replace(plan.budget, evaluations=1, repair_steps=1),
        quality_profile=replace(plan.quality_profile, max_perplexity_delta=1_000_000.0),
    )
    run = OptimizeOrchestrator(plan, tmp_path / "run.json").run(
        build_first_party_optimize_runtime(plan),
        approvals=tuple(item.code for item in plan.approvals if item.required),
    )

    assert run.outcome is WorkflowOutcome.SUPPORTED
    surgery = run.stages[5].result
    repair = run.stages[6].result
    assert surgery is not None and repair is not None
    detail = json.loads(repair.detail)
    assert detail["method"] == "lora"
    assert detail["status"] in {"accepted", "rejected"}
    assert detail["repair"]["completed_steps"] == 1
    assert detail["repair"]["resource_use"]["wall_seconds"] >= 0
    assert detail["parent_measurement"]["perplexity"] > 0
    assert detail["repaired_measurement"]["perplexity"] > 0


def test_first_party_runtime_executes_real_distillation_repair_when_requested(
    tmp_path: Path,
) -> None:
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
        repair=RepairConfig(
            method="distillation",
            target_modules=("model.layers.0.mlp.down_proj",),
            max_steps=1,
        ),
        constraints=ConstraintConfig(min_quality_retention_ratio=0.95),
    )
    plan = build_optimize_plan(settings, preset="fast", quality_profile="fast", dry_run=False)
    plan = replace(
        plan,
        budget=replace(plan.budget, evaluations=1, repair_steps=1),
        quality_profile=replace(plan.quality_profile, max_perplexity_delta=1_000_000.0),
    )
    run = OptimizeOrchestrator(plan, tmp_path / "run.json").run(
        build_first_party_optimize_runtime(plan),
        approvals=tuple(item.code for item in plan.approvals if item.required),
    )

    assert run.outcome is WorkflowOutcome.SUPPORTED
    repair = run.stages[6].result
    assert repair is not None
    detail = json.loads(repair.detail)
    assert detail["method"] == "distillation"
    assert detail["status"] in {"accepted", "rejected"}
    assert detail["repair"]["completed_steps"] == 1
    assert detail["repair"]["teacher_source"] == "teacher_inference"
    assert detail["parent_measurement"]["perplexity"] > 0
    assert detail["repaired_measurement"]["perplexity"] > 0


@pytest.mark.parametrize(
    ("scope", "runtime_scope"),
    (("attention_head", "head"), ("transformer_layer", "layer")),
)
def test_first_party_runtime_executes_real_attention_and_layer_scopes(
    tmp_path: Path, scope: str, runtime_scope: str
) -> None:
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
        search=SearchConfig(scopes=(scope,)),
        constraints=ConstraintConfig(min_quality_retention_ratio=0.95),
    )
    plan = build_optimize_plan(settings, preset="fast", quality_profile="fast", dry_run=False)
    plan = replace(
        plan,
        budget=replace(plan.budget, evaluations=1),
        quality_profile=replace(plan.quality_profile, max_perplexity_delta=1_000_000.0),
    )
    run = OptimizeOrchestrator(plan, tmp_path / "run.json").run(
        build_first_party_optimize_runtime(plan),
        approvals=tuple(item.code for item in plan.approvals if item.required),
    )

    assert run.outcome is WorkflowOutcome.SUPPORTED
    active = run.stages[4].result
    surgery = run.stages[5].result
    assert active is not None and surgery is not None
    active_detail = json.loads(active.detail)
    surgery_detail = json.loads(surgery.detail)
    assert active_detail["candidate_scope"] == runtime_scope
    assert active_detail["measurement"]["measurement_authority"] == (
        "physical_model_evaluation"
    )
    assert active_detail["measurement"]["parameter_delta"] < 0
    assert surgery_detail["candidate_scopes"] == [runtime_scope]
    assert surgery_detail["stages"][0]["evaluation"]["accepted"] is True
    assert surgery.artifact_digest is not None


def test_first_party_runtime_uses_verified_meta_surgeon_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    _write_tiny_llama(model_path)
    calibration = tmp_path / "calibration.txt"
    calibration.write_text("a b c d e a b c d e a b c d e", encoding="utf-8")

    registry = PretrainedSurgeonRegistry(tmp_path / "surgeon-registry")
    preprocessor = SurgeonPreprocessor(
        (NumericPreprocessor("weight_l1_norm", 0.0, 1.0),), (), 1, 1
    )
    model = LinearSurgeonModel(
        preprocessor.output_feature_names,
        "perplexity",
        (-1.0, 0.0),
        0.0,
        LinearConfig(max_epochs=1),
        1,
    )
    training_model = TrainingModelIdentity("tiny/source", "source-revision")
    evidence = registry.bundles.artifacts.put_bytes(b"held-out-evidence")
    bundle = registry.bundles.publish(
        model,
        preprocessor,
        DEFAULT_TARGET_SCHEMA,
        training_models=(training_model,),
        metrics={"ranking": 0.5},
        split_manifest={"version": "test"},
        provenance={"test": True},
    )
    card = PretrainedSurgeonCard(
        str(bundle.metadata.digest),
        "meta-surgeon-test",
        "test-revision",
        (training_model,),
        (str(evidence.metadata.digest),),
        1,
        1,
        1,
        ("mlp_channel",),
        ("fp32",),
        ("cpu",),
        {"minimum_support_coverage": 1.0},
        {"model_families": ["llama"]},
        "Apache-2.0",
        (("ranking", 0.5),),
        ("test bundle is not a production-quality transfer claim",),
    )
    published = registry.publish(card, key_id="test-key", secret=b"test-secret")
    monkeypatch.setenv("MODELSURGEON_TEST_META_KEY", "test-secret")
    settings = Settings(
        artifact_dir=tmp_path / "artifacts",
        model=ModelConfig(path=str(model_path), revision="test-revision", dtype="fp32"),
        calibration=CalibrationConfig(
            dataset=str(calibration), samples=2, max_sequence_length=8, seed=7
        ),
        surgeon=SurgeonConfig(
            registry_root=registry.root,
            card_digest=str(published.artifact.metadata.digest),
            signing_env="MODELSURGEON_TEST_META_KEY",
        ),
        constraints=ConstraintConfig(min_quality_retention_ratio=0.95),
    )
    plan = build_optimize_plan(settings, preset="fast", quality_profile="fast", dry_run=False)
    plan = replace(
        plan,
        budget=replace(plan.budget, evaluations=2),
        quality_profile=replace(plan.quality_profile, max_perplexity_delta=1_000_000.0),
    )
    run = OptimizeOrchestrator(plan, tmp_path / "run.json").run(
        build_first_party_optimize_runtime(plan),
        approvals=tuple(item.code for item in plan.approvals if item.required),
    )
    assert run.outcome is WorkflowOutcome.SUPPORTED
    active = run.stages[4].result
    assert active is not None
    detail = json.loads(active.detail)
    assert detail["meta_guidance"]["status"] == "verified_bundle"
    assert detail["meta_guidance"]["measurement_authority"] == "physical_evaluation"
    assert {item["method"] for item in detail["search_comparison"]["baselines"]} == {
        "meta_surgeon",
        "magnitude",
        "random",
        "no_guidance",
    }
