from __future__ import annotations

import pytest

from modelsurgeon.surgery import (
    ArtifactOutputMode,
    MetricDirection,
    RepairArtifact,
    RepairArtifactLineage,
    RepairBudget,
    RepairCost,
    RepairMethod,
    RepairMetricEvidence,
    RepairModelIdentity,
    RepairOutcomeError,
    RepairOutcomeManifest,
    RepairOutcomeStatus,
    RepairRequest,
    RepairResult,
    record_repair_result,
)


def _identity(name: str) -> RepairModelIdentity:
    return RepairModelIdentity(name, f"revision-{name}", "a" * 64, "data-v1")


def _request() -> RepairRequest:
    return RepairRequest(
        "request-1",
        "candidate-1",
        _identity("source"),
        _identity("candidate"),
        _identity("teacher"),
        RepairMethod.LOGIT_DISTILLATION,
        ArtifactOutputMode.SEPARATE_LORA,
        ("layers.0",),
        RepairBudget(100, 1_000, 60.0, 30.0, 1_000.0, 10_000),
        {"decision_id": "compatibility-1", "allowed": True},
        "heldout_perplexity",
        MetricDirection.LOWER,
        "merge_after_reload",
    )


def _lineage() -> RepairArtifactLineage:
    return RepairArtifactLineage(
        "b" * 64,
        "c" * 64,
        "d" * 64,
        "e" * 64,
        "data-v1",
        "a" * 64,
    )


def test_accepted_repair_reconciles_cost_evidence_and_lineage() -> None:
    request = _request()
    artifact = RepairArtifact(
        "adapter-1",
        "d" * 64,
        256,
        ArtifactOutputMode.SEPARATE_LORA,
        "b" * 64,
        True,
    )
    evidence = RepairMetricEvidence(
        "heldout_perplexity",
        MetricDirection.LOWER,
        10.0,
        9.7,
        0.2,
        0.2,
        0.4,
        ("model-a", "model-b"),
    )
    result = RepairResult(
        request,
        RepairOutcomeStatus.ACCEPTED,
        RepairCost(20, 800, 10.0, 5.0, 100.0, 256),
        (artifact,),
        _lineage(),
        (evidence,),
        False,
        True,
    )
    manifest = record_repair_result(RepairOutcomeManifest((request,)), result)
    assert manifest.complete
    assert result.to_record()["outcome_id"] == result.outcome_id


def test_rejected_repair_restores_candidate_and_cannot_become_parent() -> None:
    request = _request()
    result = RepairResult(
        request,
        RepairOutcomeStatus.REJECTED,
        RepairCost(5, 100, 2.0, 1.0, 10.0, 0),
        (),
        _lineage(),
        (),
        True,
        False,
        "held-out improvement was not measured",
    )
    assert not result.search_parent_eligible
    assert result.to_record()["status"] == "rejected"


def test_distillation_rejects_teacher_tokenizer_mismatch() -> None:
    with pytest.raises(RepairOutcomeError, match="tokenizer"):
        RepairRequest(
            "request-2",
            "candidate-2",
            _identity("source"),
            _identity("candidate"),
            RepairModelIdentity("teacher", "rev", "f" * 64, "data-v1"),
            RepairMethod.FEATURE_DISTILLATION,
            ArtifactOutputMode.MERGED_WEIGHTS,
            (),
            RepairBudget(10, 100, 10.0, 5.0, 100.0, 1_000),
            {"allowed": True},
            "quality",
            MetricDirection.HIGHER,
            "after_repair",
        )
