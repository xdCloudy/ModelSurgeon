"""Contract tests for physical structured-baseline matrices."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.adapters import ModelFormat
from modelsurgeon.evaluation.structured_baselines import (
    DEFAULT_STRUCTURED_BASELINE_PROTOCOL,
    ArtifactReconciliation,
    StructuredBaselineError,
    StructuredCompatibility,
    StructuredMethod,
    StructuredOutcome,
    build_default_structured_baseline_protocol,
    render_structured_baseline_protocol,
)


def test_default_protocol_retains_all_methods_targets_and_cells() -> None:
    protocol = DEFAULT_STRUCTURED_BASELINE_PROTOCOL

    assert {spec.method for spec in protocol.methods} == set(StructuredMethod)
    assert {model.family for model in protocol.models} == {"llama", "qwen"}
    assert tuple(target.fraction for target in protocol.targets) == (0.1, 0.2)
    assert protocol.seeds == (11, 23, 47)
    assert len(protocol.cells) == 192
    assert all(cell.outcome is StructuredOutcome.UNSUPPORTED for cell in protocol.cells)
    assert len(protocol.compatibility) == 64
    assert sum(
        item.outcome is StructuredCompatibility.UNKNOWN for item in protocol.compatibility
    ) == 11
    assert sum(
        item.outcome is StructuredCompatibility.UNSUPPORTED for item in protocol.compatibility
    ) == 53


def test_upstream_provenance_and_license_gap_are_explicit() -> None:
    methods = {spec.method: spec for spec in DEFAULT_STRUCTURED_BASELINE_PROTOCOL.methods}

    assert methods[StructuredMethod.LLM_PRUNER].identity.revision == (
        "128a07d977f9b205d60ab14cfbc6a78f8a8e39d2"
    )
    assert methods[StructuredMethod.SLICE_GPT].identity.revision == (
        "954700a71c599e2e5f67632865b288f007326241"
    )
    assert methods[StructuredMethod.SHORT_GPT].identity.revision == (
        "78d9615fdcae6d90368832bd0a86c49c323549b9"
    )
    assert methods[StructuredMethod.MINITRON].identity.license == "NO_DECLARED_LICENSE"
    assert methods[StructuredMethod.MINITRON].license_declared is False


def test_measured_artifact_requires_reload_and_generation_evidence() -> None:
    with pytest.raises(StructuredBaselineError, match="reload and generate"):
        ArtifactReconciliation(
            0.1,
            0.1,
            100,
            90,
            1000,
            900,
            "a" * 64,
            True,
            False,
        )


def test_measured_cell_can_be_created_only_with_complete_reconciliation() -> None:
    protocol = DEFAULT_STRUCTURED_BASELINE_PROTOCOL
    cell = protocol.cells[0]
    measured = replace(
        cell,
        outcome=StructuredOutcome.MEASURED,
        reconciliation=ArtifactReconciliation(
            cell.target.fraction,
            cell.target.fraction,
            100,
            90,
            1000,
            900,
            "a" * 64,
            True,
            True,
        ),
        runtime_seconds=1.0,
        gpu_seconds=0.5,
    )

    assert measured.cell_id == cell.cell_id
    assert measured.to_record()["outcome"] == "measured"


def test_protocol_identity_is_deterministic_and_published_evidence_matches() -> None:
    protocol = build_default_structured_baseline_protocol()
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "research"
        / "v1.1-structured-baseline-evidence-v1.json"
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert protocol.protocol_id == DEFAULT_STRUCTURED_BASELINE_PROTOCOL.protocol_id
    assert evidence["protocol_id"] == protocol.protocol_id
    assert "structured_protocol_" in render_structured_baseline_protocol(protocol)
    assert all(
        item.model_format is ModelFormat.SAFETENSORS
        for item in protocol.compatibility
        if item.outcome is StructuredCompatibility.UNKNOWN
    )
