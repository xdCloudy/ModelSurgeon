"""Contract tests for matched Wanda and SparseGPT baseline evidence."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.adapters import CompetitorBudget, ModelFormat
from modelsurgeon.evaluation.unstructured_baselines import (
    DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL,
    BaselineOutcome,
    CompatibilityOutcome,
    SparsityTarget,
    UnstructuredBaselineError,
    UnstructuredMethod,
    build_baseline_adapter,
    build_default_unstructured_baseline_protocol,
    render_unstructured_baseline_protocol,
)


def test_default_protocol_is_matched_and_retains_unsupported_cells() -> None:
    protocol = DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL

    assert len(protocol.models) == 2
    assert {model.family for model in protocol.models} == {"llama", "qwen"}
    assert {method.method for method in protocol.methods} == set(UnstructuredMethod)
    assert tuple(target.fraction for target in protocol.targets) == (0.2, 0.5)
    assert protocol.comparison_controls == ("magnitude_mean_absolute", "seeded_random")
    assert protocol.seeds == (11, 23, 47)
    assert len(protocol.cells) == 24
    assert all(cell.outcome is BaselineOutcome.UNSUPPORTED for cell in protocol.cells)
    assert all(metric.value is None for cell in protocol.cells for metric in cell.metrics)
    assert sum(
        item.outcome is CompatibilityOutcome.UNKNOWN for item in protocol.compatibility
    ) == 4
    assert sum(
        item.outcome is CompatibilityOutcome.UNSUPPORTED for item in protocol.compatibility
    ) == 4


def test_default_methods_are_revision_and_license_pinned() -> None:
    methods = {spec.method: spec for spec in DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL.methods}

    assert methods[UnstructuredMethod.WANDA].identity.revision == (
        "8e8fc87b4a2f9955baa7e76e64d5fce7fa8724a6"
    )
    assert methods[UnstructuredMethod.SPARSEGPT].identity.revision == (
        "147d2159dc4f3e9f73e47b32c04d7b3708f44436"
    )
    assert methods[UnstructuredMethod.WANDA].identity.license == "MIT"
    assert methods[UnstructuredMethod.SPARSEGPT].identity.license == "Apache-2.0"


def test_published_evidence_envelope_matches_protocol() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "research"
        / "v1.1-unstructured-baseline-evidence-v1.json"
    )
    evidence = json.loads(path.read_text(encoding="utf-8"))
    protocol = DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL

    assert evidence["protocol_id"] == protocol.protocol_id
    assert evidence["cell_count"] == len(protocol.cells)
    assert evidence["outcome_counts"] == {"unsupported": 24}
    assert evidence["performance_claim"] is None


def test_protocol_identity_is_deterministic_and_changes_with_target() -> None:
    protocol = build_default_unstructured_baseline_protocol()

    assert protocol.protocol_id == DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL.protocol_id
    assert protocol.protocol_id.startswith("baseline_protocol_")
    with pytest.raises(UnstructuredBaselineError, match="20% and 50%"):
        replace(protocol, targets=(SparsityTarget(0.2), SparsityTarget(0.4)))
    assert "protocol_id" in render_unstructured_baseline_protocol(protocol)


def test_structured_speedup_claim_is_rejected() -> None:
    spec = DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL.methods[0]

    with pytest.raises(UnstructuredBaselineError, match="structured deployment"):
        replace(spec, structured_speedup_claim=True)


def test_build_baseline_adapter_preserves_method_identity() -> None:
    spec = DEFAULT_UNSTRUCTURED_BASELINE_PROTOCOL.methods[0]
    adapter = build_baseline_adapter(
        spec,
        version_command=("python", "--version"),
        command_template=("python", "-c", "pass", "{source}", "{output}"),
        budget=CompetitorBudget(1.0, 4096, 4096, 4096, 4096),
    )

    assert adapter.identity == spec.identity
    assert adapter.model_format is ModelFormat.SAFETENSORS
    assert adapter.capability_decision(spec.operation)[0] is False
