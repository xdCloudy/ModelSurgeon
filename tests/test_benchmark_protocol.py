"""Contract tests for the frozen competitive benchmark protocol."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.evaluation.benchmark_protocol import (
    DEFAULT_BENCHMARK_PROTOCOL,
    BenchmarkProtocolError,
    ContaminationLicenseAudit,
    LicenseDecision,
    ProtocolApplicability,
    ProtocolCell,
    ProtocolExclusion,
    StatisticalPlan,
    build_default_benchmark_protocol,
    render_benchmark_protocol,
)


def test_default_protocol_is_complete_and_content_addressed() -> None:
    protocol = DEFAULT_BENCHMARK_PROTOCOL

    assert protocol.protocol_id.startswith("protocol_")
    assert protocol.audit.audit_id.startswith("audit_")
    assert len(protocol.cells) == len(protocol.ladder.targets) * len(protocol.methods) * len(
        protocol.tasks
    )
    assert {cell.applicability for cell in protocol.cells} == {
        ProtocolApplicability.SUPPORTED,
        ProtocolApplicability.UNKNOWN,
    }
    assert all(cell.license_decision is LicenseDecision.ALLOWED for cell in protocol.cells)
    assert protocol.statistical_plan.seeds == (11, 23, 47)
    assert "three seeds" in protocol.statistical_plan.power_limitation
    assert "wide intervals" in protocol.statistical_plan.precision_limitation


def test_checked_in_manifest_envelope_matches_typed_protocol() -> None:
    path = Path(__file__).parents[1] / "docs" / "research" / "v1.1-benchmark-protocol-v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))

    protocol = DEFAULT_BENCHMARK_PROTOCOL
    assert manifest["protocol_id"] == protocol.protocol_id
    assert manifest["audit_id"] == protocol.audit.audit_id
    assert manifest["cell_count"] == len(protocol.cells)
    assert manifest["applicability_counts"] == {"supported": 14, "unknown": 84}


def test_default_protocol_retains_explicit_exclusion_and_exact_metric_contract() -> None:
    protocol = build_default_benchmark_protocol()

    exclusion = protocol.audit.exclusions[0]
    assert exclusion.decision is LicenseDecision.EXCLUDED
    assert "Gemma" in exclusion.reason or "license" in exclusion.reason
    assert {metric.category.value for metric in protocol.metrics} == {
        "quality",
        "artifact",
        "runtime",
        "cost",
        "reliability",
    }
    assert {metric.unit for metric in protocol.metrics} >= {
        "nats/token",
        "bytes",
        "seconds",
        "gpu-seconds",
        "proportion",
    }
    assert all(task.revision for task in protocol.tasks)
    assert all(task.contamination_gate for task in protocol.tasks)


def test_protocol_render_is_deterministic_and_preserves_negative_cells() -> None:
    protocol = build_default_benchmark_protocol()

    assert render_benchmark_protocol(protocol) == render_benchmark_protocol(protocol)
    markdown = render_benchmark_protocol(protocol, format="markdown")
    assert protocol.protocol_id in markdown
    assert protocol.audit.audit_id in markdown
    assert "unknown" in render_benchmark_protocol(protocol)


def test_protocol_identity_changes_when_a_seed_or_budget_changes() -> None:
    protocol = DEFAULT_BENCHMARK_PROTOCOL
    changed_plan = replace(
        protocol.statistical_plan,
        seeds=(11, 23, 53),
    )
    changed = replace(protocol, statistical_plan=changed_plan)
    assert changed.protocol_id != protocol.protocol_id


def test_supported_cell_requires_allowed_license() -> None:
    with pytest.raises(BenchmarkProtocolError, match="allowed license"):
        ProtocolCell(
            "100M",
            "reference_dense",
            "heldout_perplexity",
            ProtocolApplicability.SUPPORTED,
            LicenseDecision.EXCLUDED,
            "license excluded",
        )


def test_non_supported_cell_requires_reason() -> None:
    with pytest.raises(BenchmarkProtocolError, match="explanatory reason"):
        ProtocolCell(
            "100M",
            "reference_dense",
            "arc",
            ProtocolApplicability.UNKNOWN,
            LicenseDecision.ALLOWED,
            "short",
        )


def test_statistical_plan_requires_three_seeds_and_declared_limits() -> None:
    with pytest.raises(BenchmarkProtocolError, match="at least three"):
        StatisticalPlan(
            (1, 2),
            "model x task x seed",
            1000,
            0.95,
            ("model",),
            ("report",),
            "power limitation",
            "precision limitation",
        )


def test_audit_is_content_addressed_and_requires_exclusion() -> None:
    exclusion = ProtocolExclusion(
        "model",
        "revision",
        "license",
        LicenseDecision.EXCLUDED,
        "explicitly excluded by policy",
        "https://example.test/source",
    )
    audit = ContaminationLicenseAudit(
        DEFAULT_BENCHMARK_PROTOCOL.audit.checked_on,
        "auditor",
        "policy",
        ("revision",),
        (exclusion,),
    )
    changed = replace(audit, policy="changed policy")
    assert changed.audit_id != audit.audit_id
