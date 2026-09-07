from __future__ import annotations

from modelsurgeon.evaluation import (
    AuditCell,
    AuditCellOutcome,
    AuditClaim,
    AuditMetric,
    AuditOutcome,
    AuditProtocol,
    FindingSeverity,
    audit_public_claims,
)


def _protocol(*, require_negative_result: bool = True) -> AuditProtocol:
    return AuditProtocol(
        "protocol-v1",
        ("cell-1", "cell-2"),
        ("quality",),
        ("source-v1",),
        require_negative_result=require_negative_result,
    )


def _measured(
    cell_id: str = "cell-1",
    *,
    protocol_revision: str = "protocol-v1",
    contamination_status: str = "clean",
    reproduction_status: str = "passed",
    quality: float = 0.9,
) -> AuditCell:
    return AuditCell(
        cell_id,
        protocol_revision,
        AuditCellOutcome.MEASURED,
        "a" * 64,
        ("source-v1", "config-v1"),
        (AuditMetric("quality", quality, quality - 0.05, quality + 0.05),),
        3,
        contamination_status,
        reproduction_status,
    )


def _negative() -> AuditCell:
    return AuditCell(
        "cell-2",
        "protocol-v1",
        AuditCellOutcome.NEGATIVE_RESULT,
        None,
        (),
        (),
        None,
        "unknown",
        "not_run",
        "negative result retained",
    )


def test_clean_audit_maps_claim_to_measured_confidence_bounded_cells() -> None:
    report = audit_public_claims(
        _protocol(),
        (_measured(), _negative()),
        (AuditClaim("claim-1", "quality", ("cell-1",), "higher", 0.85),),
    )

    assert report.outcome is AuditOutcome.CLEAN
    assert report.audit_id.startswith("public_audit_")
    assert report.audited_claim_ids == ("claim-1",)


def test_audit_detects_protocol_contamination_reproduction_and_missing_evidence() -> None:
    report = audit_public_claims(
        _protocol(),
        (
            _measured(
                protocol_revision="protocol-v2",
                contamination_status="contaminated",
                reproduction_status="failed",
            ),
        ),
        (AuditClaim("claim-1", "quality", ("cell-1", "cell-2"), "higher", 0.85),),
    )
    codes = {item.code for item in report.findings}

    assert report.outcome is AuditOutcome.FINDINGS
    assert {"protocol_drift", "contamination", "failed_reproduction"} <= codes
    assert "missing_cell" in codes
    assert "missing_negative_results" in codes
    assert any(item.severity is FindingSeverity.BLOCKING for item in report.findings)


def test_unsupported_cell_cannot_support_a_published_claim() -> None:
    unsupported = AuditCell(
        "cell-1",
        "protocol-v1",
        AuditCellOutcome.UNSUPPORTED,
        None,
        (),
        (),
        None,
        "unknown",
        "not_run",
        "runtime unavailable",
    )
    report = audit_public_claims(
        _protocol(require_negative_result=False),
        (unsupported, _negative()),
        (AuditClaim("claim-1", "quality", ("cell-1",), "higher", 0.85),),
    )

    assert any(item.code == "claim_unsupported_cell" for item in report.findings)
    assert report.outcome is AuditOutcome.FINDINGS
