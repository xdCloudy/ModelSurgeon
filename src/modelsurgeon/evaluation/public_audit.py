"""Read-only public claim auditing and falsification checks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

PUBLIC_AUDIT_SCHEMA_VERSION: Final[int] = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PublicAuditError(ValueError):
    """Raised when an audit input is malformed rather than merely defective."""


class AuditOutcome(StrEnum):
    CLEAN = "clean"
    FINDINGS = "findings"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKING = "blocking"


class AuditCellOutcome(StrEnum):
    MEASURED = "measured"
    NEGATIVE_RESULT = "negative_result"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PublicAuditError(f"{label} is required")
    return value


def _digest(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise PublicAuditError(f"{label} must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class AuditMetric:
    name: str
    value: float
    confidence_low: float | None = None
    confidence_high: float | None = None

    def __post_init__(self) -> None:
        _text(self.name, "audit metric name")
        if not math.isfinite(self.value):
            raise PublicAuditError("audit metrics must be finite")
        if (self.confidence_low is None) != (self.confidence_high is None):
            raise PublicAuditError("audit confidence bounds must be paired")
        low = self.confidence_low
        high = self.confidence_high
        if low is not None and high is not None and (
            not math.isfinite(low)
            or not math.isfinite(high)
            or low > self.value
            or self.value > high
        ):
            raise PublicAuditError("audit confidence bounds must contain the metric")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
        }


@dataclass(frozen=True, slots=True)
class AuditCell:
    cell_id: str
    protocol_revision: str
    outcome: AuditCellOutcome
    artifact_digest: str | None
    lineage: tuple[str, ...]
    metrics: tuple[AuditMetric, ...]
    repetitions: int | None
    contamination_status: str
    reproduction_status: str
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.cell_id, "audit cell ID")
        _text(self.protocol_revision, "audit cell protocol revision")
        names = tuple(item.name for item in self.metrics)
        if names != tuple(sorted(set(names))):
            raise PublicAuditError("audit cell metrics must be canonical")
        if self.artifact_digest is not None:
            _digest(self.artifact_digest, "audit artifact digest")
        if self.outcome is AuditCellOutcome.MEASURED:
            if self.artifact_digest is None or not self.lineage:
                raise PublicAuditError("measured cells require artifact and lineage")
            if self.repetitions is None or self.repetitions < 3:
                raise PublicAuditError("measured cells require at least three repetitions")
            if self.contamination_status not in {"clean", "unknown", "contaminated"}:
                raise PublicAuditError("contamination status is invalid")
        elif self.outcome in {AuditCellOutcome.UNSUPPORTED, AuditCellOutcome.FAILED}:
            if self.metrics or self.artifact_digest is not None:
                raise PublicAuditError("unsupported or failed cells cannot claim evidence")
            if not self.reason:
                raise PublicAuditError("terminal audit cells require a reason")
        if self.reproduction_status not in {"passed", "failed", "unknown", "not_run"}:
            raise PublicAuditError("reproduction status is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "protocol_revision": self.protocol_revision,
            "outcome": self.outcome.value,
            "artifact_digest": self.artifact_digest,
            "lineage": list(self.lineage),
            "metrics": [item.to_record() for item in self.metrics],
            "repetitions": self.repetitions,
            "contamination_status": self.contamination_status,
            "reproduction_status": self.reproduction_status,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AuditClaim:
    claim_id: str
    metric: str
    cell_ids: tuple[str, ...]
    direction: str
    threshold: float
    published: bool = True

    def __post_init__(self) -> None:
        _text(self.claim_id, "audit claim ID")
        _text(self.metric, "audit claim metric")
        if not self.cell_ids or self.cell_ids != tuple(sorted(set(self.cell_ids))):
            raise PublicAuditError("audit claim cells must be non-empty and canonical")
        if self.direction not in {"higher", "lower"} or not math.isfinite(self.threshold):
            raise PublicAuditError("audit claim direction or threshold is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "metric": self.metric,
            "cell_ids": list(self.cell_ids),
            "direction": self.direction,
            "threshold": self.threshold,
            "published": self.published,
        }


@dataclass(frozen=True, slots=True)
class AuditProtocol:
    protocol_revision: str
    expected_cell_ids: tuple[str, ...]
    declared_metrics: tuple[str, ...]
    source_revisions: tuple[str, ...]
    require_negative_result: bool = True
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        _text(self.protocol_revision, "audit protocol revision")
        if not self.expected_cell_ids or self.expected_cell_ids != tuple(
            sorted(set(self.expected_cell_ids))
        ):
            raise PublicAuditError("expected audit cells must be canonical")
        if not self.declared_metrics or self.declared_metrics != tuple(
            sorted(set(self.declared_metrics))
        ):
            raise PublicAuditError("declared audit metrics must be canonical")
        if not self.source_revisions or self.source_revisions != tuple(
            sorted(set(self.source_revisions))
        ):
            raise PublicAuditError("audit source revisions must be canonical")
        if not 0.5 < self.confidence_level < 1:
            raise PublicAuditError("audit confidence level must be between 0.5 and 1")

    def to_record(self) -> dict[str, object]:
        return {
            "protocol_revision": self.protocol_revision,
            "expected_cell_ids": list(self.expected_cell_ids),
            "declared_metrics": list(self.declared_metrics),
            "source_revisions": list(self.source_revisions),
            "require_negative_result": self.require_negative_result,
            "confidence_level": self.confidence_level,
        }


@dataclass(frozen=True, slots=True)
class AuditFinding:
    code: str
    severity: FindingSeverity
    path: str
    detail: str
    claim_impact: str

    def __post_init__(self) -> None:
        for label, value in (
            ("finding code", self.code),
            ("finding path", self.path),
            ("finding detail", self.detail),
            ("finding claim impact", self.claim_impact),
        ):
            _text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "path": self.path,
            "detail": self.detail,
            "claim_impact": self.claim_impact,
        }


@dataclass(frozen=True, slots=True)
class PublicAuditReport:
    protocol: AuditProtocol
    findings: tuple[AuditFinding, ...]
    checked_cell_ids: tuple[str, ...]
    audited_claim_ids: tuple[str, ...]
    limitations: tuple[str, ...]

    @property
    def outcome(self) -> AuditOutcome:
        if not self.checked_cell_ids:
            return AuditOutcome.UNKNOWN
        if any(item.severity is FindingSeverity.BLOCKING for item in self.findings):
            return AuditOutcome.FINDINGS
        if all(item.code == "unsupported_cell" for item in self.findings) and self.findings:
            return AuditOutcome.UNSUPPORTED
        return AuditOutcome.FINDINGS if self.findings else AuditOutcome.CLEAN

    @property
    def audit_id(self) -> str:
        payload = {
            "schema_version": PUBLIC_AUDIT_SCHEMA_VERSION,
            "protocol": self.protocol.to_record(),
            "findings": [item.to_record() for item in self.findings],
            "checked_cell_ids": list(self.checked_cell_ids),
            "audited_claim_ids": list(self.audited_claim_ids),
            "limitations": list(self.limitations),
        }
        return "public_audit_" + hashlib.sha256(_canonical(payload).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": PUBLIC_AUDIT_SCHEMA_VERSION,
            "audit_id": self.audit_id,
            "outcome": self.outcome.value,
            "protocol": self.protocol.to_record(),
            "findings": [item.to_record() for item in self.findings],
            "checked_cell_ids": list(self.checked_cell_ids),
            "audited_claim_ids": list(self.audited_claim_ids),
            "limitations": list(self.limitations),
        }


def _finding(
    findings: list[AuditFinding],
    code: str,
    severity: FindingSeverity,
    path: str,
    detail: str,
    impact: str,
) -> None:
    findings.append(AuditFinding(code, severity, path, detail, impact))


def audit_public_claims(
    protocol: AuditProtocol,
    cells: tuple[AuditCell, ...],
    claims: tuple[AuditClaim, ...],
) -> PublicAuditReport:
    """Audit claims without altering the supplied protocol, cells, or claims."""

    findings: list[AuditFinding] = []
    by_id = {cell.cell_id: cell for cell in cells}
    if len(by_id) != len(cells):
        _finding(
            findings,
            "duplicate_cell",
            FindingSeverity.BLOCKING,
            "cells",
            "cell IDs repeat",
            "claims cannot be attributed uniquely",
        )
    missing = sorted(set(protocol.expected_cell_ids) - set(by_id))
    for cell_id in missing:
        _finding(
            findings,
            "missing_cell",
            FindingSeverity.BLOCKING,
            f"cells.{cell_id}",
            "protocol cell is absent",
            "comparative coverage is incomplete",
        )
    unexpected = sorted(set(by_id) - set(protocol.expected_cell_ids))
    for cell_id in unexpected:
        _finding(
            findings,
            "unexpected_cell",
            FindingSeverity.WARNING,
            f"cells.{cell_id}",
            "cell is outside the declared protocol",
            "protocol drift requires review",
        )
    if (
        not any(cell.outcome is AuditCellOutcome.NEGATIVE_RESULT for cell in cells)
        and protocol.require_negative_result
    ):
        _finding(
            findings,
            "missing_negative_results",
            FindingSeverity.BLOCKING,
            "cells",
            "no negative-result cell is retained",
            "publication could omit failed or inconclusive evidence",
        )
    for cell in cells:
        if cell.protocol_revision != protocol.protocol_revision:
            _finding(
                findings,
                "protocol_drift",
                FindingSeverity.BLOCKING,
                f"cells.{cell.cell_id}.protocol_revision",
                "cell revision differs from protocol",
                "cell is not comparable",
            )
        if cell.outcome is AuditCellOutcome.MEASURED:
            if cell.contamination_status != "clean":
                _finding(
                    findings,
                    "contamination",
                    FindingSeverity.BLOCKING,
                    f"cells.{cell.cell_id}.contamination_status",
                    "contamination is not clean",
                    "measured claim is invalid",
                )
            if cell.reproduction_status != "passed":
                _finding(
                    findings,
                    "failed_reproduction",
                    FindingSeverity.BLOCKING,
                    f"cells.{cell.cell_id}.reproduction_status",
                    "independent reproduction did not pass",
                    "measured claim is not reproduced",
                )
            for metric in cell.metrics:
                if metric.name not in protocol.declared_metrics:
                    _finding(
                        findings,
                        "protocol_metric_drift",
                        FindingSeverity.ERROR,
                        f"cells.{cell.cell_id}.metrics.{metric.name}",
                        "metric is not declared",
                        "metric cannot support a published comparison",
                    )
        elif cell.outcome is AuditCellOutcome.UNKNOWN:
            _finding(
                findings,
                "unknown_cell",
                FindingSeverity.WARNING,
                f"cells.{cell.cell_id}",
                "cell outcome is unknown",
                "claim coverage is inconclusive",
            )
    audited_claim_ids: list[str] = []
    for claim in claims:
        if not claim.published:
            continue
        audited_claim_ids.append(claim.claim_id)
        if claim.metric not in protocol.declared_metrics:
            _finding(
                findings,
                "metric_cherry_pick",
                FindingSeverity.BLOCKING,
                f"claims.{claim.claim_id}.metric",
                "claim metric is absent from the protocol",
                "claim is not auditable",
            )
        values: list[float] = []
        for cell_id in claim.cell_ids:
            claim_cell = by_id.get(cell_id)
            if claim_cell is None:
                continue
            if claim_cell.outcome is not AuditCellOutcome.MEASURED:
                _finding(
                    findings,
                    "claim_unsupported_cell",
                    FindingSeverity.BLOCKING,
                    f"claims.{claim.claim_id}.cells.{cell_id}",
                    "claim references a non-measured cell",
                    "published comparative claim is unsupported",
                )
                continue
            claim_metric = next(
                (item for item in claim_cell.metrics if item.name == claim.metric), None
            )
            if claim_metric is None or claim_metric.confidence_low is None:
                _finding(
                    findings,
                    "missing_confidence",
                    FindingSeverity.BLOCKING,
                    f"cells.{cell_id}.metrics.{claim.metric}",
                    "claim metric lacks confidence bounds",
                    "claim is not confidence-bounded",
                )
                continue
            values.append(claim_metric.value)
        if values:
            aggregate = math.fsum(values) / len(values)
            alternative = sorted(values)[(len(values) - 1) // 2]
            supports_aggregate = (
                aggregate >= claim.threshold
                if claim.direction == "higher"
                else aggregate <= claim.threshold
            )
            supports_alternative = (
                alternative >= claim.threshold
                if claim.direction == "higher"
                else alternative <= claim.threshold
            )
            if supports_aggregate != supports_alternative:
                _finding(
                    findings,
                    "aggregation_sensitive",
                    FindingSeverity.ERROR,
                    f"claims.{claim.claim_id}",
                    "mean and median produce different directional decisions",
                    "claim depends on metric aggregation",
                )
            if not supports_aggregate:
                _finding(
                    findings,
                    "claim_not_supported",
                    FindingSeverity.BLOCKING,
                    f"claims.{claim.claim_id}",
                    "measured aggregate does not clear the threshold",
                    "published direction is falsified",
                )
    limitations = (
        (
            "Audit checks declared records and supplied artifacts only; "
            "it does not certify model safety."
        ),
    )
    return PublicAuditReport(
        protocol,
        tuple(sorted(findings, key=lambda item: (item.severity.value, item.code, item.path))),
        tuple(sorted(by_id)),
        tuple(sorted(audited_claim_ids)),
        limitations,
    )
