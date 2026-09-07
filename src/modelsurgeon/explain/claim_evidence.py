"""Deterministic claim-to-evidence explanations.

The renderer is intentionally downstream of the conversational evidence-query
boundary.  It accepts one typed :class:`EvidenceQueryResponse`, verifies that
the response is the canonical projection for its query and snapshot, and then
renders every retained row.  It never accepts provider text, dictionaries,
paths, or predicted values as an alternate evidence source.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Final

from modelsurgeon.conversation import (
    SOURCE_PRECEDENCE,
    CampaignOutcome,
    EvidenceMeasurement,
    EvidenceQuery,
    EvidenceQueryEngine,
    EvidenceQueryOutcome,
    EvidenceQueryRecord,
    EvidenceQueryResponse,
    EvidenceSnapshot,
    EvidenceUncertainty,
)
from modelsurgeon.experiments.identity import canonical_identity_json

CLAIM_EVIDENCE_EXPLANATION_SCHEMA_VERSION: Final[int] = 1


class ClaimEvidenceRendererError(ValueError):
    """Raised when a canonical query response cannot be rendered safely."""


class ClaimEvidenceIntegrityError(ClaimEvidenceRendererError):
    """Raised when a typed query response is tampered with or non-canonical."""


class ClaimEvidenceResourceError(ClaimEvidenceRendererError):
    """Raised when rendering would exceed a declared resource bound."""


class ClaimEvidenceStatus(StrEnum):
    """The visible status of one explanation block.

    ``PREDICTED`` is used only for an accepted decision that has no canonical
    measurement.  It is deliberately not a measurement status.
    """

    MEASURED = "measured"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"
    REJECTED = "rejected"
    ROLLBACK = "rollback"
    ROLLED_BACK = "rollback"


class ClaimEvidenceMeasurementStatus(StrEnum):
    """Whether a block contains measurements from the canonical envelope."""

    MEASURED = "measured"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ClaimEvidenceResourceBounds:
    """Hard limits for one claim rendering."""

    max_claims: int = 256
    max_metrics: int = 1024
    max_provenance_refs: int = 16_384
    max_output_bytes: int = 512 * 1024

    def __post_init__(self) -> None:
        for value, label in (
            (self.max_claims, "maximum claims"),
            (self.max_metrics, "maximum metrics"),
            (self.max_provenance_refs, "maximum provenance references"),
            (self.max_output_bytes, "maximum output bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ClaimEvidenceRendererError(f"{label} must be positive")
        if self.max_claims > 256:
            raise ClaimEvidenceResourceError("maximum claims exceeds the query boundary")
        if self.max_provenance_refs > 256 * 64:
            raise ClaimEvidenceResourceError(
                "maximum provenance references exceeds the query boundary"
            )

    def to_record(self) -> dict[str, int]:
        return {
            "max_claims": self.max_claims,
            "max_metrics": self.max_metrics,
            "max_provenance_refs": self.max_provenance_refs,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class ClaimEvidenceResourceUsage:
    """Measured work performed by the renderer."""

    claims: int
    metrics: int
    provenance_refs: int
    output_bytes: int
    query: Mapping[str, int]

    def __post_init__(self) -> None:
        for value, label in (
            (self.claims, "claim count"),
            (self.metrics, "metric count"),
            (self.provenance_refs, "provenance reference count"),
            (self.output_bytes, "output bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ClaimEvidenceRendererError(f"{label} must be non-negative")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.query.values()
        ):
            raise ClaimEvidenceRendererError(
                "query resource usage must contain non-negative integers"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "claims": self.claims,
            "metrics": self.metrics,
            "provenance_refs": self.provenance_refs,
            "output_bytes": self.output_bytes,
            "query": dict(self.query),
        }


@dataclass(frozen=True, slots=True)
class ClaimEvidenceBlock:
    """One claim-to-evidence block with no lossy narrative-only fields."""

    evidence_id: str
    status: ClaimEvidenceStatus
    outcome: EvidenceQueryOutcome
    measurement_status: ClaimEvidenceMeasurementStatus
    measurements: tuple[EvidenceMeasurement, ...]
    source_outcome: CampaignOutcome
    source_digest: str
    detail: str | None
    artifact_digest: str | None
    inconclusive: bool
    provenance_refs: tuple[str, ...]
    observed_at: str | None
    missing_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    state_version: int
    record_digest: str

    @property
    def unavailable(self) -> bool:
        return bool(self.missing_fields or self.unavailable_fields)

    def to_record(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "status": self.status.value,
            "outcome": self.outcome.value,
            "availability": "unavailable" if self.unavailable else "available",
            "measurement_status": self.measurement_status.value,
            "measurements": [item.to_record() for item in self.measurements],
            "source_outcome": self.source_outcome.value,
            "source_digest": self.source_digest,
            "detail": self.detail,
            "artifact_digest": self.artifact_digest,
            "inconclusive": self.inconclusive,
            "provenance_refs": list(self.provenance_refs),
            "observed_at": self.observed_at,
            "missing_fields": list(self.missing_fields),
            "unavailable_fields": list(self.unavailable_fields),
            "state_version": self.state_version,
            "record_digest": self.record_digest,
        }

    def render_line(self) -> str:
        label = self.status.value.upper()
        parts = [f"[{label}] {self.evidence_id}", f"outcome={self.outcome.value}"]
        if self.measurements:
            rendered_metrics = ", ".join(_render_measurement(item) for item in self.measurements)
            parts.append(f"measurements={rendered_metrics}")
        elif self.status is ClaimEvidenceStatus.PREDICTED:
            parts.append("no measurement supplied; prediction/decision only")
        else:
            parts.append("measurements=unavailable")
        if self.unavailable:
            fields = tuple(sorted(set(self.missing_fields + self.unavailable_fields)))
            parts.append(f"unavailable_fields={','.join(fields)}")
        if self.detail is not None:
            parts.append(f"detail={self.detail}")
        parts.append(f"source={self.source_digest}")
        parts.append(f"provenance={','.join(self.provenance_refs)}")
        return " | ".join(parts)


@dataclass(frozen=True, slots=True)
class ClaimEvidenceExplanation:
    """Canonical structured explanation and deterministic text rendering."""

    campaign_id: str
    query_id: str
    response_digest: str
    snapshot_id: str
    snapshot_digest: str
    response_status: str
    claims: tuple[ClaimEvidenceBlock, ...]
    missing_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    source_precedence: tuple[str, ...]
    source_resource_bounds: Mapping[str, object]
    resource_bounds: ClaimEvidenceResourceBounds
    resource_usage: ClaimEvidenceResourceUsage
    schema_version: int = CLAIM_EVIDENCE_EXPLANATION_SCHEMA_VERSION
    explanation_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.schema_version != CLAIM_EVIDENCE_EXPLANATION_SCHEMA_VERSION:
            raise ClaimEvidenceRendererError("unsupported claim explanation schema")
        evidence_ids = tuple(item.evidence_id for item in self.claims)
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ClaimEvidenceIntegrityError("claim evidence IDs must be unique")
        if len(self.claims) > self.resource_bounds.max_claims:
            raise ClaimEvidenceResourceError("claim count exceeds the declared bound")
        if self.resource_usage.output_bytes < 0:
            raise ClaimEvidenceRendererError("output bytes must be non-negative")
        object.__setattr__(self, "explanation_digest", _digest(self._record_without_digest()))

    def _record_without_digest(self) -> dict[str, object]:
        return {
            "record_type": "canonical_claim_evidence_explanation",
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "query_id": self.query_id,
            "response_digest": self.response_digest,
            "snapshot_id": self.snapshot_id,
            "snapshot_digest": self.snapshot_digest,
            "response_status": self.response_status,
            "claims": [item.to_record() for item in self.claims],
            "missing_fields": list(self.missing_fields),
            "unavailable_fields": list(self.unavailable_fields),
            "source_precedence": list(self.source_precedence),
            "source_resource_bounds": dict(self.source_resource_bounds),
            "resource_bounds": self.resource_bounds.to_record(),
            "resource_usage": self.resource_usage.to_record(),
        }

    def to_record(self) -> dict[str, object]:
        return {**self._record_without_digest(), "explanation_digest": self.explanation_digest}

    def canonical_json(self) -> str:
        return _canonical(self.to_record())

    def render_text(self) -> str:
        lines = [
            f"Evidence explanation for {self.campaign_id}",
            "status="
            f"{self.response_status} | response={self.response_digest} | "
            f"snapshot={self.snapshot_id}",
        ]
        lines.extend(item.render_line() for item in self.claims)
        if self.missing_fields:
            lines.append(f"[UNAVAILABLE] response missing fields={','.join(self.missing_fields)}")
        if self.unavailable_fields:
            lines.append(
                "[UNAVAILABLE] response unavailable fields="
                + ",".join(self.unavailable_fields)
            )
        lines.append(
            "resource_bounds="
            + _canonical(self.resource_bounds.to_record())
            + " | source_resource_bounds="
            + _canonical(self.source_resource_bounds)
            + " | resource_usage="
            + _canonical(self.resource_usage.to_record())
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render_text()


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ClaimEvidenceRendererError("explanation is not canonical JSON") from error


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _render_measurement(measurement: EvidenceMeasurement) -> str:
    rendered = f"{measurement.metric}={measurement.value:g}"
    if measurement.unit is not None:
        rendered += f" {measurement.unit}"
    if measurement.uncertainty is not None:
        rendered += f" ({_render_uncertainty(measurement.uncertainty)})"
    return rendered


def _render_uncertainty(uncertainty: EvidenceUncertainty) -> str:
    parts: list[str] = []
    if uncertainty.lower_bound is not None or uncertainty.upper_bound is not None:
        parts.append(
            "bounds="
            f"{_number(uncertainty.lower_bound)}..{_number(uncertainty.upper_bound)}"
        )
    if uncertainty.confidence is not None:
        parts.append(f"confidence={uncertainty.confidence:g}")
    if uncertainty.standard_error is not None:
        parts.append(f"standard_error={uncertainty.standard_error:g}")
    if uncertainty.sample_count is not None:
        parts.append(f"n={uncertainty.sample_count}")
    return ",".join(parts)


def _number(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:g}"


def _status(record: EvidenceQueryRecord) -> ClaimEvidenceStatus:
    # The query envelope has no prediction field.  An accepted row without a
    # canonical measurement is therefore explicitly prediction/decision-only;
    # it is never rendered as measured.
    outcome = record.outcome
    if outcome is EvidenceQueryOutcome.ACCEPTED:
        return (
            ClaimEvidenceStatus.MEASURED
            if record.measurements
            else ClaimEvidenceStatus.PREDICTED
        )
    return {
        EvidenceQueryOutcome.REJECTED: ClaimEvidenceStatus.REJECTED,
        EvidenceQueryOutcome.ROLLED_BACK: ClaimEvidenceStatus.ROLLBACK,
        EvidenceQueryOutcome.UNSUPPORTED: ClaimEvidenceStatus.UNSUPPORTED,
        EvidenceQueryOutcome.FAILED: ClaimEvidenceStatus.FAILED,
        EvidenceQueryOutcome.UNKNOWN: ClaimEvidenceStatus.UNKNOWN,
        EvidenceQueryOutcome.INCONCLUSIVE: ClaimEvidenceStatus.INCONCLUSIVE,
        EvidenceQueryOutcome.MISSING: ClaimEvidenceStatus.UNKNOWN,
        EvidenceQueryOutcome.UNAVAILABLE: ClaimEvidenceStatus.UNKNOWN,
    }[outcome]


def _validate_response(response: EvidenceQueryResponse) -> None:
    """Verify type, nested digests, and exact canonical query projection."""

    try:
        query = EvidenceQuery.from_record(response.query.to_record())
        snapshot = EvidenceSnapshot.from_record(response.snapshot.to_record())
        expected = EvidenceQueryEngine(snapshot).query(query)
    except Exception as error:
        if isinstance(error, ClaimEvidenceRendererError):
            raise
        raise ClaimEvidenceIntegrityError(
            "evidence query response is not a valid canonical envelope"
        ) from error
    if response.canonical_json() != expected.canonical_json():
        raise ClaimEvidenceIntegrityError(
            "evidence query response does not match its canonical report"
        )


def _block(record: EvidenceQueryRecord) -> ClaimEvidenceBlock:
    status = _status(record)
    return ClaimEvidenceBlock(
        record.evidence_id,
        status,
        record.outcome,
        ClaimEvidenceMeasurementStatus.MEASURED
        if record.measurements
        else ClaimEvidenceMeasurementStatus.UNAVAILABLE,
        record.measurements,
        record.source_outcome,
        record.source_digest,
        record.detail,
        record.artifact_digest,
        record.inconclusive,
        record.provenance_refs,
        record.observed_at,
        record.missing_fields,
        record.unavailable_fields,
        record.state_version,
        record.record_digest,
    )


def render_claim_evidence(
    response: EvidenceQueryResponse,
    *,
    resource_bounds: ClaimEvidenceResourceBounds | None = None,
) -> ClaimEvidenceExplanation:
    """Render every record in one canonical #475 response.

    The input is deliberately nominally typed and integrity-checked.  A raw
    mapping, provider answer, query, snapshot, or ad-hoc evidence object is not
    accepted.  All negative records remain in ``claims`` in response order.
    """

    if not isinstance(response, EvidenceQueryResponse):
        raise ClaimEvidenceRendererError(
            "renderer accepts only a typed EvidenceQueryResponse envelope"
        )
    _validate_response(response)
    bounds = resource_bounds or ClaimEvidenceResourceBounds()
    if len(response.records) > bounds.max_claims:
        raise ClaimEvidenceResourceError("response has more claims than the declared bound")
    claims = tuple(_block(record) for record in response.records)
    metric_count = sum(len(item.measurements) for item in claims)
    provenance_count = sum(len(item.provenance_refs) for item in claims)
    if metric_count > bounds.max_metrics:
        raise ClaimEvidenceResourceError("response has more metrics than the declared bound")
    if provenance_count > bounds.max_provenance_refs:
        raise ClaimEvidenceResourceError(
            "response has more provenance references than the declared bound"
        )
    usage = ClaimEvidenceResourceUsage(
        len(claims),
        metric_count,
        provenance_count,
        0,
        response.resource_usage,
    )
    result = ClaimEvidenceExplanation(
        response.query.campaign_id,
        response.query.query_id,
        response.response_digest,
        response.snapshot.snapshot_id,
        response.snapshot.snapshot_digest,
        response.status.value,
        claims,
        response.missing_fields,
        response.unavailable_fields,
        SOURCE_PRECEDENCE,
        {
            "campaign_budget": response.snapshot.campaign.budget.to_record(),
            "query_max_records": response.query.max_records,
        },
        bounds,
        usage,
    )
    for _ in range(3):
        output_bytes = len(result.canonical_json().encode("utf-8"))
        if output_bytes == result.resource_usage.output_bytes:
            break
        result = replace(
            result,
            resource_usage=replace(result.resource_usage, output_bytes=output_bytes),
        )
    if result.resource_usage.output_bytes > bounds.max_output_bytes:
        raise ClaimEvidenceResourceError("explanation exceeds the declared output bound")
    return result


def render_claim_evidence_text(
    response: EvidenceQueryResponse,
    *,
    resource_bounds: ClaimEvidenceResourceBounds | None = None,
) -> str:
    """Return the deterministic human-readable rendering."""

    return render_claim_evidence(response, resource_bounds=resource_bounds).render_text()


# Verb-led and noun-led aliases keep the public boundary discoverable without
# creating alternate input paths.
render_claim_evidence_explanation = render_claim_evidence


class ClaimEvidenceRenderer:
    """Reusable renderer with fixed bounds and no alternate input channel."""

    def __init__(self, resource_bounds: ClaimEvidenceResourceBounds | None = None) -> None:
        self.resource_bounds = resource_bounds or ClaimEvidenceResourceBounds()

    def render(self, response: EvidenceQueryResponse) -> ClaimEvidenceExplanation:
        return render_claim_evidence(response, resource_bounds=self.resource_bounds)

    def render_text(self, response: EvidenceQueryResponse) -> str:
        return self.render(response).render_text()


__all__ = [
    "CLAIM_EVIDENCE_EXPLANATION_SCHEMA_VERSION",
    "ClaimEvidenceBlock",
    "ClaimEvidenceExplanation",
    "ClaimEvidenceIntegrityError",
    "ClaimEvidenceMeasurementStatus",
    "ClaimEvidenceRenderer",
    "ClaimEvidenceRendererError",
    "ClaimEvidenceResourceBounds",
    "ClaimEvidenceResourceError",
    "ClaimEvidenceResourceUsage",
    "ClaimEvidenceStatus",
    "render_claim_evidence",
    "render_claim_evidence_explanation",
    "render_claim_evidence_text",
]
