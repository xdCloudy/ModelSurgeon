"""Typed, bounded queries over canonical conversational campaign evidence.

This module is deliberately a read-only projection.  It accepts an already
validated campaign snapshot (or reads one through :class:`CampaignStateStore`)
and joins only engine-owned records.  It has no path, file, subprocess, model,
provider, or transcript input, so a conversational caller cannot turn the
query path into an arbitrary data or inference channel.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Final, cast

from modelsurgeon.conversation.campaign_state import (
    CampaignEvidence,
    CampaignOutcome,
    CampaignState,
    CampaignStateError,
    CampaignStateStore,
)
from modelsurgeon.experiments.identity import canonical_identity_json

EVIDENCE_QUERY_SCHEMA_VERSION: Final[int] = 1
EVIDENCE_SNAPSHOT_SCHEMA_VERSION: Final[int] = 1
MAX_EVIDENCE_QUERY_BYTES: Final[int] = 64 * 1024
MAX_EVIDENCE_QUERY_RESULT_BYTES: Final[int] = 512 * 1024

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$")
_FORBIDDEN_FIELD = re.compile(
    r"(?i)(?:path|file|filesystem|directory|shell|command|subprocess|prompt|transcript|"
    r"message|history|llm|provider_context|secret|token|password|credential|raw)"
)

type JSONValue = (
    bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
)


class EvidenceQueryError(ValueError):
    """Base error for invalid, unsafe, stale, or over-budget queries."""


class EvidenceQueryAccessError(EvidenceQueryError):
    """Raised when a query requests a non-canonical or unsafe field."""


class EvidenceQueryIntegrityError(EvidenceQueryError):
    """Raised when a snapshot or retained record fails digest validation."""


class EvidenceQueryStaleError(EvidenceQueryError):
    """Raised when a query is evaluated against a changed canonical campaign."""


class EvidenceQueryResourceError(EvidenceQueryError):
    """Raised when a query or result exceeds a hard resource limit."""


class EvidenceQueryOutcome(StrEnum):
    """Disposition visible to explanations, including negative evidence."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    ROLLED_BACK = "rolled_back"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    INCONCLUSIVE = "inconclusive"
    MISSING = "missing"
    UNAVAILABLE = "unavailable"


class EvidenceQueryStatus(StrEnum):
    """Status of the read projection itself."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNAVAILABLE = "unavailable"


class EvidenceQueryAccess(StrEnum):
    """Only the canonical read boundary is exposed by this API."""

    READ_ONLY = "read_only"


SOURCE_PRECEDENCE: Final[tuple[str, ...]] = (
    "campaign_state",
    "campaign_evidence",
    "artifact_reference",
)
_ALLOWED_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "campaign_id",
        "session_id",
        "run_id",
        "source_model_digest",
        "spec_identity",
        "spec_digest",
        "hard_constraints",
        "lifecycle",
        "campaign_outcome",
        "state_version",
        "evidence_cursor",
        "budget",
        "evidence_id",
        "source_digest",
        "outcome",
        "source_outcome",
        "detail",
        "artifact_digest",
        "inconclusive",
        "measurements",
        "uncertainty",
        "provenance_refs",
        "observed_at",
        "missing_fields",
        "unavailable_fields",
    }
)


def _canonical(value: object, label: str) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise EvidenceQueryError(f"{label} must be canonical JSON") from error


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceQueryError(f"{label} must be non-empty text")
    return value


def _identifier(value: object, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise EvidenceQueryError(f"{label} must be a canonical identifier")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _DIGEST.fullmatch(result) is None:
        raise EvidenceQueryError(f"{label} must be a sha256 content digest")
    return result


def _timestamp(value: object, label: str) -> str:
    result = _text(value, label)
    if _TIMESTAMP.fullmatch(result) is None:
        raise EvidenceQueryError(f"{label} must be an RFC 3339 UTC timestamp")
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceQueryError(f"{label} must be a non-negative integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _nonnegative_int(value, label)
    if result == 0:
        raise EvidenceQueryError(f"{label} must be positive")
    return result


def _mapping(value: object, label: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise EvidenceQueryError(f"{label} must be a JSON object")
    try:
        decoded = json.loads(_canonical(value, label))
    except json.JSONDecodeError as error:
        raise EvidenceQueryError(f"{label} must be canonical JSON") from error
    if not isinstance(decoded, dict):
        raise EvidenceQueryError(f"{label} must be a JSON object")
    return cast(dict[str, JSONValue], decoded)


def _sorted_unique(values: Sequence[str], label: str) -> tuple[str, ...]:
    result = tuple(values)
    if result != tuple(sorted(set(result))):
        raise EvidenceQueryError(f"{label} must be sorted and unique")
    if any(not value.strip() for value in result):
        raise EvidenceQueryError(f"{label} cannot contain blank values")
    return result


@dataclass(frozen=True, slots=True)
class EvidenceQueryLimits:
    """Hard ceilings for one canonical read operation."""

    max_records: int = 256
    max_fields: int = 32
    max_provenance_refs: int = 64
    max_output_bytes: int = MAX_EVIDENCE_QUERY_RESULT_BYTES

    def __post_init__(self) -> None:
        for value, label in (
            (self.max_records, "maximum evidence records"),
            (self.max_fields, "maximum evidence fields"),
            (self.max_provenance_refs, "maximum provenance references"),
            (self.max_output_bytes, "maximum query output bytes"),
        ):
            _positive_int(value, label)
        if (
            self.max_records > 256
            or self.max_fields > 32
            or self.max_provenance_refs > 64
        ):
            raise EvidenceQueryResourceError(
                "query limits exceed the canonical boundary"
            )
        if self.max_output_bytes > MAX_EVIDENCE_QUERY_RESULT_BYTES:
            raise EvidenceQueryResourceError(
                "query output limit exceeds the canonical boundary"
            )

    def to_record(self) -> dict[str, int]:
        return {
            "max_records": self.max_records,
            "max_fields": self.max_fields,
            "max_provenance_refs": self.max_provenance_refs,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class EvidenceQuery:
    """Strict query schema; it contains identities and allowlisted fields only."""

    campaign_id: str
    evidence_ids: tuple[str, ...] = ()
    outcomes: tuple[EvidenceQueryOutcome, ...] = ()
    fields: tuple[str, ...] = ()
    expected_snapshot_id: str | None = None
    expected_state_digest: str | None = None
    max_records: int = 256
    access: EvidenceQueryAccess = EvidenceQueryAccess.READ_ONLY

    def __post_init__(self) -> None:
        _identifier(self.campaign_id, "query campaign ID")
        object.__setattr__(
            self, "evidence_ids", _sorted_unique(self.evidence_ids, "evidence IDs")
        )
        if any(not item.startswith("evidence_") for item in self.evidence_ids):
            raise EvidenceQueryError(
                "query evidence IDs must be canonical evidence IDs"
            )
        if self.outcomes != tuple(
            sorted(set(self.outcomes), key=lambda item: item.value)
        ):
            raise EvidenceQueryError("query outcomes must be sorted and unique")
        if not all(isinstance(item, EvidenceQueryOutcome) for item in self.outcomes):
            raise EvidenceQueryError("query outcomes are invalid")
        selected_fields = _sorted_unique(self.fields, "query fields")
        if len(selected_fields) > 32:
            raise EvidenceQueryResourceError("query requests too many fields")
        for item in selected_fields:
            if _FORBIDDEN_FIELD.search(item) or item not in _ALLOWED_FIELDS:
                raise EvidenceQueryAccessError(
                    f"query field {item!r} is outside the read boundary"
                )
        object.__setattr__(self, "fields", selected_fields)
        if self.expected_snapshot_id is not None:
            _identifier(self.expected_snapshot_id, "expected snapshot ID")
        if self.expected_state_digest is not None:
            _digest(self.expected_state_digest, "expected state digest")
        _positive_int(self.max_records, "query maximum records")
        if self.max_records > 256:
            raise EvidenceQueryResourceError(
                "query maximum records exceeds the canonical boundary"
            )
        if not isinstance(self.access, EvidenceQueryAccess):
            raise EvidenceQueryAccessError("query access must be read_only")
        if (
            len(_canonical(self.to_record(include_id=False), "query").encode("utf-8"))
            > MAX_EVIDENCE_QUERY_BYTES
        ):
            raise EvidenceQueryResourceError("query exceeds the hard input size limit")

    @property
    def query_id(self) -> str:
        digest = hashlib.sha256(
            _canonical(self.to_record(include_id=False), "query").encode()
        ).hexdigest()
        return "evidence_query_" + digest

    def to_record(self, *, include_id: bool = True) -> dict[str, JSONValue]:
        record: dict[str, JSONValue] = {
            "schema_version": EVIDENCE_QUERY_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "evidence_ids": list(self.evidence_ids),
            "outcomes": [item.value for item in self.outcomes],
            "fields": list(self.fields),
            "expected_snapshot_id": self.expected_snapshot_id,
            "expected_state_digest": self.expected_state_digest,
            "max_records": self.max_records,
            "access": self.access.value,
        }
        if include_id:
            record["query_id"] = self.query_id
        return record

    @classmethod
    def from_record(cls, value: object) -> EvidenceQuery:
        record = _mapping(value, "evidence query")
        expected = {
            "schema_version",
            "query_id",
            "campaign_id",
            "evidence_ids",
            "outcomes",
            "fields",
            "expected_snapshot_id",
            "expected_state_digest",
            "max_records",
            "access",
        }
        if (
            set(record) != expected
            or record["schema_version"] != EVIDENCE_QUERY_SCHEMA_VERSION
        ):
            raise EvidenceQueryError(
                "evidence query has missing, unknown, or unsupported fields"
            )
        evidence_ids = record["evidence_ids"]
        outcomes = record["outcomes"]
        fields = record["fields"]
        if not isinstance(evidence_ids, list) or not all(
            isinstance(item, str) for item in evidence_ids
        ):
            raise EvidenceQueryError("query evidence IDs must be a string array")
        if not isinstance(outcomes, list) or not all(
            isinstance(item, str) for item in outcomes
        ):
            raise EvidenceQueryError("query outcomes must be a string array")
        if not isinstance(fields, list) or not all(
            isinstance(item, str) for item in fields
        ):
            raise EvidenceQueryError("query fields must be a string array")
        evidence_id_values = cast(list[str], evidence_ids)
        outcome_values = cast(list[str], outcomes)
        field_values = cast(list[str], fields)
        try:
            parsed_outcomes = tuple(
                EvidenceQueryOutcome(item) for item in outcome_values
            )
            access = EvidenceQueryAccess(cast(str, record["access"]))
        except ValueError as error:
            raise EvidenceQueryError(
                "query contains an unknown outcome or access"
            ) from error
        result = cls(
            cast(str, record["campaign_id"]),
            tuple(evidence_id_values),
            parsed_outcomes,
            tuple(field_values),
            None
            if record["expected_snapshot_id"] is None
            else cast(str, record["expected_snapshot_id"]),
            None
            if record["expected_state_digest"] is None
            else cast(str, record["expected_state_digest"]),
            cast(int, record["max_records"]),
            access,
        )
        if record["query_id"] != result.query_id:
            raise EvidenceQueryIntegrityError(
                "query ID does not match its canonical payload"
            )
        return result


@dataclass(frozen=True, slots=True)
class EvidenceUncertainty:
    """Optional uncertainty supplied by a canonical measurement producer."""

    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence: float | None = None
    standard_error: float | None = None
    sample_count: int | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.lower_bound, "uncertainty lower bound"),
            (self.upper_bound, "uncertainty upper bound"),
            (self.confidence, "uncertainty confidence"),
            (self.standard_error, "uncertainty standard error"),
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise EvidenceQueryError(f"{label} must be finite")
        if (
            self.lower_bound is not None
            and self.upper_bound is not None
            and self.lower_bound > self.upper_bound
        ):
            raise EvidenceQueryError("uncertainty bounds are reversed")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise EvidenceQueryError(
                "uncertainty confidence must be between zero and one"
            )
        if self.standard_error is not None and self.standard_error < 0:
            raise EvidenceQueryError("uncertainty standard error cannot be negative")
        if self.sample_count is not None:
            _positive_int(self.sample_count, "uncertainty sample count")

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence": self.confidence,
            "standard_error": self.standard_error,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class EvidenceMeasurement:
    """One measured value, never a prediction or an inferred default."""

    metric: str
    value: float
    unit: str | None = None
    uncertainty: EvidenceUncertainty | None = None

    def __post_init__(self) -> None:
        _identifier(self.metric, "measurement metric")
        if (
            not isinstance(self.value, (int, float))
            or isinstance(self.value, bool)
            or not math.isfinite(self.value)
        ):
            raise EvidenceQueryError("measurement value must be finite")
        if self.unit is not None:
            _text(self.unit, "measurement unit")

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "uncertainty": None
            if self.uncertainty is None
            else self.uncertainty.to_record(),
        }


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot:
    """Immutable, digest-checked input to a replayable query."""

    campaign: CampaignState
    evidence: tuple[CampaignEvidence, ...]
    captured_at: str | None = None
    state_digest: str = field(init=False)
    evidence_digest: str = field(init=False)
    snapshot_id: str = field(init=False)
    snapshot_digest: str = field(init=False)

    def __post_init__(self) -> None:
        evidence = tuple(self.evidence)
        ids = tuple(item.evidence_id for item in evidence)
        if len(ids) != len(set(ids)):
            raise EvidenceQueryIntegrityError("snapshot evidence IDs must be unique")
        cursor = self.campaign.evidence_cursor
        if cursor.sequence != len(evidence) or (
            evidence and cursor.evidence_id != evidence[-1].evidence_id
        ):
            raise EvidenceQueryIntegrityError(
                "snapshot evidence does not match the campaign evidence cursor"
            )
        if any(
            item.source_digest != self.campaign.source_model_digest for item in evidence
        ):
            raise EvidenceQueryIntegrityError(
                "snapshot joins evidence from another source model"
            )
        if self.captured_at is not None:
            _timestamp(self.captured_at, "snapshot captured timestamp")
        state_digest = self.campaign.digest
        evidence_payload = [item.to_record() for item in evidence]
        evidence_digest = _prefixed_digest(evidence_payload)
        identity = {
            "schema_version": EVIDENCE_SNAPSHOT_SCHEMA_VERSION,
            "campaign_id": self.campaign.campaign_id,
            "state_digest": state_digest,
            "evidence_digest": evidence_digest,
            "captured_at": self.captured_at,
        }
        snapshot_id = "evidence_snapshot_" + _plain_digest(identity)
        snapshot_payload = {**identity, "snapshot_id": snapshot_id}
        snapshot_digest = _prefixed_digest(snapshot_payload)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "state_digest", state_digest)
        object.__setattr__(self, "evidence_digest", evidence_digest)
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "snapshot_digest", snapshot_digest)

    @classmethod
    def from_store(
        cls, store: CampaignStateStore, campaign_id: str
    ) -> EvidenceSnapshot:
        try:
            state = store.load(campaign_id)
            evidence = store.evidence(campaign_id)
        except CampaignStateError as error:
            raise EvidenceQueryIntegrityError(
                "canonical campaign evidence is unavailable"
            ) from error
        return cls(state, evidence)

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "record_type": "canonical_evidence_snapshot",
            "schema_version": EVIDENCE_SNAPSHOT_SCHEMA_VERSION,
            "snapshot_id": self.snapshot_id,
            "snapshot_digest": self.snapshot_digest,
            "campaign": cast(JSONValue, self.campaign.to_record()),
            "state_digest": self.state_digest,
            "evidence_digest": self.evidence_digest,
            "evidence": cast(JSONValue, [item.to_record() for item in self.evidence]),
            "captured_at": self.captured_at,
        }

    @classmethod
    def from_record(cls, value: object) -> EvidenceSnapshot:
        record = _mapping(value, "evidence snapshot")
        expected = {
            "record_type",
            "schema_version",
            "snapshot_id",
            "snapshot_digest",
            "campaign",
            "state_digest",
            "evidence_digest",
            "evidence",
            "captured_at",
        }
        if (
            set(record) != expected
            or record["record_type"] != "canonical_evidence_snapshot"
        ):
            raise EvidenceQueryIntegrityError("snapshot has missing or unknown fields")
        if record["schema_version"] != EVIDENCE_SNAPSHOT_SCHEMA_VERSION:
            raise EvidenceQueryIntegrityError(
                "unsupported evidence snapshot schema version"
            )
        evidence = record["evidence"]
        if not isinstance(evidence, list):
            raise EvidenceQueryIntegrityError("snapshot evidence must be an array")
        try:
            result = cls(
                CampaignState.from_record(record["campaign"]),
                tuple(CampaignEvidence.from_record(item) for item in evidence),
                None
                if record["captured_at"] is None
                else cast(str, record["captured_at"]),
            )
        except (CampaignStateError, EvidenceQueryError) as error:
            raise EvidenceQueryIntegrityError(
                "snapshot contains invalid canonical records"
            ) from error
        if (
            record["snapshot_id"] != result.snapshot_id
            or record["snapshot_digest"] != result.snapshot_digest
            or record["state_digest"] != result.state_digest
            or record["evidence_digest"] != result.evidence_digest
        ):
            raise EvidenceQueryIntegrityError(
                "snapshot digest does not match its content"
            )
        return result

    def assert_current(self, store: CampaignStateStore) -> None:
        current = EvidenceSnapshot.from_store(store, self.campaign.campaign_id)
        if (
            current.state_digest != self.state_digest
            or current.evidence_digest != self.evidence_digest
        ):
            raise EvidenceQueryStaleError("evidence snapshot is stale")


@dataclass(frozen=True, slots=True)
class EvidenceQueryRecord:
    """One joined, field-bounded result row."""

    evidence_id: str
    campaign_id: str
    outcome: EvidenceQueryOutcome
    source_outcome: CampaignOutcome
    source_digest: str
    detail: str | None
    artifact_digest: str | None
    inconclusive: bool
    measurements: tuple[EvidenceMeasurement, ...]
    provenance_refs: tuple[str, ...]
    observed_at: str | None
    missing_fields: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    state_version: int
    fields: Mapping[str, JSONValue] = field(default_factory=dict)
    record_digest: str = field(init=False)

    def __post_init__(self) -> None:
        _identifier(self.evidence_id, "query evidence ID")
        _identifier(self.campaign_id, "query campaign ID")
        if not isinstance(self.outcome, EvidenceQueryOutcome) or not isinstance(
            self.source_outcome, CampaignOutcome
        ):
            raise EvidenceQueryError("query record outcome is invalid")
        _digest(self.source_digest, "query source digest")
        if self.detail is not None:
            _text(self.detail, "query evidence detail")
        if self.artifact_digest is not None:
            _digest(self.artifact_digest, "query artifact digest")
        if not isinstance(self.inconclusive, bool):
            raise EvidenceQueryError("query inconclusive flag is invalid")
        metrics = tuple(item.metric for item in self.measurements)
        if metrics != tuple(sorted(set(metrics))):
            raise EvidenceQueryError("query measurements must be sorted and unique")
        refs = _sorted_unique(self.provenance_refs, "query provenance references")
        if len(refs) > 64:
            raise EvidenceQueryResourceError(
                "query record has too many provenance references"
            )
        object.__setattr__(self, "provenance_refs", refs)
        object.__setattr__(
            self,
            "missing_fields",
            _sorted_unique(self.missing_fields, "missing fields"),
        )
        object.__setattr__(
            self,
            "unavailable_fields",
            _sorted_unique(self.unavailable_fields, "unavailable fields"),
        )
        selected = _mapping(self.fields, "query record fields")
        for name in selected:
            if name not in _ALLOWED_FIELDS or _FORBIDDEN_FIELD.search(name):
                raise EvidenceQueryAccessError(
                    f"query record field {name!r} is outside the read boundary"
                )
        object.__setattr__(self, "fields", MappingProxyType(selected))
        _nonnegative_int(self.state_version, "query state version")
        object.__setattr__(
            self,
            "record_digest",
            _prefixed_digest(self.to_record(include_digest=False)),
        )

    def to_record(self, *, include_digest: bool = True) -> dict[str, JSONValue]:
        record: dict[str, JSONValue] = {
            "evidence_id": self.evidence_id,
            "campaign_id": self.campaign_id,
            "outcome": self.outcome.value,
            "source_outcome": self.source_outcome.value,
            "source_digest": self.source_digest,
            "detail": self.detail,
            "artifact_digest": self.artifact_digest,
            "inconclusive": self.inconclusive,
            "measurements": [item.to_record() for item in self.measurements],
            "provenance_refs": list(self.provenance_refs),
            "observed_at": self.observed_at,
            "missing_fields": list(self.missing_fields),
            "unavailable_fields": list(self.unavailable_fields),
            "state_version": self.state_version,
            "fields": dict(self.fields),
        }
        if include_digest:
            record["record_digest"] = self.record_digest
        return record


@dataclass(frozen=True, slots=True)
class EvidenceQueryResponse:
    """Canonical response envelope suitable for a text-model explanation."""

    query: EvidenceQuery
    snapshot: EvidenceSnapshot
    status: EvidenceQueryStatus
    records: tuple[EvidenceQueryRecord, ...]
    missing_fields: tuple[str, ...] = ()
    unavailable_fields: tuple[str, ...] = ()
    resource_usage: Mapping[str, int] = field(default_factory=dict)
    response_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.query.campaign_id != self.snapshot.campaign.campaign_id:
            raise EvidenceQueryIntegrityError("query and snapshot campaign IDs differ")
        if (
            self.query.expected_snapshot_id is not None
            and self.query.expected_snapshot_id != self.snapshot.snapshot_id
        ):
            raise EvidenceQueryStaleError(
                "query references a different evidence snapshot"
            )
        if (
            self.query.expected_state_digest is not None
            and self.query.expected_state_digest != self.snapshot.state_digest
        ):
            raise EvidenceQueryStaleError(
                "query references a stale campaign state digest"
            )
        if not isinstance(self.status, EvidenceQueryStatus):
            raise EvidenceQueryError("query response status is invalid")
        if len(self.records) > self.query.max_records:
            raise EvidenceQueryResourceError(
                "query returned more records than requested"
            )
        object.__setattr__(
            self,
            "missing_fields",
            _sorted_unique(self.missing_fields, "response missing fields"),
        )
        object.__setattr__(
            self,
            "unavailable_fields",
            _sorted_unique(self.unavailable_fields, "response unavailable fields"),
        )
        usage = _mapping(self.resource_usage, "query resource usage")
        if not all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in usage.values()
        ):
            raise EvidenceQueryError(
                "query resource usage values must be non-negative integers"
            )
        object.__setattr__(self, "resource_usage", MappingProxyType(usage))
        encoded = _canonical(
            self.to_record(include_digest=False), "query response"
        ).encode("utf-8")
        if len(encoded) > MAX_EVIDENCE_QUERY_RESULT_BYTES:
            raise EvidenceQueryResourceError(
                "query response exceeds the hard size limit"
            )
        object.__setattr__(
            self,
            "response_digest",
            _prefixed_digest(self.to_record(include_digest=False)),
        )

    def to_record(self, *, include_digest: bool = True) -> dict[str, JSONValue]:
        record: dict[str, JSONValue] = {
            "record_type": "canonical_evidence_query_response",
            "schema_version": EVIDENCE_QUERY_SCHEMA_VERSION,
            "query": self.query.to_record(),
            "snapshot_id": self.snapshot.snapshot_id,
            "snapshot_digest": self.snapshot.snapshot_digest,
            "status": self.status.value,
            "records": [item.to_record() for item in self.records],
            "missing_fields": list(self.missing_fields),
            "unavailable_fields": list(self.unavailable_fields),
            "resource_usage": dict(self.resource_usage),
            "source_precedence": list(SOURCE_PRECEDENCE),
        }
        if include_digest:
            record["response_digest"] = self.response_digest
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record(), "query response")


def _plain_digest(value: object) -> str:
    return hashlib.sha256(
        _canonical(value, "digest payload").encode("utf-8")
    ).hexdigest()


def _prefixed_digest(value: object) -> str:
    return "sha256:" + _plain_digest(value)


def _disposition(evidence: CampaignEvidence) -> EvidenceQueryOutcome:
    decision = evidence.provenance.get("decision")
    if isinstance(decision, str):
        try:
            return EvidenceQueryOutcome(decision)
        except ValueError:
            pass
    if evidence.inconclusive:
        return EvidenceQueryOutcome.INCONCLUSIVE
    if evidence.outcome is CampaignOutcome.UNSUPPORTED:
        return EvidenceQueryOutcome.UNSUPPORTED
    if evidence.outcome is CampaignOutcome.FAILED:
        return EvidenceQueryOutcome.FAILED
    if evidence.outcome is CampaignOutcome.UNKNOWN:
        return EvidenceQueryOutcome.UNKNOWN
    return EvidenceQueryOutcome.ACCEPTED


def _measurement(value: object, metric: str) -> EvidenceMeasurement | None:
    if not isinstance(value, Mapping):
        return None
    raw_value = value.get("value")
    if (
        not isinstance(raw_value, (int, float))
        or isinstance(raw_value, bool)
        or not math.isfinite(raw_value)
    ):
        return None
    uncertainty_raw = value.get("uncertainty")
    uncertainty = None
    if uncertainty_raw is not None:
        if not isinstance(uncertainty_raw, Mapping):
            return None
        try:
            uncertainty = EvidenceUncertainty(
                None
                if uncertainty_raw.get("lower_bound") is None
                else float(uncertainty_raw["lower_bound"]),
                None
                if uncertainty_raw.get("upper_bound") is None
                else float(uncertainty_raw["upper_bound"]),
                None
                if uncertainty_raw.get("confidence") is None
                else float(uncertainty_raw["confidence"]),
                None
                if uncertainty_raw.get("standard_error") is None
                else float(uncertainty_raw["standard_error"]),
                None
                if uncertainty_raw.get("sample_count") is None
                else int(uncertainty_raw["sample_count"]),
            )
        except (EvidenceQueryError, TypeError, ValueError, OverflowError):
            return None
    try:
        return EvidenceMeasurement(
            metric,
            float(raw_value),
            None if value.get("unit") is None else cast(str, value["unit"]),
            uncertainty,
        )
    except EvidenceQueryError:
        return None


def _measurements(evidence: CampaignEvidence) -> tuple[EvidenceMeasurement, ...]:
    raw = evidence.provenance.get("measurements")
    if not isinstance(raw, Mapping):
        return ()
    items: list[EvidenceMeasurement] = []
    for metric, value in sorted(raw.items()):
        if not isinstance(metric, str) or _IDENTIFIER.fullmatch(metric) is None:
            continue
        parsed = _measurement(value, metric)
        if parsed is not None:
            items.append(parsed)
    return tuple(items)


def _provenance_refs(evidence: CampaignEvidence, *, limit: int) -> tuple[str, ...]:
    values: set[str] = {evidence.evidence_id, evidence.source_digest}
    for key in (
        "run_id",
        "plan_id",
        "stage_evidence_id",
        "parent_evidence_id",
        "artifact_id",
    ):
        value = evidence.provenance.get(key)
        if isinstance(value, str) and _IDENTIFIER.fullmatch(value):
            values.add(value)
    refs = tuple(sorted(values))
    if len(refs) > limit:
        raise EvidenceQueryResourceError(
            "evidence provenance join exceeds the hard limit"
        )
    return refs


def _record(
    snapshot: EvidenceSnapshot, evidence: CampaignEvidence, fields: tuple[str, ...]
) -> EvidenceQueryRecord:
    measurements = _measurements(evidence)
    all_fields = {
        "campaign_id": snapshot.campaign.campaign_id,
        "session_id": snapshot.campaign.session_id,
        "run_id": snapshot.campaign.run_id,
        "source_model_digest": snapshot.campaign.source_model_digest,
        "spec_identity": snapshot.campaign.spec_identity,
        "spec_digest": snapshot.campaign.spec_digest,
        "hard_constraints": [dict(item) for item in snapshot.campaign.hard_constraints],
        "lifecycle": snapshot.campaign.lifecycle.value,
        "campaign_outcome": snapshot.campaign.outcome.value,
        "state_version": snapshot.campaign.state_version,
        "evidence_cursor": snapshot.campaign.evidence_cursor.to_record(),
        "budget": snapshot.campaign.budget.to_record(),
        "evidence_id": evidence.evidence_id,
        "source_digest": evidence.source_digest,
        "outcome": _disposition(evidence).value,
        "source_outcome": evidence.outcome.value,
        "detail": evidence.detail,
        "artifact_digest": evidence.artifact_digest,
        "inconclusive": evidence.inconclusive,
        "measurements": [item.to_record() for item in measurements],
        "uncertainty": [
            item.uncertainty.to_record()
            for item in measurements
            if item.uncertainty is not None
        ],
        "provenance_refs": list(_provenance_refs(evidence, limit=64)),
        "observed_at": evidence.provenance.get("observed_at")
        if isinstance(evidence.provenance.get("observed_at"), str)
        else None,
    }
    selected = (
        set(fields)
        if fields
        else {
            "evidence_id",
            "outcome",
            "source_digest",
            "detail",
            "artifact_digest",
            "inconclusive",
            "provenance_refs",
        }
    )
    missing = tuple(
        sorted(
            field_name
            for field_name in selected
            if field_name
            in {"artifact_digest", "measurements", "uncertainty", "observed_at"}
            and all_fields[field_name] in (None, [], "")
        )
    )
    unavailable = tuple(
        sorted(field_name for field_name in selected if field_name not in all_fields)
    )
    field_values = {
        field_name: cast(JSONValue, all_fields[field_name])
        for field_name in sorted(selected)
        if field_name in all_fields
    }
    return EvidenceQueryRecord(
        evidence.evidence_id,
        snapshot.campaign.campaign_id,
        _disposition(evidence),
        evidence.outcome,
        evidence.source_digest,
        cast(str | None, all_fields["detail"]),
        evidence.artifact_digest,
        evidence.inconclusive,
        measurements if "measurements" in selected else (),
        tuple(cast(list[str], all_fields["provenance_refs"])),
        cast(str | None, all_fields["observed_at"]),
        missing,
        unavailable,
        snapshot.campaign.state_version,
        field_values,
    )


class EvidenceQueryEngine:
    """Evaluate allowlisted evidence queries without mutating canonical state."""

    def __init__(
        self, snapshot: EvidenceSnapshot, *, limits: EvidenceQueryLimits | None = None
    ) -> None:
        self.snapshot = snapshot
        self.limits = limits or EvidenceQueryLimits()

    @classmethod
    def from_store(
        cls,
        store: CampaignStateStore,
        campaign_id: str,
        *,
        limits: EvidenceQueryLimits | None = None,
    ) -> EvidenceQueryEngine:
        return cls(EvidenceSnapshot.from_store(store, campaign_id), limits=limits)

    def query(self, request: EvidenceQuery) -> EvidenceQueryResponse:
        if request.campaign_id != self.snapshot.campaign.campaign_id:
            raise EvidenceQueryAccessError(
                "query campaign is outside the snapshot boundary"
            )
        if request.max_records > self.limits.max_records:
            raise EvidenceQueryResourceError(
                "query maximum records exceeds engine limit"
            )
        candidates = self.snapshot.evidence
        if request.evidence_ids:
            selected = set(request.evidence_ids)
            candidates = tuple(
                item for item in candidates if item.evidence_id in selected
            )
        if request.outcomes:
            candidates = tuple(
                item for item in candidates if _disposition(item) in request.outcomes
            )
        if len(candidates) > request.max_records:
            raise EvidenceQueryResourceError(
                "query result exceeds the requested record limit"
            )
        fields = request.fields
        if len(fields) > self.limits.max_fields:
            raise EvidenceQueryResourceError(
                "query field selection exceeds engine limit"
            )
        records = tuple(_record(self.snapshot, item, fields) for item in candidates)
        missing = tuple(
            sorted(
                {field_name for item in records for field_name in item.missing_fields}
            )
        )
        unavailable = tuple(
            sorted(
                {
                    field_name
                    for item in records
                    for field_name in item.unavailable_fields
                }
            )
        )
        response = EvidenceQueryResponse(
            request,
            self.snapshot,
            EvidenceQueryStatus.PARTIAL
            if missing or unavailable
            else EvidenceQueryStatus.COMPLETE,
            records,
            missing,
            unavailable,
            {
                "input_bytes": len(
                    _canonical(request.to_record(), "query").encode("utf-8")
                ),
                "evidence_records": len(records),
                "provenance_refs": sum(len(item.provenance_refs) for item in records),
            },
        )
        if (
            len(response.canonical_json().encode("utf-8"))
            > self.limits.max_output_bytes
        ):
            raise EvidenceQueryResourceError("query result exceeds engine output limit")
        return response

    def query_current(
        self, store: CampaignStateStore, request: EvidenceQuery
    ) -> EvidenceQueryResponse:
        """Check the immutable snapshot against the store before answering."""

        self.snapshot.assert_current(store)
        return self.query(request)


def direct_evidence_report(
    snapshot: EvidenceSnapshot, request: EvidenceQuery
) -> EvidenceQueryResponse:
    """Build the direct structured report through the same canonical projection.

    Keeping this helper public makes direct API and conversational query parity
    testable without treating a transcript or provider response as authority.
    """

    return EvidenceQueryEngine(snapshot).query(request)


__all__ = [
    "EVIDENCE_QUERY_SCHEMA_VERSION",
    "EVIDENCE_SNAPSHOT_SCHEMA_VERSION",
    "MAX_EVIDENCE_QUERY_BYTES",
    "MAX_EVIDENCE_QUERY_RESULT_BYTES",
    "SOURCE_PRECEDENCE",
    "EvidenceMeasurement",
    "EvidenceQuery",
    "EvidenceQueryAccess",
    "EvidenceQueryAccessError",
    "EvidenceQueryEngine",
    "EvidenceQueryError",
    "EvidenceQueryIntegrityError",
    "EvidenceQueryLimits",
    "EvidenceQueryOutcome",
    "EvidenceQueryRecord",
    "EvidenceQueryResourceError",
    "EvidenceQueryResponse",
    "EvidenceQueryStaleError",
    "EvidenceQueryStatus",
    "EvidenceSnapshot",
    "EvidenceUncertainty",
    "direct_evidence_report",
]
