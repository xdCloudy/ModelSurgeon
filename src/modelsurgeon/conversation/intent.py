"""Canonical, provenance-linked conversational intent records."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

CONVERSATIONAL_INTENT_SCHEMA_VERSION: Literal[1] = 1
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class IntentRecordError(ValueError):
    """Raised when an intent record is unsafe, incomplete, or not canonical."""


class IntentOutcome(StrEnum):
    EXECUTABLE = "executable"
    CLARIFICATION_REQUIRED = "clarification_required"
    UNSUPPORTED = "unsupported"
    REFUSED = "refused"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntentRecordError(f"{label} must be a non-empty string")
    return value


def _identifier(value: object, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise IntentRecordError(f"{label} is not a canonical identifier")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise IntentRecordError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _confidence(value: float, label: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
        raise IntentRecordError(f"{label} must be finite and within [0, 1]")


def _canonical(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise IntentRecordError("record values must be JSON serializable") from error


def _sorted_unique(values: tuple[str, ...], label: str) -> None:
    if values != tuple(sorted(set(values))):
        raise IntentRecordError(f"{label} must be sorted and unique")


@dataclass(frozen=True, slots=True)
class SourceSpan:
    """A source location retained so normalized fields remain auditable."""

    span_id: str
    start: int
    end: int
    text: str

    def __post_init__(self) -> None:
        _identifier(self.span_id, "source span ID")
        if isinstance(self.start, bool) or self.start < 0:
            raise IntentRecordError("source span start must be non-negative")
        if isinstance(self.end, bool) or self.end <= self.start:
            raise IntentRecordError("source span end must exceed start")
        _text(self.text, "source span text")

    def to_record(self) -> dict[str, object]:
        return {
            "span_id": self.span_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class IntentField:
    """One normalized interpretation field with explicit unit and confidence."""

    field_id: str
    value: object
    unit: str | None
    confidence: float
    source_span_ids: tuple[str, ...]
    required: bool = False

    def __post_init__(self) -> None:
        _identifier(self.field_id, "intent field ID")
        if self.unit is not None:
            _identifier(self.unit, "intent field unit")
        _confidence(self.confidence, "intent field confidence")
        _sorted_unique(self.source_span_ids, "intent field source spans")
        _canonical(self.value)

    def to_record(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "value": self.value,
            "unit": self.unit,
            "confidence": self.confidence,
            "source_span_ids": list(self.source_span_ids),
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class AmbiguityRecord:
    """An unresolved interpretation alternative, never an optimization signal."""

    ambiguity_id: str
    field_id: str
    category: str
    detail: str
    alternatives: tuple[str, ...]
    required: bool = True

    def __post_init__(self) -> None:
        _identifier(self.ambiguity_id, "ambiguity ID")
        _identifier(self.field_id, "ambiguity field ID")
        _identifier(self.category, "ambiguity category")
        _text(self.detail, "ambiguity detail")
        if not self.alternatives:
            raise IntentRecordError("ambiguity alternatives must not be empty")
        _sorted_unique(self.alternatives, "ambiguity alternatives")

    def to_record(self) -> dict[str, object]:
        return {
            "ambiguity_id": self.ambiguity_id,
            "field_id": self.field_id,
            "category": self.category,
            "detail": self.detail,
            "alternatives": list(self.alternatives),
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class InterpretationStep:
    """A deterministic, provenance-linked transformation in the interpretation."""

    step_id: str
    operation: str
    input_span_ids: tuple[str, ...]
    output_field_ids: tuple[str, ...]
    confidence: float
    provenance_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.step_id, "interpretation step ID")
        _identifier(self.operation, "interpretation operation")
        _sorted_unique(self.input_span_ids, "interpretation input spans")
        _sorted_unique(self.output_field_ids, "interpretation output fields")
        _confidence(self.confidence, "interpretation confidence")
        if not self.provenance_refs:
            raise IntentRecordError("interpretation steps require provenance references")
        _sorted_unique(self.provenance_refs, "interpretation provenance references")

    def to_record(self) -> dict[str, object]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "input_span_ids": list(self.input_span_ids),
            "output_field_ids": list(self.output_field_ids),
            "confidence": self.confidence,
            "provenance_refs": list(self.provenance_refs),
        }


@dataclass(frozen=True, slots=True)
class IntentProvenance:
    """Identity of the request, provider, interpretation code, and evidence."""

    source_revision: str
    provider_id: str
    provider_revision: str
    tool_revision: str
    evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _text(self.source_revision, "source revision")
        _identifier(self.provider_id, "provider ID")
        _text(self.provider_revision, "provider revision")
        _text(self.tool_revision, "tool revision")
        _sorted_unique(self.evidence_refs, "provenance evidence references")

    def to_record(self) -> dict[str, object]:
        return {
            "source_revision": self.source_revision,
            "provider_id": self.provider_id,
            "provider_revision": self.provider_revision,
            "tool_revision": self.tool_revision,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True, slots=True)
class IntentRecord:
    """Complete request-to-spec or request-to-refusal canonical record."""

    original_request: str
    source_spans: tuple[SourceSpan, ...]
    fields: tuple[IntentField, ...]
    ambiguities: tuple[AmbiguityRecord, ...]
    interpretation_steps: tuple[InterpretationStep, ...]
    provenance: IntentProvenance
    outcome: IntentOutcome
    diagnostics: tuple[str, ...] = ()
    emitted_spec: Mapping[str, object] | None = None
    schema_version: Literal[1] = CONVERSATIONAL_INTENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CONVERSATIONAL_INTENT_SCHEMA_VERSION:
            raise IntentRecordError("unsupported conversational intent schema version")
        _text(self.original_request, "original request")
        span_ids = tuple(item.span_id for item in self.source_spans)
        field_ids = tuple(item.field_id for item in self.fields)
        ambiguity_ids = tuple(item.ambiguity_id for item in self.ambiguities)
        step_ids = tuple(item.step_id for item in self.interpretation_steps)
        _sorted_unique(span_ids, "source span IDs")
        _sorted_unique(field_ids, "intent field IDs")
        _sorted_unique(ambiguity_ids, "ambiguity IDs")
        _sorted_unique(step_ids, "interpretation step IDs")
        spans = {item.span_id: item for item in self.source_spans}
        fields = {item.field_id: item for item in self.fields}
        for span in self.source_spans:
            if span.end > len(self.original_request):
                raise IntentRecordError("source span exceeds original request")
            if self.original_request[span.start : span.end] != span.text:
                raise IntentRecordError("source span text does not match original request")
        for field in self.fields:
            if any(span_id not in spans for span_id in field.source_span_ids):
                raise IntentRecordError("intent field references an unknown source span")
        for ambiguity in self.ambiguities:
            if ambiguity.field_id not in fields:
                raise IntentRecordError("ambiguity references an unknown intent field")
        for step in self.interpretation_steps:
            if any(span_id not in spans for span_id in step.input_span_ids):
                raise IntentRecordError("interpretation step references an unknown span")
            if any(field_id not in fields for field_id in step.output_field_ids):
                raise IntentRecordError("interpretation step references an unknown field")
            if any(ref not in self.provenance.evidence_refs for ref in step.provenance_refs):
                raise IntentRecordError("interpretation step references unknown provenance")
        _sorted_unique(self.diagnostics, "intent diagnostics")
        if self.outcome is IntentOutcome.EXECUTABLE:
            if self.emitted_spec is None:
                raise IntentRecordError("executable intent requires an emitted spec")
            if any(item.required for item in self.ambiguities):
                raise IntentRecordError("executable intent cannot retain required ambiguity")
        elif self.emitted_spec is not None:
            raise IntentRecordError("non-executable intent cannot emit a spec")
        if not self.diagnostics:
            raise IntentRecordError("intent records require a diagnostic or outcome explanation")
        if self.emitted_spec is not None:
            _canonical(dict(self.emitted_spec))

    @property
    def request_digest(self) -> str:
        return _sha256(self.original_request)

    @property
    def spec_digest(self) -> str | None:
        if self.emitted_spec is None:
            return None
        return _sha256(_canonical(dict(self.emitted_spec)))

    @property
    def intent_id(self) -> str:
        identity = self.to_record(include_identity=False)
        identity.pop("provenance", None)
        return f"intent_{hashlib.sha256(_canonical(identity).encode()).hexdigest()}"

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "request_digest": self.request_digest,
            "original_request": self.original_request,
            "source_spans": [item.to_record() for item in self.source_spans],
            "fields": [item.to_record() for item in self.fields],
            "ambiguities": [item.to_record() for item in self.ambiguities],
            "interpretation_steps": [
                item.to_record() for item in self.interpretation_steps
            ],
            "provenance": self.provenance.to_record(),
            "outcome": self.outcome.value,
            "diagnostics": list(self.diagnostics),
            "emitted_spec": None if self.emitted_spec is None else dict(self.emitted_spec),
            "spec_digest": self.spec_digest,
        }
        if include_identity:
            record["intent_id"] = self.intent_id
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record())

    @classmethod
    def from_json(cls, payload: str) -> IntentRecord:
        try:
            root = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise IntentRecordError("intent record is not valid JSON") from error
        record = _object(root, "intent record")
        expected = {
            "schema_version",
            "request_digest",
            "original_request",
            "source_spans",
            "fields",
            "ambiguities",
            "interpretation_steps",
            "provenance",
            "outcome",
            "diagnostics",
            "emitted_spec",
            "spec_digest",
            "intent_id",
        }
        if set(record) != expected:
            raise IntentRecordError("intent record has missing or unknown fields")
        if _integer(record["schema_version"], "schema_version") != 1:
            raise IntentRecordError("unsupported conversational intent schema version")
        original = _string(record["original_request"], "original_request")
        source_spans = tuple(
            _source_span(item) for item in _array(record["source_spans"], "source_spans")
        )
        fields = tuple(_field(item) for item in _array(record["fields"], "fields"))
        ambiguities = tuple(
            _ambiguity(item) for item in _array(record["ambiguities"], "ambiguities")
        )
        steps = tuple(
            _step(item)
            for item in _array(record["interpretation_steps"], "interpretation_steps")
        )
        provenance = _provenance(record["provenance"])
        try:
            outcome = IntentOutcome(_string(record["outcome"], "outcome"))
        except ValueError as error:
            raise IntentRecordError("intent outcome is unknown") from error
        emitted = record["emitted_spec"]
        if emitted is not None:
            emitted = _object(emitted, "emitted_spec")
        result = cls(
            original,
            source_spans,
            fields,
            ambiguities,
            steps,
            provenance,
            outcome,
            _strings(record["diagnostics"], "diagnostics"),
            emitted,
        )
        if _string(record["request_digest"], "request_digest") != result.request_digest:
            raise IntentRecordError("request digest does not match original request")
        spec_digest = record["spec_digest"]
        if spec_digest != result.spec_digest:
            raise IntentRecordError("spec digest does not match emitted spec")
        if _string(record["intent_id"], "intent_id") != result.intent_id:
            raise IntentRecordError("intent ID does not match canonical record")
        return result


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise IntentRecordError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise IntentRecordError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise IntentRecordError(f"{name} must be a string")
    return value


def _required_string(value: object, name: str) -> str:
    return _text(value, name)


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise IntentRecordError(f"{name} must be an integer")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    return tuple(_required_string(item, f"{name} item") for item in _array(value, name))


def _source_span(value: object) -> SourceSpan:
    record = _object(value, "source span")
    if set(record) != {"span_id", "start", "end", "text"}:
        raise IntentRecordError("source span has missing or unknown fields")
    return SourceSpan(
        _identifier(record["span_id"], "source span ID"),
        _integer(record["start"], "source span start"),
        _integer(record["end"], "source span end"),
        _text(record["text"], "source span text"),
    )


def _field(value: object) -> IntentField:
    record = _object(value, "intent field")
    if set(record) != {
        "field_id",
        "value",
        "unit",
        "confidence",
        "source_span_ids",
        "required",
    }:
        raise IntentRecordError("intent field has missing or unknown fields")
    unit = record["unit"]
    if unit is not None:
        unit = _identifier(unit, "intent field unit")
    confidence = record["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise IntentRecordError("intent field confidence must be numeric")
    required = record["required"]
    if not isinstance(required, bool):
        raise IntentRecordError("intent field required must be boolean")
    return IntentField(
        _identifier(record["field_id"], "intent field ID"),
        record["value"],
        unit,
        float(confidence),
        _strings(record["source_span_ids"], "source_span_ids"),
        required,
    )


def _ambiguity(value: object) -> AmbiguityRecord:
    record = _object(value, "ambiguity")
    expected = {"ambiguity_id", "field_id", "category", "detail", "alternatives", "required"}
    if set(record) != expected:
        raise IntentRecordError("ambiguity has missing or unknown fields")
    required = record["required"]
    if not isinstance(required, bool):
        raise IntentRecordError("ambiguity required must be boolean")
    return AmbiguityRecord(
        _identifier(record["ambiguity_id"], "ambiguity ID"),
        _identifier(record["field_id"], "ambiguity field ID"),
        _identifier(record["category"], "ambiguity category"),
        _text(record["detail"], "ambiguity detail"),
        _strings(record["alternatives"], "ambiguity alternatives"),
        required,
    )


def _step(value: object) -> InterpretationStep:
    record = _object(value, "interpretation step")
    expected = {
        "step_id",
        "operation",
        "input_span_ids",
        "output_field_ids",
        "confidence",
        "provenance_refs",
    }
    if set(record) != expected:
        raise IntentRecordError("interpretation step has missing or unknown fields")
    confidence = record["confidence"]
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        raise IntentRecordError("interpretation step confidence must be numeric")
    return InterpretationStep(
        _identifier(record["step_id"], "interpretation step ID"),
        _identifier(record["operation"], "interpretation operation"),
        _strings(record["input_span_ids"], "interpretation input spans"),
        _strings(record["output_field_ids"], "interpretation output fields"),
        float(confidence),
        _strings(record["provenance_refs"], "interpretation provenance references"),
    )


def _provenance(value: object) -> IntentProvenance:
    record = _object(value, "provenance")
    expected = {
        "source_revision",
        "provider_id",
        "provider_revision",
        "tool_revision",
        "evidence_refs",
    }
    if set(record) != expected:
        raise IntentRecordError("provenance has missing or unknown fields")
    return IntentProvenance(
        _text(record["source_revision"], "source revision"),
        _identifier(record["provider_id"], "provider ID"),
        _text(record["provider_revision"], "provider revision"),
        _text(record["tool_revision"], "tool revision"),
        _strings(record["evidence_refs"], "provenance evidence references"),
    )


__all__ = [
    "CONVERSATIONAL_INTENT_SCHEMA_VERSION",
    "AmbiguityRecord",
    "IntentField",
    "IntentOutcome",
    "IntentProvenance",
    "IntentRecord",
    "IntentRecordError",
    "InterpretationStep",
    "SourceSpan",
]
