"""Canonical mutation plan, outcome, provenance, and identity serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, cast

from modelsurgeon.graph import ComponentId, ComponentIdentityMapping, IdentityRemapError
from modelsurgeon.surgery.contracts import (
    MUTATION_SCHEMA_VERSION,
    MutationContractError,
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationPrecondition,
    MutationPrimitive,
    MutationRequest,
)

MUTATION_RECORD_SCHEMA_VERSION: Literal[1] = 1
REDACTED_LOCAL_PATH = "<redacted-local-path>"


class MutationRecordError(MutationContractError):
    """Raised when a persisted mutation record is malformed or inconsistent."""


class MutationOutcomeStatus(StrEnum):
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class MutationProvenance:
    input_revision: str
    tool_revision: str
    input_path: str | None = None

    def __post_init__(self) -> None:
        if not self.input_revision or not self.tool_revision:
            raise MutationRecordError("provenance revisions must be non-empty")
        if self.input_path is not None and not self.input_path:
            raise MutationRecordError("provenance input path must not be blank")

    def to_record(self, *, redact_local_paths: bool) -> dict[str, object]:
        path = self.input_path
        if redact_local_paths and path is not None:
            path = REDACTED_LOCAL_PATH
        return {
            "input_revision": self.input_revision,
            "tool_revision": self.tool_revision,
            "input_path": path,
        }


MutationIdentityMapping = ComponentIdentityMapping


@dataclass(frozen=True, slots=True)
class MutationOutcome:
    status: MutationOutcomeStatus
    actual_delta: MutationDelta | None
    identity_mappings: tuple[MutationIdentityMapping, ...] = ()
    detail: str | None = None

    def __post_init__(self) -> None:
        sources = tuple(item.source for item in self.identity_mappings)
        if sources != tuple(sorted(sources)) or len(sources) != len(set(sources)):
            raise MutationRecordError("identity mappings must have unique canonical sources")
        if self.status is MutationOutcomeStatus.APPLIED and self.actual_delta is None:
            raise MutationRecordError("applied outcomes require actual deltas")
        if self.detail is not None and not self.detail:
            raise MutationRecordError("outcome detail must not be blank")

    def to_record(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "actual_delta": _delta_record(self.actual_delta),
            "identity_mappings": [item.to_record() for item in self.identity_mappings],
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class MutationRunRecord:
    plan: MutationPlan
    provenance: MutationProvenance
    outcome: MutationOutcome | None = None
    schema_version: Literal[1] = MUTATION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MUTATION_RECORD_SCHEMA_VERSION:
            raise MutationRecordError(
                f"unsupported mutation record schema version {self.schema_version}"
            )

    @property
    def mutation_id(self) -> str:
        return self.plan.request.mutation_id

    def to_record(self, *, redact_local_paths: bool = True) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "mutation_id": self.mutation_id,
            "plan": _plan_record(self.plan),
            "provenance": self.provenance.to_record(
                redact_local_paths=redact_local_paths
            ),
            "outcome": None if self.outcome is None else self.outcome.to_record(),
        }

    def canonical_json(self, *, redact_local_paths: bool = True) -> str:
        return json.dumps(
            self.to_record(redact_local_paths=redact_local_paths),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, payload: str) -> MutationRunRecord:
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as error:
            raise MutationRecordError("mutation record is not valid JSON") from error
        root = _object(value, "mutation record")
        if set(root) != {"schema_version", "mutation_id", "plan", "provenance", "outcome"}:
            raise MutationRecordError("mutation record has missing or unknown fields")
        if _integer(root["schema_version"], "schema_version") != 1:
            raise MutationRecordError("unsupported mutation record schema version")
        plan = _plan(root["plan"])
        mutation_id = _string(root["mutation_id"], "mutation_id")
        if mutation_id != plan.request.mutation_id:
            raise MutationRecordError("mutation identity does not match canonical request")
        provenance = _provenance(root["provenance"])
        outcome = None if root["outcome"] is None else _outcome(root["outcome"])
        return cls(plan, provenance, outcome)


def _delta_record(delta: MutationDelta | None) -> dict[str, int] | None:
    if delta is None:
        return None
    return {
        "parameters": delta.parameters,
        "flops": delta.flops,
        "memory_bytes": delta.memory_bytes,
        "storage_bytes": delta.storage_bytes,
    }


def _plan_record(plan: MutationPlan) -> dict[str, object]:
    return {
        "request": plan.request.to_record(),
        "affected_components": [str(item) for item in plan.affected_components],
        "preconditions": [
            {"key": item.key, "expected": item.expected} for item in plan.preconditions
        ],
        "expected_delta": _delta_record(plan.expected_delta),
    }


def _object(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise MutationRecordError(f"{name} must be an object")
    return cast(dict[str, object], value)


def _array(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise MutationRecordError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MutationRecordError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MutationRecordError(f"{name} must be an integer")
    return value


def _primitive(value: object, name: str) -> MutationPrimitive:
    if value is not None and not isinstance(value, (str, int, float, bool)):
        raise MutationRecordError(f"{name} must be a primitive value")
    return value


def _delta(value: object, name: str) -> MutationDelta:
    record = _object(value, name)
    keys = {"parameters", "flops", "memory_bytes", "storage_bytes"}
    if set(record) != keys:
        raise MutationRecordError(f"{name} has missing or unknown fields")
    return MutationDelta(**{key: _integer(record[key], f"{name}.{key}") for key in keys})


def _request(value: object) -> MutationRequest:
    record = _object(value, "request")
    if set(record) != {"schema_version", "kind", "targets", "parameters"}:
        raise MutationRecordError("request has missing or unknown fields")
    if _integer(record["schema_version"], "request.schema_version") != MUTATION_SCHEMA_VERSION:
        raise MutationRecordError("unsupported mutation request schema version")
    try:
        kind = MutationKind(_string(record["kind"], "request.kind"))
    except ValueError as error:
        raise MutationRecordError("request kind is unknown") from error
    targets = tuple(
        ComponentId.parse(_string(item, "request target"))
        for item in _array(record["targets"], "request.targets")
    )
    parameters = _object(record["parameters"], "request.parameters")
    return MutationRequest(
        kind,
        targets,
        tuple(
            sorted(
                (key, _primitive(item, f"parameter {key}"))
                for key, item in parameters.items()
            )
        ),
    )


def _plan(value: object) -> MutationPlan:
    record = _object(value, "plan")
    if set(record) != {"request", "affected_components", "preconditions", "expected_delta"}:
        raise MutationRecordError("plan has missing or unknown fields")
    preconditions = []
    for item in _array(record["preconditions"], "plan.preconditions"):
        entry = _object(item, "precondition")
        if set(entry) != {"key", "expected"}:
            raise MutationRecordError("precondition has missing or unknown fields")
        preconditions.append(
            MutationPrecondition(
                _string(entry["key"], "precondition.key"),
                _primitive(entry["expected"], "precondition.expected"),
            )
        )
    return MutationPlan(
        _request(record["request"]),
        tuple(
            ComponentId.parse(_string(item, "affected component"))
            for item in _array(record["affected_components"], "plan.affected_components")
        ),
        tuple(preconditions),
        _delta(record["expected_delta"], "plan.expected_delta"),
    )


def _provenance(value: object) -> MutationProvenance:
    record = _object(value, "provenance")
    if set(record) != {"input_revision", "tool_revision", "input_path"}:
        raise MutationRecordError("provenance has missing or unknown fields")
    path = record["input_path"]
    if path is not None:
        path = _string(path, "provenance.input_path")
    return MutationProvenance(
        _string(record["input_revision"], "provenance.input_revision"),
        _string(record["tool_revision"], "provenance.tool_revision"),
        path,
    )


def _mapping(value: object) -> MutationIdentityMapping:
    record = _object(value, "identity mapping")
    if set(record) != {"source", "targets", "removed", "reason"}:
        raise MutationRecordError("identity mapping has missing or unknown fields")
    targets = tuple(
        ComponentId.parse(_string(item, "mapping target"))
        for item in _array(record["targets"], "mapping.targets")
    )
    removed = record["removed"]
    if not isinstance(removed, bool) or removed != (not targets):
        raise MutationRecordError("identity mapping removed flag disagrees with targets")
    try:
        return MutationIdentityMapping(
            ComponentId.parse(_string(record["source"], "mapping.source")),
            targets,
            _string(record["reason"], "mapping.reason"),
        )
    except IdentityRemapError as error:
        raise MutationRecordError(f"invalid identity mapping: {error}") from error


def _outcome(value: object) -> MutationOutcome:
    record = _object(value, "outcome")
    if set(record) != {"status", "actual_delta", "identity_mappings", "detail"}:
        raise MutationRecordError("outcome has missing or unknown fields")
    try:
        status = MutationOutcomeStatus(_string(record["status"], "outcome.status"))
    except ValueError as error:
        raise MutationRecordError("outcome status is unknown") from error
    actual = record["actual_delta"]
    detail = record["detail"]
    if detail is not None:
        detail = _string(detail, "outcome.detail")
    return MutationOutcome(
        status,
        None if actual is None else _delta(actual, "outcome.actual_delta"),
        tuple(_mapping(item) for item in _array(record["identity_mappings"], "mappings")),
        detail,
    )
