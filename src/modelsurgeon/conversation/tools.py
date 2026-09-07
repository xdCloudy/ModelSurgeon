"""Fail-closed, capability-scoped conversational tool schemas.

This module defines the control-plane contract only.  It deliberately has no
executor, callback, subprocess, filesystem, network, or model-session hook.
The deterministic engine remains the only authority that can implement an
operation represented by one of these schemas.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, cast

from modelsurgeon.experiments.identity import canonical_identity_json

CONVERSATIONAL_TOOL_SCHEMA_VERSION: Literal[1] = 1
TOOL_SCHEMA_VERSION: Literal[1] = CONVERSATIONAL_TOOL_SCHEMA_VERSION
MAX_TOOL_INPUT_BYTES: Final[int] = 1 << 20

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_FORBIDDEN_SCHEMA_TEXT = re.compile(
    r"(?i)(tensor|shell|python|filesystem|network|command|callback|executor|"
    r"executable|remove|delete)"
)
_SCHEMA_KEYS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "description",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "pattern",
    }
)

type JSONValue = (
    bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None
)
type JSONSchema = Mapping[str, JSONValue]


class ToolContractError(ValueError):
    """Raised when a tool contract, request, or result is unsafe or invalid."""


class ToolCapability(StrEnum):
    """Capability required to request one bounded operation."""

    INSPECT_MODEL = "inspect_model"
    PREVIEW_PLAN = "preview_plan"
    QUERY_EVIDENCE = "query_evidence"
    EXECUTE_APPROVED_PLAN = "execute_approved_plan"


class ToolName(StrEnum):
    """The finite allowlist of conversational operations."""

    INSPECT_MODEL = "inspect_model"
    PREVIEW_PLAN = "preview_plan"
    QUERY_EVIDENCE = "query_evidence"
    EXECUTE_APPROVED_PLAN = "execute_approved_plan"


class ToolAccess(StrEnum):
    READ_ONLY = "read_only"
    CONSEQUENTIAL = "consequential"


class ToolOutcome(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    REFUSED = "refused"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class ToolFailureCode(StrEnum):
    UNKNOWN_TOOL = "unknown_tool"
    UNKNOWN_SCHEMA_VERSION = "unknown_schema_version"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"
    TOOL_ID_MISMATCH = "tool_id_mismatch"
    INVALID_INPUT = "invalid_input"
    BUDGET_EXCEEDED = "budget_exceeded"
    APPROVAL_REQUIRED = "approval_required"
    EXECUTION_FAILED = "execution_failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolContractError(f"{label} must be non-empty text")
    return value


def _identifier(value: object, label: str) -> str:
    result = _text(value, label)
    if _IDENTIFIER.fullmatch(result) is None:
        raise ToolContractError(f"{label} is not a canonical identifier")
    return result


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _DIGEST.fullmatch(result) is None:
        raise ToolContractError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _canonical(value: object, label: str = "tool record") -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError) as error:
        raise ToolContractError(f"{label} must be JSON-compatible") from error


def _sorted_unique(values: tuple[str, ...], label: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ToolContractError(f"{label} must be sorted and unique")


def _sorted_outcomes(values: tuple[ToolOutcome, ...]) -> None:
    if values != tuple(sorted(set(values), key=lambda item: item.value)):
        raise ToolContractError("tool failure semantics must be sorted and unique")


def _json_object(value: object, label: str) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ToolContractError(f"{label} must be a JSON object")
    try:
        encoded = _canonical(dict(value), label)
        decoded = json.loads(encoded)
    except json.JSONDecodeError as error:  # pragma: no cover - canonical JSON is valid
        raise ToolContractError(f"{label} must be a JSON object") from error
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ToolContractError(f"{label} must be a JSON object")
    return cast(dict[str, JSONValue], decoded)


def _schema_text_is_safe(value: object, *, path: str) -> None:
    if isinstance(value, str) and _FORBIDDEN_SCHEMA_TEXT.search(value):
        raise ToolContractError(f"tool schema contains forbidden authority at {path}")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ToolContractError(f"tool schema key at {path} is not text")
            if key == "properties":
                if not isinstance(child, Mapping):
                    raise ToolContractError(f"tool schema properties at {path} must be an object")
                for property_name in child:
                    if not isinstance(property_name, str) or _FORBIDDEN_SCHEMA_TEXT.search(
                        property_name
                    ):
                        raise ToolContractError(
                            f"tool schema contains forbidden property at {path}.properties"
                        )
            _schema_text_is_safe(child, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _schema_text_is_safe(child, path=f"{path}[{index}]")


def _validate_schema_shape(schema: object, *, path: str = "schema") -> dict[str, JSONValue]:
    """Validate the deliberately small JSON-schema dialect used at this boundary."""

    root = _json_object(schema, path)
    _schema_text_is_safe(root, path=path)
    unknown = set(root) - _SCHEMA_KEYS
    if unknown:
        raise ToolContractError(f"{path} has unknown keywords: {sorted(unknown)}")
    schema_type = root.get("type")
    if schema_type not in {"object", "array", "string", "integer", "number", "boolean"}:
        raise ToolContractError(f"{path} must declare one supported type")
    for name in ("minLength", "maxLength", "minimum", "maximum", "minItems", "maxItems"):
        raw = root.get(name)
        if name in root and (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or raw < 0
        ):
            raise ToolContractError(f"{path}.{name} must be a finite non-negative number")
    if "properties" in root:
        properties = root["properties"]
        if not isinstance(properties, Mapping):
            raise ToolContractError(f"{path}.properties must be an object")
        for property_name, child in properties.items():
            _identifier(property_name, f"{path} property name")
            _validate_schema_shape(child, path=f"{path}.properties.{property_name}")
    if schema_type == "object":
        if root.get("additionalProperties") is not False:
            raise ToolContractError(f"{path} must reject additional properties")
        required = root.get("required", [])
        if not isinstance(required, list) or any(not isinstance(item, str) for item in required):
            raise ToolContractError(f"{path}.required must be a string array")
        required_names = tuple(cast(str, item) for item in required)
        _sorted_unique(required_names, f"{path}.required")
        properties = root.get("properties", {})
        if not isinstance(properties, Mapping) or any(
            item not in properties for item in required_names
        ):
            raise ToolContractError(f"{path}.required references an unknown property")
    elif "properties" in root or "required" in root or "additionalProperties" in root:
        raise ToolContractError(f"{path} uses object keywords for a non-object type")
    if "items" in root:
        if schema_type != "array":
            raise ToolContractError(f"{path}.items requires an array schema")
        _validate_schema_shape(root["items"], path=f"{path}.items")
    if "enum" in root:
        enum = root["enum"]
        if not isinstance(enum, list) or not enum:
            raise ToolContractError(f"{path}.enum must be a non-empty array")
        _canonical(enum, f"{path}.enum")
    if "pattern" in root and not isinstance(root["pattern"], str):
        raise ToolContractError(f"{path}.pattern must be text")
    return root


def _matches_schema(schema: JSONSchema, value: object, *, path: str = "input") -> None:
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(value, Mapping):
            raise ToolContractError(f"{path} must be an object")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, Mapping) or not isinstance(required, list):
            raise ToolContractError(f"{path} has an invalid object schema")
        property_names = set(cast(Mapping[str, JSONValue], properties))
        required_names = {cast(str, item) for item in required}
        unknown = set(value) - property_names
        if unknown:
            raise ToolContractError(f"{path} has unknown fields: {sorted(unknown)}")
        missing = required_names - set(value)
        if missing:
            raise ToolContractError(f"{path} is missing fields: {sorted(missing)}")
        for key, child in properties.items():
            if key in value:
                _matches_schema(cast(JSONSchema, child), value[key], path=f"{path}.{key}")
    elif schema_type == "array":
        if not isinstance(value, list):
            raise ToolContractError(f"{path} must be an array")
        if "minItems" in schema and len(value) < cast(int, schema["minItems"]):
            raise ToolContractError(f"{path} has too few items")
        if "maxItems" in schema and len(value) > cast(int, schema["maxItems"]):
            raise ToolContractError(f"{path} has too many items")
        child = schema.get("items")
        if child is not None:
            for index, item in enumerate(value):
                _matches_schema(cast(JSONSchema, child), item, path=f"{path}[{index}]")
    elif schema_type == "string":
        if not isinstance(value, str):
            raise ToolContractError(f"{path} must be text")
        if "minLength" in schema and len(value) < cast(int, schema["minLength"]):
            raise ToolContractError(f"{path} is too short")
        if "maxLength" in schema and len(value) > cast(int, schema["maxLength"]):
            raise ToolContractError(f"{path} is too long")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(cast(str, pattern), value) is None:
            raise ToolContractError(f"{path} does not match its declared pattern")
    elif schema_type == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            raise ToolContractError(f"{path} must be an integer")
    elif schema_type == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ToolContractError(f"{path} must be a number")
    elif schema_type == "boolean" and not isinstance(value, bool):
        raise ToolContractError(f"{path} must be boolean")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and value < cast(float, minimum):
            raise ToolContractError(f"{path} is below its minimum")
        if maximum is not None and value > cast(float, maximum):
            raise ToolContractError(f"{path} exceeds its maximum")
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        raise ToolContractError(f"{path} has a value outside its enum")
    if "const" in schema and value != schema["const"]:
        raise ToolContractError(f"{path} does not match its constant")


@dataclass(frozen=True, slots=True)
class ToolBudget:
    """Hard per-call limits declared by a tool and optionally tightened by a caller."""

    max_wall_seconds: float
    max_memory_bytes: int
    max_evaluation_count: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_wall_seconds, bool)
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds <= 0
        ):
            raise ToolContractError("tool wall-time budget must be finite and positive")
        for name, value in (
            ("max_memory_bytes", self.max_memory_bytes),
            ("max_evaluation_count", self.max_evaluation_count),
            ("max_output_bytes", self.max_output_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ToolContractError(f"{name} must be a positive integer")

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_memory_bytes": self.max_memory_bytes,
            "max_evaluation_count": self.max_evaluation_count,
            "max_output_bytes": self.max_output_bytes,
        }


_ALL_FAILURE_SEMANTICS: tuple[ToolOutcome, ...] = tuple(
    sorted(
        (
            ToolOutcome.CANCELLED,
            ToolOutcome.FAILED,
            ToolOutcome.REFUSED,
            ToolOutcome.TIMEOUT,
            ToolOutcome.UNKNOWN,
            ToolOutcome.UNSUPPORTED,
        ),
        key=lambda item: item.value,
    )
)


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One allowlisted tool, including its complete boundary contract."""

    name: ToolName
    owner: str
    capability: ToolCapability
    access: ToolAccess
    input_schema: JSONSchema
    output_schema: JSONSchema
    budget: ToolBudget
    failure_semantics: tuple[ToolOutcome, ...] = _ALL_FAILURE_SEMANTICS
    approval_required: bool = False
    schema_version: Literal[1] = TOOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_SCHEMA_VERSION:
            raise ToolContractError("unsupported conversational tool schema version")
        _identifier(self.owner, "tool owner")
        if self.capability.value != self.name.value:
            raise ToolContractError("tool name and capability must be paired")
        if self.access is ToolAccess.CONSEQUENTIAL and not self.approval_required:
            raise ToolContractError("consequential tools require approval")
        if self.access is ToolAccess.READ_ONLY and self.approval_required:
            raise ToolContractError("read-only tools cannot require consequential approval")
        _sorted_outcomes(self.failure_semantics)
        if ToolOutcome.SUPPORTED in self.failure_semantics:
            raise ToolContractError("failure semantics cannot contain supported")
        if set(self.failure_semantics) != set(_ALL_FAILURE_SEMANTICS):
            raise ToolContractError("tool failure semantics must enumerate every failure state")
        _validate_schema_shape(self.input_schema, path=f"{self.name.value}.input_schema")
        _validate_schema_shape(self.output_schema, path=f"{self.name.value}.output_schema")

    @property
    def tool_id(self) -> str:
        payload = self.to_record(include_identity=False)
        return f"tool_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"

    def validate_input(self, value: Mapping[str, JSONValue]) -> None:
        """Validate an untrusted request input against this tool's strict schema."""

        _matches_schema(self.input_schema, value, path=f"{self.name.value}.input")

    def validate_output(self, value: Mapping[str, JSONValue]) -> None:
        """Validate a later engine adapter's typed output before publication."""

        _matches_schema(self.output_schema, value, path=f"{self.name.value}.output")

    def to_record(self, *, include_identity: bool = True) -> dict[str, JSONValue]:
        record: dict[str, JSONValue] = {
            "schema_version": self.schema_version,
            "name": self.name.value,
            "owner": self.owner,
            "capability": self.capability.value,
            "access": self.access.value,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "budget": self.budget.to_record(),
            "failure_semantics": [item.value for item in self.failure_semantics],
            "approval_required": self.approval_required,
        }
        if include_identity:
            record["tool_id"] = self.tool_id
        return record


@dataclass(frozen=True, slots=True)
class ToolRequest:
    """A model request bound to one capability and deterministic request identity."""

    request_id: str
    name: str
    capability: str
    input: Mapping[str, JSONValue]
    budget: ToolBudget
    tool_id: str | None = None
    approval_id: str | None = None
    schema_version: Literal[1] = TOOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_SCHEMA_VERSION:
            raise ToolContractError("unsupported conversational tool schema version")
        _identifier(self.request_id, "tool request ID")
        _identifier(self.name, "tool name")
        _identifier(self.capability, "tool capability")
        if self.tool_id is not None:
            _identifier(self.tool_id, "tool ID")
        if self.approval_id is not None:
            _identifier(self.approval_id, "approval ID")
        encoded = _canonical(dict(self.input), "tool input")
        if len(encoded.encode("utf-8")) > MAX_TOOL_INPUT_BYTES:
            raise ToolContractError("tool input exceeds the hard size limit")
        expected = deterministic_tool_request_id(
            self.name,
            self.capability,
            dict(self.input),
            self.budget,
            tool_id=self.tool_id,
            approval_id=self.approval_id,
            schema_version=self.schema_version,
        )
        if self.request_id != expected:
            raise ToolContractError("tool request ID does not match canonical request")

    @classmethod
    def create(
        cls,
        definition: ToolDefinition,
        input: Mapping[str, JSONValue],
        *,
        budget: ToolBudget | None = None,
        approval_id: str | None = None,
    ) -> ToolRequest:
        selected_budget = budget or definition.budget
        name = definition.name.value
        capability = definition.capability.value
        request_id = deterministic_tool_request_id(
            name,
            capability,
            dict(input),
            selected_budget,
            tool_id=definition.tool_id,
            approval_id=approval_id,
        )
        return cls(
            request_id,
            name,
            capability,
            dict(input),
            selected_budget,
            definition.tool_id,
            approval_id,
        )

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "name": self.name,
            "capability": self.capability,
            "input": dict(self.input),
            "budget": self.budget.to_record(),
            "tool_id": self.tool_id,
            "approval_id": self.approval_id,
        }

    @classmethod
    def from_record(cls, payload: object) -> ToolRequest:
        record = _json_object(payload, "tool request")
        expected = {
            "schema_version",
            "request_id",
            "name",
            "capability",
            "input",
            "budget",
            "tool_id",
            "approval_id",
        }
        if set(record) != expected:
            raise ToolContractError("tool request has missing or unknown fields")
        version = record["schema_version"]
        if version != TOOL_SCHEMA_VERSION:
            raise ToolContractError("unsupported conversational tool schema version")
        budget = _budget_from_record(record["budget"])
        input_record = _json_object(record["input"], "tool input")
        return cls(
            cast(str, record["request_id"]),
            cast(str, record["name"]),
            cast(str, record["capability"]),
            input_record,
            budget,
            None if record["tool_id"] is None else cast(str, record["tool_id"]),
            None if record["approval_id"] is None else cast(str, record["approval_id"]),
        )


def deterministic_tool_request_id(
    name: str,
    capability: str,
    input: Mapping[str, JSONValue],
    budget: ToolBudget,
    *,
    tool_id: str | None = None,
    approval_id: str | None = None,
    schema_version: int = TOOL_SCHEMA_VERSION,
) -> str:
    """Derive a replay-stable ID from the complete bounded request envelope."""

    payload = {
        "schema_version": schema_version,
        "name": name,
        "capability": capability,
        "tool_id": tool_id,
        "approval_id": approval_id,
        "input": dict(input),
        "budget": budget.to_record(),
    }
    return f"tool_request_{hashlib.sha256(_canonical(payload).encode()).hexdigest()}"


def _budget_from_record(value: object) -> ToolBudget:
    record = _json_object(value, "tool request budget")
    expected = {
        "max_wall_seconds",
        "max_memory_bytes",
        "max_evaluation_count",
        "max_output_bytes",
    }
    if set(record) != expected:
        raise ToolContractError("tool request budget has missing or unknown fields")
    wall_seconds = record["max_wall_seconds"]
    if not isinstance(wall_seconds, (int, float)) or isinstance(wall_seconds, bool):
        raise ToolContractError("tool request wall-time budget must be numeric")
    integer_values: list[int] = []
    for name in ("max_memory_bytes", "max_evaluation_count", "max_output_bytes"):
        item = record[name]
        if not isinstance(item, int) or isinstance(item, bool):
            raise ToolContractError(f"tool request {name} must be an integer")
        integer_values.append(item)
    return ToolBudget(wall_seconds, *integer_values)


@dataclass(frozen=True, slots=True)
class ToolFailure:
    code: ToolFailureCode
    detail: str
    request_id: str
    retryable: bool = False

    def __post_init__(self) -> None:
        _text(self.detail, "tool failure detail")
        _identifier(self.request_id, "tool failure request ID")

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "code": self.code.value,
            "detail": self.detail,
            "request_id": self.request_id,
            "retryable": self.retryable,
        }


@dataclass(frozen=True, slots=True)
class ToolProvenance:
    owner: str
    tool_id: str
    request_digest: str

    def __post_init__(self) -> None:
        _identifier(self.owner, "tool provenance owner")
        _identifier(self.tool_id, "tool provenance ID")
        _digest(self.request_digest, "tool provenance request digest")

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "owner": self.owner,
            "tool_id": self.tool_id,
            "request_digest": self.request_digest,
        }


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Typed result envelope; execution implementations are intentionally elsewhere."""

    request_id: str
    name: str
    outcome: ToolOutcome
    provenance: ToolProvenance
    output: Mapping[str, JSONValue] | None = None
    failure: ToolFailure | None = None

    def __post_init__(self) -> None:
        _identifier(self.request_id, "tool result request ID")
        _identifier(self.name, "tool result name")
        if self.provenance.request_digest == "":  # pragma: no cover - constructor validates
            raise ToolContractError("tool result requires request provenance")
        if self.outcome is ToolOutcome.SUPPORTED:
            if self.output is None or self.failure is not None:
                raise ToolContractError("supported tool result requires output and no failure")
            _canonical(dict(self.output), "tool result output")
        elif self.output is not None:
            raise ToolContractError("non-supported tool result cannot contain output")
        if self.outcome is not ToolOutcome.SUPPORTED and self.failure is None:
            raise ToolContractError("non-supported tool result requires a typed failure")
        if self.failure is not None and self.failure.request_id != self.request_id:
            raise ToolContractError("tool failure request ID does not match result")

    def to_record(self) -> dict[str, JSONValue]:
        return {
            "schema_version": TOOL_SCHEMA_VERSION,
            "request_id": self.request_id,
            "name": self.name,
            "outcome": self.outcome.value,
            "provenance": self.provenance.to_record(),
            "output": None if self.output is None else dict(self.output),
            "failure": None if self.failure is None else self.failure.to_record(),
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())


@dataclass(frozen=True, slots=True)
class ToolNegotiation:
    """Pre-execution capability decision for a request."""

    request_id: str
    name: str
    outcome: ToolOutcome
    tool: ToolDefinition | None = None
    failure: ToolFailure | None = None

    def __post_init__(self) -> None:
        _identifier(self.request_id, "negotiation request ID")
        _identifier(self.name, "negotiation tool name")
        if self.outcome is ToolOutcome.SUPPORTED:
            if self.tool is None or self.failure is not None:
                raise ToolContractError("supported negotiation requires a tool and no failure")
        elif self.tool is not None or self.failure is None:
            raise ToolContractError("refused negotiation requires only a typed failure")
        if self.failure is not None and self.failure.request_id != self.request_id:
            raise ToolContractError("negotiation failure request ID does not match request")


class ToolCatalog:
    """Finite registry that validates requests without executing them."""

    def __init__(self, definitions: Sequence[ToolDefinition] | None = None) -> None:
        selected = default_tool_definitions() if definitions is None else tuple(definitions)
        names = tuple(item.name.value for item in selected)
        if len(names) != len(set(names)):
            raise ToolContractError("tool names must be unique")
        self._definitions = {item.name.value: item for item in selected}

    @property
    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._definitions.values())

    def definition(self, name: str) -> ToolDefinition | None:
        return self._definitions.get(name)

    def negotiate(self, request: ToolRequest) -> ToolNegotiation:
        definition = self._definitions.get(request.name)
        if definition is None:
            return self._refused(
                request,
                ToolOutcome.UNKNOWN,
                ToolFailureCode.UNKNOWN_TOOL,
                "tool is not in the allowlist",
            )
        if request.capability != definition.capability.value:
            return self._refused(
                request,
                ToolOutcome.UNSUPPORTED,
                ToolFailureCode.UNSUPPORTED_CAPABILITY,
                "requested capability is not the tool capability",
            )
        if request.tool_id is not None and request.tool_id != definition.tool_id:
            return self._refused(
                request,
                ToolOutcome.REFUSED,
                ToolFailureCode.TOOL_ID_MISMATCH,
                "request is bound to a different tool schema",
            )
        try:
            definition.validate_input(request.input)
        except ToolContractError as error:
            return self._refused(
                request, ToolOutcome.REFUSED, ToolFailureCode.INVALID_INPUT, str(error)
            )
        if _budget_exceeds(request.budget, definition.budget):
            return self._refused(
                request,
                ToolOutcome.REFUSED,
                ToolFailureCode.BUDGET_EXCEEDED,
                "request budget exceeds the tool declaration",
            )
        if definition.approval_required and request.approval_id is None:
            return self._refused(
                request,
                ToolOutcome.REFUSED,
                ToolFailureCode.APPROVAL_REQUIRED,
                "consequential tool requires an approval identity",
            )
        return ToolNegotiation(request.request_id, request.name, ToolOutcome.SUPPORTED, definition)

    def negotiate_record(self, payload: object) -> ToolNegotiation:
        """Decode untrusted model data and return a refusal instead of executing it."""

        try:
            request = ToolRequest.from_record(payload)
        except ToolContractError as error:
            record = payload if isinstance(payload, Mapping) else {}
            request_id = record.get("request_id") if isinstance(record, Mapping) else None
            request_name = record.get("name") if isinstance(record, Mapping) else None
            safe_request_id = (
                request_id
                if isinstance(request_id, str) and _IDENTIFIER.fullmatch(request_id)
                else "tool_request_invalid"
            )
            safe_name = (
                request_name
                if isinstance(request_name, str) and _IDENTIFIER.fullmatch(request_name)
                else "unknown"
            )
            return ToolNegotiation(
                safe_request_id,
                safe_name,
                ToolOutcome.REFUSED,
                failure=ToolFailure(
                    ToolFailureCode.UNKNOWN_SCHEMA_VERSION
                    if "schema version" in str(error)
                    else ToolFailureCode.INVALID_INPUT,
                    str(error),
                    safe_request_id,
                ),
            )
        return self.negotiate(request)

    @staticmethod
    def _refused(
        request: ToolRequest,
        outcome: ToolOutcome,
        code: ToolFailureCode,
        detail: str,
    ) -> ToolNegotiation:
        return ToolNegotiation(
            request.request_id,
            request.name,
            outcome,
            failure=ToolFailure(code, detail, request.request_id),
        )


def _budget_exceeds(requested: ToolBudget, declared: ToolBudget) -> bool:
    return (
        requested.max_wall_seconds > declared.max_wall_seconds
        or requested.max_memory_bytes > declared.max_memory_bytes
        or requested.max_evaluation_count > declared.max_evaluation_count
        or requested.max_output_bytes > declared.max_output_bytes
    )


def _object_schema(
    properties: dict[str, JSONSchema], required: tuple[str, ...]
) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {
        "type": "object",
        "properties": {key: dict(value) for key, value in properties.items()},
        "required": [cast(JSONValue, item) for item in sorted(required)],
        "additionalProperties": False,
    }
    return result


def _string_schema(*, pattern: str | None = None, max_length: int = 128) -> dict[str, JSONValue]:
    schema: dict[str, JSONValue] = {"type": "string", "minLength": 1, "maxLength": max_length}
    if pattern is not None:
        schema["pattern"] = pattern
    return schema


def default_tool_definitions() -> tuple[ToolDefinition, ...]:
    """Return the version-1 allowlist; callers cannot add a model callback here."""

    ref = _string_schema(pattern=r"[a-z][a-z0-9_.:-]{0,127}")
    digest = _string_schema(pattern=r"sha256:[0-9a-f]{64}", max_length=71)
    status: JSONSchema = {
        "type": "string",
        "enum": ["supported", "unsupported", "unknown"],
    }
    evidence_refs: JSONSchema = {"type": "array", "items": ref, "maxItems": 256}
    common_failures = _ALL_FAILURE_SEMANTICS
    return (
        ToolDefinition(
            ToolName.INSPECT_MODEL,
            "modelsurgeon.conversation",
            ToolCapability.INSPECT_MODEL,
            ToolAccess.READ_ONLY,
            _object_schema({"model_ref": ref}, ("model_ref",)),
            _object_schema(
                {
                    "model_ref": ref,
                    "status": status,
                    "capability_refs": {
                        "type": "array",
                        "items": ref,
                        "maxItems": 256,
                    },
                    "provenance_ref": ref,
                },
                ("model_ref", "status", "capability_refs", "provenance_ref"),
            ),
            ToolBudget(5.0, 64 * 1024 * 1024, 1, 64 * 1024),
            common_failures,
        ),
        ToolDefinition(
            ToolName.PREVIEW_PLAN,
            "modelsurgeon.conversation",
            ToolCapability.PREVIEW_PLAN,
            ToolAccess.READ_ONLY,
            _object_schema({"intent_id": ref, "spec_digest": digest}, ("intent_id", "spec_digest")),
            _object_schema(
                {
                    "plan_id": ref,
                    "spec_digest": digest,
                    "status": status,
                    "diagnostic": _string_schema(max_length=2048),
                    "evidence_refs": evidence_refs,
                },
                ("plan_id", "spec_digest", "status", "diagnostic", "evidence_refs"),
            ),
            ToolBudget(30.0, 256 * 1024 * 1024, 1, 256 * 1024),
            common_failures,
        ),
        ToolDefinition(
            ToolName.QUERY_EVIDENCE,
            "modelsurgeon.conversation",
            ToolCapability.QUERY_EVIDENCE,
            ToolAccess.READ_ONLY,
            _object_schema(
                {
                    "evidence_ref": ref,
                    "cursor": {"type": "string", "maxLength": 256},
                    "max_records": {"type": "integer", "minimum": 1, "maximum": 256},
                },
                ("evidence_ref",),
            ),
            _object_schema(
                {
                    "evidence_ref": ref,
                    "records": {
                        "type": "array",
                        "maxItems": 256,
                        "items": _object_schema(
                            {
                                "record_id": ref,
                                "outcome": {
                                    "type": "string",
                                    "enum": [item.value for item in ToolOutcome],
                                },
                                "digest": digest,
                            },
                            ("record_id", "outcome", "digest"),
                        ),
                    },
                    "next_cursor": {"type": "string", "maxLength": 256},
                },
                ("evidence_ref", "records"),
            ),
            ToolBudget(10.0, 64 * 1024 * 1024, 1, 512 * 1024),
            common_failures,
        ),
        ToolDefinition(
            ToolName.EXECUTE_APPROVED_PLAN,
            "modelsurgeon.conversation",
            ToolCapability.EXECUTE_APPROVED_PLAN,
            ToolAccess.CONSEQUENTIAL,
            _object_schema(
                {"plan_id": ref, "plan_digest": digest, "approval_id": ref},
                ("plan_id", "plan_digest", "approval_id"),
            ),
            _object_schema(
                {
                    "run_id": ref,
                    "outcome": {"type": "string", "enum": ["supported", "failed", "unknown"]},
                    "evidence_refs": evidence_refs,
                    "artifact_ref": {"type": "string", "maxLength": 128},
                },
                ("run_id", "outcome", "evidence_refs", "artifact_ref"),
            ),
            ToolBudget(600.0, 2 * 1024 * 1024 * 1024, 1, 512 * 1024),
            common_failures,
            approval_required=True,
        ),
    )


DEFAULT_TOOL_CATALOG = ToolCatalog()


__all__ = [
    "CONVERSATIONAL_TOOL_SCHEMA_VERSION",
    "DEFAULT_TOOL_CATALOG",
    "MAX_TOOL_INPUT_BYTES",
    "TOOL_SCHEMA_VERSION",
    "JSONSchema",
    "JSONValue",
    "ToolAccess",
    "ToolBudget",
    "ToolCapability",
    "ToolCatalog",
    "ToolContractError",
    "ToolDefinition",
    "ToolFailure",
    "ToolFailureCode",
    "ToolName",
    "ToolNegotiation",
    "ToolOutcome",
    "ToolProvenance",
    "ToolRequest",
    "ToolResult",
    "default_tool_definitions",
    "deterministic_tool_request_id",
]
