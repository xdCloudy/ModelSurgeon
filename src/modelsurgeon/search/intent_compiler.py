"""Bounded compilation from canonical conversational intent to objective contracts.

This module is deliberately a control-plane boundary.  It validates typed fields,
constructs the existing :class:`ObjectiveContract`, and never invokes search,
mutation, tensor selection, or optimizer strategy code.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from modelsurgeon.conversation import IntentOutcome, IntentRecord

from .objective_contract import (
    ConstraintDirection,
    HardConstraint,
    MetricUnit,
    ObjectiveApprovalPolicy,
    ObjectiveContract,
    ObjectiveContractError,
    ObjectiveDirection,
    ObjectiveMode,
    ObjectiveNormalization,
    ObjectivePluginBinding,
    SoftObjective,
)

INTENT_COMPILER_SCHEMA_VERSION = 1


class DiagnosticSeverity(StrEnum):
    """Severity of a compiler diagnostic."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class IntentCompilerError(ValueError):
    """Raised only for invalid compiler API arguments, not intent refusal."""


@dataclass(frozen=True, slots=True)
class CompilerDiagnostic:
    """A deterministic, provenance-linked explanation of a compiler decision."""

    code: str
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    field_id: str | None = None
    source_span_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise IntentCompilerError("compiler diagnostics require code and message")
        if self.field_id is not None and not self.field_id.strip():
            raise IntentCompilerError("compiler diagnostic field ID cannot be blank")
        if self.source_span_ids != tuple(sorted(set(self.source_span_ids))):
            raise IntentCompilerError("compiler diagnostic source spans must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "field_id": self.field_id,
            "source_span_ids": list(self.source_span_ids),
        }


@dataclass(frozen=True, slots=True)
class IntentCompilation:
    """The result of compiling one :class:`IntentRecord`."""

    intent_id: str
    outcome: IntentOutcome
    diagnostics: tuple[CompilerDiagnostic, ...]
    objective_contract: ObjectiveContract | None = None
    schema_version: int = INTENT_COMPILER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTENT_COMPILER_SCHEMA_VERSION:
            raise IntentCompilerError("unsupported intent compiler schema version")
        if not self.intent_id.strip():
            raise IntentCompilerError("intent compilation requires an intent ID")
        if not self.diagnostics:
            raise IntentCompilerError("intent compilation requires diagnostics")
        ordered = tuple(sorted(self.diagnostics, key=_diagnostic_sort_key))
        if ordered != self.diagnostics:
            raise IntentCompilerError("compiler diagnostics must be in canonical order")
        if self.outcome is IntentOutcome.EXECUTABLE and self.objective_contract is None:
            raise IntentCompilerError("executable compilation requires an objective contract")
        if self.outcome is not IntentOutcome.EXECUTABLE and self.objective_contract is not None:
            raise IntentCompilerError(
                "non-executable compilation cannot emit an objective contract"
            )

    @property
    def executable(self) -> bool:
        """Whether the result is safe to submit to the deterministic engine."""

        return self.outcome is IntentOutcome.EXECUTABLE

    @property
    def contract(self) -> ObjectiveContract | None:
        """Short alias for callers that use ``contract`` terminology."""

        return self.objective_contract

    @property
    def spec_record(self) -> dict[str, object] | None:
        """Return the existing canonical objective-contract/spec record."""

        return None if self.objective_contract is None else self.objective_contract.to_record()

    @property
    def spec(self) -> dict[str, object] | None:
        """Compatibility alias for the emitted spec record."""

        return self.spec_record

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "outcome": self.outcome.value,
            "diagnostics": [item.to_record() for item in self.diagnostics],
            "spec": self.spec_record,
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )


def compile_intent_record(intent: IntentRecord) -> IntentCompilation:
    """Compile a structured intent record into an existing objective contract.

    The function is intentionally total for a valid ``IntentRecord``: unsafe or
    unsupported input is represented by a non-executable result with diagnostics.
    It never executes a mutation, selects a tensor, calls a provider, or chooses a
    search strategy.
    """

    if not isinstance(intent, IntentRecord):
        raise IntentCompilerError("compile_intent_record requires an IntentRecord")

    diagnostics = [
        CompilerDiagnostic(
            "intent-diagnostic",
            message,
            DiagnosticSeverity.WARNING,
        )
        for message in intent.diagnostics
    ]
    if intent.outcome is not IntentOutcome.EXECUTABLE:
        diagnostics.append(
            CompilerDiagnostic(
                "intent-not-executable",
                f"intent outcome {intent.outcome.value} cannot be promoted by the compiler",
            )
        )
        return _result(intent, intent.outcome, diagnostics)

    required_ambiguities = tuple(item for item in intent.ambiguities if item.required)
    if required_ambiguities:
        diagnostics.extend(
            CompilerDiagnostic(
                "required-ambiguity",
                f"required ambiguity remains unresolved: {item.ambiguity_id}",
                field_id=item.field_id,
            )
            for item in required_ambiguities
        )
        return _result(intent, IntentOutcome.CLARIFICATION_REQUIRED, diagnostics)

    constraints: list[HardConstraint] = []
    objectives: list[SoftObjective] = []
    unsupported: list[CompilerDiagnostic] = []
    invalid: list[CompilerDiagnostic] = []
    for field in intent.fields:
        parsed = _parse_field(field.field_id, field.value, field.unit)
        if parsed is None:
            unsupported.append(
                CompilerDiagnostic(
                    "unsupported-intent-field",
                    "field is not a supported typed objective or hard-constraint declaration",
                    field_id=field.field_id,
                    source_span_ids=field.source_span_ids,
                )
            )
            continue
        kind, payload = parsed
        try:
            if kind == "hard_constraint":
                constraints.append(_hard_constraint(payload))
            else:
                objectives.append(_soft_objective(payload))
        except (ObjectiveContractError, ValueError, TypeError) as error:
            invalid.append(
                CompilerDiagnostic(
                    "invalid-contract-field",
                    str(error),
                    field_id=field.field_id,
                    source_span_ids=field.source_span_ids,
                )
            )

    if unsupported:
        diagnostics.extend(unsupported)
        return _result(intent, IntentOutcome.UNSUPPORTED, diagnostics)
    if invalid:
        diagnostics.extend(invalid)
        return _result(intent, IntentOutcome.REFUSED, diagnostics)
    if not constraints:
        diagnostics.append(
            CompilerDiagnostic(
                "missing-hard-constraint",
                "no hard constraint was supplied; the compiler will not invent a threshold",
            )
        )
    if not objectives:
        diagnostics.append(
            CompilerDiagnostic(
                "missing-soft-objective",
                "at least one soft objective is required by the objective contract",
            )
        )
    if not constraints or not objectives:
        return _result(intent, IntentOutcome.CLARIFICATION_REQUIRED, diagnostics)

    try:
        contract = ObjectiveContract(tuple(constraints), tuple(objectives))
    except ObjectiveContractError as error:
        diagnostics.append(CompilerDiagnostic("invalid-objective-contract", str(error)))
        return _result(intent, IntentOutcome.REFUSED, diagnostics)

    if intent.emitted_spec is not None:
        try:
            emitted = _contract_from_record(intent.emitted_spec)
        except (ObjectiveContractError, TypeError, ValueError) as error:
            diagnostics.append(
                CompilerDiagnostic("invalid-emitted-spec", str(error))
            )
            return _result(intent, IntentOutcome.REFUSED, diagnostics)
        if emitted.to_record() != contract.to_record():
            diagnostics.append(
                CompilerDiagnostic(
                    "emitted-spec-mismatch",
                    "emitted spec does not match the contract compiled from typed fields",
                )
            )
            return _result(intent, IntentOutcome.REFUSED, diagnostics)

    diagnostics.append(
        CompilerDiagnostic(
            "compiled-objective-contract",
            "typed intent compiled into the existing objective contract schema",
            DiagnosticSeverity.INFO,
        )
    )
    return _result(intent, IntentOutcome.EXECUTABLE, diagnostics, contract)


def compile_intent(intent: IntentRecord) -> IntentCompilation:
    """Public shorthand for :func:`compile_intent_record`."""

    return compile_intent_record(intent)


def compile_conversational_intent(intent: IntentRecord) -> IntentCompilation:
    """Descriptive alias for :func:`compile_intent_record`."""

    return compile_intent_record(intent)


def _result(
    intent: IntentRecord,
    outcome: IntentOutcome,
    diagnostics: list[CompilerDiagnostic],
    contract: ObjectiveContract | None = None,
) -> IntentCompilation:
    ordered = tuple(sorted(diagnostics, key=_diagnostic_sort_key))
    return IntentCompilation(intent.intent_id, outcome, ordered, contract)


def _diagnostic_sort_key(item: CompilerDiagnostic) -> tuple[str, str, str, str]:
    return (item.code, item.field_id or "", item.message, item.severity.value)


def _parse_field(
    field_id: str, value: object, unit: str | None
) -> tuple[str, dict[str, object]] | None:
    if isinstance(value, Mapping):
        raw = dict(value)
        kind_value = raw.pop("kind", raw.pop("type", None))
        if kind_value is None:
            kind_value = raw.pop("declaration", None)
        if not isinstance(kind_value, str):
            return None
        kind = {
            "constraint": "hard_constraint",
            "hard_constraint": "hard_constraint",
            "hard-constraint": "hard_constraint",
            "objective": "soft_objective",
            "soft_objective": "soft_objective",
            "soft-objective": "soft_objective",
            "preference": "soft_objective",
        }.get(kind_value)
        if kind is None:
            return None
        raw.setdefault("unit", unit)
        allowed = (
            {"metric", "direction", "threshold", "unit", "baseline"}
            if kind == "hard_constraint"
            else {
                "metric",
                "direction",
                "unit",
                "weight",
                "normalization",
                "baseline",
                "minimum",
                "maximum",
                "plugin",
            }
        )
        if set(raw) - allowed:
            return None
        return kind, cast(dict[str, object], raw)
    return None


def _hard_constraint(payload: Mapping[str, object]) -> HardConstraint:
    metric = _required_enum(payload, "metric", str)
    direction = ConstraintDirection(_required_enum(payload, "direction", str))
    threshold = _required(payload, "threshold")
    unit = MetricUnit(_required_enum(payload, "unit", str))
    baseline = payload.get("baseline", "absolute")
    if not isinstance(baseline, str):
        raise ObjectiveContractError("constraint baseline must be a string")
    return HardConstraint(metric, direction, cast(float, threshold), unit, baseline)


def _soft_objective(payload: Mapping[str, object]) -> SoftObjective:
    metric = _required_enum(payload, "metric", str)
    direction = ObjectiveDirection(_required_enum(payload, "direction", str))
    unit = MetricUnit(_required_enum(payload, "unit", str))
    weight = payload.get("weight", 1.0)
    normalization = ObjectiveNormalization(
        _required_enum(payload, "normalization", str)
        if "normalization" in payload
        else ObjectiveNormalization.BASELINE_RATIO.value
    )
    baseline = payload.get("baseline")
    minimum = payload.get("minimum")
    maximum = payload.get("maximum")
    plugin = _plugin(payload.get("plugin"))
    return SoftObjective(
        metric,
        direction,
        unit,
        cast(float, weight),
        normalization,
        cast(float | None, baseline),
        cast(float | None, minimum),
        cast(float | None, maximum),
        plugin,
    )


def _plugin(value: object) -> ObjectivePluginBinding | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ObjectiveContractError("objective plugin must be an object")
    if set(value) != {"name", "plugin_id", "capability", "config_digest", "trust_mode"}:
        raise ObjectiveContractError("objective plugin record has unknown or missing fields")
    return ObjectivePluginBinding(
        _required_string(value, "name"),
        _required_string(value, "plugin_id"),
        _required_string(value, "capability"),
        _required_string(value, "config_digest"),
        _required_string(value, "trust_mode"),
    )


def _contract_from_record(value: Mapping[str, object]) -> ObjectiveContract:
    expected = {
        "schema_version",
        "contract_id",
        "mode",
        "constraints",
        "objectives",
        "approval_policy",
    }
    if set(value) != expected:
        raise ObjectiveContractError("emitted spec is not an objective contract record")
    constraints = tuple(
        _hard_constraint(
            _record_mapping(
                item,
                {"metric", "direction", "threshold", "unit", "baseline"},
                "constraint",
            )
        )
        for item in _sequence(value["constraints"])
    )
    objectives = tuple(
        _soft_objective(
            _record_mapping(
                item,
                {
                    "metric",
                    "direction",
                    "unit",
                    "weight",
                    "normalization",
                    "baseline",
                    "minimum",
                    "maximum",
                    "plugin",
                },
                "objective",
            )
        )
        for item in _sequence(value["objectives"])
    )
    policy_value = value["approval_policy"]
    if not isinstance(policy_value, Mapping):
        raise ObjectiveContractError("approval policy must be an object")
    if set(policy_value) != {
        "require_execution_approval",
        "require_custom_plugin_approval",
        "allowed_plugin_names",
    }:
        raise ObjectiveContractError("approval policy has unknown or missing fields")
    policy = ObjectiveApprovalPolicy(
        _required_bool(policy_value, "require_execution_approval"),
        _required_bool(policy_value, "require_custom_plugin_approval"),
        tuple(
            _required_string_item(item, "allowed plugin name")
            for item in _sequence(policy_value["allowed_plugin_names"])
        ),
    )
    schema_version = value["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ObjectiveContractError("emitted spec schema version must be an integer")
    mode = value["mode"]
    contract_id = value["contract_id"]
    if not isinstance(mode, str) or not isinstance(contract_id, str):
        raise ObjectiveContractError("emitted spec mode and contract ID must be strings")
    contract = ObjectiveContract(
        constraints,
        objectives,
        ObjectiveMode(mode),
        policy,
        schema_version,
    )
    if contract_id != contract.contract_id:
        raise ObjectiveContractError("emitted spec contract ID does not match its contents")
    return contract


def _sequence(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ObjectiveContractError("contract record arrays must be lists")
    return value


def _record_mapping(
    value: object, expected: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ObjectiveContractError(f"{label} record has unknown or missing fields")
    return value


def _required(value: Mapping[str, object], key: str) -> object:
    if key not in value or value[key] is None:
        raise ObjectiveContractError(f"{key} is required")
    return value[key]


def _required_enum(value: Mapping[str, object], key: str, expected: type[str]) -> str:
    item = _required(value, key)
    if not isinstance(item, expected) or not item.strip():
        raise ObjectiveContractError(f"{key} must be a non-empty string")
    return item


def _required_string(value: Mapping[str, object], key: str) -> str:
    item = _required(value, key)
    if not isinstance(item, str) or not item.strip():
        raise ObjectiveContractError(f"{key} must be a non-empty string")
    return item


def _required_string_item(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObjectiveContractError(f"{label} must be a non-empty string")
    return value


def _required_bool(value: Mapping[str, object], key: str) -> bool:
    item = _required(value, key)
    if not isinstance(item, bool):
        raise ObjectiveContractError(f"{key} must be boolean")
    return item


__all__ = [
    "INTENT_COMPILER_SCHEMA_VERSION",
    "CompilerDiagnostic",
    "DiagnosticSeverity",
    "IntentCompilation",
    "IntentCompilerError",
    "compile_conversational_intent",
    "compile_intent",
    "compile_intent_record",
]
