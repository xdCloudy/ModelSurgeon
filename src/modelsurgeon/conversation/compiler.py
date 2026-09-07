"""Deterministic compilation from canonical intent records to search specs.

The compiler deliberately consumes :class:`IntentRecord`, rather than parsing
the request text.  A provider may propose normalized fields, but only the
typed fields accepted here can cross the execution boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final, cast

from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ContractMetric,
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

from .intent import (
    IntentField,
    IntentOutcome,
    IntentRecord,
)

OPTIMIZATION_SPEC_SCHEMA_VERSION: Final = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


# The compiler intentionally reuses the canonical intent outcomes so callers
# do not have to translate between two boundary vocabularies.
CompilerOutcome = IntentOutcome


class DiagnosticSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class CompilerDiagnostic:
    """Machine-readable explanation for a compiler result."""

    code: str
    severity: DiagnosticSeverity
    message: str
    field_id: str | None = None

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.code) is None:
            raise ValueError("diagnostic code must be a canonical identifier")
        if not self.message.strip():
            raise ValueError("diagnostic message is required")
        if self.field_id is not None and _IDENTIFIER.fullmatch(self.field_id) is None:
            raise ValueError("diagnostic field ID must be a canonical identifier")

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "field_id": self.field_id,
        }


@dataclass(frozen=True, slots=True)
class BudgetLimit:
    """A non-negative, named resource limit carried by an optimization spec."""

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        if _IDENTIFIER.fullmatch(self.name) is None:
            raise ValueError("budget name must be a canonical identifier")
        if isinstance(self.value, bool) or not math.isfinite(self.value) or self.value < 0:
            raise ValueError("budget value must be finite and non-negative")
        if _IDENTIFIER.fullmatch(self.unit) is None:
            raise ValueError("budget unit must be a canonical identifier")

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value, "unit": self.unit}


@dataclass(frozen=True, slots=True)
class OptimizationSpec:
    """The immutable control-plane envelope around the stable objective contract."""

    objective_contract: ObjectiveContract
    budgets: tuple[BudgetLimit, ...] = ()
    allowed_operations: tuple[str, ...] = ()
    deployment_targets: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()
    schema_version: int = OPTIMIZATION_SPEC_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != OPTIMIZATION_SPEC_SCHEMA_VERSION:
            raise ValueError("unsupported optimization spec schema version")
        if self.budgets != tuple(sorted(self.budgets, key=lambda item: item.name)):
            raise ValueError("budgets must be sorted by name")
        if self.allowed_operations != tuple(sorted(set(self.allowed_operations))):
            raise ValueError("allowed operations must be sorted and unique")
        if self.deployment_targets != tuple(sorted(set(self.deployment_targets))):
            raise ValueError("deployment targets must be sorted and unique")
        if self.provenance_refs != tuple(sorted(set(self.provenance_refs))):
            raise ValueError("provenance references must be sorted and unique")

    @property
    def contract(self) -> ObjectiveContract:
        """Compatibility alias for callers that use the contract terminology."""

        return self.objective_contract

    @property
    def spec_id(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
        return f"optimization_spec_{digest}"

    @property
    def contract_id(self) -> str:
        return self.objective_contract.contract_id

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "objective_contract": self.objective_contract.to_record(),
            "budgets": [item.to_record() for item in self.budgets],
            "allowed_operations": list(self.allowed_operations),
            "deployment_targets": list(self.deployment_targets),
            "provenance_refs": list(self.provenance_refs),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class CompilationResult:
    """Compiler output, including structured diagnostics and replayable intent."""

    outcome: CompilerOutcome
    spec: OptimizationSpec | None
    diagnostics: tuple[CompilerDiagnostic, ...]
    intent_record: IntentRecord

    @property
    def executable(self) -> bool:
        return self.outcome is CompilerOutcome.EXECUTABLE

    def to_record(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "spec": None if self.spec is None else self.spec.to_record(),
            "diagnostics": [item.to_record() for item in self.diagnostics],
            "intent_record": self.intent_record.to_record(),
        }


_METRIC_UNITS: Final[dict[str, MetricUnit]] = {
    metric.value: unit
    for metric, unit in (
        (ContractMetric.QUALITY, MetricUnit.RATIO),
        (ContractMetric.PERPLEXITY, MetricUnit.PERPLEXITY_POINTS),
        (ContractMetric.LATENCY_GAIN, MetricUnit.RATIO),
        (ContractMetric.LATENCY, MetricUnit.MILLISECONDS),
        (ContractMetric.PARAMETER_COUNT, MetricUnit.COUNT),
        (ContractMetric.PEAK_RAM, MetricUnit.BYTES),
        (ContractMetric.PEAK_VRAM, MetricUnit.BYTES),
        (ContractMetric.FILE_SIZE, MetricUnit.BYTES),
        (ContractMetric.OPTIMIZATION_TIME, MetricUnit.SECONDS),
        (ContractMetric.REPAIR_COST, MetricUnit.CURRENCY),
        (ContractMetric.ENERGY, MetricUnit.JOULES),
    )
}
_DEFAULT_OPERATIONS: Final[frozenset[str]] = frozenset(
    {"inspect", "evaluate", "search", "prune", "quantize", "repair", "export"}
)


def _field_map(record: IntentRecord) -> dict[str, IntentField]:
    return {field.field_id: field for field in record.fields}


def _diagnostic(
    code: str,
    message: str,
    field_id: str | None = None,
    *,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
) -> CompilerDiagnostic:
    return CompilerDiagnostic(code, severity, message, field_id)


def _sorted_diagnostics(items: Sequence[CompilerDiagnostic]) -> tuple[CompilerDiagnostic, ...]:
    return tuple(sorted(items, key=lambda item: (item.code, item.field_id or "", item.message)))


def _as_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    return dict(cast(Mapping[str, object], value))


def _as_string_list(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) for item in value):
        return None
    return tuple(sorted(set(value)))


def _optional_number(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    return float(value)


class IntentCompiler:
    """Compile only fields in the bounded v2.1 intent vocabulary.

    No value is read from ``IntentRecord.original_request``.  Unknown fields
    are ignored unless they are in a reserved compiler namespace, where they
    produce an explicit refusal.  Callers can narrow operation and deployment
    vocabularies to the capabilities of a particular deterministic engine.
    """

    def __init__(
        self,
        *,
        supported_operations: Sequence[str] = tuple(sorted(_DEFAULT_OPERATIONS)),
        supported_deployment_targets: Sequence[str] | None = None,
        max_fields: int = 128,
        max_request_length: int = 16_384,
    ) -> None:
        self.supported_operations = tuple(sorted(set(supported_operations)))
        self.supported_deployment_targets = (
            None
            if supported_deployment_targets is None
            else tuple(sorted(set(supported_deployment_targets)))
        )
        self.max_fields = max_fields
        self.max_request_length = max_request_length
        if any(_IDENTIFIER.fullmatch(item) is None for item in self.supported_operations):
            raise ValueError("supported operations must be canonical identifiers")
        if self.supported_deployment_targets is not None and any(
            _IDENTIFIER.fullmatch(item) is None for item in self.supported_deployment_targets
        ):
            raise ValueError("supported deployment targets must be canonical identifiers")
        if max_fields <= 0 or max_request_length <= 0:
            raise ValueError("compiler bounds must be positive")

    def compile(self, record: IntentRecord) -> CompilationResult:
        diagnostics: list[CompilerDiagnostic] = []
        if not isinstance(record, IntentRecord):
            raise TypeError("intent compiler requires an IntentRecord")
        if len(record.fields) > self.max_fields:
            diagnostics.append(_diagnostic("field_limit_exceeded", "intent field bound exceeded"))
        if len(record.original_request) > self.max_request_length:
            diagnostics.append(
                _diagnostic("request_limit_exceeded", "request length bound exceeded")
            )
        if record.outcome is not IntentOutcome.EXECUTABLE:
            diagnostics.append(
                _diagnostic(
                    "intent_not_executable",
                    f"intent record outcome is {record.outcome.value}; compilation is refused",
                )
            )
            return self._result(record, CompilerOutcome.REFUSED, None, diagnostics)
        for ambiguity in record.ambiguities:
            if ambiguity.required:
                diagnostics.append(
                    _diagnostic(
                        "required_ambiguity",
                        ambiguity.detail,
                        ambiguity.field_id,
                    )
                )
        fields = _field_map(record)
        diagnostics.extend(self._reserved_field_diagnostics(fields))
        constraints, constraint_diagnostics = self._constraints(fields)
        diagnostics.extend(constraint_diagnostics)
        objectives, objective_diagnostics = self._objectives(fields)
        diagnostics.extend(objective_diagnostics)
        budgets, budget_diagnostics = self._budgets(fields)
        diagnostics.extend(budget_diagnostics)
        operations, operation_diagnostics = self._operations(fields)
        diagnostics.extend(operation_diagnostics)
        targets, target_diagnostics = self._targets(fields)
        diagnostics.extend(target_diagnostics)
        mode, mode_diagnostics = self._mode(fields)
        diagnostics.extend(mode_diagnostics)

        if any(item.code == "required_ambiguity" for item in diagnostics):
            outcome = CompilerOutcome.CLARIFICATION_REQUIRED
        elif any(item.code.startswith("unsupported_") for item in diagnostics):
            outcome = CompilerOutcome.UNSUPPORTED
        elif any(item.severity is DiagnosticSeverity.ERROR for item in diagnostics):
            outcome = CompilerOutcome.CLARIFICATION_REQUIRED
        else:
            try:
                approval = self._approval(fields)
                contract = ObjectiveContract(
                    tuple(constraints), tuple(objectives), mode=mode, approval_policy=approval
                )
                spec = OptimizationSpec(
                    contract,
                    tuple(budgets),
                    operations,
                    targets,
                    tuple(sorted(set(record.provenance.evidence_refs))),
                )
            except (ObjectiveContractError, ValueError) as error:
                diagnostics.append(_diagnostic("invalid_spec", str(error)))
                return self._result(record, CompilerOutcome.REFUSED, None, diagnostics)
            return self._result(record, CompilerOutcome.EXECUTABLE, spec, diagnostics)
        return self._result(record, outcome, None, diagnostics)

    @staticmethod
    def _approval(fields: Mapping[str, IntentField]) -> ObjectiveApprovalPolicy:
        require_execution = fields.get("approval.require_execution")
        require_plugin = fields.get("approval.require_custom_plugin")
        allowed = fields.get("approval.allowed_plugins")
        execution_value = True if require_execution is None else require_execution.value
        plugin_value = True if require_plugin is None else require_plugin.value
        if not isinstance(execution_value, bool) or not isinstance(plugin_value, bool):
            raise ValueError("approval policy values must be boolean")
        allowed_values = () if allowed is None else _as_string_list(allowed.value)
        if allowed is not None and allowed_values is None:
            raise ValueError("allowed plugin names must be an array of strings")
        return ObjectiveApprovalPolicy(execution_value, plugin_value, allowed_values or ())

    def _result(
        self,
        record: IntentRecord,
        outcome: CompilerOutcome,
        spec: OptimizationSpec | None,
        diagnostics: Sequence[CompilerDiagnostic],
    ) -> CompilationResult:
        ordered = _sorted_diagnostics(diagnostics)
        intent_outcome = IntentOutcome(outcome.value)
        emitted = None if spec is None else spec.to_record()
        messages = tuple(sorted({item.message for item in ordered} or {outcome.value}))
        intent_record = replace(
            record,
            outcome=intent_outcome,
            diagnostics=messages,
            emitted_spec=emitted,
        )
        return CompilationResult(outcome, spec, ordered, intent_record)

    @staticmethod
    def _reserved_field_diagnostics(
        fields: Mapping[str, IntentField],
    ) -> tuple[CompilerDiagnostic, ...]:
        known_prefixes = (
            "objective.",
            "constraint.",
            "budget.",
            "deployment.",
            "operations.",
            "approval.",
        )
        return tuple(
            _diagnostic(
                "unsupported_field",
                "field is outside the bounded intent compiler vocabulary",
                field_id,
            )
            for field_id in sorted(fields)
            if field_id.startswith(known_prefixes)
            and field_id
            not in {
                "objective.metric",
                "objective.direction",
                "objective.mode",
                "objective.weight",
                "objective.normalization",
                "objective.baseline",
                "objective.minimum",
                "objective.maximum",
                "objective.plugin",
                "objective.terms",
                "constraints",
                "operations.allowed",
                "allowed_operations",
                "deployment.target",
                "deployment.targets",
                "budgets",
                "approval.require_execution",
                "approval.require_custom_plugin",
                "approval.allowed_plugins",
            }
            and not re.fullmatch(
                r"(?:constraint\.)?[a-z][a-z0-9_.-]*\.(?:minimum|maximum|threshold)", field_id
            )
            and not field_id.startswith("budget.")
        )

    def _constraints(
        self, fields: Mapping[str, IntentField]
    ) -> tuple[list[HardConstraint], list[CompilerDiagnostic]]:
        values: dict[tuple[str, ConstraintDirection], tuple[object, IntentField]] = {}
        diagnostics: list[CompilerDiagnostic] = []
        raw = fields.get("constraints")
        if raw is not None:
            entries = raw.value if isinstance(raw.value, list) else None
            if entries is None:
                diagnostics.append(
                    _diagnostic("invalid_constraints", "constraints must be an array", raw.field_id)
                )
            else:
                for index, entry in enumerate(entries):
                    item = _as_mapping(entry)
                    if item is None:
                        diagnostics.append(
                            _diagnostic(
                                "invalid_constraint",
                                f"constraint {index} must be an object",
                                raw.field_id,
                            )
                        )
                        continue
                    self._add_constraint_mapping(values, item, raw.field_id, diagnostics)
        for field_id, field in fields.items():
            match = re.fullmatch(
                r"(?:constraint\.)?([a-z][a-z0-9_.-]*)\.(minimum|maximum|threshold)", field_id
            )
            if match is None:
                continue
            metric_name, suffix = match.groups()
            direction = (
                ConstraintDirection.MINIMUM if suffix == "minimum" else ConstraintDirection.MAXIMUM
            )
            values[(metric_name, direction)] = (field.value, field)
        constraints: list[HardConstraint] = []
        for (metric_name, direction), (value, field) in sorted(values.items()):
            unit = _METRIC_UNITS.get(metric_name)
            if unit is None:
                diagnostics.append(
                    _diagnostic(
                        "unsupported_constraint_metric",
                        f"constraint metric is unsupported: {metric_name}",
                        field.field_id,
                    )
                )
                continue
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                diagnostics.append(
                    _diagnostic(
                        "invalid_constraint_threshold",
                        "constraint threshold must be finite numeric",
                        field.field_id,
                    )
                )
                continue
            if field.unit is not None and field.unit != unit.value:
                diagnostics.append(
                    _diagnostic(
                        "unit_mismatch", f"{metric_name} requires unit {unit.value}", field.field_id
                    )
                )
                continue
            constraints.append(
                HardConstraint(
                    metric_name,
                    direction,
                    float(value),
                    unit,
                    "immutable_source"
                    if metric_name not in {"peak_ram", "peak_vram", "file_size"}
                    else "absolute",
                )
            )
        if not constraints:
            diagnostics.append(
                _diagnostic(
                    "missing_hard_constraint", "at least one explicit hard constraint is required"
                )
            )
        return constraints, diagnostics

    @staticmethod
    def _add_constraint_mapping(
        values: dict[tuple[str, ConstraintDirection], tuple[object, IntentField]],
        item: Mapping[str, object],
        field_id: str,
        diagnostics: list[CompilerDiagnostic],
    ) -> None:
        metric = item.get("metric")
        direction = item.get("direction", item.get("comparison"))
        threshold = item.get("threshold")
        if not isinstance(metric, str) or not isinstance(direction, str) or threshold is None:
            diagnostics.append(
                _diagnostic(
                    "invalid_constraint",
                    "constraint requires metric, direction and threshold",
                    field_id,
                )
            )
            return
        try:
            parsed = ConstraintDirection(direction)
        except ValueError:
            diagnostics.append(
                _diagnostic(
                    "invalid_constraint_direction", "constraint direction is invalid", field_id
                )
            )
            return
        values[(metric, parsed)] = (threshold, IntentField(field_id, threshold, None, 1.0, ()))

    def _objectives(
        self, fields: Mapping[str, IntentField]
    ) -> tuple[list[SoftObjective], list[CompilerDiagnostic]]:
        diagnostics: list[CompilerDiagnostic] = []
        terms: list[Mapping[str, object]] = []
        terms_field = fields.get("objective.terms")
        if terms_field is not None:
            if not isinstance(terms_field.value, list) or not all(
                _as_mapping(item) is not None for item in terms_field.value
            ):
                diagnostics.append(
                    _diagnostic(
                        "invalid_objective_terms",
                        "objective.terms must be an array of objects",
                        terms_field.field_id,
                    )
                )
            else:
                terms = [cast(Mapping[str, object], item) for item in terms_field.value]
        else:
            metric_field = fields.get("objective.metric")
            if metric_field is not None:
                default_direction = (
                    "maximize" if metric_field.value in {"quality", "latency_gain"} else "minimize"
                )
                terms = [
                    {
                        "metric": metric_field.value,
                        "unit": metric_field.unit,
                        "direction": fields.get(
                            "objective.direction",
                            IntentField("objective.direction", default_direction, None, 1.0, ()),
                        ).value,
                        "weight": fields.get(
                            "objective.weight", IntentField("objective.weight", 1.0, None, 1.0, ())
                        ).value,
                        "normalization": fields.get(
                            "objective.normalization",
                            IntentField("objective.normalization", "baseline_ratio", None, 1.0, ()),
                        ).value,
                        "baseline": fields.get(
                            "objective.baseline",
                            IntentField("objective.baseline", None, None, 1.0, ()),
                        ).value,
                        "minimum": fields.get(
                            "objective.minimum",
                            IntentField("objective.minimum", None, None, 1.0, ()),
                        ).value,
                        "maximum": fields.get(
                            "objective.maximum",
                            IntentField("objective.maximum", None, None, 1.0, ()),
                        ).value,
                        "plugin": fields.get(
                            "objective.plugin", IntentField("objective.plugin", None, None, 1.0, ())
                        ).value,
                    }
                ]
        objectives: list[SoftObjective] = []
        for index, term in enumerate(terms):
            metric = term.get("metric")
            plugin_value = term.get("plugin")
            plugin: ObjectivePluginBinding | None = None
            if plugin_value is not None:
                plugin_record = _as_mapping(plugin_value)
                if plugin_record is None or not all(
                    isinstance(plugin_record.get(key), str)
                    for key in ("name", "plugin_id", "capability", "config_digest")
                ):
                    diagnostics.append(
                        _diagnostic(
                            "invalid_plugin_binding",
                            "objective plugin binding is incomplete",
                            "objective.plugin" if index == 0 else "objective.terms",
                        )
                    )
                    continue
                trust_mode = plugin_record.get("trust_mode", "subprocess")
                if not isinstance(trust_mode, str):
                    diagnostics.append(
                        _diagnostic(
                            "invalid_plugin_binding",
                            "objective plugin trust mode must be a string",
                            "objective.plugin",
                        )
                    )
                    continue
                try:
                    plugin = ObjectivePluginBinding(
                        cast(str, plugin_record["name"]),
                        cast(str, plugin_record["plugin_id"]),
                        cast(str, plugin_record["capability"]),
                        cast(str, plugin_record["config_digest"]),
                        trust_mode,
                    )
                except ObjectiveContractError as error:
                    diagnostics.append(
                        _diagnostic("invalid_plugin_binding", str(error), "objective.plugin")
                    )
                    continue
            if not isinstance(metric, str):
                diagnostics.append(
                    _diagnostic(
                        "unsupported_objective_metric",
                        "objective metric is unsupported",
                        "objective.metric" if index == 0 else "objective.terms",
                    )
                )
                continue
            if metric not in _METRIC_UNITS and plugin is None:
                diagnostics.append(
                    _diagnostic(
                        "unsupported_objective_metric",
                        "custom objectives require a plugin and explicit unit",
                        "objective.metric" if index == 0 else "objective.terms",
                    )
                )
                continue
            try:
                direction = ObjectiveDirection(str(term.get("direction", "minimize")))
                normalization = ObjectiveNormalization(
                    str(term.get("normalization", "baseline_ratio"))
                )
                weight = term.get("weight", 1.0)
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    raise ValueError("objective weight must be numeric")
                if metric in _METRIC_UNITS:
                    unit = _METRIC_UNITS[metric]
                elif plugin is not None and isinstance(term.get("unit"), str):
                    unit = MetricUnit(cast(str, term["unit"]))
                else:
                    raise ValueError("custom objectives require a plugin and explicit unit")
                objectives.append(
                    SoftObjective(
                        metric,
                        direction,
                        unit,
                        float(weight),
                        normalization,
                        _optional_number(term.get("baseline"), "objective baseline"),
                        _optional_number(term.get("minimum"), "objective minimum"),
                        _optional_number(term.get("maximum"), "objective maximum"),
                        plugin=plugin,
                    )
                )
            except (ValueError, ObjectiveContractError, TypeError) as error:
                diagnostics.append(
                    _diagnostic(
                        "invalid_objective",
                        str(error),
                        "objective.metric" if index == 0 else "objective.terms",
                    )
                )
        if not objectives:
            diagnostics.append(
                _diagnostic("missing_objective", "at least one explicit soft objective is required")
            )
        return objectives, diagnostics

    def _budgets(
        self, fields: Mapping[str, IntentField]
    ) -> tuple[list[BudgetLimit], list[CompilerDiagnostic]]:
        diagnostics: list[CompilerDiagnostic] = []
        values: list[tuple[str, object, str | None, str]] = []
        grouped = fields.get("budgets")
        if grouped is not None:
            if not isinstance(grouped.value, list):
                diagnostics.append(
                    _diagnostic("invalid_budgets", "budgets must be an array", grouped.field_id)
                )
            else:
                for entry in grouped.value:
                    item = _as_mapping(entry)
                    if item is None or not isinstance(item.get("name"), str):
                        diagnostics.append(
                            _diagnostic(
                                "invalid_budget",
                                "budget requires a name, value and unit",
                                grouped.field_id,
                            )
                        )
                        continue
                    unit = item.get("unit")
                    if unit is not None and not isinstance(unit, str):
                        diagnostics.append(
                            _diagnostic(
                                "invalid_budget", "budget unit must be a string", grouped.field_id
                            )
                        )
                        continue
                    values.append(
                        (cast(str, item["name"]), item.get("value"), unit, grouped.field_id)
                    )
        for field_id, field in fields.items():
            if field_id.startswith("budget."):
                values.append((field_id.removeprefix("budget."), field.value, field.unit, field_id))
        budgets: list[BudgetLimit] = []
        for name, value, unit, field_id in sorted(values):
            if unit is None:
                diagnostics.append(
                    _diagnostic("missing_budget_unit", "budget unit must be explicit", field_id)
                )
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                diagnostics.append(
                    _diagnostic("invalid_budget", "budget value must be numeric", field_id)
                )
                continue
            try:
                budgets.append(BudgetLimit(name, float(value), unit))
            except ValueError as error:
                diagnostics.append(_diagnostic("invalid_budget", str(error), field_id))
        if len({item.name for item in budgets}) != len(budgets):
            diagnostics.append(_diagnostic("duplicate_budget", "budget names must be unique"))
        return budgets, diagnostics

    def _operations(
        self, fields: Mapping[str, IntentField]
    ) -> tuple[tuple[str, ...], list[CompilerDiagnostic]]:
        field = fields.get("operations.allowed") or fields.get("allowed_operations")
        if field is None:
            return (), []
        values = _as_string_list(field.value)
        if values is None:
            return (), [
                _diagnostic(
                    "invalid_operations",
                    "allowed operations must be an array of strings",
                    field.field_id,
                )
            ]
        unsupported = tuple(item for item in values if item not in self.supported_operations)
        if unsupported:
            return (), [
                _diagnostic(
                    "unsupported_operation",
                    f"operation is not supported: {unsupported[0]}",
                    field.field_id,
                )
            ]
        return values, []

    def _targets(
        self, fields: Mapping[str, IntentField]
    ) -> tuple[tuple[str, ...], list[CompilerDiagnostic]]:
        field = fields.get("deployment.targets") or fields.get("deployment.target")
        if field is None:
            return (), []
        raw = field.value if isinstance(field.value, (list, tuple)) else [field.value]
        values = _as_string_list(raw)
        if values is None:
            return (), [
                _diagnostic(
                    "invalid_deployment_targets",
                    "deployment targets must be strings",
                    field.field_id,
                )
            ]
        if self.supported_deployment_targets is not None:
            unsupported = tuple(
                item for item in values if item not in self.supported_deployment_targets
            )
            if unsupported:
                return (), [
                    _diagnostic(
                        "unsupported_deployment_target",
                        f"deployment target is not supported: {unsupported[0]}",
                        field.field_id,
                    )
                ]
        return values, []

    @staticmethod
    def _mode(fields: Mapping[str, IntentField]) -> tuple[ObjectiveMode, list[CompilerDiagnostic]]:
        field = fields.get("objective.mode")
        if field is None:
            return ObjectiveMode.WEIGHTED, []
        try:
            return ObjectiveMode(str(field.value)), []
        except ValueError:
            return ObjectiveMode.WEIGHTED, [
                _diagnostic(
                    "invalid_objective_mode", "objective mode is unsupported", field.field_id
                )
            ]


def compile_intent(
    record: IntentRecord, *, compiler: IntentCompiler | None = None
) -> CompilationResult:
    """Compile ``record`` using the bounded default compiler."""

    return (compiler or IntentCompiler()).compile(record)


# Descriptive aliases keep the public boundary discoverable without creating
# parallel implementations or result vocabularies.
CompilationDiagnostic = CompilerDiagnostic
CompilationOutcome = CompilerOutcome
OptimizationSpecCompiler = IntentCompiler
compile_intent_record = compile_intent


__all__ = [
    "OPTIMIZATION_SPEC_SCHEMA_VERSION",
    "BudgetLimit",
    "CompilationDiagnostic",
    "CompilationOutcome",
    "CompilationResult",
    "CompilerDiagnostic",
    "CompilerOutcome",
    "DiagnosticSeverity",
    "IntentCompiler",
    "OptimizationSpec",
    "OptimizationSpecCompiler",
    "compile_intent",
    "compile_intent_record",
]
