"""Fail-closed policy evaluation for compiled conversational intent.

The intent compiler answers whether typed fields can be represented by the
existing objective contract.  This module is the policy boundary immediately
after compilation: it makes interpretation confidence and ambiguity explicit,
checks that hard constraints are complete and non-conflicting, and chooses a
stable outcome when more than one problem is present.

This layer never treats confidence as optimization evidence.  A high
confidence interpretation still has to pass the compiler and every policy
check; a low-confidence required interpretation is sent back for
clarification.  No policy result other than ``executable`` contains a spec.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from modelsurgeon.conversation import (
    AmbiguityRecord,
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
)

from .intent_compiler import (
    CompilerDiagnostic,
    DiagnosticSeverity,
    IntentCompilation,
    compile_intent_record,
)
from .objective_contract import ContractMetric, ObjectiveContract

INTENT_POLICY_SCHEMA_VERSION = 1


class IntentPolicyError(ValueError):
    """Raised for invalid policy API arguments, not policy refusals."""


class ConfidenceCategory(StrEnum):
    """Interpretation-confidence bucket; never a safety or quality result."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AmbiguityCategory(StrEnum):
    """Stable policy categories for unresolved interpretation alternatives."""

    MISSING = "missing"
    VAGUE = "vague"
    CONTRADICTORY = "contradictory"
    CONFLICTING = "conflicting"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """Inspectable confidence classification for one normalized field."""

    field_id: str
    confidence: float
    category: ConfidenceCategory
    required: bool
    source_span_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.field_id.strip():
            raise IntentPolicyError("confidence assessment requires a field ID")
        if isinstance(self.confidence, bool) or not math.isfinite(self.confidence):
            raise IntentPolicyError("confidence assessment must be finite")
        if not 0 <= self.confidence <= 1:
            raise IntentPolicyError("confidence assessment must be within [0, 1]")
        if self.source_span_ids != tuple(sorted(set(self.source_span_ids))):
            raise IntentPolicyError("confidence source spans must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "confidence": self.confidence,
            "category": self.category.value,
            "required": self.required,
            "source_span_ids": list(self.source_span_ids),
        }


@dataclass(frozen=True, slots=True)
class AmbiguityAssessment:
    """An ambiguity copied into the policy decision with source linkage."""

    ambiguity_id: str
    field_id: str
    category: AmbiguityCategory
    declared_category: str
    detail: str
    alternatives: tuple[str, ...]
    required: bool
    source_span_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.ambiguity_id.strip() or not self.field_id.strip():
            raise IntentPolicyError("ambiguity assessment requires IDs")
        if not self.declared_category.strip() or not self.detail.strip():
            raise IntentPolicyError("ambiguity assessment requires category and detail")
        if not self.alternatives:
            raise IntentPolicyError("ambiguity assessment requires alternatives")
        if self.alternatives != tuple(sorted(set(self.alternatives))):
            raise IntentPolicyError("ambiguity alternatives must be sorted and unique")
        if self.source_span_ids != tuple(sorted(set(self.source_span_ids))):
            raise IntentPolicyError("ambiguity source spans must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "ambiguity_id": self.ambiguity_id,
            "field_id": self.field_id,
            "category": self.category.value,
            "declared_category": self.declared_category,
            "detail": self.detail,
            "alternatives": list(self.alternatives),
            "required": self.required,
            "source_span_ids": list(self.source_span_ids),
        }


@dataclass(frozen=True, slots=True)
class PolicyDiagnostic:
    """A deterministic reason linked to fields, source spans and evidence."""

    code: str
    message: str
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR
    field_id: str | None = None
    source_span_ids: tuple[str, ...] = ()
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code.strip() or not self.message.strip():
            raise IntentPolicyError("policy diagnostics require code and message")
        if self.field_id is not None and not self.field_id.strip():
            raise IntentPolicyError("policy diagnostic field ID cannot be blank")
        if self.source_span_ids != tuple(sorted(set(self.source_span_ids))):
            raise IntentPolicyError("policy diagnostic source spans must be sorted and unique")
        if self.provenance_refs != tuple(sorted(set(self.provenance_refs))):
            raise IntentPolicyError("policy diagnostic provenance must be sorted and unique")

    def to_record(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity.value,
            "field_id": self.field_id,
            "source_span_ids": list(self.source_span_ids),
            "provenance_refs": list(self.provenance_refs),
        }


@dataclass(frozen=True, slots=True)
class IntentPolicyDecision:
    """Canonical, provenance-linked result of post-compilation policy checks."""

    intent_id: str
    outcome: IntentOutcome
    diagnostics: tuple[PolicyDiagnostic, ...]
    confidence: tuple[ConfidenceAssessment, ...]
    ambiguities: tuple[AmbiguityAssessment, ...]
    provenance: IntentProvenance
    compilation: IntentCompilation
    schema_version: int = INTENT_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INTENT_POLICY_SCHEMA_VERSION:
            raise IntentPolicyError("unsupported intent policy schema version")
        if not self.intent_id.strip() or self.intent_id != self.compilation.intent_id:
            raise IntentPolicyError("policy decision intent identity does not match compilation")
        if not self.diagnostics:
            raise IntentPolicyError("policy decisions require diagnostics")
        if tuple(sorted(self.confidence, key=lambda item: item.field_id)) != self.confidence:
            raise IntentPolicyError("confidence assessments must be in canonical order")
        if tuple(sorted(self.ambiguities, key=lambda item: item.ambiguity_id)) != self.ambiguities:
            raise IntentPolicyError("ambiguity assessments must be in canonical order")
        if tuple(sorted(self.diagnostics, key=_diagnostic_sort_key)) != self.diagnostics:
            raise IntentPolicyError("policy diagnostics must be in canonical order")
        if self.outcome is IntentOutcome.EXECUTABLE and self.compilation.contract is None:
            raise IntentPolicyError("executable policy decision requires a contract")
        if self.outcome is not IntentOutcome.EXECUTABLE and self.contract is not None:
            raise IntentPolicyError("non-executable policy decision cannot contain a contract")

    @property
    def contract(self) -> ObjectiveContract | None:
        """The objective contract only when policy says it is executable."""

        return self.compilation.contract if self.outcome is IntentOutcome.EXECUTABLE else None

    @property
    def executable(self) -> bool:
        return self.outcome is IntentOutcome.EXECUTABLE

    @property
    def decision_id(self) -> str:
        return "policy_" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "outcome": self.outcome.value,
            "diagnostics": [item.to_record() for item in self.diagnostics],
            "confidence": [item.to_record() for item in self.confidence],
            "ambiguities": [item.to_record() for item in self.ambiguities],
            "provenance": self.provenance.to_record(),
            "compilation": self.compilation.to_record(),
            "spec": None if self.contract is None else self.contract.to_record(),
        }
        if include_identity:
            record["decision_id"] = self.decision_id
        return record

    def canonical_json(self) -> str:
        return _canonical(self.to_record(include_identity=False))


class IntentPolicyEvaluator:
    """Evaluate one intent and its compilation with deterministic precedence."""

    def __init__(self, *, high_confidence: float = 0.9, medium_confidence: float = 0.7):
        if not 0 <= medium_confidence <= high_confidence <= 1:
            raise IntentPolicyError("confidence thresholds must satisfy 0 <= medium <= high <= 1")
        self.high_confidence = float(high_confidence)
        self.medium_confidence = float(medium_confidence)

    def evaluate(
        self,
        intent: IntentRecord,
        compilation: IntentCompilation | None = None,
    ) -> IntentPolicyDecision:
        if not isinstance(intent, IntentRecord):
            raise IntentPolicyError("policy evaluation requires an IntentRecord")
        compiled = compile_intent_record(intent) if compilation is None else compilation
        if not isinstance(compiled, IntentCompilation):
            raise IntentPolicyError("policy evaluation requires an IntentCompilation")
        if compiled.intent_id != intent.intent_id:
            raise IntentPolicyError("compilation belongs to a different intent")

        fields = {field.field_id: field for field in intent.fields}
        confidence = tuple(
            _confidence_assessment(field, self.high_confidence, self.medium_confidence)
            for field in intent.fields
        )
        ambiguities = tuple(_ambiguity_assessment(item, fields) for item in intent.ambiguities)
        diagnostics = [
            _from_compiler_diagnostic(item, fields, intent.provenance)
            for item in compiled.diagnostics
        ]
        candidates: list[tuple[IntentOutcome, str]] = []

        if intent.outcome is not IntentOutcome.EXECUTABLE:
            candidates.append((intent.outcome, "intent-input-outcome"))

        for item in ambiguities:
            if not item.required:
                continue
            if item.category is AmbiguityCategory.CONTRADICTORY:
                code = "contradictory-ambiguity"
                outcome = IntentOutcome.REFUSED
            elif item.category is AmbiguityCategory.UNSUPPORTED:
                code = "unsupported-ambiguity"
                outcome = IntentOutcome.UNSUPPORTED
            else:
                code = "clarification-needed"
                outcome = IntentOutcome.CLARIFICATION_REQUIRED
            diagnostics.append(
                _policy_diagnostic(
                    code,
                    f"required {item.category.value} ambiguity remains unresolved: "
                    f"{item.ambiguity_id}",
                    fields.get(item.field_id),
                    intent.provenance,
                )
            )
            candidates.append((outcome, code))

        hard_conflicts, hard_diags = _hard_constraint_conflicts(intent, fields, intent.provenance)
        diagnostics.extend(hard_diags)
        if hard_conflicts:
            candidates.append((IntentOutcome.REFUSED, "contradictory-hard-constraints"))

        unsupported_diags = _unsupported_declarations(intent, fields, intent.provenance)
        diagnostics.extend(unsupported_diags)
        if unsupported_diags:
            candidates.append((IntentOutcome.UNSUPPORTED, "unsupported-intent"))

        soft_conflict, soft_diags = _soft_objective_conflicts(intent, fields, intent.provenance)
        diagnostics.extend(soft_diags)

        if not _has_hard_constraint(intent):
            diagnostics.append(
                _policy_diagnostic(
                    "hard-constraint-incomplete",
                    "an executable policy decision requires at least one complete hard constraint",
                    None,
                    intent.provenance,
                )
            )
            candidates.append((IntentOutcome.CLARIFICATION_REQUIRED, "hard-constraint-incomplete"))

        for assessment in confidence:
            if assessment.required and assessment.category is ConfidenceCategory.LOW:
                diagnostics.append(
                    _policy_diagnostic(
                        "low-confidence-required-field",
                        "required interpretation confidence is low; clarification is required",
                        fields[assessment.field_id],
                        intent.provenance,
                    )
                )
                candidates.append(
                    (IntentOutcome.CLARIFICATION_REQUIRED, "low-confidence-required-field")
                )

        if soft_conflict:
            diagnostics.append(
                _policy_diagnostic(
                    "conflicting-soft-objectives",
                    "multiple soft objectives target the same metric; the preference "
                    "must be clarified",
                    None,
                    intent.provenance,
                )
            )
            candidates.append((IntentOutcome.CLARIFICATION_REQUIRED, "conflicting-soft-objectives"))

        # The compiler is authoritative for schema validity.  A soft-preference
        # collision is the one compiler refusal that is safely reclassified as
        # clarification because no hard policy is being weakened.
        if compiled.outcome is not IntentOutcome.EXECUTABLE:
            compiler_outcome = (
                IntentOutcome.CLARIFICATION_REQUIRED
                if soft_conflict
                and not hard_conflicts
                and not unsupported_diags
                and compiled.outcome is IntentOutcome.REFUSED
                else compiled.outcome
            )
            candidates.append((compiler_outcome, "compiler-outcome"))
        else:
            candidates.append((IntentOutcome.EXECUTABLE, "compiled"))

        outcome = _precedence(candidates)
        diagnostics.append(
            _policy_diagnostic(
                "policy-outcome",
                f"policy outcome is {outcome.value}; confidence does not replace "
                "contract validation",
                None,
                intent.provenance,
                severity=DiagnosticSeverity.INFO
                if outcome is IntentOutcome.EXECUTABLE
                else DiagnosticSeverity.WARNING,
            )
        )
        ordered = tuple(sorted(diagnostics, key=_diagnostic_sort_key))
        return IntentPolicyDecision(
            intent.intent_id,
            outcome,
            ordered,
            confidence,
            ambiguities,
            intent.provenance,
            compiled,
        )


def evaluate_intent_policy(
    intent: IntentRecord,
    compilation: IntentCompilation | None = None,
    *,
    high_confidence: float = 0.9,
    medium_confidence: float = 0.7,
) -> IntentPolicyDecision:
    """Evaluate policy after bounded intent compilation."""

    return IntentPolicyEvaluator(
        high_confidence=high_confidence, medium_confidence=medium_confidence
    ).evaluate(intent, compilation)


def evaluate_intent(
    intent: IntentRecord,
    compilation: IntentCompilation | None = None,
) -> IntentPolicyDecision:
    """Compatibility shorthand for :func:`evaluate_intent_policy`."""

    return evaluate_intent_policy(intent, compilation)


def _confidence_assessment(field: IntentField, high: float, medium: float) -> ConfidenceAssessment:
    category = (
        ConfidenceCategory.HIGH
        if field.confidence >= high
        else ConfidenceCategory.MEDIUM
        if field.confidence >= medium
        else ConfidenceCategory.LOW
    )
    return ConfidenceAssessment(
        field.field_id, field.confidence, category, field.required, field.source_span_ids
    )


def _ambiguity_assessment(
    item: AmbiguityRecord, fields: Mapping[str, IntentField]
) -> AmbiguityAssessment:
    normalized = item.category.replace("-", "_").lower()
    category = {
        "missing": AmbiguityCategory.MISSING,
        "vague": AmbiguityCategory.VAGUE,
        "contradictory": AmbiguityCategory.CONTRADICTORY,
        "conflicting": AmbiguityCategory.CONFLICTING,
        "unsupported": AmbiguityCategory.UNSUPPORTED,
    }.get(normalized, AmbiguityCategory.UNRESOLVED)
    field = fields.get(item.field_id)
    return AmbiguityAssessment(
        item.ambiguity_id,
        item.field_id,
        category,
        item.category,
        item.detail,
        item.alternatives,
        item.required,
        () if field is None else field.source_span_ids,
    )


def _from_compiler_diagnostic(
    item: CompilerDiagnostic,
    fields: Mapping[str, IntentField],
    provenance: IntentProvenance,
) -> PolicyDiagnostic:
    field = fields.get(item.field_id) if item.field_id is not None else None
    spans = (
        item.source_span_ids
        if item.source_span_ids
        else (() if field is None else field.source_span_ids)
    )
    return PolicyDiagnostic(
        item.code,
        item.message,
        item.severity,
        item.field_id,
        spans,
        provenance.evidence_refs,
    )


def _policy_diagnostic(
    code: str,
    message: str,
    field: IntentField | None,
    provenance: IntentProvenance,
    *,
    severity: DiagnosticSeverity = DiagnosticSeverity.ERROR,
) -> PolicyDiagnostic:
    return PolicyDiagnostic(
        code,
        message,
        severity,
        None if field is None else field.field_id,
        () if field is None else field.source_span_ids,
        provenance.evidence_refs,
    )


def _kind(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get("kind", value.get("type", value.get("declaration")))
    if not isinstance(raw, str):
        return None
    return {
        "constraint": "hard_constraint",
        "hard_constraint": "hard_constraint",
        "hard-constraint": "hard_constraint",
        "objective": "soft_objective",
        "soft_objective": "soft_objective",
        "soft-objective": "soft_objective",
        "preference": "soft_objective",
    }.get(raw)


def _metric(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _has_hard_constraint(intent: IntentRecord) -> bool:
    return any(_kind(field.value) == "hard_constraint" for field in intent.fields)


def _hard_constraint_conflicts(
    intent: IntentRecord,
    fields: Mapping[str, IntentField],
    provenance: IntentProvenance,
) -> tuple[bool, list[PolicyDiagnostic]]:
    grouped: defaultdict[str, list[tuple[IntentField, Mapping[str, object]]]] = defaultdict(list)
    for field in intent.fields:
        if _kind(field.value) == "hard_constraint" and isinstance(field.value, Mapping):
            metric = _metric(field.value.get("metric"))
            if metric is not None:
                grouped[metric].append((field, field.value))
    diagnostics: list[PolicyDiagnostic] = []
    for metric, terms in sorted(grouped.items()):
        if len(terms) < 2:
            continue
        fields_for_reason = sorted(terms, key=lambda item: item[0].field_id)
        source = fields_for_reason[0][0]
        directions = {str(term.get("direction")) for _, term in terms}
        minimums = [
            term.get("threshold")
            for _, term in terms
            if term.get("direction") == "minimum"
        ]
        maximums = [
            term.get("threshold")
            for _, term in terms
            if term.get("direction") == "maximum"
        ]
        numeric_minimums = [value for value in minimums if isinstance(value, (int, float))]
        numeric_maximums = [value for value in maximums if isinstance(value, (int, float))]
        contradictory = (
            "minimum" in directions
            and "maximum" in directions
            and all(not isinstance(value, bool) for value in numeric_minimums + numeric_maximums)
            and numeric_minimums
            and numeric_maximums
            and max(cast(float, value) for value in numeric_minimums)
            > min(cast(float, value) for value in numeric_maximums)
        )
        code = "contradictory-hard-constraints" if contradictory else "conflicting-hard-constraints"
        diagnostics.append(
            _policy_diagnostic(
                code,
                f"multiple hard constraints target metric {metric!r}; hard policy "
                "cannot choose between them",
                source,
                provenance,
            )
        )
    return bool(diagnostics), diagnostics


def _soft_objective_conflicts(
    intent: IntentRecord,
    fields: Mapping[str, IntentField],
    provenance: IntentProvenance,
) -> tuple[bool, list[PolicyDiagnostic]]:
    grouped: defaultdict[str, list[IntentField]] = defaultdict(list)
    for field in intent.fields:
        if _kind(field.value) == "soft_objective" and isinstance(field.value, Mapping):
            metric = _metric(field.value.get("metric"))
            if metric is not None:
                grouped[metric].append(field)
    diagnostics: list[PolicyDiagnostic] = []
    for metric, terms in sorted(grouped.items()):
        if len(terms) > 1:
            diagnostics.append(
                _policy_diagnostic(
                    "conflicting-soft-objectives",
                    f"multiple soft objectives target metric {metric!r}",
                    sorted(terms, key=lambda item: item.field_id)[0],
                    provenance,
                )
            )
    return bool(diagnostics), diagnostics


def _unsupported_declarations(
    intent: IntentRecord,
    fields: Mapping[str, IntentField],
    provenance: IntentProvenance,
) -> list[PolicyDiagnostic]:
    supported = {item.value for item in ContractMetric}
    diagnostics: list[PolicyDiagnostic] = []
    for field in intent.fields:
        kind = _kind(field.value)
        if kind is None:
            # The compiler supplies the canonical unsupported-field diagnostic.
            continue
        assert isinstance(field.value, Mapping)
        metric = _metric(field.value.get("metric"))
        plugin = field.value.get("plugin")
        if metric is None:
            continue
        if metric not in supported and not (kind == "soft_objective" and plugin is not None):
            diagnostics.append(
                _policy_diagnostic(
                    "unsupported-metric",
                    f"metric {metric!r} is outside the verified objective-contract boundary",
                    field,
                    provenance,
                )
            )
    return diagnostics


def _precedence(candidates: list[tuple[IntentOutcome, str]]) -> IntentOutcome:
    """Choose the safest result, independent of input or diagnostic order."""

    rank = {
        IntentOutcome.EXECUTABLE: 0,
        IntentOutcome.CLARIFICATION_REQUIRED: 1,
        IntentOutcome.UNSUPPORTED: 2,
        IntentOutcome.REFUSED: 3,
    }
    if not candidates:
        return IntentOutcome.REFUSED
    return max(candidates, key=lambda item: (rank[item[0]], item[1]))[0]


def _diagnostic_sort_key(item: PolicyDiagnostic) -> tuple[str, str, str, str, tuple[str, ...]]:
    return (item.code, item.field_id or "", item.message, item.severity.value, item.source_span_ids)


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


__all__ = [
    "INTENT_POLICY_SCHEMA_VERSION",
    "AmbiguityAssessment",
    "AmbiguityCategory",
    "ConfidenceAssessment",
    "ConfidenceCategory",
    "IntentPolicyDecision",
    "IntentPolicyError",
    "IntentPolicyEvaluator",
    "PolicyDiagnostic",
    "evaluate_intent",
    "evaluate_intent_policy",
]
