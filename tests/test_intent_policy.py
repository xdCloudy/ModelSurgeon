"""Fail-closed conversational intent policy tests."""

from __future__ import annotations

from dataclasses import replace

from modelsurgeon.conversation import (
    AmbiguityRecord,
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    InterpretationStep,
    SourceSpan,
)
from modelsurgeon.search.intent_policy import evaluate_intent_policy
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    HardConstraint,
    MetricUnit,
    ObjectiveContract,
    ObjectiveDirection,
    SoftObjective,
)

REQUEST = "retain quality while reducing latency"
PROVENANCE = IntentProvenance(
    "request-v1",
    "provider",
    "provider-v1",
    "tool-v1",
    ("evidence:request",),
)


def _field(
    field_id: str,
    value: object,
    *,
    confidence: float = 0.99,
    required: bool = True,
) -> IntentField:
    return IntentField(field_id, value, None, confidence, ("span-request",), required)


def _record(
    fields: tuple[IntentField, ...],
    *,
    ambiguity: AmbiguityRecord | None = None,
) -> IntentRecord:
    contract = ObjectiveContract(
        (
            HardConstraint(
                "quality",
                ConstraintDirection.MINIMUM,
                0.98,
                MetricUnit.RATIO,
                "immutable_source",
            ),
        ),
        (
            SoftObjective(
                "latency",
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
            ),
        ),
    )
    outcome = IntentOutcome.CLARIFICATION_REQUIRED if ambiguity else IntentOutcome.EXECUTABLE
    ordered_fields = tuple(sorted(fields, key=lambda item: item.field_id))
    return IntentRecord(
        REQUEST,
        (SourceSpan("span-request", 0, len(REQUEST), REQUEST),),
        ordered_fields,
        () if ambiguity is None else (ambiguity,),
        (
            InterpretationStep(
                "step-normalize",
                "normalize",
                ("span-request",),
                tuple(item.field_id for item in ordered_fields),
                0.99,
                ("evidence:request",),
            ),
        ),
        PROVENANCE,
        outcome,
        ("typed request",),
        contract.to_record() if ambiguity is None else None,
    )


def _valid_fields() -> tuple[IntentField, ...]:
    return (
        _field(
            "constraint.quality",
            {
                "kind": "hard_constraint",
                "metric": "quality",
                "direction": "minimum",
                "threshold": 0.98,
                "unit": "ratio",
                "baseline": "immutable_source",
            },
        ),
        _field(
            "objective.latency",
            {
                "kind": "soft_objective",
                "metric": "latency",
                "direction": "minimize",
                "unit": "milliseconds",
            },
        ),
    )


def test_valid_compilation_is_executable_and_confidence_is_not_evidence() -> None:
    decision = evaluate_intent_policy(_record(_valid_fields()))
    assert decision.outcome is IntentOutcome.EXECUTABLE
    assert decision.contract is not None
    assert all(item.category.value == "high" for item in decision.confidence)
    assert decision.decision_id.startswith("policy_")


def test_low_required_confidence_requires_clarification() -> None:
    fields = tuple(
        replace(item, confidence=0.2) if item.field_id == "constraint.quality" else item
        for item in _valid_fields()
    )
    decision = evaluate_intent_policy(_record(fields))
    assert decision.outcome is IntentOutcome.CLARIFICATION_REQUIRED
    assert decision.contract is None
    assert any(item.code == "low-confidence-required-field" for item in decision.diagnostics)


def test_contradiction_precedes_confidence_and_never_emits_spec() -> None:
    fields = (
        *_valid_fields(),
        _field(
            "constraint.quality.upper",
            {
                "kind": "hard_constraint",
                "metric": "quality",
                "direction": "maximum",
                "threshold": 0.8,
                "unit": "ratio",
            },
        ),
    )
    decision = evaluate_intent_policy(_record(fields))
    assert decision.outcome is IntentOutcome.REFUSED
    assert decision.contract is None
    assert any(item.code == "contradictory-hard-constraints" for item in decision.diagnostics)


def test_required_ambiguity_is_provenance_linked_and_safe() -> None:
    ambiguity = AmbiguityRecord(
        "ambiguity-latency",
        "objective.latency",
        "vague",
        "latency target is not bounded",
        ("ask user", "use default"),
    )
    decision = evaluate_intent_policy(_record(_valid_fields(), ambiguity=ambiguity))
    assert decision.outcome is IntentOutcome.CLARIFICATION_REQUIRED
    assert decision.contract is None
    assert decision.ambiguities[0].source_span_ids == ("span-request",)
    assert decision.diagnostics[0].provenance_refs == ("evidence:request",)
