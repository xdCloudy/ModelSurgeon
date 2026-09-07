"""Focused tests for the bounded conversational intent compiler."""

from __future__ import annotations

from modelsurgeon.conversation import (
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    SourceSpan,
)
from modelsurgeon.search import (
    DiagnosticSeverity,
    compile_intent_record,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ContractMetric,
    HardConstraint,
    MetricUnit,
    ObjectiveContract,
    ObjectiveDirection,
    ObjectiveNormalization,
    SoftObjective,
)


def _intent(
    fields: tuple[IntentField, ...],
    *,
    emitted_spec: dict[str, object] | None = None,
    outcome: IntentOutcome = IntentOutcome.EXECUTABLE,
    provider_revision: str = "provider-v1",
) -> IntentRecord:
    request = "Keep quality above 0.95 and minimize latency."
    span = SourceSpan("request", 0, len(request), request)
    return IntentRecord(
        original_request=request,
        source_spans=(span,),
        fields=fields,
        ambiguities=(),
        interpretation_steps=(),
        provenance=IntentProvenance("source-v1", "local-provider", provider_revision, "tool-v1"),
        outcome=outcome,
        diagnostics=("structured interpretation complete",),
        emitted_spec=emitted_spec,
    )


def _fields() -> tuple[IntentField, ...]:
    return (
        IntentField(
            "constraint.quality",
            {
                "kind": "hard_constraint",
                "metric": "quality",
                "direction": "minimum",
                "threshold": 0.95,
                "unit": "ratio",
            },
            "ratio",
            1.0,
            ("request",),
            True,
        ),
        IntentField(
            "objective.latency",
            {
                "kind": "preference",
                "metric": "latency",
                "direction": "minimize",
                "unit": "milliseconds",
                "normalization": "identity",
            },
            "milliseconds",
            1.0,
            ("request",),
        ),
    )


def _spec() -> dict[str, object]:
    return ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.LATENCY,
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
        ),
    ).to_record()


def test_compiler_emits_existing_contract_record_without_execution() -> None:
    result = compile_intent_record(_intent(_fields(), emitted_spec=_spec()))

    assert result.executable
    assert result.spec_record == _spec()
    assert result.objective_contract is not None
    compiled = next(
        item for item in result.diagnostics if item.code == "compiled-objective-contract"
    )
    assert compiled.severity is DiagnosticSeverity.INFO


def test_equivalent_interpretations_have_deterministic_compilation_identity() -> None:
    first = compile_intent_record(_intent(_fields(), emitted_spec=_spec()))
    second = compile_intent_record(
        _intent(_fields(), emitted_spec=_spec(), provider_revision="provider-v2")
    )

    assert first.intent_id == second.intent_id
    assert first.canonical_json() == second.canonical_json()


def test_missing_hard_constraint_is_clarification_not_an_invented_threshold() -> None:
    fields = (_fields()[1],)
    result = compile_intent_record(_intent(fields, emitted_spec={}))

    assert result.outcome is IntentOutcome.CLARIFICATION_REQUIRED
    assert result.spec_record is None
    assert any(item.code == "missing-hard-constraint" for item in result.diagnostics)


def test_unrepresentable_budget_is_explicitly_unsupported() -> None:
    budget = IntentField(
        "budget.evaluations",
        {"kind": "budget", "value": 10, "unit": "count"},
        "count",
        1.0,
        ("request",),
    )
    result = compile_intent_record(
        _intent(
            tuple(sorted((*_fields(), budget), key=lambda item: item.field_id)),
            emitted_spec=_spec(),
        )
    )

    assert result.outcome is IntentOutcome.UNSUPPORTED
    assert result.spec_record is None
    assert any(item.code == "unsupported-intent-field" for item in result.diagnostics)


def test_non_executable_input_cannot_be_promoted() -> None:
    intent = _intent(
        _fields(),
        outcome=IntentOutcome.CLARIFICATION_REQUIRED,
        emitted_spec=None,
    )
    result = compile_intent_record(intent)

    assert result.outcome is IntentOutcome.CLARIFICATION_REQUIRED
    assert result.objective_contract is None
    assert any(item.code == "intent-not-executable" for item in result.diagnostics)


def test_emitted_spec_mismatch_is_refused() -> None:
    wrong = dict(_spec())
    wrong["contract_id"] = "objective_contract_" + "0" * 64
    intent = _intent(_fields(), emitted_spec=wrong)
    result = compile_intent_record(intent)

    assert result.outcome is IntentOutcome.REFUSED
    assert result.spec_record is None
    assert any(item.code == "invalid-emitted-spec" for item in result.diagnostics)


def test_compiler_result_diagnostics_are_canonical() -> None:
    result = compile_intent_record(_intent(_fields(), emitted_spec=_spec()))

    codes = [item.code for item in result.diagnostics]
    assert codes == sorted(codes)


def test_contradictory_hard_constraints_retain_a_minimal_witness() -> None:
    fields = (
        *_fields(),
        IntentField(
            "constraint.quality.upper",
            {
                "kind": "hard_constraint",
                "metric": "quality",
                "direction": "maximum",
                "threshold": 0.80,
                "unit": "ratio",
            },
            "ratio",
            1.0,
            ("request",),
            True,
        ),
        IntentField(
            "constraint.quality.unrelated",
            {
                "kind": "hard_constraint",
                "metric": "quality",
                "direction": "minimum",
                "threshold": 0.99,
                "unit": "ratio",
            },
            "ratio",
            1.0,
            ("request",),
            True,
        ),
    )
    result = compile_intent_record(
        _intent(
            tuple(sorted(fields, key=lambda item: item.field_id)),
            emitted_spec=_spec(),
        )
    )

    assert result.outcome is IntentOutcome.REFUSED
    conflict = next(
        item for item in result.diagnostics if item.code == "contradictory-hard-constraints"
    )
    assert conflict.related_field_ids == (
        "constraint.quality",
        "constraint.quality.upper",
    )
    assert conflict.source_span_ids == ("request",)


def test_duplicate_preferences_are_unresolved_not_ordered_by_input() -> None:
    fields = (
        *_fields(),
        IntentField(
            "objective.latency.alternate",
            {
                "kind": "preference",
                "metric": "latency",
                "direction": "maximize",
                "unit": "milliseconds",
                "normalization": "identity",
            },
            "milliseconds",
            1.0,
            ("request",),
        ),
    )
    result = compile_intent_record(
        _intent(tuple(sorted(fields, key=lambda item: item.field_id)), emitted_spec=_spec())
    )

    assert result.outcome is IntentOutcome.REFUSED
    conflict = next(
        item for item in result.diagnostics if item.code == "ambiguous-preference-ordering"
    )
    assert conflict.related_field_ids == (
        "objective.latency",
        "objective.latency.alternate",
    )
