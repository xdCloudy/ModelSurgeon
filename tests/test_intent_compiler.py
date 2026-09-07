"""Tests for the bounded intent-to-contract compiler."""

from __future__ import annotations

from modelsurgeon.conversation import (
    IntentCompiler,
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    InterpretationStep,
    SourceSpan,
    compile_intent,
)


def _record(
    fields: tuple[IntentField, ...], *, outcome: IntentOutcome = IntentOutcome.EXECUTABLE
) -> IntentRecord:
    request = "reduce latency while retaining at least 0.98 quality"
    spans = (SourceSpan("span-request", 0, len(request), request),)
    return IntentRecord(
        request,
        spans,
        tuple(sorted(fields, key=lambda field: field.field_id)),
        (),
        (
            InterpretationStep(
                "step-normalize",
                "normalize",
                ("span-request",),
                tuple(sorted(field.field_id for field in fields)),
                1.0,
                ("evidence:request",),
            ),
        ),
        IntentProvenance(
            "request-v1", "test-provider", "provider-v1", "compiler-v1", ("evidence:request",)
        ),
        outcome,
        ("provider interpretation is not execution authority",),
        {"untrusted": True} if outcome is IntentOutcome.EXECUTABLE else None,
    )


def test_compiles_explicit_fields_and_is_byte_stable() -> None:
    record = _record(
        (
            IntentField("objective.metric", "latency", None, 1.0, ("span-request",)),
            IntentField("quality.minimum", 0.98, "ratio", 1.0, ("span-request",), True),
            IntentField("budget.wall_time", 60, "seconds", 1.0, ("span-request",)),
            IntentField("operations.allowed", ["evaluate", "search"], None, 1.0, ("span-request",)),
            IntentField("deployment.targets", ["cpu"], None, 1.0, ("span-request",)),
        )
    )
    first = compile_intent(record)
    second = compile_intent(record)
    assert first.executable
    assert first.spec is not None
    assert first.spec.canonical_json() == second.spec.canonical_json()
    assert first.spec.contract.constraints[0].threshold == 0.98
    assert first.spec.allowed_operations == ("evaluate", "search")
    assert first.intent_record.emitted_spec == first.spec.to_record()


def test_missing_hard_constraint_requires_clarification_and_does_not_read_text() -> None:
    record = _record((IntentField("objective.metric", "latency", None, 1.0, ("span-request",)),))
    result = compile_intent(record)
    assert result.outcome is IntentOutcome.CLARIFICATION_REQUIRED
    assert result.spec is None
    assert any(item.code == "missing_hard_constraint" for item in result.diagnostics)


def test_unsupported_operation_fails_closed() -> None:
    record = _record(
        (
            IntentField("objective.metric", "latency", None, 1.0, ("span-request",)),
            IntentField("quality.minimum", 0.98, "ratio", 1.0, ("span-request",), True),
            IntentField("operations.allowed", ["choose_tensors"], None, 1.0, ("span-request",)),
        )
    )
    result = IntentCompiler().compile(record)
    assert result.outcome is IntentOutcome.UNSUPPORTED
    assert result.spec is None
    assert any(item.code == "unsupported_operation" for item in result.diagnostics)


def test_non_executable_provider_record_is_refused() -> None:
    record = _record((), outcome=IntentOutcome.CLARIFICATION_REQUIRED)
    result = compile_intent(record)
    assert result.outcome is IntentOutcome.REFUSED
    assert result.spec is None
    assert any(item.code == "intent_not_executable" for item in result.diagnostics)
