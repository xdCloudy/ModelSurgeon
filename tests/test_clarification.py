"""Deterministic v2.4 clarification state-machine tests."""

from __future__ import annotations

import json

import pytest

from modelsurgeon.conversation import (
    AmbiguityRecord,
    ClarificationAnswer,
    ClarificationMachine,
    ClarificationStatus,
    IntentField,
    IntentOutcome,
    IntentProvenance,
    IntentRecord,
    InterpretationStep,
    SourceSpan,
    replay_clarification,
)
from modelsurgeon.search.intent_policy import evaluate_intent_policy
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

REQUEST = "optimize the model"
PROVENANCE = IntentProvenance(
    "request-v24",
    "fixture-provider",
    "fixture-provider-v1",
    "clarification-test-v1",
    ("evidence:request",),
)


def _spec() -> dict[str, object]:
    return ObjectiveContract(
        (
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
        ),
        (
            SoftObjective(
                ContractMetric.LATENCY,
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
        ),
    ).to_record()


def _intent(
    fields: tuple[IntentField, ...],
    *,
    outcome: IntentOutcome = IntentOutcome.CLARIFICATION_REQUIRED,
    ambiguities: tuple[AmbiguityRecord, ...] = (),
    emitted_spec: dict[str, object] | None = None,
) -> IntentRecord:
    span = SourceSpan("request", 0, len(REQUEST), REQUEST)
    steps = (
        InterpretationStep(
            "normalize-request",
            "normalize",
            (span.span_id,),
            tuple(item.field_id for item in fields),
            0.99,
            PROVENANCE.evidence_refs,
        ),
    )
    return IntentRecord(
        REQUEST,
        (span,),
        fields,
        ambiguities,
        steps,
        PROVENANCE,
        outcome,
        ("fixture intent",),
        emitted_spec,
    )


def _soft() -> IntentField:
    return IntentField(
        "objective.latency",
        {
            "kind": "soft_objective",
            "metric": "latency",
            "direction": "minimize",
            "unit": "milliseconds",
            "normalization": "identity",
        },
        "milliseconds",
        0.99,
        ("request",),
    )


def _hard(*, confidence: float = 0.99) -> IntentField:
    return IntentField(
        "constraint.quality",
        {
            "kind": "hard_constraint",
            "metric": "quality",
            "direction": "minimum",
            "threshold": 0.95,
            "unit": "ratio",
        },
        "ratio",
        confidence,
        ("request",),
        True,
    )


def test_incomplete_request_gets_only_necessary_question_and_becomes_executable() -> None:
    intent = _intent((_soft(),))
    machine = ClarificationMachine()
    state = machine.start(intent)

    assert state.status is ClarificationStatus.INCOMPLETE
    assert len(state.questions) == 1
    question = state.questions[0]
    assert question.category == "missing-hard-constraint"
    assert "threshold" in question.prompt

    completed = machine.answer(
        state,
        ClarificationAnswer(
            question.question_id,
            {
                "metric": "quality",
                "direction": "minimum",
                "threshold": 0.95,
                "unit": "ratio",
            },
        ),
    )
    assert completed.status is ClarificationStatus.EXECUTABLE
    assert completed.policy.contract is not None
    assert completed.policy.contract.to_record() == _spec()
    assert any(
        item.startswith("clarification-answer:")
        for item in completed.intent.provenance.evidence_refs
    )


def test_ambiguous_and_contradictory_answers_fail_closed() -> None:
    ambiguity = AmbiguityRecord(
        "ambiguity-quality-direction",
        "constraint.quality",
        "vague",
        "quality direction was not explicit",
        ("maximum", "minimum"),
    )
    intent = _intent((_hard(), _soft()), ambiguities=(ambiguity,))
    state = ClarificationMachine().start(intent)
    assert state.status is ClarificationStatus.AMBIGUOUS
    assert state.questions[0].alternatives == ("maximum", "minimum")

    contradictory = ClarificationMachine().answer(
        state,
        ClarificationAnswer(
            state.questions[0].question_id,
            {"kind": "soft_objective", "metric": "quality"},
        ),
    )
    assert contradictory.status is ClarificationStatus.CONTRADICTORY
    assert not contradictory.executable
    assert contradictory.policy.contract is None


def test_unsupported_and_repeated_answers_are_retained_and_replayable() -> None:
    state = ClarificationMachine().start(_intent((_soft(),)))
    question = state.questions[0]
    invalid = ClarificationAnswer(question.question_id, {"metric": "quality"})
    unsupported = ClarificationMachine().answer(state, invalid)
    assert unsupported.status is ClarificationStatus.UNSUPPORTED
    repeated = ClarificationMachine().answer(unsupported, invalid)
    assert repeated.status is ClarificationStatus.REPEATED
    assert [item.answer_digest for item in repeated.answers] == [
        invalid.answer_digest,
        invalid.answer_digest,
    ]
    replayed = replay_clarification(_intent((_soft(),)), (invalid, invalid))
    assert replayed.canonical_json() == repeated.canonical_json()


def test_cancellation_is_terminal_and_canonical_replay_is_byte_stable() -> None:
    state = ClarificationMachine().start(_intent((_soft(),)))
    cancelled = ClarificationMachine().cancel(state)
    assert cancelled.status is ClarificationStatus.CANCELLED
    assert cancelled.questions == ()
    with pytest.raises(ValueError, match="cancelled"):
        ClarificationMachine().answer(
            cancelled, ClarificationAnswer("unknown-question", {"value": 1})
        )

    restored = type(cancelled).from_json(cancelled.canonical_json())
    assert restored.canonical_json() == cancelled.canonical_json()
    tampered = json.loads(cancelled.canonical_json())
    tampered["status"] = "executable"
    with pytest.raises(ValueError):
        type(cancelled).from_json(json.dumps(tampered))


def test_already_executable_and_unsupported_intent_do_not_ask_questions() -> None:
    executable = _intent(
        (_hard(), _soft()),
        outcome=IntentOutcome.EXECUTABLE,
        emitted_spec=_spec(),
    )
    ready = ClarificationMachine().start(executable, evaluate_intent_policy(executable))
    assert ready.status is ClarificationStatus.EXECUTABLE
    assert ready.questions == ()

    unsupported_field = IntentField(
        "unsupported.field",
        {"kind": "unsupported_declaration", "value": "not supported"},
        None,
        0.99,
        ("request",),
    )
    unsupported = _intent(
        (_hard(), _soft(), unsupported_field),
        outcome=IntentOutcome.EXECUTABLE,
        emitted_spec=_spec(),
    )
    refused = ClarificationMachine().start(unsupported)
    assert refused.status is ClarificationStatus.UNSUPPORTED
    assert refused.questions == ()
