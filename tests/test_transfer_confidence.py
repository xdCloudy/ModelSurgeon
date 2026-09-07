from __future__ import annotations

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.surgeon import (
    CompatibilityContext,
    CompatibilityDecision,
    CompatibilityOperation,
    ConfidenceDecisionStatus,
    ConfidenceInput,
    ConfidenceObservation,
    ConfidenceOutcomeStatus,
    ConfidencePolicy,
    ModelLineageGraph,
    ModelLineageNode,
    TransferConfidenceReport,
    build_selective_risk_curve,
    decide_compatibility,
    decide_transfer_confidence,
)


def _compatibility(*, known: bool = True) -> CompatibilityDecision:
    source = CompatibilityContext(
        "source",
        "rev-source",
        ModelFamily.LLAMA,
        "arch-v1",
        "features-v1",
        1,
        "targets-v1",
        1,
        "q4_k",
        "cpu",
        1,
        (CompatibilityOperation.INFER,),
    )
    target = CompatibilityContext(
        "target",
        "rev-target",
        ModelFamily.QWEN,
        "arch-v1",
        "features-v1",
        1,
        "targets-v1",
        1,
        "q4_k",
        "cpu",
        1,
        (CompatibilityOperation.INFER,),
    )
    return decide_compatibility(
        source,
        target,
        operation=CompatibilityOperation.INFER,
        lineage=(
            ModelLineageGraph(
                (
                    ModelLineageNode("source", "rev-source", ModelFamily.LLAMA, "arch-v1"),
                    ModelLineageNode("target", "rev-target", ModelFamily.QWEN, "arch-v1"),
                )
            )
            if known
            else None
        ),
        allow_adaptation=True,
    )


def _input(
    request_id: str, *, uncertainty: float = 0.1, supported: bool = True
) -> ConfidenceInput:
    return ConfidenceInput(
        request_id,
        _compatibility(),
        0.5,
        0.45,
        0.55,
        0.9,
        1.0,
        uncertainty,
        0.02,
        supported,
        supported,
        supported,
    )


def test_confidence_fails_closed_before_learned_thresholds() -> None:
    accepted = decide_transfer_confidence(_input("accepted"))
    assert accepted.status is ConfidenceDecisionStatus.ACCEPTED
    abstained = decide_transfer_confidence(_input("uncertain", uncertainty=0.9))
    assert abstained.status is ConfidenceDecisionStatus.ABSTAINED
    unsupported = decide_transfer_confidence(_input("unsupported", supported=False))
    assert unsupported.status is ConfidenceDecisionStatus.UNSUPPORTED
    unknown = decide_transfer_confidence(
        ConfidenceInput(
            "unknown",
            _compatibility(known=False),
            0.5,
            0.45,
            0.55,
            0.99,
            0.1,
            0.01,
            0.01,
            True,
            True,
            True,
        )
    )
    assert unknown.status is ConfidenceDecisionStatus.UNKNOWN


def test_selective_risk_curve_retains_coverage_and_cost() -> None:
    policy = ConfidencePolicy()
    observations = tuple(
        ConfidenceObservation(
            decide_transfer_confidence(_input(f"request-{index}"), policy=policy),
            index == 1,
            2.0 + index,
            ConfidenceOutcomeStatus.MEASURED,
        )
        for index in range(2)
    )
    report = build_selective_risk_curve(observations, (0.05, 0.2), policy=policy)
    assert isinstance(report, TransferConfidenceReport)
    assert report.curve[0].accepted_count == 0
    assert report.curve[1].accepted_count == 2
    assert report.curve[1].coverage == 1.0
    assert report.curve[1].violation_rate == 0.5
    assert report.curve[1].target_evaluation_seconds == 5.0
