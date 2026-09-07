"""Measured final repair and quantization strategy tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from modelsurgeon.search.physical_strategy import (
    PhysicalStrategyCandidate,
    PhysicalStrategyKind,
    PhysicalStrategyOutcome,
    PhysicalStrategyRequest,
    select_physical_strategy,
)

SOURCE = "sha256:" + "a" * 64
PROVENANCE = (
    "artifact=artifact-manifest",
    "dataset=dataset-manifest",
    "deployment=llama.cpp@rev",
    "quantization=Q4_K_M:after_repair",
    "repair=distillation:teacher-1",
    "teacher=teacher-1",
)


def _candidate(
    candidate_id: str,
    kind: PhysicalStrategyKind,
    *,
    quality: float = 0.99,
    deployment_cost: float = 2.0,
    optimization_cost: float = 1.0,
    measured: bool = True,
    compatible: bool = True,
    constraints_passed: bool = True,
) -> PhysicalStrategyCandidate:
    return PhysicalStrategyCandidate(
        candidate_id,
        kind,
        measured,
        compatible,
        constraints_passed,
        quality,
        deployment_cost,
        optimization_cost,
        100,
        SOURCE,
        "sha256:" + hashlib.sha256(candidate_id.encode()).hexdigest(),
        PROVENANCE,
    )


def _request() -> PhysicalStrategyRequest:
    return PhysicalStrategyRequest(SOURCE, 0.98, 5.0, 5.0, 200)


def test_measured_controls_and_quality_cost_selection_are_required() -> None:
    candidates = (
        _candidate("final_candidate_no_repair", PhysicalStrategyKind.NO_REPAIR, quality=0.985),
        _candidate("final_candidate_quant", PhysicalStrategyKind.QUANTIZATION_ONLY, quality=0.986),
        _candidate("final_candidate_repair", PhysicalStrategyKind.REPAIR, quality=0.995),
    )
    decision = select_physical_strategy(_request(), candidates)
    assert decision.outcome is PhysicalStrategyOutcome.SELECTED
    assert decision.selected is not None
    assert decision.selected.candidate_id == "final_candidate_repair"
    assert decision.measured_controls == (
        PhysicalStrategyKind.NO_REPAIR,
        PhysicalStrategyKind.QUANTIZATION_ONLY,
    )


def test_predicted_or_failed_candidates_cannot_bypass_controls() -> None:
    candidates = (
        _candidate("final_candidate_no_repair", PhysicalStrategyKind.NO_REPAIR),
        _candidate(
            "final_candidate_quant",
            PhysicalStrategyKind.QUANTIZATION_ONLY,
            measured=False,
        ),
        _candidate("final_candidate_repair", PhysicalStrategyKind.REPAIR),
    )
    decision = select_physical_strategy(_request(), candidates)
    assert decision.outcome is PhysicalStrategyOutcome.UNKNOWN
    assert decision.selected is None
    assert any(key == "final_candidate_quant" for key, _ in decision.excluded)


def test_incompatible_control_and_budget_failure_are_explicit() -> None:
    candidates = (
        _candidate("final_candidate_no_repair", PhysicalStrategyKind.NO_REPAIR),
        _candidate(
            "final_candidate_quant",
            PhysicalStrategyKind.QUANTIZATION_ONLY,
            compatible=False,
        ),
        _candidate("final_candidate_repair", PhysicalStrategyKind.REPAIR),
    )
    decision = select_physical_strategy(_request(), candidates)
    assert decision.outcome is PhysicalStrategyOutcome.UNSUPPORTED

    budget = select_physical_strategy(
        replace(_request(), max_optimization_cost=0.5),
        (
            _candidate("final_candidate_no_repair", PhysicalStrategyKind.NO_REPAIR),
            _candidate("final_candidate_quant", PhysicalStrategyKind.QUANTIZATION_ONLY),
            _candidate("final_candidate_repair", PhysicalStrategyKind.REPAIR),
        ),
    )
    assert budget.outcome is PhysicalStrategyOutcome.FAILED
