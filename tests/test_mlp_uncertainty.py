from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.active_learning.mlp_uncertainty import (
    MLPMethodEvidence,
    MLPUncertaintyBudget,
    MLPUncertaintyMethod,
    compare_mlp_uncertainty,
    estimate_mlp_uncertainty,
)


def test_mlp_uncertainty_reports_calibration_and_active_selection_lift() -> None:
    ensemble = estimate_mlp_uncertainty(
        ((0.0, 1.0, 1.0, 3.0), (0.0, 1.0, 3.0, 3.0), (0.0, 1.0, 2.0, 3.0)),
        confidence=0.8,
        max_prediction_values=100,
    )
    dropout = estimate_mlp_uncertainty(
        ((-0.2, 0.8, 1.8, 2.8), (0.2, 1.2, 2.2, 3.2), (0.0, 1.0, 2.0, 3.0)),
        confidence=0.8,
        max_prediction_values=100,
    )
    evidence = (
        MLPMethodEvidence(MLPUncertaintyMethod.DEEP_ENSEMBLE, ensemble, 3, 3, 2.0, 100, "v1"),
        MLPMethodEvidence(MLPUncertaintyMethod.MC_DROPOUT, dropout, 1, 3, 1.0, 50, "v1"),
    )

    study = compare_mlp_uncertainty((0.0, 1.0, 2.8, 3.0), evidence, confidence=0.8)

    assert study is not None
    assert all(0.0 <= item.calibration_error <= 1.0 for item in study.candidates)
    assert all(item.active_selection_lift is not None for item in study.candidates)
    assert study.selected.evidence.predictions[0].schema_version == 1


def test_mlp_uncertainty_can_be_disabled_without_inspecting_evidence() -> None:
    disabled = replace(MLPUncertaintyBudget(), enabled=False)

    assert compare_mlp_uncertainty((), (), budget=disabled) is None


def test_mlp_prediction_budget_fails_closed() -> None:
    with pytest.raises(ValueError, match="memory budget"):
        estimate_mlp_uncertainty(((0.0, 1.0), (0.1, 1.1)), confidence=0.9, max_prediction_values=3)
