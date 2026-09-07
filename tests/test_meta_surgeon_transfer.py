from __future__ import annotations

import pytest

from modelsurgeon.evaluation import (
    REQUIRED_TRANSFER_METRICS,
    MetaSurgeonTransferError,
    TransferAxis,
    TransferControl,
    TransferFeatureView,
    TransferOutcomeStatus,
    TransferStudyMetric,
    TransferStudyModel,
    build_meta_surgeon_transfer_matrix,
    record_meta_surgeon_transfer_result,
)


def _models() -> tuple[TransferStudyModel, ...]:
    return tuple(
        TransferStudyModel(
            f"{family}-{size}",
            f"org/{family}-{size}",
            f"rev-{family}-{size}",
            family,
            size,
            "corpus-v1",
            "cuda-a100",
            512,
        )
        for family in ("llama", "qwen", "mistral", "gemma")
        for size in ("small", "large")
    )


def _metrics(value: float = 0.5) -> tuple[TransferStudyMetric, ...]:
    return tuple(
        TransferStudyMetric(name, value, value - 0.1, value + 0.1, 100, "score")
        for name in sorted(REQUIRED_TRANSFER_METRICS)
    )


def test_dense_matrix_covers_families_and_is_source_target_disjoint() -> None:
    report = build_meta_surgeon_transfer_matrix(_models())
    assert not report.complete
    assert {cell.target_family for cell in report.cells} == {
        "llama",
        "qwen",
        "mistral",
        "gemma",
    }
    assert {cell.feature_view for cell in report.cells} == set(TransferFeatureView)
    for cell in report.cells:
        assert cell.target_model_id not in cell.source_model_ids
        if cell.control is TransferControl.META:
            assert cell.axis is TransferAxis.FAMILY
            assert all(
                next(model for model in _models() if model.model_id == source).family
                != cell.target_family
                for source in cell.source_model_ids
            )


def test_transfer_results_retain_negative_and_unsupported_cells() -> None:
    report = build_meta_surgeon_transfer_matrix(_models())
    measured = record_meta_surgeon_transfer_result(
        report,
        report.cells[0].cell_id,
        status=TransferOutcomeStatus.NEGATIVE_RESULT,
        metrics=_metrics(-0.2),
        evaluations=12,
        provenance={"seed_set": [0, 1, 2]},
    )
    assert measured.cells[0].status is TransferOutcomeStatus.NEGATIVE_RESULT
    unavailable = tuple(
        TransferStudyMetric(name, reason="unsupported target architecture")
        for name in sorted(REQUIRED_TRANSFER_METRICS)
    )
    recorded = record_meta_surgeon_transfer_result(
        measured,
        measured.cells[1].cell_id,
        status=TransferOutcomeStatus.UNSUPPORTED,
        metrics=unavailable,
        evaluations=0,
        failures=("architecture-normalized feature is unsupported",),
    )
    assert recorded.cells[1].status is TransferOutcomeStatus.UNSUPPORTED
    assert not recorded.complete


def test_transfer_result_rejects_incomplete_success() -> None:
    report = build_meta_surgeon_transfer_matrix(_models())
    with pytest.raises(MetaSurgeonTransferError, match="five metrics"):
        record_meta_surgeon_transfer_result(
            report,
            report.cells[0].cell_id,
            status=TransferOutcomeStatus.MEASURED,
            metrics=_metrics()[:-1],
            evaluations=1,
        )
