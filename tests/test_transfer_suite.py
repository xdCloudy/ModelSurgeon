from __future__ import annotations

import pytest

from modelsurgeon.evaluation import (
    TransferResultStatus,
    TransferSample,
    TransferSuiteError,
    build_transfer_suite,
    record_transfer_fold_result,
)


def _samples() -> tuple[TransferSample, ...]:
    output: list[TransferSample] = []
    for family in ("llama", "qwen", "mistral"):
        for size in ("small", "large"):
            checkpoint = f"{family}-{size}"
            output.append(
                TransferSample(
                    f"sample-{family}-{size}",
                    checkpoint,
                    "rev-1",
                    family,
                    size,
                    f"lineage-{family}-{size}",
                )
            )
            output.append(
                TransferSample(
                    f"sample-{family}-{size}-second",
                    checkpoint,
                    "rev-1",
                    family,
                    size,
                    f"lineage-{family}-{size}-second",
                )
            )
            output.append(
                TransferSample(
                    f"sample-{family}-{size}-third",
                    checkpoint,
                    "rev-1",
                    family,
                    size,
                    f"lineage-{family}-{size}-third",
                )
            )
    return tuple(output)


def test_transfer_suite_is_deterministic_and_source_target_disjoint() -> None:
    first = build_transfer_suite(_samples())
    second = build_transfer_suite(_samples())
    assert first.to_record() == second.to_record()
    assert {fold.kind.value for fold in first.folds} == {
        "leave_one_checkpoint",
        "leave_one_size",
        "leave_one_family",
        "zero_target",
        "few_shot",
    }
    for fold in first.folds:
        assert not set(fold.source_sample_ids) & set(fold.target_test_ids)
        assert not set(fold.source_lineage_groups) & set(fold.target_lineage_groups)


def test_transfer_results_require_complete_metrics_and_failures() -> None:
    report = build_transfer_suite(_samples())
    metrics = {
        name: (0.5, 0.4, 0.6, "measured")
        for name in ("ranking", "calibration", "numeric_error", "frontier_quality")
    }
    result = record_transfer_fold_result(
        report,
        report.folds[0].fold_id,
        status=TransferResultStatus.SUCCEEDED,
        metrics=metrics,
        evaluations=3,
    )
    assert len(result.results) == 1
    with pytest.raises(TransferSuiteError, match="complete"):
        record_transfer_fold_result(
            report,
            report.folds[1].fold_id,
            status=TransferResultStatus.FAILED,
            metrics={"ranking": (None, None, None, "failed")},
            evaluations=0,
        )
