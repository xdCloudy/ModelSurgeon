"""Contract tests for the physical HF/GGUF deployment benchmark CLI."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.cli.deployment import (
    DeploymentResult,
    DeploymentState,
    UnsupportedDeploymentRunner,
    audit_deployment_plan,
    build_deployment_plan,
    run_deployment_plan,
)
from modelsurgeon.evaluation.deployment_benchmark import (
    DEPLOYMENT_METRICS,
    DeploymentArtifact,
    DeploymentBenchmarkError,
    DeploymentBenchmarkRecord,
    DeploymentFormat,
    DeploymentMeasurementPolicy,
    DeploymentOutcome,
    DeploymentRuntimeProfile,
    record_from_mapping,
    summarize_samples,
)


def _artifact(tmp_path: Path, name: str, format: DeploymentFormat) -> DeploymentArtifact:
    path = tmp_path / name
    path.write_bytes(b"deployment-fixture-" + name.encode())
    return DeploymentArtifact.from_path(path, format)


def _profile() -> DeploymentRuntimeProfile:
    return DeploymentRuntimeProfile(
        "test-runtime",
        hardware="test-hardware",
        policy=DeploymentMeasurementPolicy(),
    )


def test_summary_uses_deterministic_rank_p95_and_dispersion() -> None:
    summary = summarize_samples("load_time_seconds", (1, 2, 3, 4, 5, 6, 7))

    assert summary.median == 4
    assert summary.p95 == 7
    assert summary.dispersion == pytest.approx(2.0)
    assert summary.unit == "s"


@pytest.mark.parametrize(
    "kwargs",
    [{"warmups": 1}, {"repetitions": 6}],
)
def test_policy_rejects_unreproducible_repetition_counts(kwargs: dict[str, int]) -> None:
    with pytest.raises(DeploymentBenchmarkError):
        DeploymentMeasurementPolicy(**kwargs)


def test_measured_record_round_trips_with_all_shared_metrics(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, "model.safetensors", DeploymentFormat.HF)
    metrics = tuple(summarize_samples(item.name, range(1, 8)) for item in DEPLOYMENT_METRICS)
    record = DeploymentBenchmarkRecord(artifact, _profile(), DeploymentOutcome.MEASURED, metrics)

    restored = record_from_mapping(record.to_record())

    assert restored.record_id == record.record_id
    assert restored.artifact.digest == artifact.digest
    assert {item.name for item in restored.metrics} == {item.name for item in DEPLOYMENT_METRICS}


@pytest.mark.parametrize(
    "outcome",
    [
        DeploymentOutcome.UNSUPPORTED,
        DeploymentOutcome.FAILED,
        DeploymentOutcome.TIMEOUT,
        DeploymentOutcome.OOM,
        DeploymentOutcome.DRIFT,
        DeploymentOutcome.INVALID_ARTIFACT,
        DeploymentOutcome.INTERRUPTED,
    ],
)
def test_terminal_outcomes_remain_distinct(tmp_path: Path, outcome: DeploymentOutcome) -> None:
    record = DeploymentBenchmarkRecord(
        _artifact(tmp_path, f"{outcome.value}.gguf", DeploymentFormat.GGUF),
        _profile(),
        outcome,
        reason="retained for audit",
    )

    assert record_from_mapping(record.to_record()).outcome is outcome


def test_deployment_resume_preserves_completed_cells(tmp_path: Path) -> None:
    plan = build_deployment_plan(
        (_artifact(tmp_path, "model.safetensors", DeploymentFormat.HF),
         _artifact(tmp_path, "model.gguf", DeploymentFormat.GGUF)),
        _profile(),
    )
    first_cell = plan.cells[0]
    first_record = record_from_mapping(
        UnsupportedDeploymentRunner().execute(first_cell)
    )
    state = DeploymentState(
        plan.plan_id,
        (DeploymentResult(str(first_cell["cell_id"]), first_record),),
    )

    resumed = run_deployment_plan(plan, state, UnsupportedDeploymentRunner())
    audit = audit_deployment_plan(plan, resumed)

    assert len(resumed.results) == 2
    assert resumed.results[0].record.record_id == first_record.record_id
    assert audit["missing_cells"] == []
    assert audit["outcomes"][DeploymentOutcome.UNSUPPORTED.value] == 2
