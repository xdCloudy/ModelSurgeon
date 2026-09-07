from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.datasets import (
    HardwareCostDataset,
    HardwareCostDatasetError,
    HardwareCostExample,
    HardwareCostOutcome,
    HardwareCostProvenance,
    HardwareCostSplit,
    build_hardware_cost_example,
)
from modelsurgeon.evaluation.deployment_benchmark import (
    DEPLOYMENT_METRICS,
    DeploymentArtifact,
    DeploymentBenchmarkRecord,
    DeploymentFormat,
    DeploymentOutcome,
    DeploymentRuntimeProfile,
    summarize_samples,
)
from modelsurgeon.surgery import PhysicalArtifactOutcome, build_example_physical_outcomes


def _deployment(outcome: PhysicalArtifactOutcome) -> DeploymentBenchmarkRecord:
    artifact = DeploymentArtifact(
        "fixture-output",
        DeploymentFormat.HF,
        outcome.artifact.digest or "0" * 64,
        outcome.artifact.size_bytes or 1,
    )
    metrics = tuple(summarize_samples(item.name, (10,) * 7) for item in DEPLOYMENT_METRICS)
    return DeploymentBenchmarkRecord(
        artifact,
        DeploymentRuntimeProfile("fixture-runtime", hardware="fixture-hardware"),
        DeploymentOutcome.MEASURED,
        metrics,
        command=("fixture-runner", "--profile", "cpu"),
    )


def _measured_example() -> HardwareCostExample:
    hf, _ = build_example_physical_outcomes()
    deployment = _deployment(hf)
    provenance = HardwareCostProvenance(
        deployment.record_id,
        "deployment-protocol-v1",
        "runner-v1",
        deployment.command,
        {"warmups": 2, "repetitions": 7},
    )
    return build_hardware_cost_example(
        model_family="llama",
        model_revision="checkpoint-v1",
        architecture_state_id="architecture-v1",
        source_artifact_digest="a" * 64,
        physical_outcome=hf,
        deployment=deployment,
        hardware_profile_id="hardware_profile_cpu",
        hardware_context_id="hwctx_cpu",
        runtime_id="fixture-runtime",
        runtime_revision="runtime-v1",
        runtime_settings={"threads": 8, "gpu_layers": None},
        seed=11,
        provenance=provenance,
    )


def _unsupported(group: str, profile: str) -> HardwareCostExample:
    return HardwareCostExample(
        "qwen",
        "checkpoint-v2",
        "architecture-v2",
        "b" * 64,
        None,
        None,
        profile,
        f"hwctx-{profile}",
        "fixture-runtime",
        "runtime-v1",
        {"threads": 8},
        23,
        HardwareCostOutcome.UNSUPPORTED,
        (),
        None,
        group,
        "runtime unavailable on the target profile",
    )


def test_measured_example_retains_repeated_distributions_and_provenance() -> None:
    example = _measured_example()

    assert example.example_id.startswith("hardware_cost_")
    assert len(example.metrics) == len(DEPLOYMENT_METRICS)
    assert all(len(metric.samples) == 7 for metric in example.metrics)
    assert example.provenance is not None
    assert example.provenance.configuration["repetitions"] == 7


def test_builder_rejects_artifact_checksum_mismatch() -> None:
    hf, _ = build_example_physical_outcomes()
    deployment = _deployment(hf)
    provenance = HardwareCostProvenance(
        deployment.record_id,
        "deployment-protocol-v1",
        "runner-v1",
        deployment.command,
        {},
    )
    with pytest.raises(HardwareCostDatasetError, match="digests"):
        build_hardware_cost_example(
            model_family="llama",
            model_revision="checkpoint-v1",
            architecture_state_id="architecture-v1",
            source_artifact_digest="c" * 64,
            physical_outcome=replace(hf, artifact=replace(hf.artifact, digest="c" * 64)),
            deployment=deployment,
            hardware_profile_id="hardware_profile_cpu",
            hardware_context_id="hwctx_cpu",
            runtime_id="fixture-runtime",
            runtime_revision="runtime-v1",
            runtime_settings={},
            seed=11,
            provenance=provenance,
        )


def test_group_split_rejects_duplicate_lineage() -> None:
    with pytest.raises(HardwareCostDatasetError, match="leak"):
        HardwareCostSplit(("lineage-a",), ("lineage-a",), ("lineage-b",))


def test_dataset_retains_unsupported_cells_and_partitions_by_profile() -> None:
    measured = _measured_example()
    validation = _unsupported("lineage-validation", "hardware_profile_laptop")
    test = _unsupported("lineage-test", "hardware_profile_workstation")
    dataset = HardwareCostDataset(
        (measured, validation, test),
        HardwareCostSplit(
            (measured.lineage_group_id,),
            (validation.lineage_group_id,),
            (test.lineage_group_id,),
        ),
        "hardware-cost-v1",
    )

    assert dataset.dataset_id.startswith("hardware_cost_dataset_")
    assert len(dataset.partition_by_profile()) == 3
    assert dataset.to_record()["coverage"]["outcomes"][HardwareCostOutcome.UNSUPPORTED.value] == 2  # type: ignore[index]
