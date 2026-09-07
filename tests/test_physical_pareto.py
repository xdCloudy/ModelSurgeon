from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.evaluation import (
    DEFAULT_PHYSICAL_PARETO_STUDY,
    PhysicalParetoError,
    PhysicalParetoKey,
    PhysicalParetoOutcome,
    build_physical_pareto_cell,
    build_physical_pareto_study,
    compute_physical_frontier,
    render_physical_pareto_study,
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
from modelsurgeon.surgery import (
    ArtifactFormat,
    PhysicalArtifactOutcome,
    build_example_physical_outcomes,
)


def _deployment(
    outcome: PhysicalArtifactOutcome,
    tmp_path: Path,
    format: DeploymentFormat,
) -> DeploymentBenchmarkRecord:
    artifact = DeploymentArtifact(
        "fixture-output",
        format,
        outcome.artifact.digest or "0" * 64,
        outcome.artifact.size_bytes or 1,
    )
    metrics = tuple(summarize_samples(item.name, (10,) * 7) for item in DEPLOYMENT_METRICS)
    return DeploymentBenchmarkRecord(
        artifact,
        DeploymentRuntimeProfile("fixture-runtime", hardware="fixture-hardware"),
        DeploymentOutcome.MEASURED,
        metrics,
    )


def _key(family: str, format: ArtifactFormat, target: str, seed: int = 11) -> PhysicalParetoKey:
    return PhysicalParetoKey(
        "a" * 64 if family == "llama" else "b" * 64,
        family,
        format,
        "modelsurgeon",
        target,
        0.03,
        "heldout-corpus-v1",
        "quality-evaluator-v1",
        "fixture-runtime-v1",
        f"{family}-revision-v1",
        "tool-revision-v1",
        seed,
    )


def test_default_protocol_retains_every_negative_cell_without_frontier_claim() -> None:
    study = DEFAULT_PHYSICAL_PARETO_STUDY

    assert len(study.cells) == 2 * 5 * 3 * 3 * 5
    assert {cell.outcome for cell in study.cells} == {PhysicalParetoOutcome.UNSUPPORTED}
    assert study.frontier_cell_ids == ()
    assert study.superiority_claim is None
    assert render_physical_pareto_study(study) == render_physical_pareto_study(study)


def test_measured_cell_reconciles_physical_and_deployment_lineage(tmp_path: Path) -> None:
    hf, _ = build_example_physical_outcomes()
    cell = build_physical_pareto_cell(
        _key("llama", ArtifactFormat.HUGGINGFACE, "gguf_q4_k"),
        hf,
        _deployment(hf, tmp_path, DeploymentFormat.HF),
        source_artifact_size_bytes=5000,
        quality_loss=0.02,
        quality_loss_low=0.01,
        quality_loss_high=0.03,
        quality_repetitions=100,
        optimization_seconds=12.0,
    )

    assert cell.outcome is PhysicalParetoOutcome.MEASURED
    assert cell.cell_id.startswith("physical_pareto_")
    assert {metric.name for metric in cell.metrics} == {
        "source_artifact_size_bytes",
        "artifact_size_bytes",
        "active_parameters",
        "quality_loss",
        "load_time_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "latency_seconds",
        "peak_ram_bytes",
        "peak_vram_bytes",
        "disk_bytes",
        "optimization_seconds",
    }


def test_measured_cell_rejects_artifact_identity_mismatch(tmp_path: Path) -> None:
    hf, _ = build_example_physical_outcomes()
    deployment = _deployment(hf, tmp_path, DeploymentFormat.HF)
    with pytest.raises(PhysicalParetoError, match="digests"):
        build_physical_pareto_cell(
            _key("llama", ArtifactFormat.HUGGINGFACE, "gguf_q4_k"),
            replace(hf, artifact=replace(hf.artifact, digest="c" * 64)),
            deployment,
            source_artifact_size_bytes=5000,
            quality_loss=0.02,
            quality_loss_low=0.01,
            quality_loss_high=0.03,
            quality_repetitions=100,
            optimization_seconds=12.0,
        )


def test_frontier_ignores_unsupported_and_requires_interval_separation(tmp_path: Path) -> None:
    hf, gguf = build_example_physical_outcomes()
    first = build_physical_pareto_cell(
        _key("llama", ArtifactFormat.HUGGINGFACE, "gguf_q4_k"),
        hf,
        _deployment(hf, tmp_path, DeploymentFormat.HF),
        source_artifact_size_bytes=5000,
        quality_loss=0.01,
        quality_loss_low=0.01,
        quality_loss_high=0.01,
        quality_repetitions=100,
        optimization_seconds=10.0,
    )
    second = build_physical_pareto_cell(
        _key("llama", ArtifactFormat.GGUF, "gguf_q6_k", seed=23),
        gguf,
        _deployment(gguf, tmp_path, DeploymentFormat.GGUF),
        source_artifact_size_bytes=6000,
        quality_loss=0.20,
        quality_loss_low=0.20,
        quality_loss_high=0.20,
        quality_repetitions=100,
        optimization_seconds=20.0,
    )

    assert compute_physical_frontier((first, second)) == (first.cell_id,)
    unsupported = replace(
        DEFAULT_PHYSICAL_PARETO_STUDY.cells[0],
        key=first.key,
        reason="runner unavailable",
    )
    assert unsupported.outcome is PhysicalParetoOutcome.UNSUPPORTED


def test_study_recomputes_frontier_from_measured_cells(tmp_path: Path) -> None:
    hf, _ = build_example_physical_outcomes()
    measured = build_physical_pareto_cell(
        DEFAULT_PHYSICAL_PARETO_STUDY.cells[0].key,
        hf,
        _deployment(hf, tmp_path, DeploymentFormat.HF),
        source_artifact_size_bytes=5000,
        quality_loss=0.01,
        quality_loss_low=0.01,
        quality_loss_high=0.01,
        quality_repetitions=100,
        optimization_seconds=12.0,
    )
    cells = tuple(
        measured if cell.cell_id == measured.cell_id else cell
        for cell in DEFAULT_PHYSICAL_PARETO_STUDY.cells
    )
    study = build_physical_pareto_study(
        families=DEFAULT_PHYSICAL_PARETO_STUDY.families,
        methods=DEFAULT_PHYSICAL_PARETO_STUDY.methods,
        compression_targets=DEFAULT_PHYSICAL_PARETO_STUDY.compression_targets,
        quality_loss_limits=DEFAULT_PHYSICAL_PARETO_STUDY.quality_loss_limits,
        seeds=DEFAULT_PHYSICAL_PARETO_STUDY.seeds,
        protocol_revision=DEFAULT_PHYSICAL_PARETO_STUDY.protocol_revision,
        cells=cells,
    )

    assert study.frontier_cell_ids == (measured.cell_id,)
