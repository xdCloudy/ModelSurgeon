from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.evaluation import (
    V2_REQUIRED_METRICS,
    ArtifactAvailability,
    ArtifactLineageStage,
    ReferenceArtifact,
    ReferenceArtifactLineage,
    V2AuditRecord,
    V2BenchmarkBundle,
    V2BenchmarkCell,
    V2BenchmarkError,
    V2BenchmarkProtocol,
    V2Budget,
    V2CellOutcome,
    V2CompetitivenessClaim,
    V2DatasetSpec,
    V2HardwareProfile,
    V2MethodSpec,
    V2Metric,
    V2ModelSpec,
    V2PublicationOutcome,
    build_v2_benchmark_bundle,
    evaluate_v2_publication,
    render_v2_protocol,
)

DIGEST = "a" * 64


def _protocol() -> V2BenchmarkProtocol:
    return V2BenchmarkProtocol(
        revision="v2.0-rc1",
        models=(
            V2ModelSpec("llama", "small", "org/llama-small", "model-1", "Apache-2.0"),
            V2ModelSpec("llama", "medium", "org/llama-medium", "model-1", "Apache-2.0"),
            V2ModelSpec("qwen", "small", "org/qwen-small", "model-1", "Apache-2.0"),
            V2ModelSpec("qwen", "medium", "org/qwen-medium", "model-1", "Apache-2.0"),
        ),
        datasets=(V2DatasetSpec("heldout", "data-1", "language-modeling", "test", "CC-BY-4.0"),),
        methods=(
            V2MethodSpec("autonomous-v2", "tool-1", "autonomous"),
            V2MethodSpec("magnitude-v1.1", "tool-1", "baseline"),
        ),
        hardware=(V2HardwareProfile("cpu-12gb", "x86_64", "none", "runtime-1", 12),),
        budget=V2Budget(60, 2048, 12, 2_000_000_000),
        seeds=(0, 1, 2),
    )


def _lineage(*, permitted: bool = True) -> ReferenceArtifactLineage:
    stages = tuple(
        ArtifactLineageStage(stage, f"{stage}-1", DIGEST, DIGEST, DIGEST)
        for stage in ("mutation", "repair", "quantization", "runtime", "rollback")
    )
    return ReferenceArtifactLineage(stages, "Apache-2.0", permitted)


def _artifact(*, available: bool = True, permitted: bool = True) -> ReferenceArtifact:
    return ReferenceArtifact(
        "candidate",
        "artifact-1",
        "GGUF",
        ArtifactAvailability.AVAILABLE if available else ArtifactAvailability.NOT_AVAILABLE,
        DIGEST if available else None,
        100 if available else None,
        _lineage(permitted=permitted),
        None if available else "execution was not supported on this profile",
    )


def _metrics(quality: float) -> tuple[V2Metric, ...]:
    values = {
        "artifact_size_bytes": (100.0, "bytes"),
        "cost_units": (2.0, "cost_units"),
        "decode_tokens_per_second": (100.0, "tokens/second"),
        "optimization_seconds": (10.0, "seconds"),
        "prompt_tokens_per_second": (120.0, "tokens/second"),
        "quality": (quality, "score"),
        "ram_bytes": (1_000.0, "bytes"),
        "vram_bytes": (0.0, "bytes"),
    }
    return tuple(
        V2Metric(name, unit, value, value - 0.01, value + 0.01, 3)
        for name, (value, unit) in sorted(values.items())
    )


def _cell(
    protocol: V2BenchmarkProtocol, seed: int, method: V2MethodSpec, *, measured: bool
) -> V2BenchmarkCell:
    model = protocol.models[0]
    dataset = protocol.datasets[0]
    hardware = protocol.hardware[0]
    return V2BenchmarkCell(
        model,
        dataset,
        method,
        hardware,
        protocol.budget,
        seed,
        V2CellOutcome.MEASURED if measured else V2CellOutcome.NEGATIVE_RESULT,
        _artifact(available=measured),
        _metrics(0.95) if measured else (),
        None if measured else "executor was unsupported",
    )


def _complete_cells(protocol: V2BenchmarkProtocol) -> tuple[V2BenchmarkCell, ...]:
    cells: list[V2BenchmarkCell] = []
    for model in protocol.models:
        for dataset in protocol.datasets:
            for method in protocol.methods:
                for hardware in protocol.hardware:
                    for seed in protocol.seeds:
                        measured = method.kind == "autonomous" and seed == 0
                        cells.append(
                            V2BenchmarkCell(
                                model,
                                dataset,
                                method,
                                hardware,
                                protocol.budget,
                                seed,
                                V2CellOutcome.MEASURED
                                if measured
                                else V2CellOutcome.NEGATIVE_RESULT,
                                _artifact(available=measured),
                                _metrics(0.95) if measured else (),
                                None if measured else "negative or unsupported result retained",
                            )
                        )
    return tuple(cells)


def test_protocol_is_preregistered_and_rendering_does_not_claim_results() -> None:
    protocol = _protocol()

    assert protocol.expected_cell_count == 24
    assert protocol.protocol_id.startswith("v2_protocol_")
    assert "no live benchmark result" in render_v2_protocol(protocol, format="markdown")
    assert protocol.metrics == V2_REQUIRED_METRICS


def test_lineage_requires_all_mutation_and_rollback_stages() -> None:
    with pytest.raises(V2BenchmarkError, match="all ordered stages"):
        ReferenceArtifactLineage(
            (ArtifactLineageStage("mutation", "r", DIGEST, DIGEST, DIGEST),),
            "Apache-2.0",
            True,
        )

    with pytest.raises(V2BenchmarkError, match="license-prohibited"):
        _artifact(available=True, permitted=False)


def test_publication_withholds_incomplete_matrix_and_missing_audit() -> None:
    protocol = _protocol()
    cell = _cell(protocol, 0, protocol.methods[0], measured=True)
    decision = evaluate_v2_publication(protocol, (cell,), (), None)

    assert decision.outcome is V2PublicationOutcome.WITHHOLD
    assert any("incomplete matrix" in reason for reason in decision.reasons)
    assert "independent audit and replay are missing" in decision.reasons
    assert decision.decision_id.startswith("v2_publication_")


def test_competitiveness_claim_requires_confidence_bounded_deployable_cells() -> None:
    protocol = _protocol()
    candidate = _cell(protocol, 0, protocol.methods[0], measured=True)
    baseline = replace(candidate, method=protocol.methods[1], metrics=_metrics(0.95))
    claim = V2CompetitivenessClaim(
        "quality-improvement",
        "quality",
        (candidate.cell_id,),
        (baseline.cell_id,),
        "higher",
    )
    audit = V2AuditRecord("auditor-1", DIGEST, "clean", "b" * 64, True)
    cells = [
        baseline if item.cell_id == baseline.cell_id else item for item in _complete_cells(protocol)
    ]
    decision = evaluate_v2_publication(protocol, tuple(cells), (claim,), audit)

    assert decision.outcome is V2PublicationOutcome.WITHHOLD
    assert any("confidence intervals" in reason for reason in decision.reasons)


def test_complete_negative_cells_are_retained_and_claims_publish_only_when_replay_is_clean() -> (
    None
):
    protocol = _protocol()
    cells = list(_complete_cells(protocol))
    candidate = next(item for item in cells if item.method.kind == "autonomous" and item.seed == 0)
    baseline = replace(
        candidate,
        method=protocol.methods[1],
        metrics=_metrics(0.90),
    )
    cells = [baseline if item.cell_id == baseline.cell_id else item for item in cells]
    claim = V2CompetitivenessClaim(
        "quality-improvement",
        "quality",
        (candidate.cell_id,),
        (baseline.cell_id,),
        "higher",
    )
    audit = V2AuditRecord("auditor-1", DIGEST, "clean", "b" * 64, True)

    decision = evaluate_v2_publication(protocol, tuple(cells), (claim,), audit)

    assert decision.outcome is V2PublicationOutcome.PUBLISH
    assert decision.supported_claim_ids == ("quality-improvement",)
    assert any(item.outcome is V2CellOutcome.NEGATIVE_RESULT for item in cells)


def test_replay_failure_and_audit_findings_fail_closed() -> None:
    protocol = _protocol()
    audit = V2AuditRecord("auditor-1", DIGEST, "findings", "b" * 64, False, ("missing_cell",))
    decision = evaluate_v2_publication(protocol, _complete_cells(protocol), (), audit)

    assert decision.outcome is V2PublicationOutcome.WITHHOLD
    assert (
        "independent audit or replay has unresolved critical evidence defects" in decision.reasons
    )


def test_bundle_identity_is_canonical_and_recomputes_its_decision() -> None:
    protocol = _protocol()
    audit = V2AuditRecord("auditor-1", DIGEST, "clean", "b" * 64, True)
    cells = _complete_cells(protocol)

    bundle = build_v2_benchmark_bundle(protocol, cells[::-1], (), audit)

    assert isinstance(bundle, V2BenchmarkBundle)
    assert bundle.bundle_id.startswith("v2_bundle_")
    assert bundle.to_record()["decision"] == bundle.decision.to_record()
    assert bundle.cells == tuple(sorted(cells, key=lambda item: item.cell_id))
