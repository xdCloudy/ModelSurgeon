import hashlib
from pathlib import Path

import pytest

from modelsurgeon.search.deployable_state import ArtifactContainerFormat
from modelsurgeon.search.heterogeneous import (
    BenchmarkObservation,
    HeterogeneousArchitectureSpec,
    HeterogeneousCapabilities,
    HeterogeneousLayer,
    HeterogeneousOutcome,
    HeterogeneousStateError,
    MaterializationEvidenceStatus,
    ReloadEvidence,
    compile_heterogeneous_state,
    materialize_heterogeneous_state,
)


def _spec(source: Path) -> HeterogeneousArchitectureSpec:
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return HeterogeneousArchitectureSpec(
        "tiny-transformer",
        "revision-1",
        source_digest,
        ArtifactContainerFormat.HUGGINGFACE,
        "tiny-runtime",
        (
            HeterogeneousLayer(
                0,
                8,
                32,
                low_rank_factors=(("attention.output", 2),),
                quantization_codec="q4",
            ),
            HeterogeneousLayer(
                1,
                4,
                16,
                sparsity=(("mlp.down", 0.25),),
                quantization_codec="q8",
            ),
        ),
        10_000,
        20_000,
    )


def _capabilities(*, mixed: bool = True) -> HeterogeneousCapabilities:
    return HeterogeneousCapabilities(
        ArtifactContainerFormat.HUGGINGFACE,
        "tiny-runtime",
        ("q4", "q8"),
        supports_low_rank=True,
        supports_sparsity=True,
        supports_mixed_quantization=mixed,
        evidence_source="tiny-fixture-capability-v1",
    )


def test_compiler_reconciles_heterogeneous_axes_and_rejects_unsupported_cells(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable source")
    spec = _spec(source)
    compiled = compile_heterogeneous_state(spec, _capabilities())
    assert compiled.outcome is HeterogeneousOutcome.ACCEPTED
    assert compiled.state is not None
    assert compiled.state.parameter_count == sum(layer.parameter_count for layer in spec.layers)
    assert compiled.state.storage_bytes == sum(layer.storage_bytes for layer in spec.layers)
    assert (
        compiled.state.to_record()["requested_layers"]
        == compiled.state.to_record()["effective_layers"]
    )

    unsupported = compile_heterogeneous_state(spec, _capabilities(mixed=False))
    assert unsupported.outcome is HeterogeneousOutcome.UNSUPPORTED
    assert unsupported.state is None
    assert any("mixed quantization" in reason for reason in unsupported.reasons)


def test_materialization_reload_benchmark_and_source_immutability(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable source")
    output = tmp_path / "heterogeneous.bin"
    compiled = compile_heterogeneous_state(_spec(source), _capabilities()).state
    assert compiled is not None
    source_before = source.read_bytes()

    def writer(_: Path, staging: Path, state: object) -> None:
        assert state is compiled
        staging.write_bytes(b"x" * compiled.storage_bytes)

    def reloader(_: Path) -> ReloadEvidence:
        return ReloadEvidence(
            True,
            "tiny-runtime",
            compiled.effective_layers,
            compiled.parameter_count,
            compiled.storage_bytes,
            "reloaded tiny heterogeneous fixture",
        )

    def benchmark(_: Path) -> tuple[BenchmarkObservation, ...]:
        return (BenchmarkObservation("latency", 1.5, "ms", 3),)

    evidence = materialize_heterogeneous_state(
        compiled,
        source,
        output,
        writer,
        reloader,
        benchmark,
    )
    assert evidence.status is MaterializationEvidenceStatus.COMPLETE
    assert output.exists()
    assert source.read_bytes() == source_before
    assert evidence.reload is not None
    assert evidence.benchmarks[0].repetitions == 3


def test_materialization_rolls_back_on_reconciliation_failure(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"immutable source")
    output = tmp_path / "heterogeneous.bin"
    compiled = compile_heterogeneous_state(_spec(source), _capabilities()).state
    assert compiled is not None

    def writer(_: Path, staging: Path, __: object) -> None:
        staging.write_bytes(b"candidate")

    def bad_reload(_: Path) -> ReloadEvidence:
        return ReloadEvidence(
            True,
            "tiny-runtime",
            compiled.effective_layers,
            compiled.parameter_count + 1,
            compiled.storage_bytes,
            "parameter count mismatch",
        )

    evidence = materialize_heterogeneous_state(
        compiled,
        source,
        output,
        writer,
        bad_reload,
        lambda _: (),
    )
    assert evidence.status is MaterializationEvidenceStatus.UNKNOWN
    assert not output.exists()
    assert not list(tmp_path.glob("*.staging"))


def test_invalid_layer_and_same_path_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(HeterogeneousStateError, match="contiguous"):
        HeterogeneousArchitectureSpec(
            "family",
            "revision",
            "0" * 64,
            ArtifactContainerFormat.GGUF,
            "runtime",
            (HeterogeneousLayer(1, 4, 8),),
            1,
            1,
        )
    source = tmp_path / "source.bin"
    source.write_bytes(b"source")
    compiled = compile_heterogeneous_state(_spec(source), _capabilities()).state
    assert compiled is not None
    evidence = materialize_heterogeneous_state(
        compiled,
        source,
        source,
        lambda *_: None,
        lambda _: ReloadEvidence(False, "runtime", compiled.effective_layers, 1, 1, "unused"),
        lambda _: (),
    )
    assert evidence.status is MaterializationEvidenceStatus.FAILED
