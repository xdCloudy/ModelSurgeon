from __future__ import annotations

import pytest

from modelsurgeon.evaluation.microbenchmarks import (
    DEFAULT_MICROBENCHMARK_PROTOCOL,
    MicrobenchmarkError,
    MicrobenchmarkKey,
    MicrobenchmarkObservation,
    MicrobenchmarkOutcome,
    MicrobenchmarkPartition,
    MicrobenchmarkResult,
    build_microbenchmark_result,
    render_microbenchmark_protocol,
)


def _key(shape_id: str) -> MicrobenchmarkKey:
    return MicrobenchmarkKey(
        "hardware_profile_fixture",
        "hwctx_fixture",
        shape_id,
        "llama.cpp",
        "b123",
        "tool-v1",
        11,
    )


def _observations(
    *, drift: bool = False, unstable: bool = False
) -> tuple[MicrobenchmarkObservation, ...]:
    names = (
        "prefill_latency_seconds",
        "decode_latency_seconds",
        "throughput_tokens_per_second",
        "bandwidth_gbps",
        "peak_ram_bytes",
        "peak_vram_bytes",
    )
    observations = []
    for repetition in range(10):
        factor = 20.0 if unstable and repetition == 9 else 1.0
        observations.append(
            MicrobenchmarkObservation(
                repetition,
                {name: (10.0 + (repetition % 2)) * factor for name in names},
                "hwctx_fixture",
                "drifted" if drift and repetition == 9 else "stable",
            )
        )
    return tuple(observations)


def test_protocol_crosses_required_alignment_and_offload_boundaries() -> None:
    tags = {tag for shape in DEFAULT_MICROBENCHMARK_PROTOCOL.shapes for tag in shape.boundary_tags}

    assert {"tensor_core_16", "simd_8", "cache_line_64"} <= tags
    assert {"gguf_block_256", "gpu_layers_0", "gpu_layers_32"} <= tags
    assert DEFAULT_MICROBENCHMARK_PROTOCOL.repetitions == 10


def test_measured_result_has_confidence_bounds_and_content_identity() -> None:
    shape = DEFAULT_MICROBENCHMARK_PROTOCOL.shapes[0]
    result = build_microbenchmark_result(_key(shape.shape_id), shape, _observations())

    assert result.outcome is MicrobenchmarkOutcome.MEASURED
    assert result.result_id.startswith("micro_result_")
    assert all(metric.confidence_low is not None for metric in result.metrics)
    assert all(metric.repetitions == 10 for metric in result.metrics)


def test_high_variance_result_is_retained_but_marked_unstable() -> None:
    shape = DEFAULT_MICROBENCHMARK_PROTOCOL.shapes[0]
    result = build_microbenchmark_result(_key(shape.shape_id), shape, _observations(unstable=True))

    assert result.outcome is MicrobenchmarkOutcome.UNSTABLE
    assert result.reason is not None
    assert len(result.observations) == 10


def test_result_rejects_environment_drift() -> None:
    shape = DEFAULT_MICROBENCHMARK_PROTOCOL.shapes[0]

    with pytest.raises(MicrobenchmarkError, match="drifting"):
        build_microbenchmark_result(_key(shape.shape_id), shape, _observations(drift=True))


def test_partition_requires_every_shape_and_reports_negative_outcomes() -> None:
    results = tuple(
        # Unsupported cells are retained without pretending to have timings.
        MicrobenchmarkResult(
            _key(shape.shape_id),
            shape,
            MicrobenchmarkOutcome.UNSUPPORTED,
            reason="llama.cpp runtime is unavailable on this target",
        )
        for shape in DEFAULT_MICROBENCHMARK_PROTOCOL.shapes
    )
    partition = MicrobenchmarkPartition(
        DEFAULT_MICROBENCHMARK_PROTOCOL,
        "hardware_profile_fixture",
        "hwctx_fixture",
        "llama.cpp",
        "b123",
        "tool-v1",
        11,
        results,
    )

    report = partition.validity_report()
    assert report["outcomes"][MicrobenchmarkOutcome.UNSUPPORTED.value] == 5  # type: ignore[index]
    assert report["eligible_result_ids"] == []
    assert partition.partition_id.startswith("micro_partition_")
    assert render_microbenchmark_protocol() == render_microbenchmark_protocol()


def test_result_requires_ten_repetitions() -> None:
    shape = DEFAULT_MICROBENCHMARK_PROTOCOL.shapes[0]
    with pytest.raises(MicrobenchmarkError, match="ten"):
        build_microbenchmark_result(_key(shape.shape_id), shape, _observations()[:9])
