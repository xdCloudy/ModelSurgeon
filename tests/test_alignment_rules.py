from __future__ import annotations

import pytest

from modelsurgeon.evaluation.microbenchmarks import (
    DEFAULT_MICROBENCHMARK_PROTOCOL,
    MicrobenchmarkKey,
    MicrobenchmarkObservation,
    MicrobenchmarkOutcome,
    MicrobenchmarkPartition,
    MicrobenchmarkResult,
    build_microbenchmark_result,
)
from modelsurgeon.surgery import (
    AlignmentAxis,
    AlignmentConstraint,
    AlignmentEvidence,
    AlignmentRuleError,
    AlignmentRuleOutcome,
    build_alignment_rule_set,
    build_unknown_preference,
)
from modelsurgeon.surgery.alignment_rules import AlignmentPreference


def _observations() -> tuple[MicrobenchmarkObservation, ...]:
    names = (
        "prefill_latency_seconds",
        "decode_latency_seconds",
        "throughput_tokens_per_second",
        "bandwidth_gbps",
        "peak_ram_bytes",
        "peak_vram_bytes",
    )
    return tuple(
        MicrobenchmarkObservation(
            index,
            {name: 10.0 + (index % 2) for name in names},
            "hwctx_fixture",
            "stable",
        )
        for index in range(10)
    )


def _partition() -> tuple[MicrobenchmarkPartition, MicrobenchmarkResult]:
    results = []
    measured: MicrobenchmarkResult | None = None
    for index, shape in enumerate(DEFAULT_MICROBENCHMARK_PROTOCOL.shapes):
        key = MicrobenchmarkKey(
            "hardware_profile_fixture",
            "hwctx_fixture",
            shape.shape_id,
            "llama.cpp",
            "revision-v1",
            "tool-v1",
            11,
        )
        if index == 0:
            measured = build_microbenchmark_result(key, shape, _observations())
            results.append(measured)
        else:
            results.append(
                MicrobenchmarkResult(
                    key,
                    shape,
                    MicrobenchmarkOutcome.UNSUPPORTED,
                    reason="fixture runtime does not expose this shape",
                )
            )
    partition = MicrobenchmarkPartition(
        DEFAULT_MICROBENCHMARK_PROTOCOL,
        "hardware_profile_fixture",
        "hwctx_fixture",
        "llama.cpp",
        "revision-v1",
        "tool-v1",
        11,
        tuple(results),
    )
    assert measured is not None
    return partition, measured


def test_hard_legality_intersects_graph_and_codec_multiples() -> None:
    constraint = AlignmentConstraint(
        AlignmentAxis.MLP_WIDTH,
        "GGUF",
        "Q4_K_M",
        graph_multiple=8,
        codec_multiple=256,
        source_revision="codec-contract-v1",
    )

    assert constraint.legal_multiple == 256
    assert constraint.is_legal(256) is True
    assert constraint.is_legal(128) is False


def test_measured_preference_requires_training_and_heldout_evidence() -> None:
    partition, measured = _partition()
    evidence = AlignmentEvidence(
        partition.partition_id,
        measured.result_id,
        partition.profile_id,
        measured.key.shape_id,
        "training",
        0.04,
        0.12,
    )
    validation = AlignmentEvidence(
        partition.partition_id,
        measured.result_id,
        partition.profile_id,
        measured.key.shape_id,
        "validation",
        0.02,
        0.08,
    )
    preference = AlignmentPreference(
        AlignmentAxis.MLP_WIDTH,
        "GGUF",
        "Q4_K_M",
        partition.profile_id,
        256,
        AlignmentRuleOutcome.MEASURED,
        (evidence,),
        (validation,),
        false_preference_rate=0.0,
    )
    rules = build_alignment_rule_set(
        partition,
        constraints=(
            AlignmentConstraint(
                AlignmentAxis.MLP_WIDTH,
                "GGUF",
                "Q4_K_M",
                8,
                256,
                "codec-contract-v1",
            ),
        ),
        preferences=(preference,),
    )

    assert rules.decide(AlignmentAxis.MLP_WIDTH, "GGUF", "Q4_K_M", 256).legal is True
    assert rules.decide(AlignmentAxis.MLP_WIDTH, "GGUF", "Q4_K_M", 256).preferred_granularity == 256
    assert rules.decide(AlignmentAxis.MLP_WIDTH, "GGUF", "Q4_K_M", 128).outcome.value == "illegal"
    assert (
        rules.decide(AlignmentAxis.HIDDEN_DIMENSION, "GGUF", "Q4_K_M", 256).outcome.value
        == "unknown"
    )


def test_unsupported_microbenchmark_cannot_create_preference() -> None:
    partition, _ = _partition()
    unsupported = partition.results[1]
    evidence = AlignmentEvidence(
        partition.partition_id,
        unsupported.result_id,
        partition.profile_id,
        unsupported.key.shape_id,
        "training",
        0.1,
        0.2,
    )
    preference = AlignmentPreference(
        AlignmentAxis.MLP_WIDTH,
        "GGUF",
        "Q4_K_M",
        partition.profile_id,
        256,
        AlignmentRuleOutcome.MEASURED,
        (evidence,),
        (evidence,),
        false_preference_rate=0.0,
    )

    with pytest.raises(AlignmentRuleError, match="measured result"):
        build_alignment_rule_set(partition, constraints=(), preferences=(preference,))


def test_unknown_preference_is_explicit_and_has_no_granularity() -> None:
    preference = build_unknown_preference(
        AlignmentAxis.LAYER_COUNT,
        "safetensors",
        None,
        "hardware_profile_fixture",
        "no held-out evidence covers layer-count kernels",
    )

    assert preference.outcome is AlignmentRuleOutcome.UNKNOWN
    assert preference.preferred_granularity is None
