"""Matched-arm quantization-order study tests."""

from __future__ import annotations

import pytest

from modelsurgeon.evaluation.quantization_order import (
    QuantizationOrderArm,
    QuantizationOrderCell,
    QuantizationOrderMetric,
    QuantizationOrderMetricState,
    QuantizationOrderOutcome,
    QuantizationOrderStudyKey,
    build_default_quantization_order_protocol,
    build_quantization_order_study,
)


def _key(structure_id: str = "structure-1") -> QuantizationOrderStudyKey:
    return QuantizationOrderStudyKey(
        "a" * 64,
        "llama",
        "10_percent",
        "Q4_K_M",
        "corpus-1",
        "evaluator-1",
        "runtime-1",
        0,
        structure_id,
    )


def _cell(
    arm: QuantizationOrderArm, quality: float, *, key: QuantizationOrderStudyKey | None = None
) -> QuantizationOrderCell:
    values = {
        "quality": quality,
        "artifact_size_bytes": 1000.0,
        "decode_tokens_per_second": 20.0,
        "peak_ram_bytes": 2000.0,
        "optimization_seconds": 3.0,
    }
    metrics = tuple(
        QuantizationOrderMetric(name, QuantizationOrderMetricState.MEASURED, values[name])
        for name in sorted(values)
    )
    return QuantizationOrderCell(
        key or _key(),
        arm,
        QuantizationOrderOutcome.MEASURED,
        "b" * 64,
        metrics,
        "tool-revision",
    )


def test_four_matched_arms_reconcile_effects_and_preference() -> None:
    cells = tuple(
        _cell(arm, quality)
        for arm, quality in (
            (QuantizationOrderArm.QUANTIZATION_ONLY, 0.80),
            (QuantizationOrderArm.SURGERY_THEN_QUANTIZE, 0.83),
            (QuantizationOrderArm.QUANTIZE_THEN_SURGERY, 0.81),
            (QuantizationOrderArm.HIGH_PRECISION_SURGERY, 0.90),
        )
    )
    comparison = build_quantization_order_study(cells).compare(_key())

    assert comparison.outcome is QuantizationOrderOutcome.MEASURED
    assert comparison.preferred_arm is QuantizationOrderArm.HIGH_PRECISION_SURGERY
    quality_delta = next(item for item in comparison.interaction_effect if item.metric == "quality")
    assert quality_delta.delta == pytest.approx(-0.02)
    assert all(
        item.delta == item.comparison - item.baseline for item in comparison.quantization_effect
    )


def test_unsupported_arm_is_retained_without_a_coerced_comparison() -> None:
    measured = [
        _cell(QuantizationOrderArm.QUANTIZATION_ONLY, 0.8),
        _cell(QuantizationOrderArm.SURGERY_THEN_QUANTIZE, 0.8),
        _cell(QuantizationOrderArm.HIGH_PRECISION_SURGERY, 0.8),
    ]
    unsupported = QuantizationOrderCell(
        _key(),
        QuantizationOrderArm.QUANTIZE_THEN_SURGERY,
        QuantizationOrderOutcome.UNSUPPORTED,
        None,
        (),
        "tool-revision",
        "native codec cannot represent the selected post-quantization edit",
    )

    comparison = build_quantization_order_study((*measured, unsupported)).compare(_key())

    assert comparison.outcome is QuantizationOrderOutcome.UNSUPPORTED
    assert comparison.preferred_arm is None
    assert comparison.quantization_effect == ()


def test_non_equivalent_structure_is_inconclusive_and_protocol_is_preregistered() -> None:
    cells = tuple(
        _cell(
            arm,
            0.8,
            key=_key(
                "structure-2"
                if arm is QuantizationOrderArm.QUANTIZE_THEN_SURGERY
                else "structure-1"
            ),
        )
        for arm in QuantizationOrderArm
    )
    comparison = build_quantization_order_study(cells).compare(_key())
    protocol = build_default_quantization_order_protocol()

    assert comparison.outcome is QuantizationOrderOutcome.INCONCLUSIVE
    assert protocol["families"] == ["llama", "qwen"]
    assert protocol["compression_levels"] == ["10_percent", "20_percent"]
    assert protocol["seeds"] == [0, 1, 2]
