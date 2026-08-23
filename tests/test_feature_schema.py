"""Tests for feature value, sample, and precision provenance records."""

from __future__ import annotations

import json

import pytest

from modelsurgeon.features import (
    ErrorProvenance,
    FeatureKind,
    FeatureRecord,
    FeatureSampleContext,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId


def _context() -> FeatureSampleContext:
    return FeatureSampleContext(
        "org/data",
        "revision-1",
        "validation",
        ("sample-2", "sample-7"),
        "clean-v2",
        "org/tokenizer",
        "tokenizer-revision",
    )


@pytest.mark.parametrize(
    "precision",
    [
        PrecisionProvenance(
            PrecisionSource.DIRECT_QUANTIZED,
            "Q4_K",
            "int32",
            quantization="Q4_K",
            codec_version="1",
        ),
        PrecisionProvenance(
            PrecisionSource.LOCALLY_DEQUANTIZED,
            "Q8_0",
            "float32",
            quantization="Q8_0",
            codec_version="1",
            error=ErrorProvenance(0.01, 0.02, "float32", "pinned-vector-v1"),
        ),
        PrecisionProvenance(PrecisionSource.HIGH_PRECISION, "float32", "float64"),
    ],
)
def test_precision_sources_are_distinct_and_json_safe(
    precision: PrecisionProvenance,
) -> None:
    record = FeatureRecord(
        ComponentId.parse("model.layers.0.self_attn.q_proj"),
        "spectral_norm",
        FeatureKind.SCALAR,
        1.25,
        "float64",
        "spectral",
        "2",
        precision,
        _context(),
        (("rank", 4), ("approximate", True)),
    ).to_record()

    assert record["precision"]["source"] == precision.source.value  # type: ignore[index]
    assert record["sample_context"]["sample_ids"] == ["sample-2", "sample-7"]  # type: ignore[index]
    assert json.loads(json.dumps(record)) == record


def test_vector_record_preserves_order_and_forward_metadata() -> None:
    feature = FeatureRecord(
        ComponentId.parse("model.norm"),
        "moments",
        FeatureKind.VECTOR,
        (1.0, -0.5, 0.0),
        "float32",
        "moments",
        "1",
        PrecisionProvenance(PrecisionSource.HIGH_PRECISION, "float32", "float32"),
        metadata=(("future_field", "preserved"),),
    )

    assert feature.to_record()["value"] == [1.0, -0.5, 0.0]
    assert feature.to_record()["metadata"] == {"future_field": "preserved"}


def test_invalid_value_shapes_and_nonfinite_values_fail_closed() -> None:
    precision = PrecisionProvenance(
        PrecisionSource.HIGH_PRECISION, "float32", "float32"
    )
    base = (ComponentId.parse("model"), "x")

    with pytest.raises(ValueError, match="scalar"):
        FeatureRecord(*base, FeatureKind.SCALAR, (1.0,), "f32", "x", "1", precision)
    with pytest.raises(ValueError, match="non-empty"):
        FeatureRecord(*base, FeatureKind.VECTOR, (), "f32", "x", "1", precision)
    with pytest.raises(ValueError, match="finite"):
        FeatureRecord(*base, FeatureKind.SCALAR, float("nan"), "f32", "x", "1", precision)


def test_precision_contract_rejects_ambiguous_provenance() -> None:
    with pytest.raises(ValueError, match="requires exact quantization"):
        PrecisionProvenance(PrecisionSource.DIRECT_QUANTIZED, "Q4_K", "int32")
    with pytest.raises(ValueError, match="measured error"):
        PrecisionProvenance(
            PrecisionSource.LOCALLY_DEQUANTIZED,
            "Q8_0",
            "float32",
            quantization="Q8_0",
        )
    with pytest.raises(ValueError, match="cannot claim"):
        PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            "float32",
            "float32",
            quantization="Q8_0",
        )
