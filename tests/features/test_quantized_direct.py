import math
import struct

import pytest

from modelsurgeon.adapters.gguf.quantization import ByteOrder, GGMLQuantizationType
from modelsurgeon.evaluation.quantized_feature_reliability import (
    QuantizedTensorSample,
    run_quantized_feature_reliability,
)
from modelsurgeon.features.quantized_direct import (
    BytesEncodedBlockSource,
    DirectQuantizedFeature,
    DirectQuantizedFeatureName,
    EmpiricalQuantizedFeatureErrorModel,
    LocalDecodeRequired,
    extract_direct_quantized_feature,
)


def _q8_block(scale: float, byte_order: ByteOrder = ByteOrder.LITTLE) -> bytes:
    prefix = "<" if byte_order is ByteOrder.LITTLE else ">"
    return struct.pack(f"{prefix}e", scale) + bytes(range(32))


def test_exact_layout_features_record_full_coverage_and_zero_error() -> None:
    source = BytesEncodedBlockSource(
        GGMLQuantizationType.Q8_0,
        64,
        _q8_block(2.0) + _q8_block(4.0),
    )

    block_count = extract_direct_quantized_feature(
        source,
        DirectQuantizedFeatureName.BLOCK_COUNT,
    )
    bytes_per_element = extract_direct_quantized_feature(
        source,
        DirectQuantizedFeatureName.ENCODED_BYTES_PER_ELEMENT,
    )

    assert isinstance(block_count, DirectQuantizedFeature)
    assert block_count.value == 2.0
    assert block_count.coverage_fraction == 1.0
    assert block_count.estimated_error == 0.0
    assert isinstance(bytes_per_element, DirectQuantizedFeature)
    assert bytes_per_element.value == pytest.approx(34 / 32)
    assert bytes_per_element.to_record()["codec"] == "Q8_0"


def test_primary_scale_mean_is_read_directly_with_full_or_sampled_coverage() -> None:
    source = BytesEncodedBlockSource(
        GGMLQuantizationType.Q8_0,
        64,
        _q8_block(2.0) + _q8_block(4.0),
    )

    exact = extract_direct_quantized_feature(
        source,
        DirectQuantizedFeatureName.PRIMARY_SCALE_ABS_MEAN,
        max_sample_blocks=2,
    )
    sampled = extract_direct_quantized_feature(
        source,
        DirectQuantizedFeatureName.PRIMARY_SCALE_ABS_MEAN,
        max_sample_blocks=1,
    )

    assert isinstance(exact, DirectQuantizedFeature)
    assert exact.value == 3.0
    assert exact.covered_blocks == 2
    assert exact.estimated_error == 0.0
    assert exact.error_method == "exact_full_block_coverage"
    assert isinstance(sampled, DirectQuantizedFeature)
    assert sampled.value == 2.0
    assert sampled.covered_blocks == 1
    assert sampled.coverage_fraction == 0.5
    assert sampled.estimated_error == 2.0
    assert sampled.error_method == "single_sample_absolute_proxy"


def test_big_endian_primary_scale_is_respected() -> None:
    source = BytesEncodedBlockSource(
        GGMLQuantizationType.Q8_0,
        32,
        _q8_block(1.5, ByteOrder.BIG),
        ByteOrder.BIG,
    )
    outcome = extract_direct_quantized_feature(
        source,
        DirectQuantizedFeatureName.PRIMARY_SCALE_ABS_MEAN,
    )

    assert isinstance(outcome, DirectQuantizedFeature)
    assert outcome.value == 1.5


def test_decoded_semantics_require_local_decode_instead_of_approximation() -> None:
    source = BytesEncodedBlockSource(
        GGMLQuantizationType.Q8_0,
        32,
        _q8_block(1.0),
    )

    for name in (
        DirectQuantizedFeatureName.WEIGHT_MEAN,
        DirectQuantizedFeatureName.WEIGHT_VARIANCE,
        DirectQuantizedFeatureName.SPARSITY,
    ):
        outcome = extract_direct_quantized_feature(source, name)
        assert isinstance(outcome, LocalDecodeRequired)
        assert outcome.to_record()["status"] == "local_decode_required"


def test_unvalidated_codec_primary_scale_requires_decode() -> None:
    payload = struct.pack("<e", 1.0)
    source = BytesEncodedBlockSource(GGMLQuantizationType.F16, 1, payload)
    outcome = extract_direct_quantized_feature(
        source,
        DirectQuantizedFeatureName.PRIMARY_SCALE_ABS_MEAN,
    )

    assert isinstance(outcome, LocalDecodeRequired)


def test_quantized_reliability_study_compares_all_target_codecs() -> None:
    samples = tuple(
        QuantizedTensorSample(
            "model-a" if index < 2 else "model-b",
            f"tensor-{index}",
            tuple(math.sin(offset / 17.0 + index) * 0.2 for offset in range(256)),
        )
        for index in range(3)
    )

    result = run_quantized_feature_reliability(samples)

    assert result.model_count == 2
    assert {item["codec"] for item in result.codecs} == {
        "Q4_K",
        "Q5_K",
        "Q6_K",
        "Q8_0",
    }
    assert all(
        set(item["decoded_feature_error_models"])
        == {
            "weight_mean",
            "weight_variance",
            "weight_abs_mean",
            "weight_rms",
            "sparsity",
        }
        for item in result.codecs
    )
    q4 = next(item for item in result.codecs if item["codec"] == "Q4_K")
    raw_models = q4["decoded_feature_error_models"]
    assert isinstance(raw_models, dict)
    model = EmpiricalQuantizedFeatureErrorModel.from_study_record(
        GGMLQuantizationType.Q4_K,
        "weight_rms",
        raw_models["weight_rms"],
        "a" * 64,
    )
    provenance = model.precision_provenance(storage_dtype="Q4_K")
    assert provenance.error is not None
    assert provenance.error.absolute_error == model.mean_absolute_error
