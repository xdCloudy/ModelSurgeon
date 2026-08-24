"""Empirical quantized-feature bias, variance, ranking, and resource study."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, cast

from modelsurgeon.adapters.gguf import Q4_K_CODEC, Q5_K_CODEC, Q6_K_CODEC, Q8_0_CODEC
from modelsurgeon.adapters.gguf.quantization import ByteOrder
from modelsurgeon.features.quantized_direct import (
    BytesEncodedBlockSource,
    DirectQuantizedFeature,
    DirectQuantizedFeatureName,
    EncodedBlockSource,
    extract_direct_quantized_feature,
)


class QuantizedFeatureReliabilityError(ValueError):
    """Raised when tensor samples cannot form a matched codec study."""


@dataclass(frozen=True, slots=True)
class QuantizedTensorSample:
    model_identifier: str
    tensor_name: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.model_identifier
            or not self.tensor_name
            or not self.values
            or len(self.values) % 256
            or any(not math.isfinite(value) for value in self.values)
        ):
            raise QuantizedFeatureReliabilityError("quantized tensor sample is invalid")


@dataclass(frozen=True, slots=True)
class QuantizedFeatureReliabilityResult:
    sample_count: int
    model_count: int
    elements_per_sample: int
    codecs: tuple[dict[str, object], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "model_count": self.model_count,
            "elements_per_sample": self.elements_per_sample,
            "codecs": list(self.codecs),
        }


def _features(values: tuple[float, ...]) -> dict[str, float]:
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return {
        "weight_mean": mean,
        "weight_variance": variance,
        "weight_abs_mean": math.fsum(abs(value) for value in values) / len(values),
        "weight_rms": math.sqrt(math.fsum(value * value for value in values) / len(values)),
        "sparsity": sum(value == 0.0 for value in values) / len(values),
    }


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            ranks[index] = rank
        cursor = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    left_ranks, right_ranks = _ranks(left), _ranks(right)
    left_mean = math.fsum(left_ranks) / len(left_ranks)
    right_mean = math.fsum(right_ranks) / len(right_ranks)
    numerator = math.fsum(
        (one - left_mean) * (two - right_mean)
        for one, two in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(math.fsum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(math.fsum((value - right_mean) ** 2 for value in right_ranks))
    return (
        0.0 if left_scale == 0.0 or right_scale == 0.0 else numerator / (left_scale * right_scale)
    )


def _error_model(reference: list[float], candidate: list[float]) -> dict[str, object]:
    errors = [right - left for left, right in zip(reference, candidate, strict=True)]
    relative = [
        abs(error) / max(abs(value), 1e-12) for value, error in zip(reference, errors, strict=True)
    ]
    mean_bias = math.fsum(errors) / len(errors)
    return {
        "mean_bias": mean_bias,
        "error_variance": math.fsum((value - mean_bias) ** 2 for value in errors) / len(errors),
        "mean_absolute_error": math.fsum(abs(value) for value in errors) / len(errors),
        "mean_relative_absolute_error": math.fsum(relative) / len(relative),
        "maximum_absolute_error": max(abs(value) for value in errors),
        "spearman_rank_correlation": _correlation(reference, candidate),
        "sample_count": len(errors),
        "reference_dtype": "float64",
        "method": "matched_tensor_empirical_v1",
    }


def _zscore(values: list[float]) -> list[float]:
    mean = math.fsum(values) / len(values)
    scale = statistics.pstdev(values)
    if scale == 0.0:
        return [0.0] * len(values)
    return [(value - mean) / scale for value in values]


def run_quantized_feature_reliability(
    samples: tuple[QuantizedTensorSample, ...],
) -> QuantizedFeatureReliabilityResult:
    if len(samples) < 3 or len({item.model_identifier for item in samples}) < 2:
        raise QuantizedFeatureReliabilityError("study requires multiple tensors and models")
    sizes = {len(item.values) for item in samples}
    if len(sizes) != 1:
        raise QuantizedFeatureReliabilityError("tensor sample sizes must match")
    reference = [_features(item.values) for item in samples]
    codecs: tuple[Any, ...] = (Q4_K_CODEC, Q5_K_CODEC, Q6_K_CODEC, Q8_0_CODEC)
    reports: list[dict[str, object]] = []
    for codec in codecs:
        decoded_features: list[dict[str, float]] = []
        direct_scales: list[float] = []
        encoded_bytes = 0
        encode_seconds = 0.0
        direct_seconds = 0.0
        decode_feature_seconds = 0.0
        quantization_rmse: list[float] = []
        for sample in samples:
            payload = bytearray(codec.layout.encoded_size(len(sample.values)))
            started = time.perf_counter()
            codec.encode_blocks(sample.values, memoryview(payload), byte_order=ByteOrder.LITTLE)
            encode_seconds += time.perf_counter() - started
            encoded_bytes += len(payload)
            source = BytesEncodedBlockSource(
                codec.identity.quant_type, len(sample.values), bytes(payload)
            )
            started = time.perf_counter()
            direct = extract_direct_quantized_feature(
                cast(EncodedBlockSource, source),
                DirectQuantizedFeatureName.PRIMARY_SCALE_ABS_MEAN,
                max_sample_blocks=1 << 20,
            )
            direct_seconds += time.perf_counter() - started
            if not isinstance(direct, DirectQuantizedFeature):
                raise QuantizedFeatureReliabilityError("validated codec lacks direct scale")
            direct_scales.append(direct.value)
            decoded: list[float] = []
            started = time.perf_counter()
            codec.decode_blocks(memoryview(payload), decoded, byte_order=ByteOrder.LITTLE)
            decoded_tuple = tuple(decoded)
            decoded_features.append(_features(decoded_tuple))
            decode_feature_seconds += time.perf_counter() - started
            quantization_rmse.append(
                math.sqrt(
                    math.fsum(
                        (left - right) ** 2
                        for left, right in zip(sample.values, decoded_tuple, strict=True)
                    )
                    / len(sample.values)
                )
            )
        feature_models = {
            name: _error_model(
                [item[name] for item in reference],
                [item[name] for item in decoded_features],
            )
            for name in reference[0]
        }
        reference_rms = [item["weight_rms"] for item in reference]
        direct_model = _error_model(_zscore(reference_rms), _zscore(direct_scales))
        direct_model["comparison"] = "zscore_primary_scale_vs_high_precision_weight_rms"
        reports.append(
            {
                "codec": codec.identity.quant_type.value,
                "codec_identity": {
                    "quant_type": codec.identity.quant_type.value,
                    "implementation": codec.identity.implementation,
                    "version": codec.identity.version,
                    "upstream_revision": codec.identity.upstream_revision,
                },
                "decoded_feature_error_models": feature_models,
                "direct_primary_scale_model": direct_model,
                "resource_cost": {
                    "encoded_bytes": encoded_bytes,
                    "encode_seconds": encode_seconds,
                    "direct_feature_seconds": direct_seconds,
                    "decode_and_feature_seconds": decode_feature_seconds,
                    "direct_speedup_vs_decode": (
                        decode_feature_seconds / direct_seconds if direct_seconds > 0.0 else None
                    ),
                },
                "weight_quantization_rmse": {
                    "mean": math.fsum(quantization_rmse) / len(quantization_rmse),
                    "maximum": max(quantization_rmse),
                },
            }
        )
    return QuantizedFeatureReliabilityResult(
        len(samples),
        len({item.model_identifier for item in samples}),
        next(iter(sizes)),
        tuple(reports),
    )
