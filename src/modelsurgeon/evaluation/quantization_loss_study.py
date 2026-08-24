"""Matched-region surgery and requantization main-effect decomposition."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from modelsurgeon.adapters.gguf import Q4_K_CODEC, Q5_K_CODEC, Q6_K_CODEC, Q8_0_CODEC
from modelsurgeon.adapters.gguf.quantization import ByteOrder

from .quantized_feature_reliability import QuantizedTensorSample


class QuantizationLossStudyError(ValueError):
    """Raised when matched quantization-loss inputs are invalid."""


@dataclass(frozen=True, slots=True)
class EffectEstimate:
    name: str
    mean: float
    confidence_low: float
    confidence_high: float
    bootstrap_repetitions: int

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mean": self.mean,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "bootstrap_repetitions": self.bootstrap_repetitions,
        }


@dataclass(frozen=True, slots=True)
class QuantizationLossStudyResult:
    sample_count: int
    surgery_fraction: float
    codecs: tuple[dict[str, object], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "surgery_fraction": self.surgery_fraction,
            "codecs": list(self.codecs),
        }


def _decode(codec: Any, values: tuple[float, ...]) -> tuple[float, ...]:
    payload = bytearray(codec.layout.encoded_size(len(values)))
    codec.encode_blocks(values, memoryview(payload), byte_order=ByteOrder.LITTLE)
    decoded: list[float] = []
    codec.decode_blocks(memoryview(payload), decoded, byte_order=ByteOrder.LITTLE)
    return tuple(decoded)


def _mse(reference: tuple[float, ...], candidate: tuple[float, ...]) -> float:
    return math.fsum(
        (left - right) ** 2 for left, right in zip(reference, candidate, strict=True)
    ) / len(reference)


def _percentile(values: list[float], quantile: float) -> float:
    position = quantile * (len(values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _estimate(
    name: str,
    values: tuple[float, ...],
    repetitions: int,
    seed: int,
) -> EffectEstimate:
    randomizer = random.Random(seed)
    means = sorted(
        math.fsum(values[randomizer.randrange(len(values))] for _ in values) / len(values)
        for _ in range(repetitions)
    )
    return EffectEstimate(
        name,
        math.fsum(values) / len(values),
        _percentile(means, 0.025),
        _percentile(means, 0.975),
        repetitions,
    )


def run_quantization_loss_study(
    samples: tuple[QuantizedTensorSample, ...],
    *,
    surgery_stride: int = 16,
    bootstrap_repetitions: int = 1000,
    seed: int = 42,
) -> QuantizationLossStudyResult:
    """Decompose aligned-region weight MSE into surgery, requantization, and interaction."""

    if (
        len(samples) < 3
        or surgery_stride < 2
        or bootstrap_repetitions <= 0
        or len({len(item.values) for item in samples}) != 1
    ):
        raise QuantizationLossStudyError("quantization-loss study inputs are invalid")
    codecs: tuple[Any, ...] = (Q4_K_CODEC, Q5_K_CODEC, Q6_K_CODEC, Q8_0_CODEC)
    reports: list[dict[str, object]] = []
    for codec_index, codec in enumerate(codecs):
        controls: list[float] = []
        surgeries: list[float] = []
        combined: list[float] = []
        interactions: list[float] = []
        conditional_requant: list[float] = []
        for sample in samples:
            reference = sample.values
            surgery = tuple(
                0.0 if index % surgery_stride == 0 else value
                for index, value in enumerate(reference)
            )
            control_decoded = _decode(codec, reference)
            combined_decoded = _decode(codec, surgery)
            control_mse = _mse(reference, control_decoded)
            surgery_mse = _mse(reference, surgery)
            combined_mse = _mse(reference, combined_decoded)
            controls.append(control_mse)
            surgeries.append(surgery_mse)
            combined.append(combined_mse)
            interactions.append(combined_mse - control_mse - surgery_mse)
            conditional_requant.append(combined_mse - surgery_mse)
        effects = (
            _estimate(
                "requantization_main_effect_mse",
                tuple(controls),
                bootstrap_repetitions,
                seed + codec_index * 10,
            ),
            _estimate(
                "surgery_main_effect_mse",
                tuple(surgeries),
                bootstrap_repetitions,
                seed + codec_index * 10 + 1,
            ),
            _estimate(
                "surgery_plus_requantization_mse",
                tuple(combined),
                bootstrap_repetitions,
                seed + codec_index * 10 + 2,
            ),
            _estimate(
                "interaction_mse",
                tuple(interactions),
                bootstrap_repetitions,
                seed + codec_index * 10 + 3,
            ),
            _estimate(
                "requantization_effect_after_surgery_mse",
                tuple(conditional_requant),
                bootstrap_repetitions,
                seed + codec_index * 10 + 4,
            ),
        )
        reports.append(
            {
                "codec": codec.identity.quant_type.value,
                "effects": [effect.to_record() for effect in effects],
                "root_mean_squared_weight_error": {
                    "requantization_control": math.sqrt(math.fsum(controls) / len(controls)),
                    "surgery_only": math.sqrt(math.fsum(surgeries) / len(surgeries)),
                    "surgery_plus_requantization": math.sqrt(math.fsum(combined) / len(combined)),
                },
                "matched_region_elements": len(samples) * len(samples[0].values),
            }
        )
    return QuantizationLossStudyResult(
        len(samples),
        1.0 / surgery_stride,
        tuple(reports),
    )
