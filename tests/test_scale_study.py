from itertools import pairwise

import pytest

from modelsurgeon.evaluation.scale_study import (
    ScaleStudyError,
    choose_scale_default,
    evenly_spaced_indices,
)


@pytest.mark.parametrize(
    ("parameters", "device", "mode", "tokens", "quantized"),
    (
        (135_000_000, "accelerator", "full_tensor", 64, False),
        (1_500_000_000, "accelerator", "tensor", 32, False),
        (4_000_000_000, "accelerator", "streaming", 16, True),
        (7_000_000_000, "cpu", "streaming", 8, True),
    ),
)
def test_consumer_scale_defaults_are_bounded(
    parameters: int,
    device: str,
    mode: str,
    tokens: int,
    quantized: bool,
) -> None:
    result = choose_scale_default(parameters, accelerator_memory_bytes=12 << 30)

    assert result.execution_device == device
    assert result.memory_mode == mode
    assert result.calibration_tokens == tokens
    assert result.requires_quantization is quantized


def test_cpu_only_and_invalid_scale_defaults_fail_safely() -> None:
    assert (
        choose_scale_default(135_000_000, accelerator_memory_bytes=None).execution_device == "cpu"
    )
    with pytest.raises(ScaleStudyError, match="parameter count"):
        choose_scale_default(0, accelerator_memory_bytes=12 << 30)
    with pytest.raises(ScaleStudyError, match="accelerator memory"):
        choose_scale_default(1, accelerator_memory_bytes=0)


def test_large_tensor_sample_indices_are_exact_and_in_bounds() -> None:
    length = 24_903_680
    result = evenly_spaced_indices(length, 65_536)

    assert len(result) == 65_536
    assert result[0] == 0
    assert result[-1] == length - 1
    assert all(left < right for left, right in pairwise(result))


@pytest.mark.parametrize(("length", "count"), ((0, 1), (1, 0), (1, 2)))
def test_invalid_sample_indices_fail_safely(length: int, count: int) -> None:
    with pytest.raises(ScaleStudyError, match="sample"):
        evenly_spaced_indices(length, count)
