"""Tests for the pinned v1 IQ-family support boundary."""

from __future__ import annotations

import pytest

from modelsurgeon.adapters.gguf import (
    GGUF_CODEC_CONFORMANCE_VECTORS,
    IQ_SUPPORT_MATRIX,
    CodecContractError,
    IQSupportLevel,
    iq_type_support,
    require_iq_native_write_target,
    validate_codec_vector,
    validate_iq_support_matrix,
)
from modelsurgeon.adapters.gguf.quantization import GGMLQuantizationType


def test_matrix_covers_every_iq_type_and_has_explicit_decisions() -> None:
    validate_iq_support_matrix()
    assert {item.level for item in IQ_SUPPORT_MATRIX} == {
        IQSupportLevel.SUPPORTED,
        IQSupportLevel.READ_ONLY,
        IQSupportLevel.DEFERRED,
    }
    assert [item.priority for item in IQ_SUPPORT_MATRIX if item.priority] == [1, 2]
    assert all(item.codebooks and item.decision for item in IQ_SUPPORT_MATRIX)


@pytest.mark.parametrize(
    ("quant_type", "level"),
    [
        (GGMLQuantizationType.IQ2_XS, IQSupportLevel.READ_ONLY),
        (GGMLQuantizationType.IQ4_NL, IQSupportLevel.SUPPORTED),
        (GGMLQuantizationType.IQ4_XS, IQSupportLevel.SUPPORTED),
    ],
)
def test_iq2_and_iq4_samples_have_valid_pinned_vectors_and_decisions(
    quant_type: GGMLQuantizationType, level: IQSupportLevel
) -> None:
    vector = next(item for item in GGUF_CODEC_CONFORMANCE_VECTORS if item.quant_type is quant_type)
    validate_codec_vector(vector)
    support = iq_type_support(quant_type)
    assert support.level is level
    assert support.gguf_py_decoder is True
    assert support.gguf_py_encoder is False


def test_native_write_boundary_fails_closed_without_family_substitution() -> None:
    assert require_iq_native_write_target(GGMLQuantizationType.IQ4_NL).priority == 1
    with pytest.raises(CodecContractError, match="read_only"):
        require_iq_native_write_target(GGMLQuantizationType.IQ2_XS)
    with pytest.raises(CodecContractError, match="deferred"):
        require_iq_native_write_target(GGMLQuantizationType.IQ1_M)
    with pytest.raises(CodecContractError, match="not an IQ-family"):
        iq_type_support(GGMLQuantizationType.Q4_K)
