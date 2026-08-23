"""Pinned v1 support boundary for importance-quantized GGUF tensor types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.adapters.gguf.conformance import (
    GGML_UPSTREAM_REVISION,
    GGUF_PY_QUANTS_BLOB,
)
from modelsurgeon.adapters.gguf.quantization import (
    QUANT_LAYOUTS,
    CodecContractError,
    GGMLQuantizationType,
    QuantizationFamily,
)


class IQSupportLevel(StrEnum):
    SUPPORTED = "supported"
    READ_ONLY = "read_only"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class IQTypeSupport:
    quant_type: GGMLQuantizationType
    level: IQSupportLevel
    priority: int | None
    codebooks: tuple[str, ...]
    upstream_reference_decoder: bool
    upstream_reference_encoder: bool
    gguf_py_decoder: bool
    gguf_py_encoder: bool
    decision: str
    upstream_revision: str = GGML_UPSTREAM_REVISION
    gguf_py_blob: str = GGUF_PY_QUANTS_BLOB

    @property
    def native_write_target(self) -> bool:
        return self.level is IQSupportLevel.SUPPORTED


IQ_SUPPORT_MATRIX = (
    IQTypeSupport(
        GGMLQuantizationType.IQ4_NL,
        IQSupportLevel.SUPPORTED,
        1,
        ("kvalues_iq4nl",),
        True,
        True,
        True,
        False,
        "Small fixed nonlinear table; implement first with exhaustive index coverage.",
    ),
    IQTypeSupport(
        GGMLQuantizationType.IQ4_XS,
        IQSupportLevel.SUPPORTED,
        2,
        ("kvalues_iq4nl", "packed_6bit_scales"),
        True,
        True,
        True,
        False,
        "Build on IQ4_NL with explicit super-block scale packing.",
    ),
    IQTypeSupport(
        GGMLQuantizationType.IQ2_XXS,
        IQSupportLevel.READ_ONLY,
        None,
        ("iq2xxs_grid", "ksigns_iq2xs"),
        True,
        True,
        True,
        False,
        "Decode for analysis; native write waits for pinned generated grid provenance.",
    ),
    IQTypeSupport(
        GGMLQuantizationType.IQ2_XS,
        IQSupportLevel.READ_ONLY,
        None,
        ("iq2xs_grid", "ksigns_iq2xs"),
        True,
        True,
        True,
        False,
        "Decode for analysis; encoder requires large grid search and importance data.",
    ),
    IQTypeSupport(
        GGMLQuantizationType.IQ2_S,
        IQSupportLevel.READ_ONLY,
        None,
        ("iq2s_grid", "ksigns_iq2xs"),
        True,
        True,
        True,
        False,
        "Decode for analysis; defer native mutation until grid conformance is pinned.",
    ),
    IQTypeSupport(
        GGMLQuantizationType.IQ3_XXS,
        IQSupportLevel.DEFERRED,
        None,
        ("iq3xxs_grid", "ksigns_iq2xs"),
        True,
        True,
        True,
        False,
        "Lower practical priority and large generated grid dependency.",
    ),
    IQTypeSupport(
        GGMLQuantizationType.IQ3_S,
        IQSupportLevel.DEFERRED,
        None,
        ("iq3s_grid", "ksigns_iq2xs"),
        True,
        True,
        True,
        False,
        "Lower practical priority and multiple packed index/scale fields.",
    ),
    IQTypeSupport(
        GGMLQuantizationType.IQ1_S,
        IQSupportLevel.DEFERRED,
        None,
        ("iq1s_grid", "iq1s_delta_table"),
        True,
        True,
        True,
        False,
        "One-bit grid search is importance-matrix-sensitive and not a v1 write target.",
    ),
    IQTypeSupport(
        GGMLQuantizationType.IQ1_M,
        IQSupportLevel.DEFERRED,
        None,
        ("iq1s_grid", "iq1m_scale_table"),
        True,
        True,
        True,
        False,
        "Most complex one-bit scale packing; defer native mutation beyond v1.",
    ),
)

_BY_TYPE = {item.quant_type: item for item in IQ_SUPPORT_MATRIX}


def iq_type_support(quant_type: GGMLQuantizationType) -> IQTypeSupport:
    try:
        return _BY_TYPE[quant_type]
    except KeyError as error:
        raise CodecContractError(f"{quant_type.value} is not an IQ-family tensor type") from error


def require_iq_native_write_target(quant_type: GGMLQuantizationType) -> IQTypeSupport:
    support = iq_type_support(quant_type)
    if not support.native_write_target:
        raise CodecContractError(
            f"{quant_type.value} native writes are {support.level.value}; "
            "codec substitution is forbidden"
        )
    return support


def validate_iq_support_matrix() -> None:
    expected = {
        quant_type
        for quant_type, layout in QUANT_LAYOUTS.items()
        if layout.family is QuantizationFamily.IQ
    }
    if set(_BY_TYPE) != expected or len(_BY_TYPE) != len(IQ_SUPPORT_MATRIX):
        raise CodecContractError("IQ support matrix must cover every exact IQ tensor type once")
    priorities = [item.priority for item in IQ_SUPPORT_MATRIX if item.priority is not None]
    if priorities != list(range(1, len(priorities) + 1)):
        raise CodecContractError("supported IQ implementation priorities must be contiguous")
