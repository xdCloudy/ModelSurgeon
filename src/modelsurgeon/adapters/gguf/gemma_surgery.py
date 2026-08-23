"""Finite Gemma-family GGUF physical-surgery compatibility adapter."""

from __future__ import annotations

from dataclasses import dataclass

from modelsurgeon.adapters.family import ModelFamily
from modelsurgeon.adapters.gguf.architecture import (
    CouplingGroup,
    GGUFArchitectureError,
    ResolvedGGUFArchitecture,
    TensorMapping,
    TensorRole,
    resolve_gguf_architecture,
)
from modelsurgeon.adapters.gguf.container import GGUFContainer
from modelsurgeon.adapters.gguf.discovery import (
    GGUFDiscovery,
    GGUFModelShape,
    discover_gguf_components,
)

GEMMA_GGUF_SURGERY_CONTRACT_VERSION = 1


class GemmaGGUFSurgeryError(GGUFArchitectureError):
    """Raised when a Gemma variant is outside the physical surgery boundary."""


@dataclass(frozen=True, slots=True)
class GemmaGGUFVariant:
    architecture: str
    generation: int
    native_surgery: bool
    constraint: str


GEMMA_GGUF_VARIANTS = (
    GemmaGGUFVariant("gemma", 1, True, "dense-rmsnorm"),
    GemmaGGUFVariant("gemma2", 2, False, "extra-norm-and-attention-dimensions"),
    GemmaGGUFVariant("gemma3", 3, False, "local-global-attention-and-extra-norms"),
)


@dataclass(frozen=True, slots=True)
class GemmaAttentionGeometry:
    head_count: int
    kv_head_count: int
    head_width: int
    kv_width: int
    query_heads_per_kv: int


@dataclass(frozen=True, slots=True)
class GemmaLayerSurgeryLayout:
    layer_index: int
    tensors: tuple[TensorMapping, ...]
    coupling_groups: tuple[CouplingGroup, ...]


@dataclass(frozen=True, slots=True)
class GemmaGGUFSurgeryAdapter:
    variant: GemmaGGUFVariant
    architecture: ResolvedGGUFArchitecture
    shape: GGUFModelShape
    attention: GemmaAttentionGeometry
    layers: tuple[GemmaLayerSurgeryLayout, ...]
    output_weight_present: bool
    contract_version: int = GEMMA_GGUF_SURGERY_CONTRACT_VERSION

    def layer(self, index: int) -> GemmaLayerSurgeryLayout:
        if index < 0 or index >= len(self.layers):
            raise GemmaGGUFSurgeryError(
                f"Gemma block {index} is outside block_count {len(self.layers)}"
            )
        return self.layers[index]

    def to_record(self) -> dict[str, str | int | bool]:
        return {
            "family": ModelFamily.GEMMA.value,
            "architecture": self.variant.architecture,
            "generation": self.variant.generation,
            "constraint": self.variant.constraint,
            "contract_version": self.contract_version,
            "layer_count": len(self.layers),
            "head_count": self.attention.head_count,
            "kv_head_count": self.attention.kv_head_count,
            "head_width": self.attention.head_width,
            "query_heads_per_kv": self.attention.query_heads_per_kv,
            "output_weight_present": self.output_weight_present,
        }


_ROLES = (
    TensorRole.INPUT_NORM,
    TensorRole.POST_ATTENTION_NORM,
    TensorRole.ATTENTION_Q,
    TensorRole.ATTENTION_K,
    TensorRole.ATTENTION_V,
    TensorRole.ATTENTION_O,
    TensorRole.MLP_GATE,
    TensorRole.MLP_UP,
    TensorRole.MLP_DOWN,
)


def _variant(name: str) -> GemmaGGUFVariant:
    try:
        variant = next(item for item in GEMMA_GGUF_VARIANTS if item.architecture == name)
    except StopIteration as error:
        raise GemmaGGUFSurgeryError(f"unknown Gemma GGUF variant {name!r}") from error
    if not variant.native_surgery:
        raise GemmaGGUFSurgeryError(
            f"Gemma variant {name!r} requires unsupported {variant.constraint} mappings"
        )
    return variant


def _geometry(shape: GGUFModelShape) -> GemmaAttentionGeometry:
    if shape.attention_heads % shape.kv_heads:
        raise GemmaGGUFSurgeryError(
            "Gemma attention head count must be divisible by KV head count"
        )
    width = shape.embedding_length // shape.attention_heads
    return GemmaAttentionGeometry(
        shape.attention_heads,
        shape.kv_heads,
        width,
        width * shape.kv_heads,
        shape.attention_heads // shape.kv_heads,
    )


def build_gemma_gguf_surgery_adapter(discovery: GGUFDiscovery) -> GemmaGGUFSurgeryAdapter:
    if discovery.family is not ModelFamily.GEMMA:
        raise GemmaGGUFSurgeryError("Gemma GGUF surgery requires explicit family gemma")
    if discovery.contract_version != GEMMA_GGUF_SURGERY_CONTRACT_VERSION:
        raise GemmaGGUFSurgeryError(
            f"unsupported Gemma GGUF contract version {discovery.contract_version}"
        )
    variant = _variant(discovery.architecture)
    architecture = resolve_gguf_architecture(
        variant.architecture, family=ModelFamily.GEMMA
    )
    layers: list[GemmaLayerSurgeryLayout] = []
    for index in range(discovery.shape.layers):
        prefix = f"blk.{index}."
        mappings = tuple(
            tensor.mapping
            for tensor in discovery.tensors
            if tensor.descriptor.name.startswith(prefix)
            and tensor.descriptor.name.endswith(".weight")
        )
        for role in _ROLES:
            if sum(mapping.role is role for mapping in mappings) != 1:
                raise GemmaGGUFSurgeryError(
                    f"Gemma block {index} has an incomplete {role.value} mapping"
                )
        layers.append(
            GemmaLayerSurgeryLayout(index, mappings, architecture.coupling_groups(index))
        )
    names = {tensor.descriptor.name for tensor in discovery.tensors}
    return GemmaGGUFSurgeryAdapter(
        variant,
        architecture,
        discovery.shape,
        _geometry(discovery.shape),
        tuple(layers),
        "output.weight" in names,
    )


def load_gemma_gguf_surgery_adapter(container: GGUFContainer) -> GemmaGGUFSurgeryAdapter:
    discovery = discover_gguf_components(container, family=ModelFamily.GEMMA)
    return build_gemma_gguf_surgery_adapter(discovery)
