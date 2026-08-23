"""Explicit Qwen2/Qwen3 dense GGUF architecture surgery adapter."""

from __future__ import annotations

from dataclasses import dataclass

from modelsurgeon.adapters.family import ModelFamily
from modelsurgeon.adapters.gguf.architecture import (
    CouplingGroup,
    GGUFArchitectureError,
    MetadataSemantic,
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

QWEN_GGUF_SURGERY_CONTRACT_VERSION = 1


class QwenGGUFSurgeryError(GGUFArchitectureError):
    """Raised when Qwen metadata is outside the finite surgery support matrix."""


@dataclass(frozen=True, slots=True)
class QwenGGUFVariant:
    architecture: str
    generation: int
    mixture_of_experts: bool
    native_surgery: bool


QWEN_GGUF_VARIANTS = (
    QwenGGUFVariant("qwen2", 2, False, True),
    QwenGGUFVariant("qwen3", 3, False, True),
    QwenGGUFVariant("qwen2moe", 2, True, False),
    QwenGGUFVariant("qwen3moe", 3, True, False),
)


@dataclass(frozen=True, slots=True)
class QwenAttentionGeometry:
    head_count: int
    kv_head_count: int
    head_width: int
    kv_width: int
    query_heads_per_kv: int


@dataclass(frozen=True, slots=True)
class QwenLayerSurgeryLayout:
    layer_index: int
    tensors: tuple[TensorMapping, ...]
    coupling_groups: tuple[CouplingGroup, ...]

    def tensor(self, role: TensorRole) -> TensorMapping:
        matches = tuple(item for item in self.tensors if item.role is role)
        if len(matches) != 1:
            raise QwenGGUFSurgeryError(
                f"Qwen block {self.layer_index} requires exactly one {role.value} "
                f"weight mapping, found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class QwenGGUFSurgeryAdapter:
    variant: QwenGGUFVariant
    architecture: ResolvedGGUFArchitecture
    shape: GGUFModelShape
    attention: QwenAttentionGeometry
    layers: tuple[QwenLayerSurgeryLayout, ...]
    tensor_names: tuple[str, ...]
    contract_version: int = QWEN_GGUF_SURGERY_CONTRACT_VERSION

    def layer(self, index: int) -> QwenLayerSurgeryLayout:
        if index < 0 or index >= len(self.layers):
            raise QwenGGUFSurgeryError(
                f"Qwen block {index} is outside block_count {len(self.layers)}"
            )
        return self.layers[index]

    def metadata_updates(
        self, changes: tuple[tuple[MetadataSemantic, int], ...]
    ) -> tuple[tuple[str, int], ...]:
        return self.architecture.metadata_updates(changes)

    def to_record(self) -> dict[str, str | int | bool]:
        return {
            "family": ModelFamily.QWEN.value,
            "architecture": self.variant.architecture,
            "generation": self.variant.generation,
            "mixture_of_experts": self.variant.mixture_of_experts,
            "contract_version": self.contract_version,
            "layer_count": len(self.layers),
            "head_count": self.attention.head_count,
            "kv_head_count": self.attention.kv_head_count,
            "head_width": self.attention.head_width,
            "query_heads_per_kv": self.attention.query_heads_per_kv,
        }


_REQUIRED_LAYER_ROLES = (
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


def _variant(architecture: str) -> QwenGGUFVariant:
    try:
        variant = next(item for item in QWEN_GGUF_VARIANTS if item.architecture == architecture)
    except StopIteration as error:
        raise QwenGGUFSurgeryError(
            f"unsupported Qwen GGUF architecture variant {architecture!r}"
        ) from error
    if not variant.native_surgery:
        raise QwenGGUFSurgeryError(
            f"Qwen architecture {architecture!r} is MoE and requires explicit "
            "expert/router physical mappings before native surgery"
        )
    return variant


def _geometry(shape: GGUFModelShape) -> QwenAttentionGeometry:
    if shape.attention_heads % shape.kv_heads:
        raise QwenGGUFSurgeryError(
            f"Qwen head count {shape.attention_heads} must be divisible by KV head "
            f"count {shape.kv_heads}"
        )
    head_width = shape.embedding_length // shape.attention_heads
    return QwenAttentionGeometry(
        shape.attention_heads,
        shape.kv_heads,
        head_width,
        head_width * shape.kv_heads,
        shape.attention_heads // shape.kv_heads,
    )


def _layer(
    discovery: GGUFDiscovery,
    architecture: ResolvedGGUFArchitecture,
    index: int,
) -> QwenLayerSurgeryLayout:
    prefix = f"blk.{index}."
    mappings = tuple(
        tensor.mapping
        for tensor in discovery.tensors
        if tensor.descriptor.name.startswith(prefix)
        and tensor.descriptor.name.endswith(".weight")
    )
    for role in _REQUIRED_LAYER_ROLES:
        count = sum(mapping.role is role for mapping in mappings)
        if count != 1:
            raise QwenGGUFSurgeryError(
                f"Qwen block {index} requires exactly one {role.value} weight, found {count}"
            )
    groups = architecture.coupling_groups(index)
    names = {mapping.tensor_name for mapping in mappings}
    missing = sorted(
        target.tensor_name
        for group in groups
        for target in group.targets
        if target.tensor_name not in names
    )
    if missing:
        raise QwenGGUFSurgeryError(
            f"Qwen block {index} coupling targets are missing: " + ", ".join(missing)
        )
    return QwenLayerSurgeryLayout(index, mappings, groups)


def build_qwen_gguf_surgery_adapter(discovery: GGUFDiscovery) -> QwenGGUFSurgeryAdapter:
    """Build an explicit dense Qwen2/Qwen3 physical-surgery view."""

    if discovery.family is not ModelFamily.QWEN:
        raise QwenGGUFSurgeryError("Qwen GGUF surgery requires explicit family qwen")
    if discovery.contract_version != QWEN_GGUF_SURGERY_CONTRACT_VERSION:
        raise QwenGGUFSurgeryError(
            f"unsupported Qwen GGUF contract version {discovery.contract_version}; "
            f"expected {QWEN_GGUF_SURGERY_CONTRACT_VERSION}"
        )
    variant = _variant(discovery.architecture)
    architecture = resolve_gguf_architecture(
        variant.architecture, family=ModelFamily.QWEN
    )
    layers = tuple(
        _layer(discovery, architecture, index) for index in range(discovery.shape.layers)
    )
    return QwenGGUFSurgeryAdapter(
        variant,
        architecture,
        discovery.shape,
        _geometry(discovery.shape),
        layers,
        tuple(sorted(tensor.descriptor.name for tensor in discovery.tensors)),
    )


def load_qwen_gguf_surgery_adapter(container: GGUFContainer) -> QwenGGUFSurgeryAdapter:
    discovery = discover_gguf_components(container, family=ModelFamily.QWEN)
    return build_qwen_gguf_surgery_adapter(discovery)
