"""Mistral GGUF surgery adapter with explicit sliding-window metadata."""

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
from modelsurgeon.adapters.gguf.container import GGUFContainer, GGUFValueType
from modelsurgeon.adapters.gguf.discovery import (
    GGUFDiscovery,
    GGUFModelShape,
    discover_gguf_components,
)

MISTRAL_GGUF_SURGERY_CONTRACT_VERSION = 1


class MistralGGUFSurgeryError(GGUFArchitectureError):
    """Raised when a Mistral file cannot be safely mapped for surgery."""


@dataclass(frozen=True, slots=True)
class MistralAttentionGeometry:
    head_count: int
    kv_head_count: int
    head_width: int
    kv_width: int
    query_heads_per_kv: int
    sliding_window: int


@dataclass(frozen=True, slots=True)
class MistralLayerSurgeryLayout:
    layer_index: int
    tensors: tuple[TensorMapping, ...]
    coupling_groups: tuple[CouplingGroup, ...]


@dataclass(frozen=True, slots=True)
class MistralGGUFSurgeryAdapter:
    architecture: ResolvedGGUFArchitecture
    shape: GGUFModelShape
    attention: MistralAttentionGeometry
    layers: tuple[MistralLayerSurgeryLayout, ...]
    legacy_llama_prefix: bool
    contract_version: int = MISTRAL_GGUF_SURGERY_CONTRACT_VERSION

    def layer(self, index: int) -> MistralLayerSurgeryLayout:
        if index < 0 or index >= len(self.layers):
            raise MistralGGUFSurgeryError(
                f"Mistral block {index} is outside block_count {len(self.layers)}"
            )
        return self.layers[index]

    def to_record(self) -> dict[str, str | int | bool]:
        return {
            "family": ModelFamily.MISTRAL.value,
            "architecture": self.architecture.architecture,
            "metadata_prefix": self.architecture.metadata_prefix,
            "legacy_llama_prefix": self.legacy_llama_prefix,
            "contract_version": self.contract_version,
            "layer_count": len(self.layers),
            "head_count": self.attention.head_count,
            "kv_head_count": self.attention.kv_head_count,
            "head_width": self.attention.head_width,
            "query_heads_per_kv": self.attention.query_heads_per_kv,
            "sliding_window": self.attention.sliding_window,
        }


_REQUIRED_ROLES = (
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


def _geometry(shape: GGUFModelShape, window: int) -> MistralAttentionGeometry:
    if window <= 0:
        raise MistralGGUFSurgeryError("Mistral sliding window must be a positive integer")
    if shape.attention_heads % shape.kv_heads:
        raise MistralGGUFSurgeryError(
            f"Mistral head count {shape.attention_heads} must be divisible by KV head "
            f"count {shape.kv_heads}"
        )
    width = shape.embedding_length // shape.attention_heads
    return MistralAttentionGeometry(
        shape.attention_heads,
        shape.kv_heads,
        width,
        width * shape.kv_heads,
        shape.attention_heads // shape.kv_heads,
        window,
    )


def _layer(
    discovery: GGUFDiscovery, architecture: ResolvedGGUFArchitecture, index: int
) -> MistralLayerSurgeryLayout:
    prefix = f"blk.{index}."
    mappings = tuple(
        tensor.mapping
        for tensor in discovery.tensors
        if tensor.descriptor.name.startswith(prefix)
        and tensor.descriptor.name.endswith(".weight")
    )
    for role in _REQUIRED_ROLES:
        count = sum(mapping.role is role for mapping in mappings)
        if count != 1:
            raise MistralGGUFSurgeryError(
                f"Mistral block {index} requires exactly one {role.value} weight, "
                f"found {count}"
            )
    groups = architecture.coupling_groups(index)
    names = {mapping.tensor_name for mapping in mappings}
    if any(
        target.tensor_name not in names for group in groups for target in group.targets
    ):
        raise MistralGGUFSurgeryError(
            f"Mistral block {index} has an incomplete physical coupling group"
        )
    return MistralLayerSurgeryLayout(index, mappings, groups)


def build_mistral_gguf_surgery_adapter(
    discovery: GGUFDiscovery, *, sliding_window: int
) -> MistralGGUFSurgeryAdapter:
    if discovery.family is not ModelFamily.MISTRAL:
        raise MistralGGUFSurgeryError("Mistral GGUF surgery requires explicit family mistral")
    if discovery.contract_version != MISTRAL_GGUF_SURGERY_CONTRACT_VERSION:
        raise MistralGGUFSurgeryError(
            f"unsupported Mistral GGUF contract version {discovery.contract_version}"
        )
    if discovery.architecture not in {"mistral", "llama"}:
        raise MistralGGUFSurgeryError(
            f"unsupported Mistral GGUF architecture {discovery.architecture!r}"
        )
    architecture = resolve_gguf_architecture(
        discovery.architecture, family=ModelFamily.MISTRAL
    )
    return MistralGGUFSurgeryAdapter(
        architecture,
        discovery.shape,
        _geometry(discovery.shape, sliding_window),
        tuple(
            _layer(discovery, architecture, index)
            for index in range(discovery.shape.layers)
        ),
        discovery.architecture == "llama",
    )


def load_mistral_gguf_surgery_adapter(
    container: GGUFContainer,
) -> MistralGGUFSurgeryAdapter:
    discovery = discover_gguf_components(container, family=ModelFamily.MISTRAL)
    architecture = resolve_gguf_architecture(
        discovery.architecture, family=ModelFamily.MISTRAL
    )
    key = f"{architecture.metadata_prefix}.attention.sliding_window"
    entry = container.metadata_entry(key)
    if (
        entry is None
        or entry.value_type not in {GGUFValueType.UINT32, GGUFValueType.UINT64}
        or not isinstance(entry.value, int)
        or isinstance(entry.value, bool)
        or entry.value <= 0
    ):
        raise MistralGGUFSurgeryError(
            f"required Mistral GGUF metadata {key!r} must be a positive integer"
        )
    return build_mistral_gguf_surgery_adapter(
        discovery, sliding_window=entry.value
    )
