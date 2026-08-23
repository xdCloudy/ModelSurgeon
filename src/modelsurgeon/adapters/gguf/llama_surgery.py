"""Fail-closed Llama GGUF architecture view for physical surgery planning."""

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

LLAMA_GGUF_SURGERY_CONTRACT_VERSION = 1


class LlamaGGUFSurgeryError(GGUFArchitectureError):
    """Raised when a GGUF cannot be represented by the Llama surgery contract."""


@dataclass(frozen=True, slots=True)
class LlamaAttentionGeometry:
    """Validated MHA/GQA widths used by physical attention edits."""

    head_count: int
    kv_head_count: int
    head_width: int
    query_width: int
    kv_width: int
    query_heads_per_kv: int


@dataclass(frozen=True, slots=True)
class LlamaLayerSurgeryLayout:
    """All required physical mappings and coupled axes for one Llama block."""

    layer_index: int
    tensors: tuple[TensorMapping, ...]
    coupling_groups: tuple[CouplingGroup, ...]

    def tensor(self, role: TensorRole) -> TensorMapping:
        matches = tuple(item for item in self.tensors if item.role is role)
        if len(matches) != 1:
            raise LlamaGGUFSurgeryError(
                f"Llama block {self.layer_index} requires exactly one {role.value} "
                f"weight mapping, found {len(matches)}"
            )
        return matches[0]


@dataclass(frozen=True, slots=True)
class LlamaGGUFSurgeryAdapter:
    """Versioned, serializable-shape Llama mapping for downstream surgery plans."""

    architecture: ResolvedGGUFArchitecture
    shape: GGUFModelShape
    attention: LlamaAttentionGeometry
    layers: tuple[LlamaLayerSurgeryLayout, ...]
    tensor_names: tuple[str, ...]
    output_weight_present: bool
    contract_version: int = LLAMA_GGUF_SURGERY_CONTRACT_VERSION

    def layer(self, index: int) -> LlamaLayerSurgeryLayout:
        if index < 0 or index >= len(self.layers):
            raise LlamaGGUFSurgeryError(
                f"Llama block {index} is outside block_count {len(self.layers)}"
            )
        return self.layers[index]

    def metadata_updates(
        self, changes: tuple[tuple[MetadataSemantic, int], ...]
    ) -> tuple[tuple[str, int], ...]:
        return self.architecture.metadata_updates(changes)

    def rename_tensor_blocks(
        self, tensor_name: str, layer_map: tuple[tuple[int, int | None], ...]
    ) -> str | None:
        return self.architecture.rename_tensor_blocks(tensor_name, layer_map)

    def to_record(self) -> dict[str, str | int | bool]:
        return {
            "family": ModelFamily.LLAMA.value,
            "architecture": self.architecture.architecture,
            "contract_version": self.contract_version,
            "layer_count": len(self.layers),
            "head_count": self.attention.head_count,
            "kv_head_count": self.attention.kv_head_count,
            "head_width": self.attention.head_width,
            "query_heads_per_kv": self.attention.query_heads_per_kv,
            "output_weight_present": self.output_weight_present,
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


def _attention_geometry(shape: GGUFModelShape) -> LlamaAttentionGeometry:
    if shape.attention_heads % shape.kv_heads:
        raise LlamaGGUFSurgeryError(
            f"Llama attention head count {shape.attention_heads} must be divisible by "
            f"KV head count {shape.kv_heads} for grouped-query attention"
        )
    head_width = shape.embedding_length // shape.attention_heads
    return LlamaAttentionGeometry(
        shape.attention_heads,
        shape.kv_heads,
        head_width,
        shape.embedding_length,
        head_width * shape.kv_heads,
        shape.attention_heads // shape.kv_heads,
    )


def _layer_layout(
    discovery: GGUFDiscovery,
    architecture: ResolvedGGUFArchitecture,
    layer: int,
) -> LlamaLayerSurgeryLayout:
    prefix = f"blk.{layer}."
    mappings = tuple(
        tensor.mapping
        for tensor in discovery.tensors
        if tensor.descriptor.name.startswith(prefix)
        and tensor.descriptor.name.endswith(".weight")
    )
    for role in _REQUIRED_LAYER_ROLES:
        matches = [mapping for mapping in mappings if mapping.role is role]
        if len(matches) != 1:
            raise LlamaGGUFSurgeryError(
                f"Llama block {layer} requires exactly one {role.value} weight, "
                f"found {len(matches)}"
            )
    groups = architecture.coupling_groups(layer)
    names = {mapping.tensor_name for mapping in mappings}
    for group in groups:
        missing = [
            target.tensor_name
            for target in group.targets
            if target.tensor_name not in names
        ]
        if missing:
            raise LlamaGGUFSurgeryError(
                f"Llama coupling group {group.group_id!r} is missing: " + ", ".join(missing)
            )
    return LlamaLayerSurgeryLayout(layer, mappings, groups)


def build_llama_gguf_surgery_adapter(
    discovery: GGUFDiscovery,
) -> LlamaGGUFSurgeryAdapter:
    """Build the strict Llama surgery view from reconciled GGUF discovery."""

    if discovery.family is not ModelFamily.LLAMA or discovery.architecture != "llama":
        raise LlamaGGUFSurgeryError(
            "Llama GGUF surgery requires explicit family llama and architecture llama"
        )
    if discovery.contract_version != LLAMA_GGUF_SURGERY_CONTRACT_VERSION:
        raise LlamaGGUFSurgeryError(
            f"unsupported Llama GGUF contract version {discovery.contract_version}; "
            f"expected {LLAMA_GGUF_SURGERY_CONTRACT_VERSION}"
        )
    architecture = resolve_gguf_architecture("llama", family=ModelFamily.LLAMA)
    names = tuple(sorted(tensor.descriptor.name for tensor in discovery.tensors))
    required_global = {"token_embd.weight", "output_norm.weight"}
    missing_global = sorted(required_global.difference(names))
    if missing_global:
        raise LlamaGGUFSurgeryError(
            "Llama GGUF is missing required global tensors: " + ", ".join(missing_global)
        )
    layers = tuple(
        _layer_layout(discovery, architecture, layer)
        for layer in range(discovery.shape.layers)
    )
    return LlamaGGUFSurgeryAdapter(
        architecture,
        discovery.shape,
        _attention_geometry(discovery.shape),
        layers,
        names,
        "output.weight" in names,
    )


def load_llama_gguf_surgery_adapter(container: GGUFContainer) -> LlamaGGUFSurgeryAdapter:
    """Discover and validate one GGUF as an explicit Llama surgery target."""

    discovery = discover_gguf_components(container, family=ModelFamily.LLAMA)
    return build_llama_gguf_surgery_adapter(discovery)
