"""Discover canonical component and coupling graphs from a GGUF descriptor index."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import reduce
from itertools import combinations
from operator import mul

from modelsurgeon.adapters.family import ModelFamily
from modelsurgeon.adapters.gguf.architecture import (
    AxisSemantic,
    CouplingKind,
    MetadataSemantic,
    ResolvedGGUFArchitecture,
    TensorMapping,
    TensorRole,
    resolve_gguf_architecture,
)
from modelsurgeon.adapters.gguf.container import (
    GGUFContainer,
    GGUFMetadataEntry,
    GGUFTensorDescriptor,
    GGUFValueType,
)
from modelsurgeon.graph import (
    ComponentGraph,
    ComponentId,
    ConstraintKind,
    EdgeKind,
    GraphEdge,
    GraphNode,
    MutationConstraint,
)

_MAX_PARAMETER_COUNT = (1 << 63) - 1
_LAYER_NAME = re.compile(r"blk\.(?P<layer>\d+)\.")


class GGUFDiscoveryError(ValueError):
    """Base error for GGUF metadata, tensor, or graph reconciliation failures."""


class MissingGGUFMetadataError(GGUFDiscoveryError):
    """Raised when required explicit architecture metadata is absent or mistyped."""


class MissingGGUFTensorError(GGUFDiscoveryError):
    """Raised when an architecture-required physical tensor is absent."""


class GGUFTensorShapeError(GGUFDiscoveryError):
    """Raised when descriptor dimensions disagree with architecture metadata."""


@dataclass(frozen=True, slots=True)
class GGUFModelShape:
    layers: int
    embedding_length: int
    feed_forward_length: int
    attention_heads: int
    kv_heads: int
    key_head_length: int | None = None
    value_head_length: int | None = None

    @property
    def key_length(self) -> int:
        return self.key_head_length or self.embedding_length // self.attention_heads

    @property
    def value_length(self) -> int:
        return self.value_head_length or self.embedding_length // self.attention_heads


@dataclass(frozen=True, slots=True)
class GGUFTensorComponent:
    descriptor: GGUFTensorDescriptor
    mapping: TensorMapping
    component_id: ComponentId
    element_count: int


@dataclass(frozen=True, slots=True)
class GGUFDiscovery:
    family: ModelFamily
    architecture: str
    contract_version: int
    shape: GGUFModelShape
    tensors: tuple[GGUFTensorComponent, ...]
    parameter_count: int
    storage_bytes: int
    graph: ComponentGraph

    def to_record(self) -> dict[str, str | int]:
        return {
            "family": self.family.value,
            "architecture": self.architecture,
            "contract_version": self.contract_version,
            "layer_count": self.shape.layers,
            "tensor_count": len(self.tensors),
            "parameter_count": self.parameter_count,
            "storage_bytes": self.storage_bytes,
            "graph_node_count": len(self.graph.nodes),
            "coupling_count": len(self.graph.constraints),
        }


def _required_entry(container: GGUFContainer, key: str) -> GGUFMetadataEntry:
    entry = container.metadata_entry(key)
    if entry is None:
        raise MissingGGUFMetadataError(f"required GGUF metadata {key!r} is missing")
    return entry


def _required_string(container: GGUFContainer, key: str) -> str:
    entry = _required_entry(container, key)
    if entry.value_type is not GGUFValueType.STRING or not isinstance(entry.value, str):
        raise MissingGGUFMetadataError(f"GGUF metadata {key!r} must be a string")
    if not entry.value:
        raise MissingGGUFMetadataError(f"GGUF metadata {key!r} must not be empty")
    return entry.value


def _required_positive_int(container: GGUFContainer, key: str) -> int:
    entry = _required_entry(container, key)
    if entry.value_type not in (GGUFValueType.UINT32, GGUFValueType.UINT64):
        raise MissingGGUFMetadataError(f"GGUF metadata {key!r} must be UINT32 or UINT64")
    if not isinstance(entry.value, int) or isinstance(entry.value, bool) or entry.value <= 0:
        raise MissingGGUFMetadataError(f"GGUF metadata {key!r} must be a positive integer")
    return entry.value


def _optional_positive_int(container: GGUFContainer, key: str) -> int | None:
    if container.metadata_entry(key) is None:
        return None
    return _required_positive_int(container, key)


def _model_shape(
    container: GGUFContainer,
    architecture: ResolvedGGUFArchitecture,
) -> GGUFModelShape:
    def value(semantic: MetadataSemantic) -> int:
        return _required_positive_int(container, architecture.metadata_key(semantic))

    heads = value(MetadataSemantic.HEAD_COUNT)
    kv_heads = value(MetadataSemantic.KV_HEAD_COUNT)
    embedding = value(MetadataSemantic.EMBEDDING_LENGTH)
    key_head_length = _optional_positive_int(
        container, architecture.metadata_key(MetadataSemantic.KEY_LENGTH)
    )
    value_head_length = _optional_positive_int(
        container, architecture.metadata_key(MetadataSemantic.VALUE_LENGTH)
    )
    if kv_heads > heads:
        raise GGUFTensorShapeError(
            f"KV head count {kv_heads} exceeds attention head count {heads}"
        )
    if (key_head_length is None or value_head_length is None) and embedding % heads:
        raise GGUFTensorShapeError(
            f"embedding length {embedding} is not divisible by attention head count "
            f"{heads} and explicit head lengths are incomplete"
        )
    return GGUFModelShape(
        layers=value(MetadataSemantic.BLOCK_COUNT),
        embedding_length=embedding,
        feed_forward_length=value(MetadataSemantic.FEED_FORWARD_LENGTH),
        attention_heads=heads,
        kv_heads=kv_heads,
        key_head_length=key_head_length,
        value_head_length=value_head_length,
    )


def _expected_axis_size(
    semantic: AxisSemantic,
    role: TensorRole,
    shape: GGUFModelShape,
) -> int | None:
    if semantic in {
        AxisSemantic.INPUT_FEATURE,
        AxisSemantic.OUTPUT_FEATURE,
        AxisSemantic.HIDDEN_FEATURE,
    }:
        return shape.embedding_length
    if semantic is AxisSemantic.ATTENTION_HEAD:
        return (
            shape.value_length
            if role is TensorRole.ATTENTION_O
            else shape.key_length
        ) * shape.attention_heads
    if semantic is AxisSemantic.KV_HEAD:
        return (
            shape.value_length
            if role is TensorRole.ATTENTION_V
            else shape.key_length
        ) * shape.kv_heads
    if semantic is AxisSemantic.MLP_CHANNEL:
        return shape.feed_forward_length
    return None


def _parameter_id(mapping: TensorMapping) -> ComponentId:
    suffix = mapping.tensor_name.rsplit(".", maxsplit=1)[-1]
    return mapping.component_id.child(suffix)


def _map_tensor(
    descriptor: GGUFTensorDescriptor,
    architecture: ResolvedGGUFArchitecture,
    shape: GGUFModelShape,
) -> GGUFTensorComponent:
    try:
        mapping = architecture.map_tensor(descriptor.name)
    except ValueError as error:
        raise GGUFDiscoveryError(str(error)) from error
    if len(mapping.axes) != len(descriptor.dimensions):
        raise GGUFTensorShapeError(
            f"tensor {descriptor.name!r} rank {len(descriptor.dimensions)} does not match "
            f"the {len(mapping.axes)} declared architecture axes"
        )
    if tuple(axis.index for axis in mapping.axes) != tuple(range(len(mapping.axes))):
        raise GGUFTensorShapeError(
            f"tensor {descriptor.name!r} architecture axes are not contiguous"
        )
    for axis in mapping.axes:
        expected = _expected_axis_size(axis.semantic, mapping.role, shape)
        actual = descriptor.dimensions[axis.index]
        if expected is not None and actual != expected:
            raise GGUFTensorShapeError(
                f"tensor {descriptor.name!r} axis {axis.index} ({axis.semantic.value}) "
                f"has size {actual}, expected {expected}"
            )
    element_count = reduce(mul, descriptor.dimensions, 1)
    return GGUFTensorComponent(
        descriptor,
        mapping,
        _parameter_id(mapping),
        element_count,
    )


def _node_kind(component_id: ComponentId, tensor: GGUFTensorComponent | None = None) -> str:
    if tensor is not None:
        return "parameter"
    text = str(component_id)
    leaf = component_id.segments[-1].value
    if text == "model":
        return "model"
    if re.fullmatch(r"model\.layers\.\d+", text):
        return "transformer_layer"
    if leaf == "self_attn":
        return "attention"
    if leaf == "mlp":
        return "mlp"
    if leaf in {"input_layernorm", "post_attention_layernorm", "norm"}:
        return "normalization"
    if leaf in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}:
        return "projection"
    if leaf == "embed_tokens":
        return "embedding"
    if leaf == "lm_head":
        return "output_projection"
    return "module"


def _tensor_attributes(tensor: GGUFTensorComponent) -> tuple[tuple[str, str | int], ...]:
    attributes: list[tuple[str, str | int]] = [
        ("physical_name", tensor.descriptor.name),
        ("role", tensor.mapping.role.value),
        ("ggml_type", tensor.descriptor.quant_type.value),
        ("rank", len(tensor.descriptor.dimensions)),
        ("element_count", tensor.element_count),
        ("storage_bytes", tensor.descriptor.byte_size),
    ]
    for axis, dimension in zip(
        tensor.mapping.axes, tensor.descriptor.dimensions, strict=True
    ):
        attributes.append((f"dimension_{axis.index}", dimension))
        attributes.append((f"axis_{axis.index}", axis.semantic.value))
    return tuple(attributes)


def _required_tensor_names(shape: GGUFModelShape) -> tuple[str, ...]:
    names = ["token_embd.weight", "output_norm.weight"]
    for layer in range(shape.layers):
        prefix = f"blk.{layer}"
        names.extend(
            (
                f"{prefix}.attn_norm.weight",
                f"{prefix}.ffn_norm.weight",
                f"{prefix}.attn_q.weight",
                f"{prefix}.attn_k.weight",
                f"{prefix}.attn_v.weight",
                f"{prefix}.attn_output.weight",
                f"{prefix}.ffn_gate.weight",
                f"{prefix}.ffn_up.weight",
                f"{prefix}.ffn_down.weight",
            )
        )
    return tuple(names)


def _hierarchy_edges(nodes: dict[ComponentId, GraphNode]) -> list[GraphEdge]:
    edges: list[GraphEdge] = []
    for component_id in sorted(nodes):
        parent = component_id.parent
        if parent is None:
            continue
        edges.append(GraphEdge(component_id, parent, EdgeKind.PARENT))
        edges.append(GraphEdge(parent, component_id, EdgeKind.CHILD))
    return edges


def _build_graph(
    tensors: tuple[GGUFTensorComponent, ...],
    architecture: ResolvedGGUFArchitecture,
    shape: GGUFModelShape,
    parameter_count: int,
    storage_bytes: int,
) -> ComponentGraph:
    nodes: dict[ComponentId, GraphNode] = {
        ComponentId.parse("model"): GraphNode(
            ComponentId.parse("model"),
            "model",
            (
                ("family", architecture.contract.family.value),
                ("architecture", architecture.architecture),
                ("parameter_count", parameter_count),
                ("storage_bytes", storage_bytes),
                ("tensor_count", len(tensors)),
            ),
        )
    }
    tensor_by_name: dict[str, GGUFTensorComponent] = {}
    for tensor in tensors:
        if tensor.component_id in nodes:
            raise GGUFDiscoveryError(
                f"multiple physical tensors map to {tensor.component_id}"
            )
        tensor_by_name[tensor.descriptor.name] = tensor
        nodes[tensor.component_id] = GraphNode(
            tensor.component_id,
            _node_kind(tensor.component_id, tensor),
            _tensor_attributes(tensor),
        )
        ancestor = tensor.component_id.parent
        while ancestor is not None:
            if ancestor not in nodes:
                attributes: tuple[tuple[str, str | int], ...] = ()
                if _node_kind(ancestor) == "transformer_layer":
                    attributes = (("layer_index", int(ancestor.segments[-1].value)),)
                nodes[ancestor] = GraphNode(ancestor, _node_kind(ancestor), attributes)
            ancestor = ancestor.parent

    edges = _hierarchy_edges(nodes)
    constraints: list[MutationConstraint] = []
    constraint_kind = {
        CouplingKind.ATTENTION_HEAD: ConstraintKind.SAME_HEAD_SET,
        CouplingKind.KV_HEAD: ConstraintKind.SAME_HEAD_SET,
        CouplingKind.MLP_CHANNEL: ConstraintKind.SAME_HIDDEN_SIZE,
    }
    for layer in range(shape.layers):
        for group in architecture.coupling_groups(layer):
            members: list[ComponentId] = []
            parameters: list[tuple[str, str | int]] = [
                ("coupling_kind", group.kind.value)
            ]
            for index, target in enumerate(group.targets):
                indexed_tensor = tensor_by_name.get(target.tensor_name)
                if indexed_tensor is None:
                    raise MissingGGUFTensorError(
                        f"coupling group {group.group_id!r} requires tensor "
                        f"{target.tensor_name!r}"
                    )
                if indexed_tensor.mapping.role is not target.role:
                    raise GGUFDiscoveryError(
                        f"tensor {target.tensor_name!r} has role "
                        f"{indexed_tensor.mapping.role.value}, "
                        f"expected {target.role.value}"
                    )
                if target.axis >= len(indexed_tensor.mapping.axes):
                    raise GGUFTensorShapeError(
                        f"coupling target {target.tensor_name!r} axis {target.axis} is outside "
                        f"rank {len(indexed_tensor.mapping.axes)}"
                    )
                members.append(indexed_tensor.component_id)
                parameters.append(
                    (f"target_{index}", f"{target.tensor_name}:axis:{target.axis}")
                )
            ordered_members = tuple(sorted(members))
            constraints.append(
                MutationConstraint(
                    group.group_id,
                    constraint_kind[group.kind],
                    ordered_members,
                    parameters=tuple(parameters),
                )
            )
            for left, right in combinations(ordered_members, 2):
                edges.append(GraphEdge(left, right, EdgeKind.COUPLED))
                edges.append(
                    GraphEdge(
                        left,
                        right,
                        EdgeKind.CONSTRAINED,
                        attributes=(("constraint_id", group.group_id),),
                    )
                )
    return ComponentGraph.build(tuple(nodes.values()), tuple(edges), tuple(constraints))


def discover_gguf_components(
    container: GGUFContainer,
    *,
    family: ModelFamily | None = None,
) -> GGUFDiscovery:
    """Reconcile GGUF metadata and descriptors into a canonical physical graph."""
    architecture_name = _required_string(container, "general.architecture")
    try:
        architecture = resolve_gguf_architecture(architecture_name, family=family)
    except ValueError as error:
        raise GGUFDiscoveryError(str(error)) from error
    shape = _model_shape(container, architecture)

    physical_names = {tensor.name for tensor in container.tensors}
    missing = [name for name in _required_tensor_names(shape) if name not in physical_names]
    if missing:
        raise MissingGGUFTensorError(
            "GGUF architecture-required tensors are missing: " + ", ".join(missing)
        )
    for name in physical_names:
        matched = _LAYER_NAME.match(name)
        if matched is not None and int(matched.group("layer")) >= shape.layers:
            raise GGUFTensorShapeError(
                f"tensor {name!r} references a block outside block_count {shape.layers}"
            )

    tensors = tuple(
        _map_tensor(descriptor, architecture, shape)
        for descriptor in container.tensors
    )
    parameter_count = sum(tensor.element_count for tensor in tensors)
    storage_bytes = sum(tensor.descriptor.byte_size for tensor in tensors)
    if parameter_count > _MAX_PARAMETER_COUNT:
        raise GGUFDiscoveryError("GGUF parameter total exceeds signed 64-bit range")
    graph = _build_graph(
        tensors,
        architecture,
        shape,
        parameter_count,
        storage_bytes,
    )
    graph_parameter_count = 0
    for node in graph.nodes:
        if node.kind != "parameter":
            continue
        element_count = dict(node.attributes).get("element_count")
        if not isinstance(element_count, int) or isinstance(element_count, bool):
            raise GGUFDiscoveryError(
                f"parameter node {node.component_id} has no integer element count"
            )
        graph_parameter_count += element_count
    if graph_parameter_count != parameter_count:
        raise GGUFDiscoveryError(
            f"graph parameter total {graph_parameter_count} does not reconcile with "
            f"descriptor total {parameter_count}"
        )
    return GGUFDiscovery(
        architecture.contract.family,
        architecture.architecture,
        architecture.contract.version,
        shape,
        tensors,
        parameter_count,
        storage_bytes,
        graph,
    )
