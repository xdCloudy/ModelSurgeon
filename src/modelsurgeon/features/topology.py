"""Deterministic topology features derived only from the persisted component graph."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.graph.schema import ComponentGraph, ConstraintKind, EdgeKind, GraphNode

TOPOLOGY_FEATURE_EXTRACTOR_VERSION = "1"


class TopologyFeatureError(ValueError):
    """Raised when a component graph cannot yield deterministic topology features."""


@dataclass(frozen=True, slots=True)
class TopologyFeatures:
    component_id: ComponentId
    depth: int
    position: int
    normalized_position: float
    sibling_position: int
    normalized_sibling_position: float
    degree: int
    in_degree: int
    out_degree: int
    coupling_set_size: int
    shape_roles: tuple[str, ...]
    layer_index: int | None
    normalized_layer_index: float | None

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "depth": self.depth,
            "position": self.position,
            "normalized_position": self.normalized_position,
            "sibling_position": self.sibling_position,
            "normalized_sibling_position": self.normalized_sibling_position,
            "degree": self.degree,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "coupling_set_size": self.coupling_set_size,
            "shape_roles": list(self.shape_roles),
            "layer_index": self.layer_index,
            "normalized_layer_index": self.normalized_layer_index,
        }

    def feature_records(self) -> tuple[FeatureRecord, ...]:
        precision = PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            "graph_schema_v1",
            "float64",
        )
        metadata = (("shape_roles", "|".join(self.shape_roles)),)
        values = (
            ("topology_depth", float(self.depth)),
            ("topology_position", float(self.position)),
            ("topology_normalized_position", self.normalized_position),
            ("topology_sibling_position", float(self.sibling_position)),
            ("topology_normalized_sibling_position", self.normalized_sibling_position),
            ("topology_degree", float(self.degree)),
            ("topology_in_degree", float(self.in_degree)),
            ("topology_out_degree", float(self.out_degree)),
            ("topology_coupling_set_size", float(self.coupling_set_size)),
        )
        records = [
            FeatureRecord(
                self.component_id,
                name,
                FeatureKind.SCALAR,
                value,
                "float64",
                "graph_topology",
                TOPOLOGY_FEATURE_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            )
            for name, value in values
        ]
        if self.normalized_layer_index is not None:
            records.append(
                FeatureRecord(
                    self.component_id,
                    "topology_normalized_layer_index",
                    FeatureKind.SCALAR,
                    self.normalized_layer_index,
                    "float64",
                    "graph_topology",
                    TOPOLOGY_FEATURE_EXTRACTOR_VERSION,
                    precision,
                    metadata=(("layer_index", self.layer_index), *metadata),
                )
            )
        return tuple(records)


def _layer_index(component_id: ComponentId, node: GraphNode) -> int | None:
    explicit = dict(node.attributes).get("layer_index")
    if isinstance(explicit, int) and not isinstance(explicit, bool) and explicit >= 0:
        return explicit
    segments = tuple(segment.value for segment in component_id)
    for index, segment in enumerate(segments[:-1]):
        if segment != "layers":
            continue
        candidate = segments[index + 1]
        if isinstance(candidate, int):
            return candidate
    return None


def _shape_roles(graph: ComponentGraph, node: GraphNode) -> tuple[str, ...]:
    roles = {f"kind:{node.kind}"}
    attributes = dict(node.attributes)
    if "element_count" in attributes:
        roles.add("parameter_elements")
    for key in attributes:
        if key.endswith("_index"):
            roles.add(key.removesuffix("_index"))
    for constraint in graph.constraints:
        if node.component_id not in constraint.members:
            continue
        if constraint.kind is ConstraintKind.SAME_HEAD_SET:
            roles.add("head_set")
        elif constraint.kind is ConstraintKind.SAME_HIDDEN_SIZE:
            roles.add("hidden_size")
        elif constraint.kind is ConstraintKind.SHAPE_EQUALITY:
            roles.add("shape_equality")
        elif constraint.kind is ConstraintKind.GROUPED_MUTATION:
            roles.add("grouped_mutation")
        else:
            roles.add(f"constraint:{constraint.kind.value}")
    return tuple(sorted(roles))


def _coupling_sizes(graph: ComponentGraph) -> dict[ComponentId, int]:
    adjacency: dict[ComponentId, set[ComponentId]] = defaultdict(set)
    for edge in graph.edges:
        if edge.kind is not EdgeKind.COUPLED:
            continue
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)

    sizes: dict[ComponentId, int] = {}
    for node in graph.nodes:
        origin = node.component_id
        if origin in sizes:
            continue
        seen = {origin}
        queue = deque([origin])
        while queue:
            current = queue.popleft()
            for neighbour in adjacency.get(current, ()):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                queue.append(neighbour)
        size = len(seen)
        for member in seen:
            sizes[member] = size
    return sizes


def extract_topology_features(graph: ComponentGraph) -> tuple[TopologyFeatures, ...]:
    """Extract stable topology features without consulting framework/model objects."""

    canonical = ComponentGraph.build(graph.nodes, graph.edges, graph.constraints)
    nodes = canonical.nodes
    if not nodes:
        raise TopologyFeatureError("topology extraction requires at least one graph node")

    node_ids = tuple(node.component_id for node in nodes)
    position_by_id = {component_id: index for index, component_id in enumerate(node_ids)}
    sibling_groups: dict[ComponentId | None, list[ComponentId]] = defaultdict(list)
    for component_id in node_ids:
        sibling_groups[component_id.parent].append(component_id)
    for siblings in sibling_groups.values():
        siblings.sort()

    in_edges: dict[ComponentId, int] = defaultdict(int)
    out_edges: dict[ComponentId, int] = defaultdict(int)
    neighbours: dict[ComponentId, set[ComponentId]] = defaultdict(set)
    for edge in canonical.edges:
        out_edges[edge.source] += 1
        in_edges[edge.target] += 1
        neighbours[edge.source].add(edge.target)
        neighbours[edge.target].add(edge.source)

    coupling_sizes = _coupling_sizes(canonical)
    layer_indices = sorted(
        {
            index
            for node in nodes
            if (index := _layer_index(node.component_id, node)) is not None
        }
    )
    layer_position = {value: index for index, value in enumerate(layer_indices)}

    denominator = max(1, len(nodes) - 1)
    output: list[TopologyFeatures] = []
    for node in nodes:
        component_id = node.component_id
        position = position_by_id[component_id]
        siblings = sibling_groups[component_id.parent]
        sibling_position = siblings.index(component_id)
        sibling_denominator = max(1, len(siblings) - 1)
        layer_index = _layer_index(component_id, node)
        normalized_layer_index: float | None = None
        if layer_index is not None:
            normalized_layer_index = (
                0.0
                if len(layer_indices) <= 1
                else layer_position[layer_index] / (len(layer_indices) - 1)
            )
        output.append(
            TopologyFeatures(
                component_id=component_id,
                depth=len(component_id.segments) - 1,
                position=position,
                normalized_position=position / denominator,
                sibling_position=sibling_position,
                normalized_sibling_position=sibling_position / sibling_denominator,
                degree=len(neighbours[component_id]),
                in_degree=in_edges[component_id],
                out_degree=out_edges[component_id],
                coupling_set_size=coupling_sizes[component_id],
                shape_roles=_shape_roles(canonical, node),
                layer_index=layer_index,
                normalized_layer_index=normalized_layer_index,
            )
        )
    return tuple(output)
