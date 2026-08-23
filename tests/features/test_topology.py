from modelsurgeon.features.topology import extract_topology_features
from modelsurgeon.graph import (
    ComponentGraph,
    ComponentId,
    ConstraintKind,
    EdgeKind,
    GraphEdge,
    GraphNode,
    GraphProvenance,
    MutationConstraint,
    dump_component_graph,
    load_component_graph,
)


def _graph() -> ComponentGraph:
    model = ComponentId.parse("model")
    layers = ComponentId.parse("model.layers")
    layer0 = ComponentId.parse("model.layers.0")
    attention0 = ComponentId.parse("model.layers.0.self_attn")
    q_proj = ComponentId.parse("model.layers.0.self_attn.q_proj")
    k_proj = ComponentId.parse("model.layers.0.self_attn.k_proj")
    coupled_left, coupled_right = sorted((q_proj, k_proj))
    layer1 = ComponentId.parse("model.layers.1")
    nodes = (
        GraphNode(model, "model"),
        GraphNode(layers, "module"),
        GraphNode(layer0, "transformer_layer"),
        GraphNode(attention0, "attention", (("layer_index", 0),)),
        GraphNode(q_proj, "projection", (("layer_index", 0),)),
        GraphNode(k_proj, "projection", (("layer_index", 0),)),
        GraphNode(layer1, "transformer_layer"),
    )
    edges = (
        GraphEdge(layers, model, EdgeKind.PARENT),
        GraphEdge(model, layers, EdgeKind.CHILD),
        GraphEdge(layer0, layers, EdgeKind.PARENT),
        GraphEdge(layers, layer0, EdgeKind.CHILD),
        GraphEdge(attention0, layer0, EdgeKind.PARENT),
        GraphEdge(layer0, attention0, EdgeKind.CHILD),
        GraphEdge(q_proj, attention0, EdgeKind.PARENT),
        GraphEdge(attention0, q_proj, EdgeKind.CHILD),
        GraphEdge(k_proj, attention0, EdgeKind.PARENT),
        GraphEdge(attention0, k_proj, EdgeKind.CHILD),
        GraphEdge(layer1, layers, EdgeKind.PARENT),
        GraphEdge(layers, layer1, EdgeKind.CHILD),
        GraphEdge(coupled_left, coupled_right, EdgeKind.COUPLED),
    )
    constraints = (
        MutationConstraint(
            "layer0-head-set",
            ConstraintKind.SAME_HEAD_SET,
            tuple(sorted((q_proj, k_proj))),
        ),
    )
    return ComponentGraph.build(nodes, edges, constraints)


def test_extracts_depth_positions_degree_coupling_roles_and_layer_index() -> None:
    features = {item.component_id: item for item in extract_topology_features(_graph())}

    root = features[ComponentId.parse("model")]
    q_proj = features[ComponentId.parse("model.layers.0.self_attn.q_proj")]
    k_proj = features[ComponentId.parse("model.layers.0.self_attn.k_proj")]
    layer1 = features[ComponentId.parse("model.layers.1")]

    assert root.depth == 0
    assert root.position == 0
    assert root.normalized_position == 0.0
    assert q_proj.depth == 4
    assert q_proj.coupling_set_size == 2
    assert k_proj.coupling_set_size == 2
    assert q_proj.degree == 2
    assert "head_set" in q_proj.shape_roles
    assert "layer" in q_proj.shape_roles
    assert q_proj.layer_index == 0
    assert q_proj.normalized_layer_index == 0.0
    assert layer1.layer_index == 1
    assert layer1.normalized_layer_index == 1.0


def test_features_are_identical_after_graph_serialization_round_trip() -> None:
    graph = _graph()
    before = tuple(item.to_record() for item in extract_topology_features(graph))
    serialized = dump_component_graph(
        graph,
        GraphProvenance("fixture", "1", "revision"),
    )
    reloaded = load_component_graph(serialized).graph
    after = tuple(item.to_record() for item in extract_topology_features(reloaded))

    assert after == before


def test_feature_records_include_shape_role_provenance() -> None:
    features = extract_topology_features(_graph())
    q_proj = next(
        item
        for item in features
        if item.component_id == ComponentId.parse("model.layers.0.self_attn.q_proj")
    )
    records = q_proj.feature_records()

    assert any(record.name == "topology_normalized_layer_index" for record in records)
    assert all("shape_roles" in dict(record.metadata) for record in records)
