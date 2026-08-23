"""Tests for canonical seeded single-component mutation candidate enumeration."""

from __future__ import annotations

from modelsurgeon.experiments.candidates import (
    CandidateEnumeratorConfig,
    CandidateFilter,
    CandidateScope,
    enumerate_mutation_candidates,
)
from modelsurgeon.experiments.identity import derive_run_identity
from modelsurgeon.graph import (
    ComponentGraph,
    ComponentId,
    ConstraintKind,
    GraphNode,
    MutationConstraint,
)


def _id(value: str) -> ComponentId:
    return ComponentId.parse(value)


def _graph() -> ComponentGraph:
    nodes = (
        GraphNode(_id("model"), "model"),
        GraphNode(_id("model.layers.0"), "transformer_layer"),
        GraphNode(
            _id("model.layers.0.self_attn.head.0"),
            "attention_head",
            (("head_index", 0), ("layer_index", 0)),
        ),
        GraphNode(
            _id("model.layers.0.mlp.channel.0"),
            "mlp_channel",
            (("channel_index", 0), ("layer_index", 0)),
        ),
        GraphNode(_id("model.layers.0.self_attn.q_proj"), "projection"),
        GraphNode(_id("model.layers.0.input_norm"), "normalization"),
        GraphNode(_id("model.layers.0.weight"), "parameter"),
        GraphNode(_id("model.layers.1"), "transformer_layer"),
        GraphNode(
            _id("model.layers.1.self_attn.head.0"),
            "attention_head",
            (("head_index", 0), ("layer_index", 1)),
        ),
        GraphNode(
            _id("model.layers.1.mlp.channel.0"),
            "mlp_channel",
            (("channel_index", 0), ("layer_index", 1)),
        ),
        GraphNode(_id("model.layers.1.self_attn.q_proj"), "projection"),
        GraphNode(_id("model.layers.1.kv_head.0"), "kv_head"),
        GraphNode(_id("model.residual_path.0"), "residual_path"),
    )
    constraint = MutationConstraint(
        "q-proj-norm-coupling",
        ConstraintKind.GROUPED_MUTATION,
        tuple(sorted((_id("model.layers.0.self_attn.q_proj"), _id("model.layers.0.input_norm")))),
    )
    return ComponentGraph.build(nodes, constraints=(constraint,))


def _run_id() -> str:
    return derive_run_identity("exp_" + "a" * 64).run_id


def test_enumeration_is_deterministic_unique_and_uses_canonical_request_ids() -> None:
    config = CandidateEnumeratorConfig(seed=42)
    first = enumerate_mutation_candidates(_graph(), _run_id(), config)
    second = enumerate_mutation_candidates(_graph(), _run_id(), config)

    assert first.to_record() == second.to_record()
    assert first.eligible_count == 8
    assert len(first.candidates) == 8
    assert len({item.candidate_id for item in first.candidates}) == 8
    assert len({item.mutation_id for item in first.candidates}) == 8
    assert all(item.request.targets == (item.component_id,) for item in first.candidates)
    assert all(item.request.parameters[0][0] == "candidate_scope" for item in first.candidates)


def test_unsupported_nodes_are_excluded_with_exact_counts() -> None:
    report = enumerate_mutation_candidates(
        _graph(),
        _run_id(),
        CandidateEnumeratorConfig(seed=0),
    )
    exclusions = {item.reason: item.count for item in report.exclusions}

    assert exclusions == {
        "unsupported-kind:kv_head": 1,
        "unsupported-kind:model": 1,
        "unsupported-kind:parameter": 1,
        "unsupported-kind:residual_path": 1,
    }
    assert report.graph_node_count == 13


def test_scope_layer_kind_and_prefix_filters_are_applied_before_sampling() -> None:
    report = enumerate_mutation_candidates(
        _graph(),
        _run_id(),
        CandidateEnumeratorConfig(
            seed=7,
            filters=CandidateFilter(
                scopes=(CandidateScope.ATTENTION_HEAD,),
                include_kinds=("attention_head",),
                include_prefixes=(_id("model.layers"),),
                exclude_prefixes=(_id("model.layers.0"),),
                layer_indices=(1,),
            ),
        ),
    )

    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.scope is CandidateScope.ATTENTION_HEAD
    assert candidate.component_id == _id("model.layers.1.self_attn.head.0")
    assert candidate.layer_index == 1


def test_seeded_limit_is_bounded_repeatable_and_changes_selection() -> None:
    first = enumerate_mutation_candidates(
        _graph(),
        _run_id(),
        CandidateEnumeratorConfig(seed=10, max_candidates=3),
    )
    repeated = enumerate_mutation_candidates(
        _graph(),
        _run_id(),
        CandidateEnumeratorConfig(seed=10, max_candidates=3),
    )
    other_seed = enumerate_mutation_candidates(
        _graph(),
        _run_id(),
        CandidateEnumeratorConfig(seed=11, max_candidates=3),
    )

    assert [item.candidate_id for item in first.candidates] == [
        item.candidate_id for item in repeated.candidates
    ]
    assert len(first.candidates) == 3
    assert first.eligible_count == 8
    assert {item.reason: item.count for item in first.exclusions}["sampled-out"] == 5
    assert [item.candidate_id for item in first.candidates] != [
        item.candidate_id for item in other_seed.candidates
    ]


def test_component_planner_closure_is_preserved_in_candidate() -> None:
    report = enumerate_mutation_candidates(
        _graph(),
        _run_id(),
        CandidateEnumeratorConfig(
            seed=0,
            filters=CandidateFilter(
                scopes=(CandidateScope.COMPONENT,),
                include_prefixes=(_id("model.layers.0.self_attn.q_proj"),),
            ),
        ),
    )

    assert len(report.candidates) == 1
    candidate = report.candidates[0]
    assert candidate.component_id == _id("model.layers.0.self_attn.q_proj")
    assert candidate.affected_components == tuple(
        sorted(
            (
                _id("model.layers.0.input_norm"),
                _id("model.layers.0.self_attn.q_proj"),
            )
        )
    )
    assert candidate.constraint_ids == ("q-proj-norm-coupling",)


def test_missing_logical_metadata_is_counted_not_emitted() -> None:
    graph = ComponentGraph.build(
        (
            GraphNode(_id("model"), "model"),
            GraphNode(
                _id("model.layers.0.self_attn.head.0"),
                "attention_head",
                (("layer_index", 0),),
            ),
        )
    )
    report = enumerate_mutation_candidates(
        graph,
        _run_id(),
        CandidateEnumeratorConfig(seed=0),
    )

    assert not report.candidates
    assert {item.reason: item.count for item in report.exclusions} == {
        "invalid-metadata:head": 1,
        "unsupported-kind:model": 1,
    }


def test_transformer_layer_index_is_derived_from_canonical_path() -> None:
    report = enumerate_mutation_candidates(
        ComponentGraph.build((GraphNode(_id("model.layers.9"), "transformer_layer"),)),
        _run_id(),
        CandidateEnumeratorConfig(seed=0),
    )

    assert report.candidates[0].scope is CandidateScope.TRANSFORMER_LAYER
    assert report.candidates[0].layer_index == 9
    parameters = dict(report.candidates[0].request.parameters)
    assert parameters == {"candidate_scope": "layer", "layer_index": 9}
