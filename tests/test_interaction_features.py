"""Tests for bounded pre-mutation interaction feature extraction."""

from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.datasets import (
    InteractionExample,
    InteractionKind,
    InteractionMetric,
    InteractionOutcome,
    InteractionProvenance,
)
from modelsurgeon.features import (
    InteractionFeatureBudget,
    InteractionFeatureCell,
    InteractionFeatureError,
    InteractionFeatureKind,
    extract_interaction_features,
)
from modelsurgeon.features.schema import FeatureSampleContext
from modelsurgeon.features.topology import TopologyFeatures
from modelsurgeon.graph import ComponentId


def _component(name: str) -> ComponentId:
    return ComponentId.parse(f"model.layers.0.mlp.{name}.weight")


def _interaction(outcome: InteractionOutcome = InteractionOutcome.MEASURED) -> InteractionExample:
    measured = outcome is InteractionOutcome.MEASURED
    metrics = (InteractionMetric("quality_loss", "quality_loss", 0.4),) if measured else ()
    non_additivity = (InteractionMetric("quality_loss", "quality_loss", 0.1),) if measured else ()
    return InteractionExample(
        kind=InteractionKind.ORDERED_PAIR,
        model_family="tiny-family",
        model_revision="revision",
        root_state_id="state_" + "1" * 64,
        parent_state_id="state_" + "1" * 64,
        result_state_id="state_" + "2" * 64,
        source_artifact_digest="a" * 64,
        corpus_id="corpus",
        hardware_profile_id="cpu",
        runtime_id="runtime",
        seed=0,
        mutation_ids=("A", "B"),
        ancestor_state_ids=("state_" + "1" * 64,),
        outcome=outcome,
        metrics=metrics,
        provenance=InteractionProvenance(
            ("benchmark-1",), "protocol", "tool", ("runner",), {"budget": "fixed"}
        ),
        non_additivity_metrics=non_additivity,
        lineage_group_id="lineage",
        reason=None if measured else "second mutation unsupported",
    )


def _context() -> FeatureSampleContext:
    return FeatureSampleContext(
        "heldout-text",
        "revision",
        "train",
        ("sample-1",),
        "preprocess-v1",
        "tokenizer",
        "tokenizer-revision",
    )


def _topology(component_id: ComponentId, position: int) -> TopologyFeatures:
    return TopologyFeatures(
        component_id,
        depth=3,
        position=position,
        normalized_position=float(position),
        sibling_position=position,
        normalized_sibling_position=float(position),
        degree=1,
        in_degree=1,
        out_degree=0,
        coupling_set_size=2,
        shape_roles=("kind:weight",),
        layer_index=0,
        normalized_layer_index=0.0,
    )


def _cell(outcome: InteractionOutcome = InteractionOutcome.MEASURED) -> InteractionFeatureCell:
    left = _component("up_proj")
    right = _component("down_proj")
    return InteractionFeatureCell(
        _interaction(outcome),
        left,
        right,
        _context(),
        left_activation=(1.0, 0.0),
        right_activation=(0.0, 1.0),
        left_gradient=(1.0, 1.0),
        right_gradient=(2.0, 2.0),
        left_topology=_topology(left, 0),
        right_topology=_topology(right, 2),
        redundancy_score=0.75,
    )


def test_interaction_features_are_deterministic_and_retain_targets() -> None:
    record = extract_interaction_features((_cell(),), InteractionFeatureBudget(max_pairs=1))[0]
    values = {item.name: item.value for item in record.values}

    assert record.outcome is InteractionFeatureKind.MEASURED
    assert values["activation_overlap"] == 0.0
    assert values["gradient_overlap"] == pytest.approx(1.0)
    assert values["redundancy_score"] == 0.75
    assert values["topology_distance"] == 4.0
    assert values["cumulative_error_quality_loss"] == 0.1
    assert values["order_sensitive"] == 1.0
    assert record.feature_id == extract_interaction_features((_cell(),))[0].feature_id
    assert record.to_record()["budget"]["pair_generation"] == "explicit_bounded_cells_only"


def test_failed_target_is_explicit_and_missing_pre_mutation_evidence_is_not_zero_filled() -> None:
    cell = replace(
        _cell(InteractionOutcome.FAILED),
        left_activation=None,
        right_activation=None,
        left_gradient=None,
        right_gradient=None,
        left_topology=None,
        right_topology=None,
        redundancy_score=None,
    )

    record = extract_interaction_features((cell,))[0]

    assert record.outcome is InteractionFeatureKind.FAILED
    assert "activation_overlap" in record.missing_features
    assert "cumulative_error" in record.missing_features
    assert all(item.name not in record.missing_features for item in record.values)


def test_post_mutation_and_unbounded_cells_fail_closed() -> None:
    with pytest.raises(InteractionFeatureError, match="pre-mutation"):
        replace(_cell(), source_phase="post_mutation")

    with pytest.raises(InteractionFeatureError, match="pair budget"):
        extract_interaction_features((_cell(), _cell()), InteractionFeatureBudget(max_pairs=1))

    with pytest.raises(InteractionFeatureError, match="RAM"):
        InteractionFeatureBudget(max_pairs=1, block_size=1, max_ram_bytes=1_024)
