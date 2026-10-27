from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.surgeon import (
    StructuralAblation,
    StructuralExample,
    StructuralModelError,
    StructuralModelKind,
    StructuralOutcomeStatus,
    StructuralStudyConfig,
    evaluate_structural_models,
)


def _example(index: int, split: str, target: float | None = None) -> StructuralExample:
    return StructuralExample(
        example_id=f"example-{split}-{index}",
        parent_state_id=f"state-{split}-{index}",
        model_revision=f"model-{split}-{index}",
        lineage_group_id=f"lineage-{split}-{index}",
        split=split,
        node_features=((float(index), 1.0), (float(index) + 0.5, 2.0)),
        edges=((0, 1),) if index % 2 else (),
        order_history=(float(index), float(index) + 0.25),
        parameter_count=100 + index,
        target=float(index) if target is None else target,
    )


def test_structural_study_is_bounded_deterministic_and_records_ablations() -> None:
    examples = tuple(
        [_example(index, "train") for index in range(5)]
        + [_example(index, "validation") for index in range(5, 8)]
        + [_example(index, "test") for index in range(8, 12)]
    )
    config = StructuralStudyConfig(bootstrap_repetitions=20, training_steps=20, max_nodes=4)
    first = evaluate_structural_models(examples, config)
    second = evaluate_structural_models(examples, config)

    assert first.to_json() == second.to_json()
    assert {item.kind for item in first.evaluations} == set(StructuralModelKind)
    assert len(first.evaluations[0].predictors) == 5
    assert {item.ablation for item in first.ablations} == set(StructuralAblation)
    assert all(item.resources.deterministic_inference for item in first.evaluations)
    assert first.selected_model is None


def test_structural_study_rejects_leakage_and_censored_training() -> None:
    with pytest.raises(StructuralModelError, match="five unique"):
        StructuralStudyConfig(seeds=(0, 1, 2, 3))

    examples = (
        _example(1, "validation"),
        _example(2, "test"),
        StructuralExample(
            example_id="censored",
            parent_state_id="state-censored",
            model_revision="model-censored",
            lineage_group_id="lineage-censored",
            split="train",
            node_features=((1.0, 0.0),),
            edges=(),
            order_history=(),
            parameter_count=10,
            target=None,
            status=StructuralOutcomeStatus.UNKNOWN,
        ),
    )
    with pytest.raises(StructuralModelError, match="no measured"):
        evaluate_structural_models(examples, StructuralStudyConfig(training_steps=5))


def test_model_selection_uses_validation_not_test_outcomes() -> None:
    def signal_example(index: int, split: str) -> StructuralExample:
        target = float(index % 2)
        return StructuralExample(
            example_id=f"signal-{split}-{index}",
            parent_state_id=f"signal-state-{split}-{index}",
            model_revision=f"signal-model-{split}-{index}",
            lineage_group_id=f"signal-lineage-{split}-{index}",
            split=split,
            node_features=((1.0, 1.0), (1.0, 1.0)),
            edges=((0, 1),) if target else (),
            order_history=(target, target),
            parameter_count=100,
            target=target,
        )

    # These labels isolate the graph/history feature path from pooled node features.
    training = [signal_example(index, "train") for index in range(20)]
    validation = [signal_example(index, "validation") for index in range(12)]
    test = [signal_example(index, "test") for index in range(12)]
    flipped_test = [replace(item, target=1.0 - float(item.target)) for item in test]
    config = StructuralStudyConfig(training_steps=500, bootstrap_repetitions=100, max_nodes=4)

    original = evaluate_structural_models((*training, *validation, *test), config)
    changed_test = evaluate_structural_models((*training, *validation, *flipped_test), config)

    assert original.selected_model is StructuralModelKind.LINEAR_TOPOLOGY_HISTORY
    assert changed_test.selected_model is original.selected_model
    selected = next(item for item in original.evaluations if item.kind is original.selected_model)
    assert selected.validation_metric("mae").value < 0.5
    assert selected.metric("mae").value < 1e-4
    changed = next(
        item for item in changed_test.evaluations if item.kind is changed_test.selected_model
    )
    assert changed.metric("mae").value > 0.5
