from __future__ import annotations

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
