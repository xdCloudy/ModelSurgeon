from __future__ import annotations

from dataclasses import replace

import pytest

from modelsurgeon.datasets import (
    InteractionDataset,
    InteractionDatasetError,
    InteractionExample,
    InteractionKind,
    InteractionMetric,
    InteractionOutcome,
    InteractionProvenance,
    InteractionSplit,
    build_cumulative_interaction,
    build_pairwise_interaction,
)


def _digest(index: int) -> str:
    return f"{index:064x}"


def _metric(quality: float, latency: float) -> tuple[InteractionMetric, ...]:
    return (
        InteractionMetric("latency_seconds", "s", latency),
        InteractionMetric("quality_loss", "quality_loss", quality),
    )


def _provenance(index: int) -> InteractionProvenance:
    return InteractionProvenance(
        (f"benchmark-{index}",),
        "protocol-v1",
        "tool-v1",
        ("runner", "--seed", str(index)),
        {"budget": "fixed"},
    )


def _single(index: int, mutation_id: str, *, result_state: str) -> InteractionExample:
    return InteractionExample(
        InteractionKind.SINGLE,
        "tiny-family",
        "model-revision-train",
        "state_" + "1" * 64,
        "state_" + "1" * 64,
        result_state,
        _digest(10),
        "corpus-train",
        "profile-cpu",
        "runtime",
        0,
        (mutation_id,),
        ("state_" + "1" * 64,),
        InteractionOutcome.MEASURED,
        _metric(0.4 if mutation_id == "A" else 0.3, 1.0 if mutation_id == "A" else 0.8),
        _provenance(index),
        lineage_group_id="lineage-train",
    )


def _pair() -> InteractionExample:
    return InteractionExample(
        InteractionKind.ORDERED_PAIR,
        "tiny-family",
        "model-revision-train",
        "state_" + "1" * 64,
        "state_" + "2" * 64,
        "state_" + "4" * 64,
        _digest(10),
        "corpus-train",
        "profile-cpu",
        "runtime",
        0,
        ("A", "B"),
        ("state_" + "1" * 64, "state_" + "2" * 64),
        InteractionOutcome.MEASURED,
        _metric(0.9, 2.1),
        _provenance(3),
        lineage_group_id="lineage-train",
    )


def test_pairwise_reconciliation_retains_non_additivity_and_order() -> None:
    first = _single(1, "A", result_state="state_" + "2" * 64)
    second = _single(2, "B", result_state="state_" + "3" * 64)
    pair = build_pairwise_interaction(first, second, _pair())

    assert pair.mutation_ids == ("A", "B")
    assert pair.reconciled
    expected = {item.name: item.value for item in pair.expected_additive_metrics}
    non_additive = {item.name: item.value for item in pair.non_additivity_metrics}
    assert expected == {"latency_seconds": 1.8, "quality_loss": 0.7}
    assert non_additive == pytest.approx({"latency_seconds": 0.3, "quality_loss": 0.2})
    assert pair.example_id == build_pairwise_interaction(first, second, _pair()).example_id


def test_cumulative_reconciliation_sums_independent_single_targets() -> None:
    singles = (
        _single(1, "A", result_state="state_" + "2" * 64),
        _single(2, "B", result_state="state_" + "3" * 64),
        _single(3, "C", result_state="state_" + "4" * 64),
    )
    cumulative = replace(
        _pair(),
        kind=InteractionKind.CUMULATIVE,
        mutation_ids=("A", "B", "C"),
        metrics=_metric(1.4, 3.1),
    )

    result = build_cumulative_interaction(singles, cumulative)

    assert result.reconciled
    assert {item.name: item.value for item in result.expected_additive_metrics} == {
        "latency_seconds": 2.6,
        "quality_loss": 1.0,
    }
    assert {item.name: item.value for item in result.non_additivity_metrics} == pytest.approx(
        {"latency_seconds": 0.5, "quality_loss": 0.4}
    )


def test_dataset_retains_failures_and_rejects_cross_split_ancestors() -> None:
    first = _single(1, "A", result_state="state_" + "2" * 64)
    second = _single(2, "B", result_state="state_" + "3" * 64)
    pair = build_pairwise_interaction(first, second, _pair())
    failed = replace(
        second,
        outcome=InteractionOutcome.FAILED,
        metrics=(),
        reason="second mutation unsupported by native codec",
        lineage_group_id="lineage-train",
    )
    validation = replace(
        first,
        model_revision="model-revision-validation",
        source_artifact_digest=_digest(11),
        corpus_id="corpus-validation",
        root_state_id="state_" + "5" * 64,
        parent_state_id="state_" + "5" * 64,
        ancestor_state_ids=("state_" + "5" * 64,),
        lineage_group_id="lineage-validation",
    )
    test = replace(
        validation,
        model_revision="model-revision-test",
        source_artifact_digest=_digest(12),
        corpus_id="corpus-test",
        root_state_id="state_" + "6" * 64,
        parent_state_id="state_" + "6" * 64,
        ancestor_state_ids=("state_" + "6" * 64,),
        lineage_group_id="lineage-test",
    )
    dataset = InteractionDataset(
        (first, second, pair, failed, validation, test),
        InteractionSplit(("lineage-train",), ("lineage-validation",), ("lineage-test",)),
        "interaction-v1",
    )
    assert dataset.to_record()["coverage"] == {
        "examples": 6,
        "outcomes": {"measured": 5, "unsupported": 0, "failed": 1, "unknown": 0},
        "kinds": {"single": 5, "ordered_pair": 1, "cumulative": 0},
        "reconciled_pairs": 1,
    }

    leaking = replace(test, ancestor_state_ids=first.ancestor_state_ids)
    with pytest.raises(InteractionDatasetError, match="leakage"):
        InteractionDataset(
            (first, second, pair, failed, validation, leaking),
            InteractionSplit(("lineage-train",), ("lineage-validation",), ("lineage-test",)),
            "interaction-v1",
        )
