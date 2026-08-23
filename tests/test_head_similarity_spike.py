"""Tests for bounded attention-head redundancy metric selection."""

from __future__ import annotations

import pytest

from modelsurgeon.features.head_similarity_spike import (
    HeadSimilarityBudget,
    HeadSimilarityMethod,
    HeadSimilarityProbe,
    HeadSimilaritySpikeError,
    default_head_similarity_probes,
    evaluate_head_similarity_methods,
)


def test_two_small_models_select_output_correlation_with_recorded_costs() -> None:
    report = evaluate_head_similarity_methods(
        default_head_similarity_probes(),
        HeadSimilarityBudget(max_workspace_bytes=4096, max_elapsed_seconds=1.0),
    )

    assert report.models == ("tiny_attention_a", "tiny_attention_b")
    assert len(report.results) == 6
    assert report.recommendation is HeadSimilarityMethod.OUTPUT_CORRELATION
    assert "recommend output_correlation" in report.rationale

    aggregates = {item.method: item for item in report.aggregates}
    output = aggregates[HeadSimilarityMethod.OUTPUT_CORRELATION]
    subspace = aggregates[HeadSimilarityMethod.SUBSPACE_PROJECTION]
    weights = aggregates[HeadSimilarityMethod.WEIGHT_COSINE]

    assert output.predictive_spearman == pytest.approx(1.0)
    assert subspace.predictive_spearman == pytest.approx(1.0)
    assert output.ranking_stability == pytest.approx(1.0)
    assert subspace.ranking_stability == pytest.approx(1.0)
    assert weights.predictive_spearman < output.predictive_spearman
    assert output.max_workspace_bytes < subspace.max_workspace_bytes
    assert output.total_operation_units < subspace.total_operation_units
    assert all(item.total_elapsed_seconds >= 0.0 for item in report.aggregates)
    assert all(item.feasible for item in report.aggregates)

    record = report.to_record()
    assert record["recommendation"] == "output_correlation"
    assert record["models"] == ["tiny_attention_a", "tiny_attention_b"]


def test_tight_workspace_budget_rejects_all_head_metrics() -> None:
    report = evaluate_head_similarity_methods(
        default_head_similarity_probes(),
        HeadSimilarityBudget(max_workspace_bytes=1, max_elapsed_seconds=1.0),
    )

    assert report.recommendation is None
    assert "reject all metrics" in report.rationale
    assert not any(item.feasible for item in report.aggregates)


def test_head_similarity_spike_requires_two_model_probes() -> None:
    first = default_head_similarity_probes()[0]
    with pytest.raises(HeadSimilaritySpikeError, match="requires two model probes"):
        evaluate_head_similarity_methods((first,))

    with pytest.raises(HeadSimilaritySpikeError, match="at least three pairs"):
        HeadSimilarityProbe("bad", first.pairs[:2])
