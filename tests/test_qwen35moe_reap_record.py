from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from modelsurgeon.adapters.huggingface import Qwen35MoeReapError, Qwen35MoeReapResult


def _record() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metric": "mean_router_weighted_expert_output_l2",
        "router_weight_semantics": "qwen35_top_k_weights_renormalized_per_token",
        "scores_by_layer": [[0.25, None], [0.5, 0.75]],
        "routed_counts_by_layer": [[3, 0], [2, 1]],
    }


def test_reap_score_record_round_trips_without_torch() -> None:
    record = _record()

    result = Qwen35MoeReapResult.from_record(record)

    assert result.scores_by_layer == ((0.25, None), (0.5, 0.75))
    assert result.routed_counts_by_layer == ((3, 0), (2, 1))
    assert result.to_record() == record


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 1.0),
        ("schema_version", 2),
        ("metric", "unknown"),
        ("router_weight_semantics", "full_softmax_probabilities"),
        ("scores_by_layer", []),
        ("scores_by_layer", [[0.25, None], [0.5]]),
        ("routed_counts_by_layer", [[3, 0], [2]]),
        ("routed_counts_by_layer", [[3, False], [2, 1]]),
        ("scores_by_layer", [[0.25, 0.5], [0.5, 0.75]]),
        ("scores_by_layer", [[0.25, None], [0.5, None]]),
        ("scores_by_layer", [[0.25, float("nan")], [0.5, 0.75]]),
        ("scores_by_layer", [[0.25, -0.1], [0.5, 0.75]]),
        ("scores_by_layer", [[0.25, 10**400], [0.5, 0.75]]),
        ("scores_by_layer", [[0.25, True], [0.5, 0.75]]),
    ],
)
def test_reap_score_record_rejects_malformed_evidence(field: str, value: object) -> None:
    record = deepcopy(_record())
    record[field] = value

    with pytest.raises(Qwen35MoeReapError):
        Qwen35MoeReapResult.from_record(record)
