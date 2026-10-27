"""Pure ranking tests for the physical-search activation baseline."""

from __future__ import annotations

import math

from modelsurgeon.adapters.huggingface.block_influence import BlockInfluenceScore
from modelsurgeon.experiments.candidates import CandidateScope, MutationCandidate
from modelsurgeon.first_party_optimize_runtime import (
    _activation_magnitude_order,
    _activation_weighted_order,
    _shortgpt_layer_order,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import MutationKind, MutationRequest


def test_activation_magnitude_order_uses_lowest_finite_channel_signal() -> None:
    order = _activation_magnitude_order(
        ("channel-b", "channel-a", "channel-c", "missing"),
        {
            "channel-b": (
                {"name": "activation_abs_mean", "value": 0.2},
                {"name": "activation_abs_mean", "value": 0.4},
            ),
            "channel-a": ({"name": "activation_abs_mean", "value": 0.1},),
            "channel-c": (
                {"name": "activation_abs_mean", "value": -0.5},
                {"name": "activation_abs_mean", "value": 10**10000},
                {"name": "activation_abs_mean", "value": math.nan},
                {"name": "activation_abs_mean", "value": True},
                {"name": "activation_rms", "value": 0.0},
            ),
        },
    )

    assert order == ("channel-a", "channel-b")


def test_activation_weighted_order_combines_weight_and_calibration_signal() -> None:
    order = _activation_weighted_order(
        ("channel-a", "channel-b", "channel-c", "missing"),
        {
            "channel-a": (
                {"name": "weight_l2_norm", "value": 10.0},
                {"name": "activation_rms", "value": 0.1},
            ),
            "channel-b": (
                {"name": "weight_l2_norm", "value": 2.0},
                {"name": "activation_rms", "value": 1.0},
            ),
            "channel-c": (
                {"name": "weight_l2_norm", "value": 10**10000},
                {"name": "weight_l2_norm", "value": math.inf},
                {"name": "activation_rms", "value": 0.0},
            ),
        },
    )

    assert order == ("channel-a", "channel-b")


def test_shortgpt_order_ranks_layer_candidates_by_lowest_block_influence() -> None:
    candidates = tuple(_layer_candidate(layer) for layer in (1, 0, 2))
    scores = (
        BlockInfluenceScore(layer_index=0, score=0.05, token_count=50),
        BlockInfluenceScore(layer_index=1, score=0.01, token_count=50),
        BlockInfluenceScore(layer_index=2, score=0.30, token_count=50),
    )

    assert _shortgpt_layer_order(candidates, scores) == (
        "cand_1",
        "cand_0",
        "cand_2",
    )


def _layer_candidate(layer: int) -> MutationCandidate:
    component = ComponentId.parse(f"model.layers.{layer}")
    request = MutationRequest(
        MutationKind.MASK,
        (component,),
        (("candidate_scope", "transformer_layer"), ("layer_index", layer)),
    )
    return MutationCandidate(
        f"cand_{layer}",
        CandidateScope.TRANSFORMER_LAYER,
        component,
        "transformer_layer",
        layer,
        request,
        (component,),
        (),
    )
