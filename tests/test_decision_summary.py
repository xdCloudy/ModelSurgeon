from __future__ import annotations

import json

import pytest

from modelsurgeon.active_learning.candidate_scoring import CandidateScore
from modelsurgeon.experiments.candidates import CandidateScope, MutationCandidate
from modelsurgeon.explain import (
    AttributionUnavailable,
    DecisionSummaryError,
    QuantizationContext,
    attribute_predictions,
    generate_mutation_decision_summary,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgeon.models import LinearConfig, LinearSurgeonModel
from modelsurgeon.surgery.contracts import MutationDelta, MutationKind, MutationRequest


def _candidate(identifier: str = "cand_example") -> MutationCandidate:
    component = ComponentId.parse("model.layers.0.mlp.channel.3")
    request = MutationRequest(MutationKind.REMOVE, (component,))
    return MutationCandidate(
        identifier,
        CandidateScope.MLP_CHANNEL,
        component,
        "mlp_channel",
        0,
        request,
        (component,),
        (),
    )


def _score(identifier: str = "cand_example") -> CandidateScore:
    return CandidateScore(
        identifier,
        0.7,
        (("latency", -0.2), ("perplexity", 0.01)),
        0.75,
        0.8,
        0.05,
    )


def test_json_and_human_summary_share_values_and_rank_evidence_deterministically() -> None:
    model = LinearSurgeonModel(
        ("num:a", "missing:a", "cat:family=llama"),
        "perplexity",
        (1.0, -4.0, 2.0),
        0.0,
        LinearConfig(),
        1,
    )
    attribution = attribute_predictions(model, ((2.0, 1.0, 1.0),))
    summary = generate_mutation_decision_summary(
        _candidate(),
        _score(),
        expected_delta=MutationDelta(-10, -20, -30, -40),
        attribution=attribution,
        quantization_context=QuantizationContext("gguf", "Q4_K_M", "direct", True),
        top_evidence_count=2,
    )
    record = json.loads(summary.to_json())
    assert record == summary.to_record()
    assert [item["feature_name"] for item in record["attribution"]["top_evidence"]] == [
        "missing:a",
        "cat:family=llama",
    ]
    text = summary.to_text()
    assert "Safe probability: 0.8" in text
    assert "storage_bytes: -40" in text
    assert "codec: Q4_K_M" in text
    assert summary.to_text() == text


def test_unknown_values_are_labeled_and_identity_mismatch_fails() -> None:
    unavailable = AttributionUnavailable("mlp", "not additive")
    summary = generate_mutation_decision_summary(
        _candidate(),
        _score(),
        expected_delta=None,
        attribution=unavailable,
        quantization_context=None,
    )
    assert summary.expected_delta.status == "unknown"
    assert "parameters: unknown" in summary.to_text()
    assert "codec: unknown" in summary.to_text()
    assert summary.to_record()["attribution"]["reason"] == "not additive"  # type: ignore[index]
    with pytest.raises(DecisionSummaryError, match="identities"):
        generate_mutation_decision_summary(
            _candidate(),
            _score("cand_other"),
            expected_delta=None,
            attribution=unavailable,
            quantization_context=None,
        )
