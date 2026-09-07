"""Golden and boundary tests for claim-to-evidence explanations."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from modelsurgeon.conversation import (
    CampaignBudget,
    CampaignEvidence,
    CampaignOutcome,
    CampaignSpec,
    EvidenceCursor,
    EvidenceQuery,
    EvidenceQueryEngine,
    EvidenceQueryResponse,
    EvidenceSnapshot,
    direct_evidence_report,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.explain import (
    ClaimEvidenceMeasurementStatus,
    ClaimEvidenceRendererError,
    ClaimEvidenceResourceBounds,
    ClaimEvidenceResourceError,
    ClaimEvidenceStatus,
    render_claim_evidence,
)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()


def _evidence(
    evidence_id: str,
    outcome: CampaignOutcome,
    *,
    decision: str | None = None,
    measured: bool = False,
    inconclusive: bool = False,
) -> CampaignEvidence:
    provenance: dict[str, object] = {
        "record_type": "claim_renderer_fixture",
        "source": "trusted-test",
        "run_id": "run_claim_renderer",
        "plan_id": "plan_claim_renderer",
    }
    if decision is not None:
        provenance["decision"] = decision
    if measured:
        provenance["measurements"] = {
            "quality": {
                "value": 0.97,
                "unit": "score",
                "uncertainty": {
                    "lower_bound": 0.96,
                    "upper_bound": 0.98,
                    "confidence": 0.95,
                    "standard_error": 0.01,
                    "sample_count": 3,
                },
            }
        }
        provenance["observed_at"] = "2026-09-07T12:00:00Z"
    return CampaignEvidence(
        evidence_id,
        "sha256:" + "a" * 64,
        outcome,
        "claim renderer fixture: " + outcome.value,
        provenance,
        artifact_digest="sha256:" + "b" * 64 if decision == "accepted" else None,
        inconclusive=inconclusive,
    )


def _snapshot() -> EvidenceSnapshot:
    spec_payload = {"constraints": [{"metric": "quality", "minimum": 0.95}]}
    spec = CampaignSpec(
        "claim_renderer_contract",
        _digest(spec_payload),
        spec_payload,
        (spec_payload["constraints"][0],),  # type: ignore[arg-type]
    )
    state = new_campaign_state(
        session_id="chat_claim_renderer",
        run_id="run_claim_renderer",
        source_model_digest="sha256:" + "a" * 64,
        spec=spec,
        policy_state={"record_type": "claim_renderer_policy", "revision": "v1"},
        provider_context={"provider_id": "fixture"},
        budget=CampaignBudget(60.0, 4096, 8, 8192),
        campaign_id="campaign_claim_renderer",
    )
    evidence = (
        _evidence(
            "evidence_accepted",
            CampaignOutcome.SUPPORTED,
            decision="accepted",
            measured=True,
        ),
        _evidence(
            "evidence_predicted",
            CampaignOutcome.SUPPORTED,
            decision="accepted",
        ),
        _evidence(
            "evidence_rejected",
            CampaignOutcome.SUPPORTED,
            decision="rejected",
            measured=True,
        ),
        _evidence("evidence_rollback", CampaignOutcome.FAILED, decision="rolled_back"),
        _evidence("evidence_unknown", CampaignOutcome.UNKNOWN),
        _evidence("evidence_inconclusive", CampaignOutcome.UNKNOWN, inconclusive=True),
        _evidence("evidence_unsupported", CampaignOutcome.UNSUPPORTED),
        _evidence("evidence_failed", CampaignOutcome.FAILED),
    )
    state = replace(
        state,
        evidence_cursor=EvidenceCursor(len(evidence), evidence[-1].evidence_id),
        state_version=len(evidence),
    )
    return EvidenceSnapshot(state, evidence)


def _response() -> EvidenceQueryResponse:
    snapshot = _snapshot()
    query = EvidenceQuery(
        snapshot.campaign.campaign_id,
        fields=(
            "detail",
            "measurements",
            "observed_at",
            "outcome",
            "provenance_refs",
            "source_outcome",
            "uncertainty",
        ),
        expected_snapshot_id=snapshot.snapshot_id,
        expected_state_digest=snapshot.state_digest,
    )
    return EvidenceQueryEngine(snapshot).query(query)


def test_golden_statuses_preserve_all_evidence_and_never_promote_predictions() -> None:
    response = _response()
    explanation = render_claim_evidence(response)

    assert tuple(item.evidence_id for item in explanation.claims) == tuple(
        item.evidence_id for item in response.records
    )
    by_id = {item.evidence_id: item for item in explanation.claims}
    assert by_id["evidence_accepted"].status is ClaimEvidenceStatus.MEASURED
    assert by_id["evidence_predicted"].status is ClaimEvidenceStatus.PREDICTED
    assert (
        by_id["evidence_predicted"].measurement_status
        is ClaimEvidenceMeasurementStatus.UNAVAILABLE
    )
    assert by_id["evidence_predicted"].measurements == ()
    assert by_id["evidence_rejected"].status is ClaimEvidenceStatus.REJECTED
    assert by_id["evidence_rollback"].status is ClaimEvidenceStatus.ROLLBACK
    assert by_id["evidence_unknown"].status is ClaimEvidenceStatus.UNKNOWN
    assert by_id["evidence_inconclusive"].status is ClaimEvidenceStatus.INCONCLUSIVE
    assert by_id["evidence_unsupported"].status is ClaimEvidenceStatus.UNSUPPORTED
    assert by_id["evidence_failed"].status is ClaimEvidenceStatus.FAILED
    assert by_id["evidence_accepted"].measurements[0].unit == "score"
    assert by_id["evidence_accepted"].measurements[0].uncertainty is not None
    assert by_id["evidence_predicted"].unavailable

    text = explanation.render_text()
    for label in (
        "MEASURED",
        "PREDICTED",
        "REJECTED",
        "ROLLBACK",
        "UNKNOWN",
        "INCONCLUSIVE",
        "UNSUPPORTED",
        "FAILED",
    ):
        assert f"[{label}]" in text
    assert "quality=0.97 score" in text
    assert "bounds=0.96..0.98" in text
    assert "prediction/decision only" in text
    assert "measurements=unavailable" in text
    assert "run_claim_renderer" in text


def test_text_rendering_matches_golden_fixture() -> None:
    expected = (
        Path(__file__).parent / "golden" / "claim_evidence_renderer_v1.txt"
    ).read_text(encoding="utf-8")
    assert render_claim_evidence(_response()).render_text() + "\n" == expected


def test_unavailable_fields_are_explicit_and_negative_evidence_is_retained() -> None:
    response = _response()
    explanation = render_claim_evidence(response)

    assert "measurements" in explanation.missing_fields
    assert all(item.unavailable_fields == () for item in explanation.claims)
    assert "[UNAVAILABLE] response missing fields=measurements" in explanation.render_text()
    assert len(explanation.claims) == 8


def test_tampered_and_untrusted_inputs_fail_closed() -> None:
    response = _response()
    with pytest.raises(ClaimEvidenceRendererError, match="typed EvidenceQueryResponse"):
        render_claim_evidence({})  # type: ignore[arg-type]
    tampered = replace(
        response,
        records=(replace(response.records[0], detail="tampered"), *response.records[1:]),
    )  # type: ignore[attr-defined]
    with pytest.raises(ClaimEvidenceRendererError, match="canonical report"):
        render_claim_evidence(tampered)


def test_deterministic_replay_and_canonical_report_parity() -> None:
    response = _response()
    restored = EvidenceQueryResponse.from_record(
        response.to_record(),
        snapshot=EvidenceSnapshot.from_record(response.snapshot.to_record()),
    )
    replay = EvidenceQueryEngine(response.snapshot).query(response.query)
    direct = direct_evidence_report(response.snapshot, response.query)
    first = render_claim_evidence(response)
    second = render_claim_evidence(replay)

    assert response.canonical_json() == replay.canonical_json()
    assert response.canonical_json() == direct.canonical_json()
    assert restored.canonical_json() == response.canonical_json()
    assert first.canonical_json() == second.canonical_json()
    assert first.render_text() == second.render_text()
    assert first.response_digest == direct.response_digest
    assert (
        first.source_resource_bounds["campaign_budget"]
        == response.snapshot.campaign.budget.to_record()
    )


def test_renderer_resource_bounds_are_hard() -> None:
    response = _response()
    with pytest.raises(ClaimEvidenceResourceError):
        render_claim_evidence(response, resource_bounds=ClaimEvidenceResourceBounds(max_claims=1))  # type: ignore[arg-type]
